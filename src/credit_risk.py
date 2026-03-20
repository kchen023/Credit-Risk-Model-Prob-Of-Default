"""
Credit Risk Model - Core Functions
===================================
Pure functions extracted from the notebook to enable unit testing.
Covers three modelling approaches:
  - Reduced-form logistic regression (simulated data)
  - Reduced-form logistic regression (UCI bankruptcy data)
  - Structural Merton (1974) Distance-to-Default via Monte Carlo
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.stats.mstats import winsorize


# ---------------------------------------------------------------------------
# Market-Implied PD (Credit Triangle)
# ---------------------------------------------------------------------------

def get_market_implied_pd(bond_yield: float, risk_free_rate: float,
                          recovery_rate: float = 0.40) -> float:
    """
    Calculate market-implied probability of default from bond spreads.

    Uses the simplified credit triangle formula:
        PD = Spread / (1 - Recovery Rate)

    Args:
        bond_yield:     YTM of the corporate bond (e.g. 0.065 for 6.5%).
        risk_free_rate: Yield on the matching risk-free benchmark.
        recovery_rate:  Assumed recovery in default (0 ≤ R < 1). Default 40%.

    Returns:
        Implied PD clamped to [0, 1].  Negative spreads return 0.

    Raises:
        ValueError: If recovery_rate is not in [0, 1).
    """
    if not (0.0 <= recovery_rate < 1.0):
        raise ValueError(
            f"recovery_rate must be in [0, 1), got {recovery_rate}"
        )
    spread = bond_yield - risk_free_rate
    implied_pd = spread / (1.0 - recovery_rate)
    return max(0.0, implied_pd)


# ---------------------------------------------------------------------------
# Mathematical helpers
# ---------------------------------------------------------------------------

def sigmoid(z: np.ndarray) -> np.ndarray:
    """Logistic sigmoid: maps any real value to (0, 1)."""
    return 1.0 / (1.0 + np.exp(-z))


def compute_logit_score(features: np.ndarray, coefficients: np.ndarray,
                        intercept: float = 0.0) -> np.ndarray:
    """
    Compute linear combination z = intercept + features @ coefficients.

    Args:
        features:     Shape (n,) or (n, k) array of feature values.
        coefficients: Shape (k,) array of model coefficients.
        intercept:    Scalar intercept / bias term.

    Returns:
        Shape (n,) array of logit scores.
    """
    return intercept + np.dot(features, coefficients)


# ---------------------------------------------------------------------------
# Synthetic data generation (Method 1)
# ---------------------------------------------------------------------------

def generate_synthetic_default_data(n_samples: int = 2000,
                                    seed: int = 101) -> pd.DataFrame:
    """
    Generate a synthetic credit-risk dataset with known ground-truth
    coefficients.  Mirrors the "God Mode" simulation in the notebook.

    Ground-truth logit:
        z = 4·leverage − 6·profitability − 0.8·coverage + 2.5·volatility − 1.5

    Args:
        n_samples: Number of observations.
        seed:      NumPy random seed for reproducibility.

    Returns:
        DataFrame with columns: leverage, profitability, coverage,
        volatility, default (0/1).
    """
    np.random.seed(seed)

    data = {
        "leverage":      np.random.normal(0.40, 0.15, n_samples),
        "profitability": np.random.normal(0.10, 0.05, n_samples),
        "coverage":      np.random.lognormal(1.0, 0.5, n_samples),
        "volatility":    np.random.normal(0.30, 0.10, n_samples),
    }
    df = pd.DataFrame(data)

    logit_z = (
        4.0 * df["leverage"]
        - 6.0 * df["profitability"]
        - 0.8 * df["coverage"]
        + 2.5 * df["volatility"]
        - 1.5
    )
    prob = sigmoid(logit_z)
    df["default"] = (np.random.rand(n_samples) < prob).astype(int)
    return df


# ---------------------------------------------------------------------------
# Data pre-processing (Method 2)
# ---------------------------------------------------------------------------

def preprocess_features(df: pd.DataFrame, feature_cols: list,
                        winsorize_limits: tuple = (0.01, 0.01)
                        ) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Winsorize extreme values then z-score standardise the given columns.

    Args:
        df:               Input DataFrame.
        feature_cols:     Column names to transform.
        winsorize_limits: (lower, upper) tail fractions to winsorise.

    Returns:
        (transformed_df, fitted_scaler) – the DataFrame is a copy; the
        scaler can be applied to new data at prediction time.
    """
    df_out = df.copy()
    for col in feature_cols:
        df_out[col] = winsorize(df_out[col], limits=list(winsorize_limits))

    scaler = StandardScaler()
    df_out[feature_cols] = scaler.fit_transform(df_out[feature_cols])
    return df_out, scaler


# ---------------------------------------------------------------------------
# Merton structural model (Method 3)
# ---------------------------------------------------------------------------

def compute_asset_parameters(market_cap: float, total_debt: float,
                              sigma_equity: float) -> tuple[float, float]:
    """
    Estimate Merton asset value and asset volatility from observable equity
    metrics (naïve approximation).

        V_A    = Market Cap + Total Debt
        σ_A    = σ_E × (E / V_A)

    Args:
        market_cap:   Equity market capitalisation.
        total_debt:   Book value of total debt.
        sigma_equity: Annualised equity return volatility.

    Returns:
        (asset_value, asset_volatility)

    Raises:
        ValueError: If market_cap or total_debt is negative.
    """
    if market_cap < 0:
        raise ValueError("market_cap must be non-negative")
    if total_debt < 0:
        raise ValueError("total_debt must be non-negative")

    V_A = market_cap + total_debt
    if V_A == 0:
        raise ValueError("Combined asset value is zero; cannot compute σ_A")
    sigma_A = sigma_equity * (market_cap / V_A)
    return V_A, sigma_A


def run_merton_simulation(V_A: float, sigma_A: float, r: float,
                          T: float = 1.0, num_sims: int = 10_000,
                          num_steps: int = 252,
                          seed: int = 42) -> np.ndarray:
    """
    Simulate Geometric Brownian Motion paths for firm asset value.

        dV = r·V·dt + σ_A·V·dW   (risk-neutral measure)

    Args:
        V_A:       Initial asset value.
        sigma_A:   Asset volatility (annualised).
        r:         Risk-free rate (annualised).
        T:         Time horizon in years.
        num_sims:  Number of Monte Carlo paths.
        num_steps: Number of time steps (252 ≈ trading days in a year).
        seed:      NumPy random seed.

    Returns:
        Array of shape (num_steps + 1, num_sims) containing all paths.
    """
    np.random.seed(seed)
    dt = T / num_steps

    paths = np.zeros((num_steps + 1, num_sims))
    paths[0] = V_A

    for t in range(1, num_steps + 1):
        z = np.random.standard_normal(num_sims)
        drift = (r - 0.5 * sigma_A ** 2) * dt
        shock = sigma_A * np.sqrt(dt) * z
        paths[t] = paths[t - 1] * np.exp(drift + shock)

    return paths


def compute_structural_pd(ending_values: np.ndarray,
                          default_threshold: float) -> float:
    """
    Estimate probability of default as the fraction of simulated paths
    whose terminal asset value falls below the default threshold (total debt).

    Args:
        ending_values:     1-D array of terminal asset values.
        default_threshold: Threshold below which default is declared.

    Returns:
        Estimated probability of default in [0, 1].
    """
    return float(np.mean(ending_values < default_threshold))
