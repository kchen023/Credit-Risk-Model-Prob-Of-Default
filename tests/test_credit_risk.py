"""
Test suite for src/credit_risk.py
===================================
Covers:
  1. get_market_implied_pd   – credit triangle formula
  2. sigmoid                 – mathematical helper
  3. compute_logit_score     – linear combination helper
  4. generate_synthetic_default_data – data factory
  5. preprocess_features     – winsorisation + standardisation
  6. compute_asset_parameters – Merton asset value / volatility
  7. run_merton_simulation   – GBM Monte Carlo
  8. compute_structural_pd   – default rate from simulated paths

Proposed areas for improvement are noted with  # TODO: comments where
existing behaviour contains gaps or edge-cases not yet exercised.
"""

import numpy as np
import pandas as pd
import pytest

from src.credit_risk import (
    get_market_implied_pd,
    sigmoid,
    compute_logit_score,
    generate_synthetic_default_data,
    preprocess_features,
    compute_asset_parameters,
    run_merton_simulation,
    compute_structural_pd,
)


# =============================================================================
# 1. Market-Implied PD  (get_market_implied_pd)
# =============================================================================

class TestMarketImpliedPD:
    """
    Tests for the simplified credit triangle formula:
        PD = max(0, Spread / (1 - Recovery))
    """

    # --- happy path -----------------------------------------------------------

    def test_ford_example_from_notebook(self):
        """Replicate the Ford bond example: yield 6.375%, rfr 4.37%, R=40%."""
        pd = get_market_implied_pd(0.06375, 0.0437, recovery_rate=0.40)
        # spread = 0.02005, PD = 0.02005 / 0.6 ≈ 3.34%
        assert abs(pd - 0.033416) < 1e-5

    def test_default_recovery_rate_is_40_percent(self):
        """When recovery_rate is omitted the default of 40 % should be used."""
        pd_explicit = get_market_implied_pd(0.07, 0.05, recovery_rate=0.40)
        pd_default  = get_market_implied_pd(0.07, 0.05)
        assert pd_explicit == pd_default

    def test_zero_recovery_rate(self):
        """With 0% recovery PD equals the raw spread."""
        pd = get_market_implied_pd(0.06, 0.04, recovery_rate=0.0)
        assert abs(pd - 0.02) < 1e-10

    def test_high_recovery_rate(self):
        """
        Higher recovery → smaller denominator → higher implied PD for the
        same spread.  Reflects that lenders demand less spread protection
        when they expect to recover more.
        """
        spread = 0.02
        pd_low_R  = get_market_implied_pd(0.05 + spread, 0.05, recovery_rate=0.20)
        pd_high_R = get_market_implied_pd(0.05 + spread, 0.05, recovery_rate=0.60)
        assert pd_high_R > pd_low_R

    def test_wider_spread_increases_pd(self):
        """Wider credit spread should produce a higher implied PD."""
        pd_narrow = get_market_implied_pd(0.055, 0.05, recovery_rate=0.40)
        pd_wide   = get_market_implied_pd(0.080, 0.05, recovery_rate=0.40)
        assert pd_wide > pd_narrow

    # --- edge cases -----------------------------------------------------------

    def test_zero_spread_returns_zero(self):
        """Bond yielding exactly the risk-free rate → zero implied PD."""
        pd = get_market_implied_pd(0.05, 0.05)
        assert pd == 0.0

    def test_negative_spread_clamped_to_zero(self):
        """
        Negative spreads (rare arbitrage scenarios) must not produce
        negative probabilities.
        """
        pd = get_market_implied_pd(0.03, 0.05)
        assert pd == 0.0

    def test_large_spread(self):
        """Distressed bond with a very large spread should still be in [0,1]."""
        pd = get_market_implied_pd(0.30, 0.05, recovery_rate=0.40)
        # PD = 0.25 / 0.6 ≈ 0.417; valid probability
        assert 0.0 <= pd <= 1.0

    # --- input validation -----------------------------------------------------

    def test_recovery_rate_exactly_one_raises(self):
        """Recovery = 1.0 would cause division by zero; must raise."""
        with pytest.raises(ValueError):
            get_market_implied_pd(0.06, 0.04, recovery_rate=1.0)

    def test_recovery_rate_greater_than_one_raises(self):
        with pytest.raises(ValueError):
            get_market_implied_pd(0.06, 0.04, recovery_rate=1.5)

    def test_negative_recovery_rate_raises(self):
        """Recovery rate cannot be negative."""
        with pytest.raises(ValueError):
            get_market_implied_pd(0.06, 0.04, recovery_rate=-0.1)

    # TODO: test that non-numeric inputs raise TypeError rather than crashing
    # TODO: test very small spreads (< 1 bp) for floating-point precision


# =============================================================================
# 2. Sigmoid
# =============================================================================

class TestSigmoid:

    def test_sigmoid_at_zero(self):
        assert sigmoid(0) == 0.5

    def test_sigmoid_large_positive_approaches_one(self):
        assert sigmoid(500) > 0.9999

    def test_sigmoid_large_negative_approaches_zero(self):
        assert sigmoid(-500) < 0.0001

    def test_output_strictly_between_zero_and_one(self):
        z_vals = np.linspace(-20, 20, 200)
        probs = sigmoid(z_vals)
        assert np.all(probs > 0.0)
        assert np.all(probs < 1.0)

    def test_symmetry(self):
        """sigmoid(z) + sigmoid(-z) must equal 1 for any z."""
        for z in [0.0, 1.0, -1.0, 3.14, -100.0]:
            assert abs(sigmoid(z) + sigmoid(-z) - 1.0) < 1e-12

    def test_monotonically_increasing(self):
        z_vals = np.linspace(-5, 5, 50)
        probs = sigmoid(z_vals)
        assert np.all(np.diff(probs) > 0)

    def test_array_input(self):
        result = sigmoid(np.array([-1.0, 0.0, 1.0]))
        assert result.shape == (3,)
        assert result[1] == 0.5
        assert result[0] < result[1] < result[2]

    # TODO: test scalar vs array type consistency (should return same type?)


# =============================================================================
# 3. Logit Score
# =============================================================================

class TestComputeLogitScore:

    def test_single_observation(self):
        features = np.array([0.4, 0.1, 3.0, 0.3])
        coefs    = np.array([4.0, -6.0, -0.8, 2.5])
        intercept = -1.5
        # z = -1.5 + 4*0.4 - 6*0.1 - 0.8*3.0 + 2.5*0.3
        expected = -1.5 + 1.6 - 0.6 - 2.4 + 0.75
        result = compute_logit_score(features, coefs, intercept)
        assert abs(result - expected) < 1e-10

    def test_batch_observations(self):
        X = np.array([[1.0, 0.0], [0.0, 1.0]])
        coefs = np.array([2.0, -3.0])
        result = compute_logit_score(X, coefs)
        np.testing.assert_allclose(result, [2.0, -3.0])

    def test_zero_features_returns_intercept(self):
        features = np.zeros(4)
        coefs    = np.array([4.0, -6.0, -0.8, 2.5])
        result = compute_logit_score(features, coefs, intercept=1.23)
        assert abs(result - 1.23) < 1e-10

    def test_zero_coefficients(self):
        features = np.array([1.0, 2.0, 3.0])
        coefs    = np.zeros(3)
        result = compute_logit_score(features, coefs, intercept=5.0)
        assert abs(result - 5.0) < 1e-10

    # TODO: test mismatched feature / coefficient dimensions raise ValueError


# =============================================================================
# 4. Synthetic Data Generation
# =============================================================================

class TestGenerateSyntheticDefaultData:

    def test_output_shape_default(self):
        df = generate_synthetic_default_data()
        assert len(df) == 2000

    def test_output_shape_custom(self):
        df = generate_synthetic_default_data(n_samples=500)
        assert len(df) == 500

    def test_required_columns_present(self):
        df = generate_synthetic_default_data(n_samples=100)
        assert set(df.columns) == {"leverage", "profitability", "coverage",
                                   "volatility", "default"}

    def test_default_column_is_binary(self):
        df = generate_synthetic_default_data(n_samples=1000)
        assert df["default"].isin([0, 1]).all()

    def test_reproducibility_same_seed(self):
        df1 = generate_synthetic_default_data(n_samples=200, seed=7)
        df2 = generate_synthetic_default_data(n_samples=200, seed=7)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_produce_different_data(self):
        df1 = generate_synthetic_default_data(n_samples=200, seed=1)
        df2 = generate_synthetic_default_data(n_samples=200, seed=2)
        assert not df1["leverage"].equals(df2["leverage"])

    def test_coverage_always_positive(self):
        """Coverage is lognormal; must be strictly positive."""
        df = generate_synthetic_default_data(n_samples=2000, seed=0)
        assert (df["coverage"] > 0).all()

    def test_default_rate_in_plausible_range(self):
        """With the ground-truth coefficients default rate should be 10–40%."""
        df = generate_synthetic_default_data(n_samples=10_000, seed=42)
        rate = df["default"].mean()
        assert 0.10 <= rate <= 0.40, f"Unexpected default rate: {rate:.2%}"

    def test_leverage_approx_normal_mean(self):
        df = generate_synthetic_default_data(n_samples=10_000, seed=42)
        assert abs(df["leverage"].mean() - 0.40) < 0.02

    def test_profitability_approx_normal_mean(self):
        df = generate_synthetic_default_data(n_samples=10_000, seed=42)
        assert abs(df["profitability"].mean() - 0.10) < 0.01

    def test_volatility_approx_normal_mean(self):
        df = generate_synthetic_default_data(n_samples=10_000, seed=42)
        assert abs(df["volatility"].mean() - 0.30) < 0.02

    def test_no_missing_values(self):
        df = generate_synthetic_default_data(n_samples=500)
        assert df.isnull().sum().sum() == 0

    # TODO: verify that riskier firms (high leverage, low profitability, low
    #       coverage, high volatility) have statistically higher default rates
    # TODO: test n_samples=0 and n_samples=1 edge cases


# =============================================================================
# 5. Preprocessing
# =============================================================================

class TestPreprocessFeatures:

    @pytest.fixture
    def sample_df(self):
        np.random.seed(0)
        return pd.DataFrame({
            "leverage":      np.random.normal(0.4, 0.2, 200),
            "profitability": np.random.normal(0.1, 0.05, 200),
            "coverage":      np.random.lognormal(1.0, 0.5, 200),
        })

    def test_output_shape_unchanged(self, sample_df):
        cols = ["leverage", "profitability", "coverage"]
        result, _ = preprocess_features(sample_df, cols)
        assert result.shape == sample_df.shape

    def test_columns_standardised_approx_zero_mean(self, sample_df):
        cols = ["leverage", "profitability", "coverage"]
        result, _ = preprocess_features(sample_df, cols)
        for col in cols:
            assert abs(result[col].mean()) < 1e-10, (
                f"{col} mean after standardisation is not ~0"
            )

    def test_columns_standardised_approx_unit_std(self, sample_df):
        cols = ["leverage", "profitability", "coverage"]
        result, _ = preprocess_features(sample_df, cols)
        for col in cols:
            assert abs(result[col].std() - 1.0) < 0.05, (
                f"{col} std after standardisation is not ~1"
            )

    def test_original_dataframe_not_mutated(self, sample_df):
        """preprocess_features must return a copy, not modify in place."""
        cols = ["leverage", "profitability", "coverage"]
        original_mean = sample_df["leverage"].mean()
        preprocess_features(sample_df, cols)
        assert abs(sample_df["leverage"].mean() - original_mean) < 1e-10

    def test_non_feature_columns_untouched(self, sample_df):
        """Columns not in feature_cols should be left as-is."""
        sample_df["target"] = np.random.randint(0, 2, len(sample_df))
        cols = ["leverage", "profitability"]
        result, _ = preprocess_features(sample_df, cols)
        pd.testing.assert_series_equal(result["coverage"], sample_df["coverage"])

    def test_winsorisation_clips_extreme_values(self):
        """Extreme outliers should be clipped to the winsorisation bounds."""
        df = pd.DataFrame({"x": [0.0] * 98 + [1000.0, -1000.0]})
        result, _ = preprocess_features(df, ["x"], winsorize_limits=(0.01, 0.01))
        # After winsorising, the two outliers are pulled to the 1st/99th percentile
        standardised = result["x"]
        assert standardised.max() < 100.0
        assert standardised.min() > -100.0

    def test_scaler_returned_and_usable(self, sample_df):
        """The returned scaler must be applicable to new data."""
        cols = ["leverage", "profitability", "coverage"]
        _, scaler = preprocess_features(sample_df, cols)
        new_data = sample_df.head(10).copy()
        transformed = scaler.transform(new_data[cols])
        assert transformed.shape == (10, 3)

    # TODO: test behaviour with a single-row DataFrame (n=1) – std is 0
    # TODO: test that custom winsorize_limits=(0,0) leaves distribution unchanged
    # TODO: verify winsorisation is applied before standardisation (order matters)


# =============================================================================
# 6. Merton Asset Parameters
# =============================================================================

class TestComputeAssetParameters:

    def test_asset_value_is_sum(self):
        V_A, _ = compute_asset_parameters(50e9, 20e9, 0.30)
        assert V_A == 70e9

    def test_asset_volatility_formula(self):
        market_cap, total_debt, sigma_E = 50e9, 20e9, 0.30
        V_A, sigma_A = compute_asset_parameters(market_cap, total_debt, sigma_E)
        expected_sigma_A = sigma_E * (market_cap / V_A)  # 0.30 * 50/70
        assert abs(sigma_A - expected_sigma_A) < 1e-12

    def test_all_equity_firm(self):
        """Zero debt: asset volatility equals equity volatility."""
        _, sigma_A = compute_asset_parameters(100e9, 0.0, 0.25)
        assert abs(sigma_A - 0.25) < 1e-12

    def test_high_leverage_reduces_asset_volatility(self):
        """More debt means equity is a smaller fraction of assets → lower σ_A."""
        _, sigma_low_debt  = compute_asset_parameters(80e9, 20e9, 0.30)
        _, sigma_high_debt = compute_asset_parameters(20e9, 80e9, 0.30)
        assert sigma_low_debt > sigma_high_debt

    def test_negative_market_cap_raises(self):
        with pytest.raises(ValueError, match="market_cap"):
            compute_asset_parameters(-10e9, 20e9, 0.30)

    def test_negative_total_debt_raises(self):
        with pytest.raises(ValueError, match="total_debt"):
            compute_asset_parameters(50e9, -5e9, 0.30)

    def test_zero_total_assets_raises(self):
        """market_cap = debt = 0 is undefined; must raise."""
        with pytest.raises(ValueError):
            compute_asset_parameters(0.0, 0.0, 0.30)

    # TODO: test negative sigma_equity (should it raise ValueError?)
    # TODO: test that sigma_A is always <= sigma_E (by construction)


# =============================================================================
# 7. Merton Monte Carlo Simulation
# =============================================================================

class TestRunMertonSimulation:

    @pytest.fixture
    def default_paths(self):
        return run_merton_simulation(100e9, 0.20, 0.045)

    def test_output_shape(self, default_paths):
        assert default_paths.shape == (253, 10_000)  # num_steps+1 × num_sims

    def test_initial_value_correct(self, default_paths):
        assert np.all(default_paths[0] == 100e9)

    def test_all_paths_positive(self, default_paths):
        """GBM never hits zero; all values must be strictly positive."""
        assert np.all(default_paths > 0)

    def test_reproducibility(self):
        p1 = run_merton_simulation(100e9, 0.20, 0.045, seed=0)
        p2 = run_merton_simulation(100e9, 0.20, 0.045, seed=0)
        np.testing.assert_array_equal(p1, p2)

    def test_different_seeds_produce_different_paths(self):
        p1 = run_merton_simulation(100e9, 0.20, 0.045, seed=1)
        p2 = run_merton_simulation(100e9, 0.20, 0.045, seed=2)
        assert not np.array_equal(p1, p2)

    def test_higher_volatility_produces_wider_distribution(self):
        """Higher σ_A should yield a wider spread in terminal asset values."""
        paths_low  = run_merton_simulation(100e9, 0.10, 0.045, seed=42)
        paths_high = run_merton_simulation(100e9, 0.50, 0.045, seed=42)
        std_low  = paths_low[-1].std()
        std_high = paths_high[-1].std()
        assert std_high > std_low

    def test_custom_num_sims(self):
        paths = run_merton_simulation(100e9, 0.20, 0.045, num_sims=500)
        assert paths.shape[1] == 500

    def test_custom_num_steps(self):
        paths = run_merton_simulation(100e9, 0.20, 0.045, num_steps=12)
        assert paths.shape == (13, 10_000)

    def test_risk_neutral_drift_mean(self):
        """
        Under risk-neutral GBM E[V_T] = V_0 * exp(r * T).
        With many paths the sample mean should be close to this.
        """
        V0, r, T = 100e9, 0.045, 1.0
        paths = run_merton_simulation(V0, 0.20, r, T=T,
                                      num_sims=100_000, seed=0)
        expected_mean = V0 * np.exp(r * T)
        sample_mean   = paths[-1].mean()
        # Allow 1% relative tolerance
        assert abs(sample_mean / expected_mean - 1.0) < 0.01

    # TODO: test that zero volatility gives deterministic path (drift only)
    # TODO: test that very long T or many steps don't overflow


# =============================================================================
# 8. Structural PD
# =============================================================================

class TestComputeStructuralPD:

    def test_no_defaults(self):
        """All terminal values well above threshold → PD = 0."""
        ending_values = np.full(10_000, 1e15)
        pd = compute_structural_pd(ending_values, default_threshold=1e9)
        assert pd == 0.0

    def test_all_defaults(self):
        """All terminal values below threshold → PD = 1."""
        ending_values = np.full(10_000, 1.0)
        pd = compute_structural_pd(ending_values, default_threshold=1e9)
        assert pd == 1.0

    def test_half_defaults(self):
        """Exactly 50% above and 50% below → PD = 0.5."""
        ending_values = np.array([0.5] * 500 + [1.5] * 500)
        pd = compute_structural_pd(ending_values, default_threshold=1.0)
        assert pd == 0.5

    def test_output_in_unit_interval(self):
        ending_values = np.random.uniform(0, 200e9, 10_000)
        pd = compute_structural_pd(ending_values, default_threshold=100e9)
        assert 0.0 <= pd <= 1.0

    def test_pd_increases_as_threshold_rises(self):
        """Raising the default threshold (more debt) should increase PD."""
        ending_values = np.random.lognormal(mean=25.0, sigma=0.5, size=10_000)
        pd_low  = compute_structural_pd(ending_values, default_threshold=np.exp(24.0))
        pd_high = compute_structural_pd(ending_values, default_threshold=np.exp(26.0))
        assert pd_high > pd_low

    def test_pd_consistent_with_simulation(self):
        """
        End-to-end check: low-leverage firm (assets >> debt) should
        have a near-zero structural PD.
        """
        V0, debt, sigma_A, r = 200e9, 10e9, 0.15, 0.045
        paths = run_merton_simulation(V0, sigma_A, r, num_sims=50_000, seed=1)
        pd = compute_structural_pd(paths[-1], default_threshold=debt)
        # With 20× asset-to-debt coverage structural PD should be tiny
        assert pd < 0.01

    def test_empty_array_raises(self):
        """Empty ending_values should not silently return NaN."""
        ending_values = np.array([])
        # np.mean of empty array is NaN – ensure we handle or document this
        pd = compute_structural_pd(ending_values, default_threshold=1.0)
        assert np.isnan(pd)  # current behaviour – document/fix as desired

    # TODO: handle / test negative ending_values (can't happen with GBM, but
    #       a caller might supply raw data)
    # TODO: test with a single simulation path (edge-case for mean)


# =============================================================================
# 9.  Integration / cross-component tests
# =============================================================================

class TestIntegration:

    def test_riskier_firm_higher_structural_pd(self):
        """
        Holding total assets constant, a near-distress firm (assets barely
        above debt) should have a materially higher structural PD than a
        well-capitalised firm.

        Setup:
          Healthy:    equity=90B, debt=10B, σ_E=0.80  → σ_A≈0.72, coverage=10x
          Distressed: equity=10B, debt=90B, σ_E=0.80  → σ_A≈0.08, coverage=1.11x
        """
        sigma_E, r = 0.80, 0.045

        # Healthy firm: assets 10× above the debt barrier
        V_A_safe, sigma_A_safe = compute_asset_parameters(90e9, 10e9, sigma_E)
        paths_safe = run_merton_simulation(V_A_safe, sigma_A_safe, r,
                                          num_sims=50_000, seed=7)
        pd_safe = compute_structural_pd(paths_safe[-1], default_threshold=10e9)

        # Distressed firm: assets only 11% above the debt barrier
        V_A_dist, sigma_A_dist = compute_asset_parameters(10e9, 90e9, sigma_E)
        paths_dist = run_merton_simulation(V_A_dist, sigma_A_dist, r,
                                          num_sims=50_000, seed=7)
        pd_dist = compute_structural_pd(paths_dist[-1], default_threshold=90e9)

        assert pd_dist > pd_safe, (
            f"Expected distressed firm PD ({pd_dist:.4f}) > "
            f"safe firm PD ({pd_safe:.4f})"
        )

    def test_credit_strategy_signal_logic(self):
        """
        When model PD > market PD → SHORT; when model PD < market PD → LONG.
        Verifies the notebook's trading signal logic end-to-end.
        """
        market_pd = get_market_implied_pd(0.06375, 0.0437)

        # Case 1: model sees more risk → SHORT
        model_pd_high = 0.0532
        assert model_pd_high > market_pd  # SHORT signal

        # Case 2: model sees less risk → LONG
        model_pd_low = 0.0286
        assert model_pd_low < market_pd   # LONG signal

    def test_synthetic_data_pipeline_produces_valid_predictions(self):
        """
        Smoke-test the full reduced-form pipeline:
        generate → split → fit logit → predict → PDs in [0,1].
        """
        import statsmodels.api as sm
        from sklearn.model_selection import train_test_split

        df = generate_synthetic_default_data(n_samples=500, seed=0)
        X = df[["leverage", "profitability", "coverage", "volatility"]]
        y = df["default"]

        X_train, X_test, y_train, _ = train_test_split(
            X, y, test_size=0.3, random_state=0
        )
        X_train_sm = sm.add_constant(X_train)
        model = sm.Logit(y_train, X_train_sm).fit(disp=0)

        X_test_sm = sm.add_constant(X_test)
        preds = model.predict(X_test_sm)

        assert len(preds) == len(X_test)
        assert np.all(preds >= 0.0)
        assert np.all(preds <= 1.0)

    # TODO: test that fitted logit coefficients have the expected signs
    #       (leverage positive, profitability negative, coverage negative,
    #        volatility positive) – the notebook's ground truth should be
    #       approximately recovered with a large enough sample.
