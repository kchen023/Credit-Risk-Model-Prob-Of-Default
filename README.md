# Bottom-Up Credit Strategy: Reduced Form Credit Model & Structural Credit Model
### Project Overview
This repository implements a complete exercise on **credit risk modeling** on **probability of default** and derive **long/short strategy** based on model results vs. market-implied result. 

Explictly in the industry, **bottom-up credit strategy** defines the universe of eligible bonds, then categorize the eligible universe into different groups, and then relative value analysis is employed to find bonds that are relatively undervalued or overvalued. 

In the relative value analysis family, two typical methodologies include **reduced form credit model** and **structural credit model**, where **reduced form models** use both the observed **company-specific variables** (such as financial ratios and recovery assumptions) and the **macroeconomic variables** (such as economic growth and market volatility) - where probability of default is exogenous and estimated from **regression**, while **structural model** use **Monte Carlo simulation** along with **economic theory** as well as **market information** to derive paths of company's financial conditions, and then exogenously calculate the probability of default by comparing asset value with liability value. 

My analysis is conducted on both 
**(1) simulated datasets** (for theoretical validation) and 
**(2) real data of publicly traded companies** (for empirical calibration), demonstrating a complete workflow from raw data ingestion to signal generation.

## 📌 Project Overview
This repository implements a dual-framework approach to quantitative credit risk modeling, designed to replicate the rigorous standards of **credit strategy in the industry**. By combining **econometric modeling** (Reduced Form) with **Monte Carlo simulation** (Structural Models), this project generates tradable credit signals (Long/Short) by identifying discrepancies between fundamental default risk and market-implied pricing.

The analysis is conducted on both **simulated datasets** (for theoretical validation) and **real data of publicly traded companies** (for empirical calibration), demonstrating a complete workflow from raw data ingestion to signal generation.

---

## 1. The Modeling Framework

### 🔹 Part A: Reduced Form Model (Logistic Regression)
* **Objective:** To estimate the probability of default (PD) by analyzing **financial ratios** derived from **financial statement analysis**.
* **Model Specification:** The probability of default, $P(Y=1)$, is modeled as a function of exogenous covariates using the Logistic function:

$$
P(Default|X) = \frac{1}{1 + e^{-(\beta_0 + \beta_{Lev}X_{Lev} + \beta_{Prof}X_{Prof} + \dots )}}
$$

  Where $\beta$ coefficients are estimated via **Maximum Likelihood Estimation (MLE)** to maximize the log-likelihood function of the observed default events.

* **Data Complexity & Engineering:**
    * Addressed **data complexity** in the UCI Corporate Bankruptcy dataset (real industry data) through rigorous preprocessing.
    * Applied **Winsorization** to mitigate the impact of outliers common in corporate accounting data.
    * Performed **Standardization** (Z-score scaling) to allow for direct comparison of coefficient sensitivity across variables with vastly different scales.

> **Key Finding:** Empirical results revealed that **Financial Ratios - Leverage, Profitability, Coverage**, and **Volatility** often serve as a stronger predictor of corporate survival than pure Leverage metrics in specific market regimes.

### 🔹 Part B: Structural Model (Merton 1974)
* **Objective:** To estimate the "Distance-to-Default" by viewing firm equity as a **Call Option** on its underlying assets.
* **Model Dynamics (GBM):** The unobservable Asset Value ($V_t$) is modeled as a **Geometric Brownian Motion** stochastic process:

$$
dV_t = \mu V_t dt + \sigma_A V_t dZ_t
$$

  Where:
  * $\mu$: Asset drift (expected return).
  * $\sigma_A$: Asset volatility (derived from Equity volatility).
  * $dZ_t$: A standard Wiener process (random shock).

* **Methodology:**
    1.  Reverse-engineered unobservable Asset Value ($V_A$) and Asset Volatility ($\sigma_A$) from observable market inputs using the relation $\sigma_A \approx \sigma_E \times \frac{E}{V}$.
    2.  Deployed **Monte Carlo simulation** to generate **10,000 stochastic paths** of asset value evolution over a $T=1$ year horizon.
    3.  **Default Condition:** A firm defaults if the simulated asset value at maturity falls below the face value of debt ($V_T < D$).

---

## 2. Model Outputs

### Reduced Form: Key Outputs
**Part A Simulated Data and Logistic Regression**

<p align="center">
  <img src="img/logistic_simulated_data.png" width="750">
</p>

**Part B Real Company Data and Logistic Regression**

<p align="center">
  <img src="img/logistic_industry_data.png" width="750">
</p>



### Structural Model (Merton): Monte Carlo Asset Paths

<p align="center">
  <img src="img/merton_paths_Ford.png" width="750">
</p>

In this case study of Ford Moter, Monte Carlo simulation generates stochastic asset value paths and compares them
against the debt barrier, producing a structural probability of default (PD) and an intuitive distance-to-default interpretation. 

Given that all simulated 10,000 paths witnesses the assets value sufficient to cover debt (shown as asset paths are above the "red" debt level), the probability of default is zero.

### Strategy Signal (Model PD vs Market-Implied PD)

<p align="center">
  <img src="img/credit_signal.png" width="750">
</p>

Signal = Model PD - Market PD. Positive signals indicate underpriced risk
(short credit), while negative signals indicate potential relative value (long credit).

---

## 3. Variable Selection & Economic Rationale
Following the **Campbell-Hilscher-Szilagyi (2008)** framework, variables were selected to capture Solvency, Profitability, and Market Valuation.

| Variable Category | Proxy Used (Code) | Economic Rationale | Expected Sign |
| :--- | :--- | :--- | :--- |
| **Solvency** | `leverage` (Debt / Capital) | Measures the distance to the bankruptcy barrier. | **(+) Positive** |
| **Profitability** | `profitability` (EBITDA / Assets) | Internal cash flow generation capacity to service debt. | **(-) Negative** |
| **Coverage** | `coverage` (EBITDA / Interest) | Immediate liquidity cushion against interest payments. | **(-) Negative** |
| **Market Risk** | `volatility` (12M Std Dev) | Market's forward-looking assessment of asset uncertainty. | **(+) Positive** |

---

## 4. Credit Strategy & Signal Generation
The core objective of this project is to derive an actionable **credit strategy** by comparing the *Model-Implied PD* against the *Market-Implied PD*.

$$
Signal = Model_{PD} - Market_{PD}
$$

* **Credit Spread Estimation:** Market-Implied PDs are reverse-engineered from corporate bond yields and credit spreads (Yield - Risk-Free Rate) using standard hazard rate approximations.

### 📉 Trading Logic

| Condition | Interpretation | Strategy | Instrument |
| :--- | :--- | :--- | :--- |
| **Model PD > Market PD** | Market is underpricing risk (Asset is Overvalued). | **Short Credit** | Buy CDS / Short Bond |
| **Model PD < Market PD** | Market is overpricing risk (Asset is Undervalued). | **Long Credit** | Sell CDS / Buy Bond |

---

## 5. Technical Implementation
* **Languages & Libraries:** Python (`pandas`, `numpy`, `statsmodels`, `scipy`, `sklearn`).
* **Data Sources:**
    * **Simulated Data:** Generated synthetic datasets to validate model parameters ("God Mode").
    * **Real Industry Data:** Integrated the UCI Machine Learning Repository API (`ucimlrepo`) and `yfinance` for live market data.
* **Visualization:** `matplotlib` for plotting Monte Carlo simulation paths and analyzing asset value distributions.

---



*Disclaimer: This project is for academic and demonstration purposes only and does not constitute financial advice.*
