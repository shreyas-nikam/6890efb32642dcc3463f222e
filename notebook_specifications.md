
# Jupyter Notebook Specification: Macro-Credit Model Builder

## 1. Notebook Overview

**Learning Goals:**

This notebook aims to guide users through building and evaluating macro-economic time series models for forecasting Probability of Default (PD) and Loss Given Default (LGD). It emphasizes compliance with IFRS 9/Basel requirements by incorporating macroeconomic scenarios. The goal is to provide a practical understanding of how macroeconomic factors impact credit risk.

**Expected Outcomes:**

Upon completion of this notebook, users will be able to:

1.  Understand the regulatory necessity for forward-looking macro-credit models in PD/LGD estimation and stress testing.
2.  Prepare credit risk and macroeconomic data for modeling, including stationarity transformations and structural break adjustments.
3.  Identify and rank relevant macroeconomic drivers for credit risk.
4.  Build, compare, and interpret ARDL, VAR, and ARIMAX models for credit risk forecasting.
5.  Apply model selection techniques based on information criteria and residual diagnostics.
6.  Generate baseline and stress-scenario forecasts for PD/LGD and visualize their potential impact.

## 2. Mathematical and Theoretical Foundations

### 2.1 Stationarity and Data Transformations

Many time series models require stationary data. Stationarity implies that the statistical properties of a time series, such as the mean and variance, do not change over time.  We will use the Augmented Dickey-Fuller (ADF) and Phillips-Perron (PP) tests to check for stationarity. Non-stationary data often require transformations. Common transformations include:

*   **Logarithmic Transformation:**  Used to stabilize variance, especially when dealing with exponential growth. If $X_t$ is a time series, the logarithmic transformation is $Y_t = \ln(X_t)$.

*   **Differencing:** Used to remove trends. The first difference is defined as $\Delta X_t = X_t - X_{t-1}$. Higher-order differencing can be applied if the first difference is not stationary.

*   **Percentage Change:** Used for rates. The percentage change is defined as $((X_t - X_{t-1})/X_{t-1}) * 100$.

### 2.2 Autoregressive Distributed Lag (ARDL) Model

The ARDL model combines autoregressive (AR) terms of the dependent variable with lagged values of independent variables. A general ARDL(p, q) model can be represented as:

$$Y_t = \alpha + \sum_{i=1}^{p} \phi_i Y_{t-i} + \sum_{j=0}^{q} \beta_j X_{t-j} + \epsilon_t$$

where:
*   $Y_t$ is the dependent variable at time $t$.
*   $X_t$ is the independent variable at time $t$.
*   $p$ is the order of the autoregressive component.
*   $q$ is the order of the distributed lag component.
*   $\phi_i$ are the coefficients of the lagged dependent variables.
*   $\beta_j$ are the coefficients of the lagged independent variables.
*   $\alpha$ is the intercept.
*   $\epsilon_t$ is the error term.

### 2.3 Vector Autoregression (VAR) Model

A VAR model is a multivariate time series model that captures the interdependencies among multiple variables. A VAR(p) model can be represented as:

$$Y_t = c + A_1 Y_{t-1} + A_2 Y_{t-2} + ... + A_p Y_{t-p} + \epsilon_t$$

where:
*   $Y_t$ is a vector of $k$ variables at time $t$.
*   $c$ is a $k \times 1$ vector of intercepts.
*   $A_i$ are $k \times k$ coefficient matrices.
*   $p$ is the order of the VAR model.
*   $\epsilon_t$ is a $k \times 1$ vector of error terms.

### 2.4 ARIMAX Model

An ARIMAX model extends the ARIMA model by including exogenous variables. The general form of an ARIMAX(p, d, q) model is:

$$\phi(B)(1-B)^d Y_t = \theta(B) \epsilon_t + C(B)X_t$$

Where:
*   $Y_t$ is the time series being modeled.
*   $X_t$ is the exogenous variable.
*   $B$ is the backshift operator ($BY_t = Y_{t-1}$).
*   $\phi(B)$ is the autoregressive polynomial of order $p$.
*   $\theta(B)$ is the moving average polynomial of order $q$.
*   $d$ is the order of integration (number of differences required for stationarity).
*   $C(B)$ represents the impact of the exogenous variable $X_t$.
*   $\epsilon_t$ is the error term.

### 2.5 Model Selection Criteria

*   **Akaike Information Criterion (AIC):** A measure of the relative quality of statistical models for a given set of data. AIC is defined as:

$$AIC = 2k - 2\ln(L)$$

where:
*   $k$ is the number of parameters in the model.
*   $L$ is the maximized value of the likelihood function for the model.

*   **Bayesian Information Criterion (BIC):** Similar to AIC, but with a stronger penalty for model complexity.  BIC is defined as:

$$BIC = \ln(n)k - 2\ln(L)$$

where:
*   $n$ is the number of data points.
*   $k$ is the number of parameters in the model.
*   $L$ is the maximized value of the likelihood function for the model.

### 2.6 Residual Diagnostics

*   **Durbin-Watson Test:** Tests for autocorrelation in the residuals of a regression model.

*   **Ljung-Box Test:**  A more general test for autocorrelation.

*   **White's Test:** Tests for heteroscedasticity (non-constant variance) in the residuals.

*   **ARCH Test:** Tests for autoregressive conditional heteroscedasticity.

*   **Jarque-Bera Test:** Tests for normality of the residuals.

## 3. Code Requirements

### 3.1 Expected Libraries

The following Python libraries are expected to be used:

*   **pandas:** For data manipulation and analysis.
*   **numpy:** For numerical computations.
*   **statsmodels:** For statistical modeling and time series analysis.
*   **joblib:** For serializing and deserializing models.
*   **matplotlib:** For creating static, interactive, and animated visualizations in Python.
*   **seaborn:** For making statistical graphics.
*   **arch:** For volatility modeling (ARCH/GARCH).
*   **pmdarima:** For ARIMA model selection.
*   **yaml:** For handling YAML configurations

### 3.2 Data Ingestion and Cleaning

*   **Input:** The notebook should be able to ingest the Taiwan credit card default dataset and macroeconomic data (GDP growth, CPI inflation, unemployment rate, etc.).
*   **Output:** A cleaned and merged pandas DataFrame `taiwan_macro_panel.parquet` containing quarterly PD, LGD (optional), and macroeconomic data from 2015 Q1 to 2024 Q4. This DataFrame should be saved to a file.
*   **Algorithms/Functions:**
    *   Functions to download or load the datasets.
    *   Functions to merge and align datasets based on dates.
    *   Functions to handle missing data.

### 3.3 Data Transformation

*   **Input:** The merged DataFrame from the previous step.
*   **Output:** A transformed DataFrame with stationary time series.  The transformations applied should be stored in a `data_transform.yaml` file.
*   **Algorithms/Functions:**
    *   Functions to apply log-diff, percent-change, or other transformations.
    *   Functions to perform ADF and PP tests.
    *   Functions to store the transformation metadata in `data_transform.yaml`.

### 3.4 Exploratory Data Analysis (EDA)

*   **Input:** The transformed DataFrame.
*   **Output:**
    *   Time-series plots of PD vs. macro variables.
    *   Correlation heatmap of transformed variables.
    *   ACF and PACF plots for PD.
*   **Algorithms/Functions:**
    *   Functions to generate time series plots.
    *   Functions to calculate and visualize the correlation matrix.
    *   Functions to generate ACF and PACF plots.

### 3.5 Model Building

*   **Input:** The transformed DataFrame.
*   **Output:** Fitted ARDL, VAR, and ARIMAX models.
*   **Algorithms/Functions:**
    *   Functions to fit ARDL models using `statsmodels`.
    *   Functions to fit VAR models using `statsmodels`.
    *   Functions to fit ARIMAX models using `pmdarima` or `statsmodels`.
    *   Functions for lag selection (e.g., using information criteria).

### 3.6 Model Diagnostics

*   **Input:** Fitted models and residuals.
*   **Output:**
    *   Residual ACF plots.
    *   QQ-plots of residuals.
    *   Bar plots of ARCH LM test p-values.
*   **Algorithms/Functions:**
    *   Functions to calculate and plot residual ACF.
    *   Functions to generate QQ-plots.
    *   Functions to perform and visualize ARCH LM tests.
    *   Functions for Durbin-Watson, Ljung-Box, Jarque-Bera tests.

### 3.7 Scenario Forecasting

*   **Input:** Fitted model, `data_transform.yaml`, and user-specified macro scenarios.
*   **Output:** Forecasts of PD/LGD under baseline and stress scenarios.
*   **Algorithms/Functions:**
    *   Functions to generate forecasts using the fitted models.
    *   Functions to incorporate macro scenarios into the forecasts.
    *   Functions to transform forecast back to original space using `data_transform.yaml`.
    *   Functions to generate fan charts/confidence intervals.

### 3.8 Visualization

*   **Input:** Baseline and stress scenario PD pathways.
*   **Output:**
    *   Plots of baseline vs. stress scenario PD pathways with fan-charts/confidence intervals.
    *   Table of stress multipliers.

### 3.9 Model Serialization
*   **Input:** Fitted models
*   **Output:** Serialized models (.pkl files) in the `models/` directory using `joblib`.

## 4. Additional Notes or Instructions

*   **Data Assumptions:** Assume quarterly frequency for all time series.
*   **Stationarity:** Ensure all time series are stationary before modeling.
*   **Reproducibility:** Fix random seeds for reproducibility and record library versions.
*   **Model Storage:** Store fitted models in the `models/` directory.
*   **Transformation Storage**: Store transformation metadata in `data_transform.yaml`.
*   **Naming Conventions:** Use consistent naming conventions for variables and functions.  Example: `taiwan_pd_quarterly.csv` for the quarterly PD data, `PD_ARDL_L1U2.pkl` for the serialized ARDL model.
*   **User Instructions:**
    *   Include clear markdown explanations for each step.
    *   Comment code blocks thoroughly.
*   **Stress Scenarios:** stress scenarios should represent reasonable macroeconomic downturns.
*   **Model Hand-off:** Clearly state the selected champion model and required files for subsequent use.  For example: "The selected champion model is PD_ARDL_L1U2.pkl stored in models/. Part 2 will begin by loading this file together with data_transform.yaml."
*   **Structural Breaks:** Incorporate regime dummies for structural breaks (e.g., COVID-19, VAT launch).
*   **References:** At the end of the notebook provide a reference section with links or citations used. Example:
```html
   1. Taiwan Credit Card Default Dataset: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)
```
