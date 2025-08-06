
# Technical Specification for ARIMAX Credit Risk Modeler Jupyter Notebook

## 1. Notebook Overview

This Jupyter Notebook, titled `Macro_Model_Part1.ipynb`, serves as an interactive platform for estimating and diagnosing an ARIMAX model. The primary objective is to predict `Segment A default-rate` using macroeconomic variables like `GDP_Growth_YoY_%` and `Unemployment_%` as exogenous regressors. The notebook guides users through data loading, pre-processing, model estimation, and diagnostic checks, emphasizing the application of time series modeling to financial credit risk data.

### Learning Goals

Upon completion of this notebook, users will be able to:
*   Understand the process of loading and aligning diverse time series datasets for macroeconomic modeling.
*   Identify and apply appropriate stationarity transformations to time series data, validating them using statistical tests.
*   Construct and estimate an $ARIMAX(p,d,q)$ model for credit default rates, incorporating relevant macroeconomic exogenous variables.
*   Utilize information criteria (AIC and BIC) to systematically select optimal model orders and exogenous lag structures.
*   Perform comprehensive diagnostic checks on model residuals, including visual inspection (ACF/PACF plots) and formal statistical tests (Ljung-Box test), to ensure model adequacy.
*   Persist a fitted ARIMAX model for subsequent use in further analysis or validation steps.

### Expected Outcomes

The notebook will produce:
*   A pre-processed and stationary time series dataset ready for ARIMAX modeling.
*   An estimated $ARIMAX(p,d,q)$ model for `Segment A default-rate`.
*   A clear display of AIC and BIC values to aid in model order selection.
*   Visual diagnostic plots (ACF and PACF) of model residuals.
*   Statistical results from the Ljung-Box test for residual autocorrelation.
*   A serialized Python object of the fitted ARIMAX model, saved as `macro_pd_arimax_taiwan.pkl`.

## 2. Mathematical and Theoretical Foundations

This section will provide the necessary theoretical background for understanding the ARIMAX model and its associated diagnostic tools.

### 2.1 Stationarity of Time Series

A time series is said to be stationary if its statistical properties, such as mean, variance, and autocorrelation, are constant over time. Most time series models, including ARIMAX, assume stationarity. Non-stationary series often exhibit trends, seasonality, or changing variance, which can lead to spurious regressions.

*   **Transformations for Stationarity**:
    *   **Year-over-Year (YoY) Percentage Change**: For variables like GDP or CPI, calculating the YoY percentage change helps remove trends and seasonal patterns, transforming a level series into a growth rate.
        $$ Y_{t}^{YoY} = \frac{Y_t - Y_{t-4}}{Y_{t-4}} \times 100 $$
    *   **Log-Differencing**: For variables that exhibit exponential growth or whose variance increases with the mean, taking the natural logarithm and then differencing can achieve stationarity.
        $$ \Delta \ln(Y_t) = \ln(Y_t) - \ln(Y_{t-1}) $$

*   **Unit Root Tests**: These statistical tests are used to determine if a time series is stationary.
    *   **Augmented Dickey-Fuller (ADF) Test**: The null hypothesis ($H_0$) for the ADF test is that a unit root is present (i.e., the series is non-stationary). A small p-value (typically $< 0.05$) indicates rejection of $H_0$, suggesting stationarity.
    *   **Phillips-Perron (PP) Test**: Similar to the ADF test, the PP test also tests the null hypothesis of a unit root but is non-parametric, making it robust to heteroscedasticity and autocorrelation in the error term.

### 2.2 The ARIMAX Model

The Autoregressive Integrated Moving Average with Exogenous Inputs (ARIMAX) model is a powerful tool for forecasting a time series based on its own past values, past forecast errors, and the current and past values of other explanatory (exogenous) variables.

The general form of an $ARIMAX(p,d,q)$ model with $m$ exogenous regressors can be expressed as:
$$ (1 - \sum_{i=1}^{p} \phi_i L^i) (1 - L)^d Y_t = c + (1 + \sum_{j=1}^{q} \theta_j L^j) \epsilon_t + \sum_{k=1}^{m} \beta_k X_{k,t} $$
Where:
*   $Y_t$: The target variable (e.g., `Segment A default-rate`) at time $t$.
*   $L$: The lag operator, such that $L^i Y_t = Y_{t-i}$.
*   $p$: The order of the autoregressive (AR) part, indicating the number of past observations of $Y_t$ used in the model.
*   $\phi_i$: The coefficients for the autoregressive terms.
*   $d$: The order of differencing (I for integrated part), indicating the number of non-seasonal differences needed to achieve stationarity.
*   $q$: The order of the moving average (MA) part, indicating the number of past forecast errors ($\epsilon_t$) used in the model.
*   $\theta_j$: The coefficients for the moving average terms.
*   $c$: A constant term.
*   $\epsilon_t$: The white noise error term (residuals) at time $t$, assumed to be independently and identically distributed with a mean of zero and constant variance.
*   $X_{k,t}$: The $k$-th exogenous variable (e.g., `GDP_Growth_YoY_%`, `Unemployment_%`) at time $t$. These variables can also be lagged (e.g., $X_{k,t-l}$ for lag $l$).
*   $\beta_k$: The coefficients for the exogenous variables.

### 2.3 Model Order Selection Criteria

Information criteria help in selecting the most appropriate model order ($p, d, q$) and exogenous variable lags by balancing model fit with complexity. Lower values generally indicate a better model.

*   **Akaike Information Criterion (AIC)**:
    $$ AIC = -2 \ln(L) + 2k $$
*   **Bayesian Information Criterion (BIC)**:
    $$ BIC = -2 \ln(L) + k \ln(n) $$
    Where:
    *   $L$: The maximum likelihood of the estimated model.
    *   $k$: The number of parameters in the model.
    *   $n$: The number of observations used for estimation.
    BIC penalizes complex models more heavily than AIC, making it tend to select simpler models.

### 2.4 Residual Diagnostics

After fitting an ARIMAX model, it is crucial to perform diagnostic checks on its residuals to ensure that they resemble white noise, indicating that all relevant information has been captured by the model.

*   **Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) Plots**:
    *   **ACF**: Measures the correlation between a time series and its lagged values. For white noise residuals, the ACF should show no significant correlations beyond lag 0.
    *   **PACF**: Measures the correlation between a time series and its lagged values, after removing the effect of intermediate lags. For white noise residuals, the PACF should also show no significant correlations.
    Significant spikes in ACF or PACF plots of residuals suggest remaining autocorrelation, implying the model is misspecified.

*   **Ljung-Box Test**:
    This is a formal statistical test for assessing whether there is significant autocorrelation in the residuals at various lags. The test statistic $Q$ is given by:
    $$ Q = n(n+2) \sum_{k=1}^{h} \frac{\hat{\rho}_k^2}{n-k} $$
    Where:
    *   $n$: The number of observations.
    *   $\hat{\rho}_k$: The sample autocorrelation coefficient of the residuals at lag $k$.
    *   $h$: The number of lags being tested.
    The null hypothesis ($H_0$) is that the residuals are independently distributed (i.e., the autocorrelations are all zero). A large p-value (typically $> 0.05$) indicates that we cannot reject $H_0$, supporting the assumption of no remaining autocorrelation. A small p-value would suggest that the model is inadequate.

## 3. Code Requirements

The notebook will be structured logically into distinct sections, each addressing a specific phase of the ARIMAX modeling process.

### 3.1 Expected Libraries

The following Python libraries will be used:
*   `pandas`: For data manipulation and analysis, especially with time series.
*   `numpy`: For numerical operations.
*   `statsmodels`: The primary library for time series modeling, specifically `statsmodels.tsa.arima.model.ARIMAX`. Also for ADF, PP tests, and Ljung-Box test.
*   `matplotlib.pyplot`: For creating static, interactive, and animated visualizations.
*   `seaborn`: For enhancing data visualizations.
*   `pickle`: For serializing and deserializing Python object structures.
*   `pandas_datareader`: for fetching macro time-series directly from public sources.

### 3.2 Input/Output Expectations

*   **Input Data**:
**Input data** (two CSVs to be placed in `/data/`):
*  `segmentA_default_rates_taiwan.csv` – quarterly fields `Quarter`, `Default_Rate_%`, `Exposure_TWD` covering 2015 Q1 – 2025 Q2.<br>• `taiwan_macro_quarterly.csv` – quarterly macro drivers `Quarter`, `GDP_Growth_YoY_%`, `Unemployment_%`, `CPI_YoY_%`, `Policy_Rate_%` for the same horizon.
* Both files must jointly provide **≥ 10 years** of quarterly observations to meet variability guidance from the MMG document.

*   **Output Data**:
    *   Transformed time series data for modeling.
    *   Tables displaying ADF and PP test results for all series.
    *   Tables/displays of AIC and BIC values for different ARIMAX model configurations.
    *   Residual diagnostics plots (ACF, PACF).
    *   Table presenting Ljung-Box test results.
    *   A serialized ARIMAX model object (`macro_pd_arimax_taiwan.pkl`) saved in a dedicated `/models/` directory.

### 3.3 Algorithms and Functions to be Implemented

The notebook will conceptually follow these steps (without writing Python code directly):

#### **Section 01: Data Loading and Initial Exploration (`01_load_data`)**

*   **Markdown Explanation**:
    *   Introduction to the dataset and its importance for credit risk modeling.
    *   Description of `Segment A default-rate` as the target variable and selected macro drivers (`GDP_Growth_YoY_%`, `Unemployment_%`) as exogenous variables.
    *   Instruction to load the provided quarterly dataset.
*   **Code Sections**:
* python<br># STEP 0 – data acquisition / load\nimport pandas as pd\ncr_path = 'data/segmentA_default_rates_taiwan.csv'\nmac_path = 'data/taiwan_macro_quarterly.csv'\ncr = pd.read_csv(cr_path, parse_dates=['Quarter'])\nmac = pd.read_csv(mac_path, parse_dates=['Quarter'])\nraw_df = (cr.merge(mac, on='Quarter', how='inner')\n .sort_values('Quarter')\n .set_index('Quarter'))\nraw_df.head()\n

    *   Load the dataset into a pandas DataFrame.
    *   Display the head of the DataFrame and its info/describe to understand structure and data types.
    *   Filter the dataset to include only `Segment A` default rates.
    *   Perform calendar alignment and forward-fill any missing data points to ensure a continuous time series.
*   **Visualization**:
    *   **Overlay plot**: Display `Segment A default-rate` versus `GDP_Growth_YoY_%` over time.
        *   Include markers for significant events or regime shifts (e.g., COVID-19 pandemic, IFRS-9 implementation) to facilitate visual inspection of potential structural breaks.
        *   **Purpose**: Visual regime check and economic intuition.

#### **Section 02: Data Pre-processing for Stationarity (`02_preprocess`)**

*   **Markdown Explanation**:
    *   Explanation of time series stationarity and its importance for ARIMAX models.
    *   Rationale for chosen transformations (YoY % for growth rates, log-differencing where appropriate).
    *   Introduction to ADF and PP unit root tests.
* Ensure macro and default-rate series are calendar-aligned **before** transformation; forward-fill any sparse quarters.
*   **Code Sections**:
    *   Apply necessary transformations to the target series (`Default_Rate_%`) and selected exogenous variables (`GDP_Growth_YoY_%`, `Unemployment_%`) to achieve stationarity.
    *   For each transformed series:
        *   Perform the Augmented Dickey-Fuller (ADF) test.
        *   Perform the Phillips-Perron (PP) test.
        *   Display the results of these tests (test statistic, p-value) in a clear table format.
        *   Justify the chosen transformation based on test outcomes.
*   **Visualization**:
    *   Plots of transformed series to visually confirm stationarity (e.g., no clear trend, constant variance).

#### **Section 03: ARIMAX Model Estimation and Diagnostics (`03_train_arimax`)**

*   **Markdown Explanation**:
    *   Detailed explanation of the ARIMAX model components ($p,d,q$) and exogenous variables.
    *   Guidance on selecting optimal model orders using AIC and BIC.
    *   Importance of residual diagnostics (ACF, PACF, Ljung-Box) for model validation.
*   **Code Sections**:
    *   **Reusable Helper Function**: Implement a function `train_arimax(df, target_col, exog_cols, order)`:
        *   **Input**: `df` (DataFrame with target and exogenous series), `target_col` (string, name of the target column), `exog_cols` (list of strings, names of exogenous columns), `order` (tuple `(p,d,q)` for ARIMAX).
        *   **Process**:
            *   Instantiate and fit the `ARIMAX` model using `statsmodels`.
            *   Extract model summary, AIC, BIC.
            *   Perform residual diagnostics (ACF/PACF plots, Ljung-Box test).
        *   **Output**: Returns the fitted model object and a DataFrame containing diagnostic test results (e.g., Ljung-Box p-values for different lags).
    *   **Model Order Selection Interface**:
        *   Provide interactive elements (e.g., using `ipywidgets` concepts for sliders or text input) for users to specify `p`, `d`, `q` orders and to select which exogenous variables (and their lags if applicable) to include.
        *   Call `train_arimax` with user-defined parameters.
        *   Display the AIC and BIC values prominently for each estimated model configuration.
        *   Guide the user to iteratively adjust parameters to minimize AIC/BIC.
    *   **Residual Diagnostics Display**:
        *   Automatically generate and display the **Residual ACF plot** and **Residual PACF plot** for the fitted model.
        *   Present the results of the **Ljung-Box test** for residual autocorrelation, including p-values, in a clear table format for multiple lags (e.g., up to 10 or 20 lags).
        *   Interpret the diagnostic results to confirm model adequacy (i.e., residuals resemble white noise).
    *   **Model Export**:
        *   After a satisfactory model is identified, implement code to save the fitted model object using `pickle`.
        *   Save path: `/models/macro_pd_arimax_taiwan.pkl`.
        *   Include markdown explaining how to reload this `.pkl` file in subsequent notebooks.
*   **Visualization**:
    *   **Residual ACF Plot**: Verify no remaining autocorrelation.
    *   **Residual PACF Plot**: Further verify no remaining autocorrelation.
    *   Tables for AIC/BIC values.
    *   Table for Ljung-Box test results.

## 4. Additional Notes or Instructions

*   **Assumptions**:
    *   The provided dataset is provided and tailored for this exercise, covering the specified time horizon (2015 Q1 – 2025 Q2).
    *   The primary target for modeling is `Segment A default-rate`.
    *   The key exogenous regressors are `GDP_Growth_YoY_%` and `Unemployment_%`.

*   **Constraints**:
    *   **Notebook Layout**: The notebook must follow the logical flow of `01_load_data`, `02_preprocess`, `03_train_arimax` as distinct, well-documented sections within a single `Macro_Model_Part1.ipynb` file.
    *   **Python Version and Libraries**: Adherence to Python 3.10 and `statsmodels` version ≥ 0.15 is required.
    *   **Random Seed**: For reproducibility, `RANDOM_SEED = 42` should be set at the beginning of the notebook where applicable (though less critical for deterministic `statsmodels` fitting).
    *   **Code Standards**: All code should follow PEP-8 guidelines, include type hints, and utilize docstrings for functions.
    *   **Model Persistence**: The final fitted model must be saved as `macro_pd_arimax_taiwan.pkl` in a directory named `/models/`.

*   **Customization Instructions**:
    *   Users will be guided to experiment with different `p`, `d`, `q` orders and the inclusion of various exogenous lags by observing the real-time display of AIC and BIC values.
    *   Instructions will explain how to interpret AIC/BIC: lower values generally indicate a better balance between model fit and complexity.
    *   Users should also be encouraged to critically evaluate the residual diagnostic plots and Ljung-Box test results to ensure the selected model adequately captures the time series dynamics.
```