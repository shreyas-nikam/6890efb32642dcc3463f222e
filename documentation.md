id: 6890efb32642dcc3463f222e_documentation
summary: Lab 4.1: Macro-Economic Models - Development Documentation
feedback link: https://docs.google.com/forms/d/e/1FAIpQLSfWkOK-in_bMMoHSZfcIvAeO58PAH9wrDqcxnJABHaxiDqhSA/viewform?usp=sf_link
environments: Web
status: Published
# Building a Macro-Economic Credit Risk Forecasting Application with Streamlit and ARIMAX

## Introduction to QuLab: Macro-Economic Models for Credit Risk Forecasting
Duration: 05:00

Welcome to this codelab! In this session, you will learn how to build and understand a Streamlit application focused on applying macro-economic models for credit risk forecasting. This application, named "QuLab," uses a synthetic dataset to illustrate the process of estimating and diagnosing an ARIMAX (Autoregressive Integrated Moving Average with eXogenous inputs) model for predicting segment default rates, integrating key macroeconomic variables.

Understanding and forecasting credit risk is paramount in financial institutions. Traditional credit risk models often rely on internal factors, but macro-economic conditions (like GDP growth and unemployment rates) play a crucial role, especially during economic downturns. This codelab will guide you through the process of:

*   **Data Handling**: Loading and exploring time series data.
*   **Time Series Pre-processing**: Ensuring data stationarity, a critical prerequisite for many time series models, using techniques like differencing and statistical tests (ADF, KPSS).
*   **ARIMAX Model Estimation**: Building a robust forecasting model that combines the strengths of ARIMA for time series patterns with the predictive power of exogenous macroeconomic indicators. We will cover model order selection using information criteria (AIC, BIC).
*   **Model Diagnostics**: Evaluating the model's performance by analyzing its residuals using tools like the Ljung-Box test, ensuring the model captures all relevant information.
*   **Model Persistence**: Saving trained models for future inference and deployment.

By the end of this codelab, you will have a comprehensive understanding of how to develop an interactive Streamlit application for macro-economic credit risk modeling, equipped with the knowledge of underlying statistical concepts and best practices.

<aside class="positive">
<b>Why is this important?</b>
This application demonstrates a practical approach to integrating macroeconomic data into credit risk models, making forecasts more resilient and insightful, especially in volatile economic environments. The use of Streamlit allows for rapid prototyping and deployment of interactive data science applications.
</aside>

### Application Architecture Overview

The QuLab application is structured into a main `app.py` file which acts as a navigator, routing to different functionalities implemented in separate Python files within the `application_pages` directory. This modular design promotes maintainability and scalability.

Here's a high-level overview of the application's structure:

```
├── app.py
└── application_pages/
    ├── __init__.py
    ├── page1.py
    ├── page2.py
    ├── page3.py
    └── page4.py
```

*   `app.py`: The main entry point. Sets up the Streamlit page configuration, displays the title, and manages navigation between different pages via a sidebar.
*   `application_pages/page1.py`: Handles data loading, synthetic data generation, and initial data exploration with visualizations.
*   `application_pages/page2.py`: Implements data pre-processing steps, specifically focusing on achieving stationarity for time series data using differencing and conducting unit root tests.
*   `application_pages/page3.py`: Contains the core logic for ARIMAX model estimation, including parameter selection, model fitting, and comprehensive residual diagnostics.
*   `application_pages/page4.py`: Manages the persistence of the trained model, allowing users to download it.

### Getting Started

To run this application locally, you will need Python installed (preferably Python 3.8+).

1.  **Save the files:** Save the provided `app.py` and the `application_pages` directory with its contents in a local folder.

2.  **Create a virtual environment (recommended):**
    ```console
    python -m venv venv
    source venv/bin/activate # On Windows: .\venv\Scripts\activate
    ```

3.  **Install dependencies:**
    The application uses `streamlit`, `pandas`, `numpy`, `matplotlib`, and `statsmodels`. You can install them using pip:
    ```console
    pip install streamlit pandas numpy matplotlib statsmodels
    ```

4.  **Run the Streamlit application:**
    Navigate to the directory containing `app.py` in your terminal and run:
    ```console
    streamlit run app.py
    ```
    This will open the application in your default web browser.

## Data Loading and Initial Exploration
Duration: 07:00

This step focuses on generating and understanding the synthetic dataset used throughout the application. The `application_pages/page1.py` script is responsible for this, providing an interactive way to view raw data, its statistical summary, and visual trends.

### Synthetic Data Generation

The application uses a synthetic dataset to mimic real-world Taiwan credit risk and macroeconomic data. This allows for reproducible demonstrations without relying on sensitive financial data. The data includes:

*   **Segment A Default Rate**: A simulated default rate for a specific credit segment. It includes a general increasing trend and a simulated recessionary spike.
*   **GDP Growth YoY%**: Simulated year-over-year GDP growth, including cyclical patterns and a dip during the simulated recession.
*   **Unemployment%**: Simulated unemployment rate, inversely correlated with GDP growth.

The data generation process ensures that the time series exhibit characteristics typical of economic data, such as trends, seasonality, and responses to economic events.

Let's look at the relevant code snippet from `application_pages/page1.py`:

```python
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def run_page1():
    st.header("1. Data Loading and Initial Exploration")
    st.markdown("This section loads synthetic time series data mimicking Taiwan credit risk and macroeconomic data. It displays initial data insights and visualizations.")

    RANDOM_SEED = 42
    start_date = '2015-01-01'
    quarters = pd.date_range(start=start_date, periods=40, freq='QS-JAN')

    np.random.seed(RANDOM_SEED)
    default_rate = 1.5 + 0.05 * np.arange(len(quarters)) + np.random.normal(0, 0.2, len(quarters))
    recession_start = 20
    recession_end = 24
    default_rate[recession_start:recession_end] += np.random.normal(1.0, 0.3, recession_end - recession_start)

    gdp_growth = 2.5 + 1.0 * np.sin(np.arange(len(quarters)) * 2 * np.pi / 8) + np.random.normal(0, 0.5, len(quarters))
    gdp_growth[recession_start:recession_end] = np.random.normal(-2.0, 1.0, recession_end - recession_start)

    unemployment = 5.0 - 0.5 * gdp_growth + np.random.normal(0, 0.3, len(quarters))
    unemployment = np.clip(unemployment, 2.0, 12.0)

    df_synthetic = pd.DataFrame({
        'Quarter': quarters,
        'Segment A Default Rate': default_rate,
        'GDP_Growth_YoY_%': gdp_growth,
        'Unemployment_%': unemployment
    })
    df_synthetic.set_index('Quarter', inplace=True)
    df_synthetic.index.freq = 'QS-JAN'

    st.subheader("Synthetic Dataset Overview:")
    st.dataframe(df_synthetic.head())
    st.write(f"Dataset shape: {df_synthetic.shape}")
    st.write("Basic statistics:")
    st.dataframe(df_synthetic.describe())

    # Visualization of synthetic time series data
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    axes[0].plot(df_synthetic.index, df_synthetic['Segment A Default Rate'], linewidth=2, color='red', label='Segment A Default Rate')
    axes[0].set_title('Segment A Default Rate Over Time', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Default Rate (%)')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(df_synthetic.index, df_synthetic['GDP_Growth_YoY_%'], linewidth=2, color='blue', label='GDP Growth YoY%')
    axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1].set_title('GDP Growth Year-over-Year (%)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('GDP Growth (%)')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    ax3 = axes[2]
    ax3_twin = ax3.twinx()
    line1 = ax3.plot(df_synthetic.index, df_synthetic['Segment A Default Rate'], linewidth=2, color='red', label='Default Rate')
    line2 = ax3_twin.plot(df_synthetic.index, df_synthetic['GDP_Growth_YoY_%'], linewidth=2, color='blue', label='GDP Growth')
    ax3.set_title('Default Rate vs GDP Growth Over Time', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Default Rate (%)', color='red')
    ax3_twin.set_ylabel('GDP Growth (%)', color='blue')
    ax3.set_xlabel('Quarter')
    ax3.grid(True, alpha=0.3)
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
```

### Key Takeaways from Page 1

When you navigate to "Data Loading & Exploration" in the Streamlit app, you will see:

*   **Dataset Head**: The first few rows of the generated `df_synthetic` DataFrame.
*   **Dataset Shape and Basic Statistics**: Provides an overview of the data dimensions and summary statistics (mean, std, min, max, quartiles) for each column.
*   **Time Series Plots**:
    *   Separate plots for 'Segment A Default Rate' and 'GDP Growth YoY%' showing their evolution over time. Observe the trends and the impact of the simulated recession.
    *   A combined plot of 'Default Rate vs GDP Growth Over Time' using a twin y-axis. This visualization is crucial for visually identifying potential relationships between the target variable and an exogenous variable. You should observe an inverse relationship during the recessionary period.

<aside class="positive">
<b>Tip:</b> Always start with thorough data exploration. Visualizations help in identifying trends, seasonality, outliers, and potential relationships between variables, which are crucial for subsequent modeling decisions.
</aside>

## Data Pre-processing for Stationarity
Duration: 10:00

Stationarity is a fundamental concept in time series analysis. A stationary time series is one whose statistical properties (mean, variance, autocorrelation) do not change over time. Many time series models, including ARIMAX, assume that the underlying process generating the data is stationary. Non-stationary series can lead to spurious regressions and unreliable forecasts.

This step, handled by `application_pages/page2.py`, demonstrates how to test for stationarity and apply transformations (differencing) to achieve it.

### Understanding Stationarity and Unit Root Tests

The application uses two common unit root tests to assess stationarity:

1.  **Augmented Dickey-Fuller (ADF) Test**:
    *   **Null Hypothesis ($H_0$)**: The time series has a unit root (is non-stationary).
    *   **Alternative Hypothesis ($H_1$)**: The time series is stationary.
    *   **Interpretation**: If the p-value is less than or equal to a chosen significance level (e.g., 0.05), we reject $H_0$, concluding the series is stationary.

2.  **Kwiatkowski-Phillips-Schmidt-Shin (KPSS) Test**:
    *   **Null Hypothesis ($H_0$)**: The time series is stationary around a deterministic trend (or mean-stationary if `regression='c'`).
    *   **Alternative Hypothesis ($H_1$)**: The time series is non-stationary.
    *   **Interpretation**: If the p-value is less than or equal to a chosen significance level (e.g., 0.05), we reject $H_0$, concluding the series is non-stationary.

<aside class="negative">
<b>Important Note:</b> ADF and KPSS tests have opposite null hypotheses. If both tests suggest stationarity (ADF rejects $H_0$, KPSS fails to reject $H_0$), you can be reasonably confident. If both suggest non-stationarity, it's clear. If they conflict, further investigation or transformations might be needed.
</aside>

### Transformations for Stationarity

A common technique to achieve stationarity is **differencing**. This involves subtracting the previous observation from the current observation. For quarterly data, year-over-year differencing (lag 4) might also be considered for seasonality, but for this specific example, a simple first-order differencing is applied to the target variable.

*   **First-Order Differencing**:
    $$ \Delta Y_t = Y_t - Y_{t-1} $$
    The application specifically transforms the 'Segment A Default Rate' into `'Default_Rate_Diff'`. The exogenous variables like 'GDP_Growth_YoY_%' are often already in a form that is considered stationary (e.g., growth rates).

Let's examine the code from `application_pages/page2.py`:

```python
import streamlit as st
import pandas as pd
import statsmodels.stats.diagnostic as smd # For adfuller and kpss

def run_page2():
    st.header("2. Data Pre-processing for Stationarity")
    st.markdown("This section defines a helper function for unit root tests and applies differencing to the target variable to achieve stationarity, then verifies it.")

    def run_unit_root_tests(series, name):
        """
        Performs Augmented Dickey-Fuller (ADF) and Kwiatkowski-Phillips-Schmidt-Shin (KPSS) tests
        for stationarity on a given time series.
        """
        st.write(f"\n Unit Root Tests for: **{name}** ")

        # ADF Test
        # Null Hypothesis ($H_0$): The series has a unit root (non-stationary).
        # If p-value <= 0.05, reject $H_0$ (series is stationary).
        adf_result = smd.adfuller(series.dropna(), autolag='AIC')
        st.write("Augmented Dickey-Fuller Test:")
        st.write(f"  ADF Statistic: {adf_result[0]:.4f}")
        st.write(f"  P-value: {adf_result[1]:.4f}")
        st.write(f"  Critical Values (1%, 5%, 10%): {adf_result[4]}")
        st.write(f"  Result: {'**Stationary**' if adf_result[1] <= 0.05 else '**Non-Stationary**'} (based on ADF)")

        # KPSS Test
        # Null Hypothesis ($H_0$): The series is stationary around a deterministic trend (trend-stationary).
        # If p-value <= 0.05, reject $H_0$ (series is non-stationary).
        try:
            kpss_result = smd.kpss(series.dropna(), regression='c', nlags='auto')
            st.write("\nKPSS Test:")
            st.write(f"  KPSS Statistic: {kpss_result[0]:.4f}")
            st.write(f"  P-value: {kpss_result[1]:.4f}")
            st.write(f"  Critical Values (10%, 5%, 2.5%, 1%): {kpss_result[3]}")
            st.write(f"  Result: {'**Non-Stationary**' if kpss_result[1] <= 0.05 else '**Stationary**'} (based on KPSS)")
        except Exception as e:
            st.error(f"\nKPSS Test failed: {e}")
        
        return adf_result, kpss_result if 'kpss_result' in locals() else None

    # ... (Rest of the run_page2 function handling button click and data generation)
    if st.button("Apply Transformations and Test Stationarity"):
        # Load the synthetic data (same as in page1.py)
        # ... (data generation code from page1.py) ...

        st.subheader("Testing stationarity of original time series:")
        run_unit_root_tests(df_synthetic['Segment A Default Rate'], 'Segment A Default Rate')
        run_unit_root_tests(df_synthetic['GDP_Growth_YoY_%'], 'GDP_Growth_YoY_%')
        run_unit_root_tests(df_synthetic['Unemployment_%'], 'Unemployment_%')

        df_transformed = df_synthetic.copy()
        df_transformed['Default_Rate_Diff'] = df_transformed['Segment A Default Rate'].diff(1)
        df_transformed = df_transformed.drop(columns=['Segment A Default Rate'])
        df_transformed = df_transformed.dropna()
        df_transformed.index = pd.to_datetime(df_transformed.index)
        df_transformed.index.freq = 'QS-JAN'

        st.subheader("Testing stationarity of transformed time series:")
        run_unit_root_tests(df_transformed['Default_Rate_Diff'], 'Default_Rate_Diff')
        run_unit_root_tests(df_transformed['GDP_Growth_YoY_%'], 'GDP_Growth_YoY_% (Transformed)')

        st.session_state['df_transformed'] = df_transformed
```

### Navigating Page 2

1.  Navigate to "Data Pre-processing (Stationarity)" in the sidebar.
2.  Click the "Apply Transformations and Test Stationarity" button.
3.  Observe the output:
    *   The results of ADF and KPSS tests for the original 'Segment A Default Rate', 'GDP_Growth_YoY_%', and 'Unemployment_%' will be displayed. You should see that 'Segment A Default Rate' is likely non-stationary.
    *   The `Default_Rate_Diff` (first difference of the default rate) is then calculated.
    *   The tests are run again on `Default_Rate_Diff` and 'GDP_Growth_YoY_%'. The goal is for `Default_Rate_Diff` to now appear stationary.
4.  The `df_transformed` DataFrame is stored in `st.session_state` so it can be used in the next step (ARIMAX model estimation).

## ARIMAX Model Estimation and Diagnostics
Duration: 15:00

This is the core of the application, implemented in `application_pages/page3.py`. It allows you to estimate an ARIMAX model and perform crucial diagnostic checks.

### The ARIMAX Model

ARIMAX is an extension of the ARIMA model that includes the concept of "exogenous" (external) variables. It is particularly useful when the target time series is influenced by other independent time series.

The general form of an $ARIMAX(p,d,q)$ model with $m$ exogenous regressors can be expressed as:
$$ (1 - \sum_{i=1}^{p} \phi_i L^i) (1 - L)^d Y_t = c + (1 + \sum_{j=1}^{q} \theta_j L^j) \epsilon_t + \sum_{k=1}^{m} \beta_k X_{k,t} $$
Where:
*   $Y_t$: The target variable at time $t$.
*   $L$: The lag operator, such that $L Y_t = Y_{t-1}$.
*   $p$: The order of the autoregressive (AR) part. This indicates the number of past observations of the target variable to include in the model.
*   $\phi_i$: AR coefficients.
*   $d$: The order of differencing (I for integrated part). This ensures stationarity.
*   $q$: The order of the moving average (MA) part. This indicates the number of past error terms to include.
*   $\theta_j$: MA coefficients.
*   $c$: A constant term.
*   $\epsilon_t$: The white noise error term (residuals) at time $t$. This is the part of $Y_t$ that the model cannot explain.
*   $X_{k,t}$: The $k$-th exogenous variable at time $t$.
*   $\beta_k$: The coefficients for the exogenous variables, indicating their impact on $Y_t$.

### Model Order Selection Criteria

Choosing the correct orders for $p$, $d$, and $q$ (and potentially seasonal orders) is crucial. Information criteria help in selecting the best model among several candidates:

*   **Akaike Information Criterion (AIC)**:
    $$ AIC = -2 \ln(L) + 2k $$
*   **Bayesian Information Criterion (BIC)**:
    $$ BIC = -2 \ln(L) + k \ln(n) $$
    Where: $L$ is the maximum likelihood of the model, $k$ is the number of parameters in the model, and $n$ is the number of observations. Lower values for AIC and BIC generally indicate a better model fit relative to the number of parameters used. BIC penalizes more heavily for additional parameters than AIC, leading to more parsimonious models.

### Residual Diagnostics (Ljung-Box Test)

After fitting a model, it's essential to check its residuals (the difference between actual and predicted values). Ideally, residuals should be white noise, meaning they are independently and identically distributed with a mean of zero and constant variance. Any remaining patterns in the residuals indicate that the model has not fully captured the information in the data.

The **Ljung-Box test** is used to check for autocorrelation in the residuals.
The Ljung-Box test statistic $Q$ is given by:
$$ Q = n(n+2) \sum_{k=1}^{h} \frac{\hat{\rho}_k^2}{n-k} $$
Where: $n$ is the number of observations, $\hat{\rho}_k$ is the sample autocorrelation at lag $k$, and $h$ is the number of lags tested.
*   **Null Hypothesis ($H_0$)**: The residuals are independently distributed (no significant autocorrelation).
*   **Alternative Hypothesis ($H_1$)**: Some autocorrelations are not zero.
*   **Interpretation**: A large p-value (typically $> 0.05$) indicates that we fail to reject the null hypothesis, supporting the conclusion that the residuals are white noise, implying a good model fit. A small p-value indicates remaining autocorrelation, suggesting the model may need refinement (e.g., adjusting $p$ or $q$ orders).

### Code Walkthrough for Page 3

Here's an overview of the key functions in `application_pages/page3.py`:

```python
import streamlit as st
import pandas as pd
import statsmodels.tsa.api as smt # For plot_acf/pacf
import statsmodels.stats.diagnostic as smd # For acorr_ljungbox
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Helper functions for validation, data preparation, fitting, and diagnostics
def validate_arimax_inputs(df, target_col, exog_cols, order): # ...
def prepare_arimax_data(df, target_col, exog_cols): # ...
def fit_arimax_model(endog_series, exog_df, order): # ...
def compute_residual_diagnostics(fitted_model): # ...
def plot_residual_diagnostics(fitted_model, model_name="ARIMAX"): # ...
def interpret_ljung_box_results(ljung_box_results, significance_level=0.05): # ...

def train_arimax(df, target_col, exog_cols, order):
    """Orchestrates the ARIMAX model training and diagnostics."""
    with st.spinner(f"Training ARIMAX{order} Model..."):
        validate_arimax_inputs(df, target_col, exog_cols, order)
        endog_series, exog_df = prepare_arimax_data(df, target_col, exog_cols)
        fitted_model = fit_arimax_model(endog_series, exog_df, order)
        ljung_box_results = compute_residual_diagnostics(fitted_model)
    st.success(f"ARIMAX{order} model training completed!")
    return fitted_model, ljung_box_results

def run_page3():
    st.header("3. ARIMAX Model Estimation and Diagnostics")
    st.markdown("This section allows users to specify and estimate an ARIMAX(p,d,q) model...")

    st.subheader("ARIMAX Model Equation:")
    # ... (equation and explanation) ...

    st.subheader("Model Order Selection Criteria:")
    # ... (AIC/BIC equations and explanation) ...

    st.subheader("Residual Diagnostics (Ljung-Box Test):")
    # ... (Ljung-Box equation and explanation) ...

    target_column = 'Default_Rate_Diff'
    exogenous_columns = ['GDP_Growth_YoY_%']

    st.write(f"**Target variable:** `{target_column}`")
    st.write(f"**Exogenous variables:** `{exogenous_columns}`")

    p = st.number_input("Enter ARIMA Order (p):", min_value=0, value=1, key='p_input')
    d = st.number_input("Enter ARIMA Order (d):", min_value=0, value=0, key='d_input')
    q = st.number_input("Enter ARIMA Order (q):", min_value=0, value=0, key='q_input')
    order = (p, d, q)

    if 'df_transformed' in st.session_state and st.button("Train Selected ARIMAX Model", key='train_button'):
        df_transformed = st.session_state['df_transformed']
        try:
            fitted_model, ljung_box_results = train_arimax(df_transformed, target_column, exogenous_columns, order)
            st.session_state['fitted_model'] = fitted_model
            st.session_state['ljung_box_results'] = ljung_box_results

            st.subheader(f"ARIMAX{order} Model Summary:")
            st.code(fitted_model.summary().as_text())

            st.subheader("Residual Diagnostics:")
            plot_residual_diagnostics(fitted_model, f"ARIMAX{order}")

            st.write("\nLjung-Box Test Results:")
            st.dataframe(ljung_box_results)

            significance_level = st.slider("Significance Level for Ljung-Box Test:", min_value=0.01, max_value=0.10, value=0.05, step=0.01, key='lb_level')
            interpretation = interpret_ljung_box_results(ljung_box_results, significance_level=significance_level)
            st.markdown(f"**Interpretation:** {interpretation['message']}")

            if 'model_comparison_data' not in st.session_state:
                st.session_state['model_comparison_data'] = []
            st.session_state['model_comparison_data'].append({
                'Model': f'ARIMAX{order}',
                'AIC': fitted_model.aic,
                'BIC': fitted_model.bic,
                'Log-Likelihood': fitted_model.llf,
                'Parameters': len(fitted_model.params)
            })
            st.subheader("Model Comparison Table:")
            comparison_df = pd.DataFrame(st.session_state['model_comparison_data'])
            st.dataframe(comparison_df.round(2))

            best_model_idx = comparison_df['AIC'].idxmin()
            best_model_name = comparison_df.loc[best_model_idx, 'Model']
            st.write(f"\n**Best model based on AIC:** {best_model_name}")

        except Exception as e:
            st.error(f"Error during model training or diagnostics: {e}")
    else:
        st.info("Please load synthetic data first and then train the model.")
```

### Navigating Page 3

1.  Navigate to "ARIMAX Model Estimation & Diagnostics" in the sidebar.
2.  Ensure you have previously clicked "Apply Transformations and Test Stationarity" on Page 2, as this page relies on the `df_transformed` DataFrame in `st.session_state`.
3.  **Specify ARIMAX Order (p, d, q)**:
    *   `p` (AR order): Number of lagged observations of the dependent variable.
    *   `d` (Differencing order): Corresponds to the differencing applied on Page 2. Since `Default_Rate_Diff` is already first-differenced, `d` should typically be 0 here when modeling `Default_Rate_Diff`. If you were modeling the original series directly, `d` would be 1.
    *   `q` (MA order): Number of lagged forecast errors.
    *   Start with a simple model, e.g., (1,0,0) or (0,0,1), and iterate based on diagnostics and information criteria. A common starting point after differencing is looking at ACF/PACF plots of the *differenced* series (though not explicitly plotted in the app, this is a standard step in a full analysis).
4.  Click "Train Selected ARIMAX Model".
5.  **Review Output**:
    *   **Model Summary**: This extensive table provides coefficients for AR, MA, and exogenous terms, their standard errors, p-values, and confidence intervals. Crucially, it also provides **AIC** and **BIC** values for model comparison.
    *   **Residual Diagnostics Plots**: ACF (Autocorrelation Function) and PACF (Partial Autocorrelation Function) plots of the model residuals. For a good model, these plots should show no significant spikes (i.e., residuals should resemble white noise).
    *   **Ljung-Box Test Results**: A table showing the Q-statistic and p-value for the Ljung-Box test at specified lags. A high p-value (e.g., > 0.05) indicates no significant autocorrelation in residuals.
    *   **Interpretation**: The app provides a direct interpretation of the Ljung-Box test based on the selected significance level.
    *   **Model Comparison Table**: This table accumulates results from multiple model runs, allowing you to compare AIC, BIC, and other metrics across different ARIMAX orders. Aim for the model with the lowest AIC/BIC that also passes residual diagnostic checks.

<aside class="positive">
<b>Experiment:</b> Try different combinations of (p, d, q) values and observe how AIC, BIC, and the Ljung-Box test results change. This iterative process is key to finding the optimal model. Remember, 'd' refers to the differencing applied *within* the ARIMA model. If your input series is already differenced, 'd' should be 0.
</aside>

## Model Persistence
Duration: 03:00

Once you have identified a well-performing ARIMAX model, you'll want to save it for future use, such as making predictions on new data or deploying it in a production environment. This step, implemented in `application_pages/page4.py`, allows you to download the trained model.

### Saving and Loading Models

The application uses Python's `pickle` module to serialize (save) the fitted `statsmodels` ARIMAX model object and deserialize (load) it later. `pickle` is a standard way to store Python objects.

Here's the relevant code from `application_pages/page4.py`:

```python
import streamlit as st
import pickle
import os

def run_page4():
    st.header("4. Model Persistence")
    st.markdown("You can download the best-performing fitted ARIMAX model for future use.")

    if 'fitted_model' in st.session_state and st.session_state['fitted_model'] is not None:
        best_model = st.session_state['fitted_model']
        
        # Create a dummy directory for saving locally in Streamlit environment
        output_dir = './models_temp/'
        os.makedirs(output_dir, exist_ok=True)
        model_filename = os.path.join(output_dir, 'macro_pd_arimax_taiwan.pkl')

        try:
            with open(model_filename, 'wb') as file:
                pickle.dump(best_model, file)
            
            with open(model_filename, 'rb') as file:
                st.download_button(
                    label="Download Fitted ARIMAX Model (.pkl)",
                    data=file,
                    file_name="macro_pd_arimax_taiwan.pkl",
                    mime="application/octet-stream"
                )
            st.success(f"Model ready for download.")
            st.info(f"Model details: Type - {best_model.__class__.__name__}, AIC - {best_model.aic:.2f}")

        except Exception as e:
            st.error(f"Error preparing model for download: {e}")
    else:
        st.warning("No model has been successfully fitted yet to save.")
```

### Navigating Page 4

1.  Navigate to "Model Persistence" in the sidebar.
2.  Ensure you have successfully trained a model on Page 3, as the download button will only appear if a `fitted_model` object exists in `st.session_state`.
3.  Click the "Download Fitted ARIMAX Model (.pkl)" button.
4.  The model file (`macro_pd_arimax_taiwan.pkl`) will be downloaded to your browser's default download location.

<aside class="positive">
<b>Next Steps:</b> Once downloaded, this `.pkl` file can be loaded into any Python environment using `pickle.load()` to perform forecasts or integrate the model into a larger system without re-training.
</aside>

```python
import pickle

# To load the model later
with open('macro_pd_arimax_taiwan.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# You can now use loaded_model for forecasting
# For example: loaded_model.predict(start='2025-01-01', end='2025-03-31', exog=new_exog_data)
```

## Conclusion
Duration: 02:00

Congratulations! You have successfully completed this codelab on building a macro-economic credit risk forecasting application using Streamlit and ARIMAX models.

You have learned:
*   How to structure a Streamlit application with multiple pages.
*   The process of generating and exploring synthetic time series data.
*   The importance of stationarity for time series modeling and how to achieve it through differencing.
*   The application of unit root tests (ADF, KPSS) to verify stationarity.
*   The theoretical foundation and practical estimation of ARIMAX models, including the role of exogenous variables.
*   How to evaluate model performance using information criteria (AIC, BIC) and essential residual diagnostics like ACF/PACF plots and the Ljung-Box test.
*   The method for persisting trained machine learning models for future use.

This comprehensive guide provides a solid foundation for developing more sophisticated time series forecasting applications, particularly in financial contexts where macroeconomic factors are critical. Feel free to extend this application by incorporating more complex exogenous variables, implementing model selection automation, or adding forecast visualization features.
