
# Streamlit Application Requirements Specification

## 1. Application Overview

The purpose of this Streamlit application is to provide an interactive platform for estimating and diagnosing an ARIMAX (Autoregressive Integrated Moving Average with Exogenous Inputs) model for credit risk forecasting. Specifically, it aims to predict **Segment A default rates** by incorporating key macroeconomic variables.

**Objectives:**
*   **Data Loading & Alignment:** Enable users to load and view a synthetic dataset simulating Taiwan's quarterly credit-risk data alongside core macroeconomic drivers (GDP growth %, Unemployment %).
*   **Stationarity Transformation & Verification:** Guide users through applying stationarity transformations (e.g., differencing) and verifying these transformations using formal statistical tests (Augmented Dickey-Fuller (ADF) and Phillips-Perron (PP) tests).
*   **ARIMAX Model Estimation:** Allow users to specify and estimate an $ARIMAX(p,d,q)$ model for the transformed default rate, utilizing macroeconomic variables as exogenous regressors. The model order selection will be guided by information criteria (AIC/BIC).
*   **Residual Diagnostics:** Provide comprehensive diagnostic checks on model residuals, including visual inspection via Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) plots, and formal statistical testing using the Ljung-Box test.
*   **Model Persistence:** Facilitate the saving of the best-performing fitted ARIMAX model for future use and deployment.

## 2. User Interface Requirements

The Streamlit application will feature a clear, step-by-step layout to guide the user through the modeling process.

### Layout and Navigation Structure
*   The application will be structured into logical sections corresponding to the key steps: "Data Loading & Exploration", "Data Pre-processing (Stationarity)", and "ARIMAX Model Estimation & Diagnostics", and "Model Persistence".
*   These sections can be navigated sequentially, possibly via a sidebar or expandable sections.

### Input Widgets and Controls
*   **Data Loading:**
    *   A button to trigger the loading and display of the synthetic dataset.
        *   `st.button("Load Synthetic Data")`
*   **ARIMAX Order Selection:**
    *   Numeric input fields for users to specify the $p, d, q$ orders of the ARIMAX model.
        *   `st.number_input("ARIMA Order (p):", min_value=0, value=1)`
        *   `st.number_input("ARIMA Order (d):", min_value=0, value=0)`
        *   `st.number_input("ARIMA Order (q):", min_value=0, value=0)`
    *   *(Future Enhancement):* Dropdowns or multi-select widgets to select target and exogenous variables from the loaded dataset if a more flexible data input mechanism is introduced.
*   **Model Training Trigger:**
    *   A button to initiate the ARIMAX model training process with the selected orders.
        *   `st.button("Train ARIMAX Model")`
*   **Significance Level Input:**
    *   A slider or number input for setting the significance level for interpreting Ljung-Box test results.
        *   `st.slider("Significance Level for Ljung-Box Test:", min_value=0.01, max_value=0.10, value=0.05, step=0.01)`

### Visualization Components
*   **Initial Data Visualization:**
    *   **Overlay Plot:** An interactive plot displaying "Segment A Default Rate" and "GDP Growth YoY%" over time, allowing for visual inspection of their relationship.
    *   **Individual Time Series Plots:** Separate plots for "Segment A Default Rate", "GDP Growth YoY%", and "Unemployment %".
*   **Stationarity Test Results:**
    *   Tables displaying the results of the ADF and KPSS tests for original and transformed time series.
*   **Model Diagnostics:**
    *   **Residual ACF Plot:** A plot of the Autocorrelation Function of the model residuals.
    *   **Residual PACF Plot:** A plot of the Partial Autocorrelation Function of the model residuals.
    *   **Ljung-Box Test Results Table:** A table showing the Ljung-Box test statistics and p-values for various lags.
*   **Model Comparison:**
    *   A table summarizing key metrics (AIC, BIC, Log-Likelihood, Number of Parameters) for different models that have been trained (if multiple runs are performed and stored).

### Interactive Elements and Feedback Mechanisms
*   **Progress Indicators:** Display loading spinners or progress bars during data processing and model training to provide user feedback on ongoing operations.
*   **Model Summary Display:** Present the detailed `statsmodels` summary of the fitted ARIMAX model.
*   **Diagnostic Interpretation:** Provide clear textual interpretations for unit root tests and Ljung-Box test results (e.g., "Series is Stationary", "Model is adequate", "Model may need refinement").
*   **Error Handling:** Display informative error messages if model fitting or data processing fails.
*   **Model Download:** A button to allow users to download the fitted ARIMAX model as a `.pkl` file.

## 3. Additional Requirements

*   **Real-time Updates and Responsiveness:** The application should be responsive, updating displayed results and visualizations dynamically as users interact with input widgets or trigger model training.
*   **Annotation and Tooltip Specifications:**
    *   All plots must have clear titles, labeled axes, and legends.
    *   Explanations for mathematical formulas will be provided using LaTeX markdown.
    *   Interpretations for statistical test results (ADF, KPSS, Ljung-Box) and model evaluation criteria (AIC, BIC) will be displayed.
*   **Model Persistence:** The application must allow for the serialization and download of the best-performing fitted ARIMAX model, enabling its reuse outside the application.

## 4. Notebook Content and Code Requirements

This section outlines the specific code components from the Jupyter Notebook and their integration into the Streamlit application.

### Global Variables and Libraries
*   **Random Seed:** `RANDOM_SEED = 42` will be set globally for reproducibility.
*   **Required Libraries:**
    ```python
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import statsmodels.tsa.api as smt
    import statsmodels.stats.diagnostic as smd
    import pickle
    import os
    import streamlit as st
    from ucimlrepo import fetch_ucirepo # For synthetic data loading
    ```

### 1. Data Loading and Initial Exploration (`01_load_data`)

*   **Description:** This section loads or generates the synthetic time series data mimicking the Taiwan credit risk and macroeconomic dataset. It displays initial data insights and visualizations.
*   **Streamlit Integration:** This will be the initial section of the app. A button can trigger the data generation and display.
*   **Relevant Code:**
    ```python
    # Synthetic Time Series Data Generation (to be executed on app startup or button click)
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

    # Display (using Streamlit components)
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
    plt.close(fig) # Prevent duplicate plots
    ```

### 2. Data Pre-processing for Stationarity (`02_preprocess`)

*   **Description:** This section defines a helper function for unit root tests and applies differencing to the target variable to achieve stationarity, then verifies it.
*   **Streamlit Integration:** This section will guide the user through the transformations and display test results.
*   **Relevant Code:**
    ```python
    def run_unit_root_tests(series, name):
        """
        Performs Augmented Dickey-Fuller (ADF) and Kwiatkowski-Phillips-Schmidt-Shin (KPSS) tests
        for stationarity on a given time series.
        """
        st.write(f"\n--- Unit Root Tests for: **{name}** ---")

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

    st.subheader("Transformations for Stationarity:")
    st.markdown(r"""
    *   **Year-over-Year (YoY) Percentage Change**:
        $$ Y_{t}^{YoY} = \frac{Y_t - Y_{t-4}}{Y_{t-4}} \times 100 $$
    *   **Log-Differencing**:
        $$ \Delta \ln(Y_t) = \ln(Y_t) - \ln(Y_{t-1}) $$
    """)

    if st.button("Apply Transformations and Test Stationarity"):
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

        st.session_state['df_transformed'] = df_transformed # Store for next step
    ```

### 3. ARIMAX Model Estimation and Diagnostics (`03_train_arimax`)

*   **Description:** This section contains the core logic for ARIMAX model training, summary display, and residual diagnostics.
*   **Streamlit Integration:** This will be the main interactive component, allowing users to select parameters and view results.
*   **Relevant Code:**
    ```python
    def validate_arimax_inputs(df, target_col, exog_cols, order):
        # ... (function as in notebook)
        pass

    def prepare_arimax_data(df, target_col, exog_cols):
        # ... (function as in notebook)
        pass

    def fit_arimax_model(endog_series, exog_df, order):
        # ... (function as in notebook)
        pass

    def compute_residual_diagnostics(fitted_model):
        # ... (function as in notebook)
        pass

    def plot_residual_diagnostics(fitted_model, model_name="ARIMAX"):
        fig, axes = plt.subplots(1, 2, figsize=(16, 4))
        smt.graphics.plot_acf(fitted_model.resid, lags=10, ax=axes[0], title=f'{model_name} Residuals ACF')
        smt.graphics.plot_pacf(fitted_model.resid, lags=10, ax=axes[1], title=f'{model_name} Residuals PACF')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    def interpret_ljung_box_results(ljung_box_results, significance_level=0.05):
        # ... (function as in notebook)
        pass

    def train_arimax(df, target_col, exog_cols, order):
        # This function orchestrates the calls to other helper functions
        with st.spinner(f"Training ARIMAX{order} Model..."):
            validate_arimax_inputs(df, target_col, exog_cols, order)
            endog_series, exog_df = prepare_arimax_data(df, target_col, exog_cols)
            fitted_model = fit_arimax_model(endog_series, exog_df, order)
            ljung_box_results = compute_residual_diagnostics(fitted_model)
        st.success(f"ARIMAX{order} model training completed!")
        return fitted_model, ljung_box_results

    st.subheader("ARIMAX Model Equation:")
    st.markdown(r"""
    The general form of an $ARIMAX(p,d,q)$ model with $m$ exogenous regressors can be expressed as:
    $$ (1 - \sum_{i=1}^{p} \phi_i L^i) (1 - L)^d Y_t = c + (1 + \sum_{j=1}^{q} \theta_j L^j) \epsilon_t + \sum_{k=1}^{m} \beta_k X_{k,t} $$
    Where:
    *   $Y_t$: The target variable at time $t$.
    *   $L$: The lag operator.
    *   $p$: The order of the autoregressive (AR) part.
    *   $\phi_i$: AR coefficients.
    *   $d$: The order of differencing (I for integrated part).
    *   $q$: The order of the moving average (MA) part.
    *   $\theta_j$: MA coefficients.
    *   $c$: A constant term.
    *   $\epsilon_t$: The white noise error term (residuals) at time $t$.
    *   $X_{k,t}$: The $k$-th exogenous variable at time $t$.
    *   $\beta_k$: The coefficients for the exogenous variables.
    """)

    st.subheader("Model Order Selection Criteria:")
    st.markdown(r"""
    *   **Akaike Information Criterion (AIC)**:
        $$ AIC = -2 \ln(L) + 2k $$
    *   **Bayesian Information Criterion (BIC)**:
        $$ BIC = -2 \ln(L) + k \ln(n) $$
    Where: $L$ is the maximum likelihood, $k$ is the number of parameters, and $n$ is the number of observations. Lower values for AIC and BIC indicate a better model.
    """)

    st.subheader("Residual Diagnostics (Ljung-Box Test):")
    st.markdown(r"""
    The Ljung-Box test statistic $Q$ is given by:
    $$ Q = n(n+2) \sum_{k=1}^{h} \frac{\hat{\rho}_k^2}{n-k} $$
    Where: $n$ is the number of observations, $\hat{\rho}_k$ is the sample autocorrelation at lag $k$, and $h$ is the number of lags tested. A large p-value (typically $> 0.05$) supports no remaining autocorrelation.
    """)

    # Model Parameters setup
    target_column = 'Default_Rate_Diff'
    # Note: The provided notebook uses only 'GDP_Growth_YoY_%' as exogenous.
    # The application can be extended to allow selection of 'Unemployment_%' as well.
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
            st.session_state['fitted_model'] = fitted_model # Store for persistence
            st.session_state['ljung_box_results'] = ljung_box_results # Store for display

            st.subheader(f"ARIMAX{order} Model Summary:")
            st.code(fitted_model.summary().as_text())

            st.subheader("Residual Diagnostics:")
            plot_residual_diagnostics(fitted_model, f"ARIMAX{order}")

            st.write("\nLjung-Box Test Results:")
            st.dataframe(ljung_box_results)
            
            significance_level = st.slider("Significance Level for Ljung-Box Test:", min_value=0.01, max_value=0.10, value=0.05, step=0.01, key='lb_level')
            interpretation = interpret_ljung_box_results(ljung_box_results, significance_level=significance_level)
            st.markdown(f"**Interpretation:** {interpretation['message']}")
            if interpretation['status'] == 'needs_refinement':
                st.write(f"  Problematic lags: {interpretation.get('problematic_lags', [])}")

            # For model comparison, results could be appended to a list in session_state over multiple runs.
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

### 4. Model Persistence

*   **Description:** This section allows the user to save the currently best-performing fitted ARIMAX model.
*   **Streamlit Integration:** A download button will be provided to facilitate saving.
*   **Relevant Code:**
    ```python
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

This specification outlines the necessary components and their integration for building the Streamlit application based on the provided Jupyter Notebook and requirements.
