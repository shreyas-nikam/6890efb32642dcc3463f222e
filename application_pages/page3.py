
import streamlit as st
import pandas as pd
import statsmodels.tsa.api as smt
import statsmodels.stats.diagnostic as smd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

def validate_arimax_inputs(df, target_col, exog_cols, order):
    """Validates that the target and exogenous columns exist in the DataFrame."""
    if target_col not in df.columns:
        raise ValueError(f"Target column \"{target_col}\" not found in DataFrame.")
    for col in exog_cols:
        if col not in df.columns:
            raise ValueError(f"Exogenous column \"{col}\" not found in DataFrame.")
    if not all(isinstance(o, int) and o >= 0 for o in order):
        raise ValueError("ARIMA order must be a tuple of non-negative integers (p, d, q).")

def prepare_arimax_data(df, target_col, exog_cols):
    """Prepares the endogenous and exogenous data for ARIMAX model fitting."""
    endog_series = df[target_col]
    exog_df = df[exog_cols]
    return endog_series, exog_df


def fit_arimax_model(endog_series, exog_df, order):
    """Fits an ARIMAX model to the given data."""
    try:
        model = ARIMA(endog_series, exog=exog_df, order=order)
        fitted_model = model.fit()
        return fitted_model
    except Exception as e:
        raise RuntimeError(f"Error fitting ARIMAX model: {e}")

def compute_residual_diagnostics(fitted_model):
    """Computes Ljung-Box test for residual autocorrelation."""
    ljung_box_results = smd.acorr_ljungbox(fitted_model.resid, lags=[10], return_df=True)
    return ljung_box_results

def plot_residual_diagnostics(fitted_model, model_name="ARIMAX"):
    """Plots ACF and PACF of the model residuals."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 4))
    smt.graphics.plot_acf(fitted_model.resid, lags=10, ax=axes[0], title=f'{model_name} Residuals ACF')
    smt.graphics.plot_pacf(fitted_model.resid, lags=10, ax=axes[1], title=f'{model_name} Residuals PACF')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

def interpret_ljung_box_results(ljung_box_results, significance_level=0.05):
    """Interprets the Ljung-Box test results."""
    p_value = ljung_box_results['lb_pvalue'].iloc[0]  # Get the p-value for the first lag
    if p_value > significance_level:
        message = "The Ljung-Box test indicates that the residuals are likely independently distributed (no significant autocorrelation)."
        status = "adequate"
    else:
        message = "The Ljung-Box test suggests that there is significant autocorrelation in the residuals. The model may need refinement."
        status = "needs_refinement"

    interpretation = {
        "message": message,
        "status": status
    }
    return interpretation


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
    st.markdown("""
    In this step, we build an **ARIMAX(p,d,q)** model to forecast the differenced default rate, using macroeconomic variables as additional inputs.  
    This model combines the strengths of ARIMA (capturing patterns in the target series) with the influence of external factors such as GDP growth.

    You’ll start by selecting the ARIMA order — the values of **p**, **d**, and **q** that define how the model uses past values, differences, and past errors.  
    Once the model is trained, we evaluate it using **AIC** and **BIC**, where lower values mean a better balance between accuracy and complexity.

    But good fit is not enough — we also check the model’s residuals.  
    The **Ljung–Box test** and the **ACF/PACF plots** help confirm whether the model has captured all meaningful patterns, leaving residuals that are effectively random.  
    If the p-value in the Ljung–Box test is above 0.05, it suggests no significant autocorrelation remains, meaning the model is statistically adequate.

    By the end of this step, you’ll see:
    - A complete model summary.
    - Residual diagnostics to assess model adequacy.
    - A comparison table showing how different models perform, with the best one highlighted and ready for download on the next page.
    """)



    # Model Parameters setup
    target_column = 'Default_Rate_Diff'
    # Note: The provided notebook uses only 'GDP_Growth_YoY(%)' as exogenous.
    # The application can be extended to allow selection of 'Unemployment(%)' as well.
    exogenous_columns = ['GDP_Growth_YoY(%)']

    p = st.number_input("Enter ARIMA Order (p):", min_value=0, value=1, key='p_input')
    d = st.number_input("Enter ARIMA Order (d):", min_value=0, value=0, key='d_input')
    q = st.number_input("Enter ARIMA Order (q):", min_value=0, value=0, key='q_input')
    order = (p, d, q)

    trained = False
    if 'df_transformed' in st.session_state and st.button("Train Selected ARIMAX Model", key='train_button'):
        df_transformed = st.session_state['df_transformed']
        try:
            fitted_model, ljung_box_results = train_arimax(df_transformed, target_column, exogenous_columns, order)
            st.session_state['fitted_model'] = fitted_model  # Store for persistence
            st.session_state['ljung_box_results'] = ljung_box_results  # Store for display
            trained = True
        except Exception as e:
            st.error(f"Error during model training or diagnostics: {e}")

    if 'ljung_box_results' in st.session_state:
        # Show model summary and diagnostics if available
        if 'fitted_model' in st.session_state:
            fitted_model = st.session_state['fitted_model']
            st.subheader(f"ARIMAX{order} Model Summary:")
            st.code(fitted_model.summary().as_text())
            st.markdown("""
            **Model Summary Explanation:**
            - The table above shows the estimated coefficients for each parameter in the ARIMAX model, along with their standard errors, z-values, and p-values.
            - Lower AIC/BIC values indicate a better model fit with less complexity.
            - Significant coefficients (p < 0.05) suggest those variables have a meaningful impact on the target.
            - Review the log-likelihood, AIC, and BIC to compare models.
            """)

            st.subheader("Residual Diagnostics:")
            plot_residual_diagnostics(fitted_model, f"ARIMAX{order}")
            st.markdown("""
            **Residual Diagnostics Explanation:**
            - The ACF and PACF plots above show the autocorrelation and partial autocorrelation of the model residuals.
            - Ideally, residuals should not show significant autocorrelation, indicating the model has captured all meaningful patterns.
            - Use the Ljung-Box test below to statistically confirm if residuals are random (p > 0.05 is desired).
            """)

        st.subheader("Ljung-Box Test Results:")
        st.markdown("""
        The Ljung-Box test checks whether the residuals from the ARIMAX model are random (i.e., show no significant autocorrelation). A high p-value (above the selected significance level) suggests the model has adequately captured the patterns in the data.
        """)
        ljung_box_results = st.session_state['ljung_box_results']
        st.dataframe(ljung_box_results)

        significance_level = st.slider("Significance Level for Ljung-Box Test:", min_value=0.01, max_value=0.10, value=0.05, step=0.01, key='lb_level')
        interpretation = interpret_ljung_box_results(ljung_box_results, significance_level=significance_level)
        st.markdown(f"**Interpretation:** {interpretation['message']}")
        if interpretation['status'] == 'needs_refinement':
            st.write(f"  Problematic lags: {interpretation.get('problematic_lags', [])}")

        # For model comparison, results could be appended to a list in session_state over multiple runs.
        if 'fitted_model' in st.session_state:
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
            st.markdown("""
            This table summarizes the performance of each ARIMAX model you have trained. Compare AIC and BIC values to select the best model—lower values indicate a better fit with less complexity. The best model based on AIC is highlighted below.
            """)
            comparison_df = pd.DataFrame(st.session_state['model_comparison_data'])
            st.dataframe(comparison_df.round(2))

            best_model_idx = comparison_df['AIC'].idxmin()
            best_model_name = comparison_df.loc[best_model_idx, 'Model']
            st.write(f"\n**Best model based on AIC:** {best_model_name}")
    else:
        st.info("Please load synthetic data first and then train the model.")

if __name__ == "__main__":
    run_page3()
