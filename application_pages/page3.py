
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
    st.markdown("This section allows users to specify and estimate an ARIMAX(p,d,q) model for the transformed default rate, utilizing macroeconomic variables as exogenous regressors. The model order selection will be guided by information criteria (AIC/BIC).")

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
            st.session_state['fitted_model'] = fitted_model  # Store for persistence
            st.session_state['ljung_box_results'] = ljung_box_results  # Store for display

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

if __name__ == "__main__":
    run_page3()
