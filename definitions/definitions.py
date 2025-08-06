import pandas as pd
import statsmodels.tsa.arima.model as sm_arima
import statsmodels.graphics.tsaplots as tsaplots
import statsmodels.stats.diagnostic as sm_diagnostic
import matplotlib.pyplot as plt

def train_arimax(df, target_col, exog_cols, order):
    """
    Instantiates and fits an ARIMAX model, performs residual diagnostics.
    
    Arguments:
        df (pd.DataFrame): DataFrame with target and exogenous variables.
        target_col (str): Name of the target variable column.
        exog_cols (list[str]): List of exogenous variable column names.
        order (tuple[int, int, int]): ARIMAX order (p, d, q).
    
    Output:
        tuple: FittedARIMAXModel object and diagnostic results DataFrame.
    """

    # Data Preparation
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in the DataFrame.")
    endog = df[target_col]

    exog = None
    if exog_cols:
        missing_exog_cols = [col for col in exog_cols if col not in df.columns]
        if missing_exog_cols:
            raise KeyError(f"Columns not found: {missing_exog_cols}")
        exog = df[exog_cols]

    # Model Instantiation and Fitting
    try:
        model = sm_arima.ARIMAX(endog=endog, exog=exog, order=order)
        fitted_model = model.fit()
    except Exception:
        # Re-raise exceptions from statsmodels' ARIMAX constructor or fit method
        raise

    # Comprehensive Residual Diagnostics
    residuals = fitted_model.resid

    # ACF and PACF plots
    # Determine appropriate lags based on residuals length, max 40
    max_lags_for_plots = max(1, min(len(residuals) // 2 - 1, 40))
    
    plt.figure(figsize=(10, 4))
    tsaplots.plot_acf(residuals, lags=max_lags_for_plots, ax=plt.gca(), title='ACF of Residuals')
    plt.tight_layout()

    plt.figure(figsize=(10, 4))
    tsaplots.plot_pacf(residuals, lags=max_lags_for_plots, ax=plt.gca(), title='PACF of Residuals')
    plt.tight_layout()

    # Ljung-Box test
    # Common lags for Ljung-Box. Ensure lags do not exceed residuals length - 1.
    ljung_box_lags = [lag for lag in [10, 20, 30] if lag <= len(residuals) - 1]
    if not ljung_box_lags and len(residuals) > 1: # Ensure at least one lag if data allows
        ljung_box_lags = [len(residuals) - 1]
    elif not ljung_box_lags: # If residuals are too short for any lag (e.g., length <= 1)
        # Create an empty DataFrame for diagnostics if no valid lags can be tested
        diagnostic_results = pd.DataFrame(columns=['Ljung-Box Statistic', 'P-value'], index=pd.Index([], name='Lag'))
        return fitted_model, diagnostic_results

    ljung_box_df = sm_diagnostic.acorr_ljungbox(residuals, lags=ljung_box_lags, return_df=True)

    # Format the Ljung-Box results DataFrame
    diagnostic_results = ljung_box_df.rename(columns={
        'lb_stat': 'Ljung-Box Statistic',
        'lb_pvalue': 'P-value'
    })
    diagnostic_results.index.name = 'Lag'

    # Extract model summary, AIC, and BIC (accessed for the purpose of the docstring)
    _ = fitted_model.summary()
    _ = fitted_model.aic
    _ = fitted_model.bic

    return fitted_model, diagnostic_results