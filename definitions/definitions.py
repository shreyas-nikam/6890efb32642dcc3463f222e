import pandas as pd
import numpy as np
import statsmodels.tsa.api as smt
import statsmodels.stats.diagnostic as smd

def train_arimax(df, target_col, exog_cols, order):
    """
    Trains an ARIMAX model using `statsmodels` with the specified target, exogenous variables, and model order.
    It then performs and provides residual diagnostics, including Ljung-Box test results, along with model selection criteria like AIC and BIC.

    Arguments:
        df (pandas.DataFrame): The DataFrame containing the target and exogenous time series.
        target_col (str): The name of the target column in `df`.
        exog_cols (list[str]): A list of column names in `df` to be used as exogenous variables.
        order (tuple): A tuple `(p, d, q)` specifying the order of the ARIMAX model, where p is the AR order, d is the differencing order, and q is the MA order.

    Output:
        tuple: A tuple containing the fitted `statsmodels` ARIMAX model object and a pandas.DataFrame with Ljung-Box test results for the residuals, including p-values for various lags.
    """

    # 1. Input Validation
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    if df.empty:
        raise ValueError("Input DataFrame cannot be empty.")

    if not isinstance(target_col, str):
        raise TypeError("target_col must be a string.")
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in DataFrame.")

    if not isinstance(exog_cols, list):
        raise TypeError("exog_cols must be a list of strings.")
    for col in exog_cols:
        if not isinstance(col, str):
            raise TypeError(f"All elements in exog_cols must be strings, found: {type(col)}")
        if col not in df.columns:
            raise KeyError(f"Exogenous column '{col}' not found in DataFrame.")

    if not isinstance(order, tuple) or len(order) != 3:
        raise TypeError("order must be a tuple of length 3 (p, d, q).")
    for val in order:
        if not isinstance(val, int) or val < 0:
            raise TypeError("All elements in order (p, d, q) must be non-negative integers.")

    # 2. Prepare data
    endog_series = df[target_col]
    
    # Check for non-numeric target data that cannot be coerced
    if not pd.api.types.is_numeric_dtype(endog_series):
        temp_endog = pd.to_numeric(endog_series, errors='coerce')
        if temp_endog.isnull().any(): # Check if coercion introduced NaNs, implying truly non-numeric data
            raise ValueError(f"Target column '{target_col}' contains non-numeric data that cannot be coerced to numeric.")
        endog_series = temp_endog # Use coerced numeric series if successful
        
    exog_df = None
    if exog_cols:
        exog_df = df[exog_cols]
        # Check for non-numeric exogenous data that cannot be coerced
        for col in exog_cols:
            if not pd.api.types.is_numeric_dtype(exog_df[col]):
                temp_exog_col = pd.to_numeric(exog_df[col], errors='coerce')
                if temp_exog_col.isnull().any():
                    raise ValueError(f"Exogenous column '{col}' contains non-numeric data that cannot be coerced to numeric.")
                exog_df[col] = temp_exog_col # Use coerced numeric column if successful
    
    # 3. Model Training
    # statsmodels ARIMA handles NaNs by dropping rows where NaNs are present in endog or exog.
    # If the remaining data is insufficient after dropping NaNs, statsmodels will raise an appropriate error.
    try:
        model = smt.ARIMA(endog=endog_series, exog=exog_df, order=order)
        fitted_model = model.fit()
    except (ValueError, np.linalg.LinAlgError, RuntimeError, TypeError) as e:
        # Catch common errors from statsmodels during fitting (e.g., data issues,
        # insufficient observations after differencing/dropping NaNs, numerical instability).
        # Re-raise the original exception as per test requirements.
        raise e

    # 4. Residual Diagnostics (Ljung-Box Test)
    residuals = fitted_model.resid
    
    # The Ljung-Box test requires a sufficient number of residuals.
    # It typically needs `lags + 1` observations at minimum.
    # Set a reasonable default for maximum lags (e.g., 20), ensuring it's less than `len(residuals)`.
    if len(residuals) < 2: 
        # Not enough data for meaningful Ljung-Box test (needs at least 2 residuals for 1 lag).
        ljung_box_results = pd.DataFrame(columns=['lb_stat', 'lb_pvalue', 'bp_stat', 'bp_pvalue'])
    else:
        # Determine the maximum number of lags to test.
        # Ensure it's at least 1, and less than the number of residuals.
        max_lags_for_test = min(20, len(residuals) - 1) 
        
        # If max_lags_for_test ends up being 0 (e.g., len(residuals) is 1), 
        # acorr_ljungbox will still raise an error, so ensure it's at least 1 if we proceed.
        if max_lags_for_test < 1:
            ljung_box_results = pd.DataFrame(columns=['lb_stat', 'lb_pvalue', 'bp_stat', 'bp_pvalue'])
        else:
            ljung_box_results = smd.acorr_ljungbox(residuals, lags=max_lags_for_test, return_df=True)
            # The 'lb_pvalue' column name is already correct when return_df=True.

    return fitted_model, ljung_box_results