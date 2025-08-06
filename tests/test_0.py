import pytest
import pandas as pd
import numpy as np
import statsmodels.tsa.api as smt

# Placeholder for the module import
from definition_cc5e64457e604768a0eeceee5615e5c5 import train_arimax

# Helper function to create a dummy DataFrame for testing
def _create_dummy_df(num_rows=50, include_exog=True, include_nan=False, non_numeric=False):
    """Creates a dummy pandas DataFrame for testing ARIMAX functions."""
    dates = pd.date_range(start='2010-01-01', periods=num_rows, freq='QS') # Quarterly frequency
    data = {
        'target_col': np.random.rand(num_rows) * 100 + 10 # Ensure values are positive for potential log transformations
    }
    if include_exog:
        data['exog_col_1'] = np.random.rand(num_rows) * 50
        data['exog_col_2'] = np.random.rand(num_rows) * 20
    
    df = pd.DataFrame(data, index=dates)
    df.index.name = 'Quarter'
    
    if non_numeric and num_rows > 0:
        df.loc[df.index[0], 'target_col'] = 'abc' # Introduce non-numeric value
    
    if include_nan and num_rows > 5:
        # Introduce NaNs in the middle of the series
        df.loc[df.index[int(num_rows/2)], 'target_col'] = np.nan
        if include_exog:
            df.loc[df.index[int(num_rows/2) + 1], 'exog_col_1'] = np.nan
            
    return df

@pytest.fixture
def dummy_df_with_exog():
    return _create_dummy_df(num_rows=100, include_exog=True)

@pytest.fixture
def dummy_df_no_exog():
    return _create_dummy_df(num_rows=100, include_exog=False)

@pytest.fixture
def dummy_df_small():
    return _create_dummy_df(num_rows=10, include_exog=True)


# Test Case 1: Happy Path with exogenous variables
def test_train_arimax_happy_path_with_exog(dummy_df_with_exog):
    """
    Tests the function with valid data, target, exogenous variables, and model order.
    Asserts that a fitted model object and a Ljung-Box DataFrame are returned,
    and that their basic properties (type, content) are as expected.
    """
    target_col = 'target_col'
    exog_cols = ['exog_col_1', 'exog_col_2']
    order = (1, 1, 1) # (p, d, q)

    fitted_model, ljung_box_results = train_arimax(dummy_df_with_exog, target_col, exog_cols, order)

    # Assertions for the fitted model object
    # Expecting an instance of statsmodels results wrapper, or at least an object with common attributes
    assert hasattr(fitted_model, 'aic'), "Fitted model should have an AIC attribute"
    assert hasattr(fitted_model, 'bic'), "Fitted model should have a BIC attribute"
    assert hasattr(fitted_model, 'summary'), "Fitted model should have a summary method"
    assert fitted_model.df_model > 0, "Model should have positive degrees of freedom"

    # Assertions for Ljung-Box results DataFrame
    assert isinstance(ljung_box_results, pd.DataFrame), "Ljung-Box results should be a DataFrame"
    assert not ljung_box_results.empty, "Ljung-Box results DataFrame should not be empty"
    assert 'lb_pvalue' in ljung_box_results.columns, "Ljung-Box DataFrame should contain 'lb_pvalue' column"
    assert ljung_box_results['lb_pvalue'].dtype in [np.float64, np.float32], "Ljung-Box p-values should be numeric"


# Test Case 2: Happy Path without exogenous variables (effectively an ARIMA model)
def test_train_arimax_no_exog(dummy_df_no_exog):
    """
    Tests the function's behavior when no exogenous variables are provided.
    It should gracefully handle an empty exog_cols list, effectively training an ARIMA model.
    """
    target_col = 'target_col'
    exog_cols = [] # No exogenous variables
    order = (1, 1, 1)

    fitted_model, ljung_box_results = train_arimax(dummy_df_no_exog, target_col, exog_cols, order)

    assert hasattr(fitted_model, 'aic'), "Fitted model should have an AIC attribute"
    assert isinstance(ljung_box_results, pd.DataFrame), "Ljung-Box results should be a DataFrame"
    assert not ljung_box_results.empty, "Ljung-Box results DataFrame should not be empty"


# Test Case 3: Invalid Input Types
@pytest.mark.parametrize("df_input, target_col, exog_cols, order, expected_error", [
    (None, 'target_col', ['exog_col_1'], (1,1,1), TypeError), # df is None
    (_create_dummy_df(), 123, ['exog_col_1'], (1,1,1), TypeError), # target_col is int
    (_create_dummy_df(), 'target_col', 'exog_col_1', (1,1,1), TypeError), # exog_cols is str
    (_create_dummy_df(), 'target_col', ['exog_col_1'], [1,1,1], TypeError), # order is list
    (_create_dummy_df(), 'target_col', ['exog_col_1'], (1,1), TypeError), # order wrong length (too short)
    (_create_dummy_df(), 'target_col', ['exog_col_1'], (1,1,1,1), TypeError), # order wrong length (too long)
])
def test_train_arimax_invalid_input_types(df_input, target_col, exog_cols, order, expected_error):
    """
    Tests scenarios where input arguments have incorrect data types, expecting TypeErrors.
    """
    with pytest.raises(expected_error):
        train_arimax(df_input, target_col, exog_cols, order)


# Test Case 4: Invalid Column Names or Data Content
@pytest.mark.parametrize("df_modifier, target_col, exog_cols, order, expected_error", [
    (lambda: _create_dummy_df(), 'non_existent_target', ['exog_col_1'], (1,1,1), KeyError), # Target col not in df
    (lambda: _create_dummy_df(), 'target_col', ['non_existent_exog'], (1,1,1), KeyError), # Exog col not in df
    (lambda: _create_dummy_df(num_rows=0), 'target_col', ['exog_col_1'], (1,1,1), ValueError), # Empty DataFrame
    (lambda: _create_dummy_df(non_numeric=True), 'target_col', ['exog_col_1'], (1,1,1), (ValueError, TypeError)), # Non-numeric data
    (lambda: _create_dummy_df(include_nan=True, num_rows=20), 'target_col', ['exog_col_1'], (1,1,1), (ValueError, RuntimeError, TypeError)), # NaNs, statsmodels might raise an error if not enough data after dropna
])
def test_train_arimax_invalid_data_or_columns(df_modifier, target_col, exog_cols, order, expected_error):
    """
    Tests scenarios where column names are incorrect, DataFrame is empty, or contains non-numeric/NaN data.
    """
    df = df_modifier()
    with pytest.raises(expected_error):
        train_arimax(df, target_col, exog_cols, order)


# Test Case 5: Insufficient Data or Invalid Order Values
@pytest.mark.parametrize("order, expected_error", [
    ((1, 1, 1), (ValueError, np.linalg.LinAlgError, RuntimeError)), # Small df for this order
    ((-1, 1, 1), ValueError), # Negative p order
    ((1, -1, 1), ValueError), # Negative d order
    ((1, 1, -1), ValueError), # Negative q order
    ((5, 5, 5), (ValueError, np.linalg.LinAlgError, RuntimeError)), # Very high order for small df
])
def test_train_arimax_insufficient_data_or_invalid_order_values(dummy_df_small, order, expected_error):
    """
    Tests cases where the DataFrame is too small for the specified model order,
    or where the ARIMAX order parameters (p, d, q) are invalid (e.g., negative).
    """
    target_col = 'target_col'
    exog_cols = ['exog_col_1'] # Include exog for more realistic small data scenario
    
    with pytest.raises(expected_error):
        train_arimax(dummy_df_small, target_col, exog_cols, order)