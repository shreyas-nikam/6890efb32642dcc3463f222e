import pytest
import pandas as pd
import numpy as np

# This block is for the placeholder as per instructions.
from definition_f26f837dce3e4351a07661c3d873d6cb import inverse_transform_forecast

# --- Helper functions for calculating EXPECTED values in tests ---
# These functions simulate the inverse logic that 'inverse_transform_forecast' should implement.
# They are internal to the test file and are used to derive the expected output,
# independent of the actual function's implementation.

def _inverse_log_helper(series: pd.Series) -> pd.Series:
    """Helper to calculate inverse log transformation."""
    return np.exp(series)

def _inverse_diff_helper(transformed_series: pd.Series, original_last_value: float, order: int = 1) -> pd.Series:
    """
    Helper to calculate inverse differencing.
    transformed_series: The differenced forecast series.
    original_last_value: The last value of the *original* (or pre-differenced) series
                         from which differencing started.
    """
    if order != 1:
        raise NotImplementedError("Only order 1 differencing inverse is supported for test helpers.")
    
    inverted_series = pd.Series(index=transformed_series.index, dtype=float)
    current_value = original_last_value
    for i, diff_val in enumerate(transformed_series):
        current_value += diff_val
        inverted_series.iloc[i] = current_value
    return inverted_series

# --- Test Cases ---

# Test 1: Basic functionality - Inverse Log transformation
def test_inverse_log_transformation():
    """
    Tests the inverse transformation for a column that was log-transformed.
    Original data is provided but not directly used in the inverse log calculation,
    as it's a point-wise inverse.
    """
    # Setup: Forecast values are already log-transformed
    dates_forecast = pd.date_range(start='2025-01-01', periods=3, freq='QS')
    transformed_forecast = pd.DataFrame({'value': [np.log(10.0), np.log(15.0), np.log(20.0)]}, index=dates_forecast)
    
    # Original data before transformation (not directly used for log inverse calculation, but needed by function signature)
    dates_original = pd.date_range(start='2024-01-01', periods=3, freq='QS')
    original_data = pd.DataFrame({'value': [5.0, 6.0, 8.0]}, index=dates_original)
    
    transformation_metadata = {'value': [{'type': 'log'}]}

    # Expected output: Each forecast value should be exp(itself)
    expected_forecast_values = _inverse_log_helper(transformed_forecast['value'])
    expected_forecast = pd.DataFrame({'value': expected_forecast_values}, index=dates_forecast)

    # Execute and Assert
    result_forecast = inverse_transform_forecast(transformed_forecast, original_data, transformation_metadata)
    pd.testing.assert_frame_equal(result_forecast, expected_forecast)

# Test 2: Basic functionality - Inverse Differencing transformation (Order 1)
def test_inverse_diff_transformation():
    """
    Tests the inverse differencing transformation.
    This requires the last value from the original data to correctly reconstruct the series.
    """
    # Setup: Original data from which differencing started
    dates_original = pd.date_range(start='2024-01-01', periods=5, freq='QS')
    original_data = pd.DataFrame({'value': [100.0, 102.0, 105.0, 107.0, 110.0]}, index=dates_original)
    
    # Forecast values are the predicted differences
    dates_forecast = pd.date_range(start='2025-01-01', periods=3, freq='QS')
    transformed_forecast = pd.DataFrame({'value': [2.0, 3.0, 4.0]}, index=dates_forecast)
    transformation_metadata = {'value': [{'type': 'diff', 'order': 1}]}

    # Expected output calculation:
    # Get the last original value to serve as the base for inverse differencing
    last_original_value = original_data['value'].iloc[-1] # This is 110.0
    
    # Apply inverse differencing to the forecast
    # 110 + 2 = 112
    # 112 + 3 = 115
    # 115 + 4 = 119
    expected_forecast_values = _inverse_diff_helper(transformed_forecast['value'], last_original_value)
    expected_forecast = pd.DataFrame({'value': expected_forecast_values}, index=dates_forecast)

    # Execute and Assert
    result_forecast = inverse_transform_forecast(transformed_forecast, original_data, transformation_metadata)
    pd.testing.assert_frame_equal(result_forecast, expected_forecast)

# Test 3: Edge Case - Combined transformations (Log then Diff)
def test_inverse_log_then_diff_transformation():
    """
    Tests a scenario where a column underwent multiple transformations (log then diff).
    The inverse operations must be applied in reverse order (inverse diff then inverse log),
    and original_data must be correctly used to establish the base for the cumulative inverse diff.
    """
    # Setup: Original data before any transformation
    dates_original = pd.date_range(start='2024-01-01', periods=3, freq='QS')
    original_data = pd.DataFrame({'value': [100.0, 110.0, 121.0]}, index=dates_original) # Last value is 121.0
    
    # Forecast values are the predicted log-differences
    dates_forecast = pd.date_range(start='2025-01-01', periods=2, freq='QS')
    transformed_forecast = pd.DataFrame({'value': [0.10, 0.11]}, index=dates_forecast) # Example: log(121 * (1 + 0.10)) - log(121) = 0.10
    
    # Metadata specifies the forward transformations in order of application
    transformation_metadata = {'value': [{'type': 'log'}, {'type': 'diff', 'order': 1}]}

    # Expected output calculation:
    # 1. Get the last original value from original_data: 121.0
    # 2. Apply the *first forward transformation* (log) to this last original value to get the base for inverse diff
    #    This is crucial: inverse differencing needs the last value from the series *before it was differenced*.
    #    Since the original data was log-transformed then differenced, the base for inverse diff is the log of the last original value.
    base_for_inverse_diff = np.log(original_data['value'].iloc[-1]) # np.log(121.0) approx 4.79579
    
    # 3. Apply inverse differencing to the transformed forecast using this base
    #    This will give us the forecast values in the *log-transformed* space.
    #    Forecasted log values: [4.79579 + 0.10 = 4.89579, 4.89579 + 0.11 = 5.00579]
    temp_inverse_diff_values = _inverse_diff_helper(transformed_forecast['value'], base_for_inverse_diff)
    
    # 4. Apply inverse log to the result of step 3 to get values in the original scale
    #    Final forecast: [exp(4.89579) approx 133.0489, exp(5.00579) approx 149.2783]
    expected_forecast_values = _inverse_log_helper(temp_inverse_diff_values)
    expected_forecast = pd.DataFrame({'value': expected_forecast_values}, index=dates_forecast)

    # Execute and Assert (use rtol for floating-point comparisons)
    result_forecast = inverse_transform_forecast(transformed_forecast, original_data, transformation_metadata)
    pd.testing.assert_frame_equal(result_forecast, expected_forecast, rtol=1e-5)

# Test 4: Edge Case - Empty transformed forecast DataFrame
def test_empty_transformed_forecast():
    """
    Tests handling of an empty transformed forecast DataFrame.
    The function should return an empty DataFrame with the same structure.
    """
    # Setup
    dates_forecast = pd.date_range(start='2025-01-01', periods=0, freq='QS') # Empty index
    transformed_forecast = pd.DataFrame({'value': []}, index=dates_forecast) # Empty DataFrame
    
    dates_original = pd.date_range(start='2024-01-01', periods=3, freq='QS')
    original_data = pd.DataFrame({'value': [10.0, 20.0, 30.0]}, index=dates_original)
    
    transformation_metadata = {'value': [{'type': 'log'}]}

    # Expected output: An empty DataFrame mirroring the input structure
    expected_forecast = pd.DataFrame({'value': []}, index=dates_forecast)

    # Execute and Assert
    result_forecast = inverse_transform_forecast(transformed_forecast, original_data, transformation_metadata)
    pd.testing.assert_frame_equal(result_forecast, expected_forecast)

# Test 5: Expected functionality - Column not found in metadata (should pass through unchanged)
def test_column_not_in_metadata_passthrough():
    """
    Tests that columns present in the transformed_forecast but not specified in
    the transformation_metadata are passed through to the output unchanged.
    """
    # Setup
    dates_forecast = pd.date_range(start='2025-01-01', periods=3, freq='QS')
    transformed_forecast = pd.DataFrame({
        'transformed_col': [np.log(10.0), np.log(15.0), np.log(20.0)], # This column will be inverse-transformed
        'untransformed_col': [100.0, 110.0, 120.0]                       # This column should pass through as is
    }, index=dates_forecast)
    
    dates_original = pd.date_range(start='2024-01-01', periods=3, freq='QS')
    original_data = pd.DataFrame({
        'transformed_col': [5.0, 6.0, 8.0],
        'untransformed_col': [90.0, 95.0, 100.0]
    }, index=dates_original)
    
    transformation_metadata = {'transformed_col': [{'type': 'log'}]} # Metadata only for 'transformed_col'

    # Expected output: 'transformed_col' is inverse-log transformed; 'untransformed_col' is identical to input
    expected_transformed_col_values = _inverse_log_helper(transformed_forecast['transformed_col'])
    expected_forecast = pd.DataFrame({
        'transformed_col': expected_transformed_col_values,
        'untransformed_col': [100.0, 110.0, 120.0]
    }, index=dates_forecast)

    # Execute and Assert
    result_forecast = inverse_transform_forecast(transformed_forecast, original_data, transformation_metadata)
    pd.testing.assert_frame_equal(result_forecast, expected_forecast)