import pytest
import pandas as pd
import numpy as np

# DO NOT REPLACE or REMOVE this block
from definition_12c3e16c358545d8a7db3b8e3996998d import apply_transformations
# DO NOT REPLACE or REMOVE this block

def test_apply_transformations_standard_cases():
    """
    Test case 1: Happy path with multiple columns and different valid transformations.
    Covers 'log_diff', 'percent_change', and 'diff'.
    """
    data = {
        'date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05']),
        'gdp': [100, 105, 110.25, 115.76, 121.55], # ~5% growth
        'cpi': [50, 51, 52.02, 53.06, 54.12], # ~2% growth
        'unemployment': [10, 11, 10.5, 12, 11.5]
    }
    df = pd.DataFrame(data).set_index('date')
    
    transform_map = {
        'gdp': 'log_diff',
        'cpi': 'percent_change',
        'unemployment': 'diff'
    }
    
    transformed_df = apply_transformations(df.copy(), transform_map)
    
    # Expected calculations
    expected_gdp_transformed = np.log(df['gdp']).diff()
    expected_cpi_transformed = df['cpi'].pct_change() * 100
    expected_unemployment_transformed = df['unemployment'].diff()
    
    # Assertions using pandas.testing for series equality
    pd.testing.assert_series_equal(transformed_df['gdp'], expected_gdp_transformed, check_dtype=False, rtol=1e-05, atol=1e-08)
    pd.testing.assert_series_equal(transformed_df['cpi'], expected_cpi_transformed, check_dtype=False, rtol=1e-05, atol=1e-08)
    pd.testing.assert_series_equal(transformed_df['unemployment'], expected_unemployment_transformed, check_dtype=False, rtol=1e-05, atol=1e-08)
    
    # Ensure other columns not in transform_map remain unchanged (not applicable here as all are transformed)
    # Ensure dataframe index is preserved
    pd.testing.assert_index_equal(transformed_df.index, df.index)


def test_apply_transformations_empty_dataframe():
    """
    Test case 2: Edge case - Input an empty DataFrame.
    Expected: An empty DataFrame with the same columns.
    """
    df = pd.DataFrame({'col_a': [], 'col_b': []})
    transform_map = {'col_a': 'diff'}
    
    transformed_df = apply_transformations(df.copy(), transform_map)
    
    assert transformed_df.empty
    assert list(transformed_df.columns) == list(df.columns)


@pytest.mark.parametrize("dataframe_input, transform_map_input, expected_outcome", [
    # Test case 3a: Column specified in transform_map does not exist in DataFrame
    (pd.DataFrame({'A': [1,2,3], 'B': [4,5,6]}), {'C': 'diff'}, pytest.raises(KeyError)),
    # Test case 3b: Invalid transformation type string
    (pd.DataFrame({'A': [1,2,3]}), {'A': 'unsupported_transform'}, pytest.raises(ValueError)),
    # Test case 3c: Empty transform_map (should return original DataFrame unchanged)
    (pd.DataFrame({'A': [1,2,3], 'B': [4,5,6]}), {}, pd.DataFrame({'A': [1,2,3], 'B': [4,5,6]}))
])
def test_apply_transformations_invalid_inputs(dataframe_input, transform_map_input, expected_outcome):
    """
    Test case 3: Edge cases - Column not found, invalid transformation type, and empty transform_map.
    Uses parametrize to cover multiple scenarios within one test function.
    """
    if isinstance(expected_outcome, type) and issubclass(expected_outcome, Exception):
        with expected_outcome:
            apply_transformations(dataframe_input, transform_map_input)
    else:  # This branch handles the case where transform_map is empty
        result_df = apply_transformations(dataframe_input, transform_map_input)
        pd.testing.assert_frame_equal(result_df, expected_outcome)


def test_apply_transformations_non_numeric_data():
    """
    Test case 4: Edge case - Column contains non-numeric data when a numeric transformation is applied.
    Expected: A TypeError or ValueError from pandas' underlying operations.
    """
    data = {
        'date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']),
        'value': [10, 20, 'thirty'] # Non-numeric string
    }
    df = pd.DataFrame(data).set_index('date')
    transform_map = {'value': 'diff'}
    
    # Expecting pandas to raise a TypeError or ValueError when attempting arithmetic on non-numeric data
    with pytest.raises((TypeError, ValueError)): 
        apply_transformations(df, transform_map)

def test_apply_transformations_single_row_and_nan_results():
    """
    Test case 5: Edge case - DataFrame with a single row, and behavior when transformations
    (like diff, log_diff, pct_change) inherently produce NaNs (e.g., first row or operations with NaNs).
    """
    # Test with a single-row DataFrame
    data_single_row = {
        'date': pd.to_datetime(['2023-01-01']),
        'metric_a': [100],
        'metric_b': [50]
    }
    df_single_row = pd.DataFrame(data_single_row).set_index('date')
    
    transform_map_single = {
        'metric_a': 'diff',
        'metric_b': 'log_diff'
    }
    
    transformed_single_row_df = apply_transformations(df_single_row.copy(), transform_map_single)
    
    # For a single row, 'diff' and 'log_diff' will always result in NaN
    assert transformed_single_row_df['metric_a'].isna().all()
    assert transformed_single_row_df['metric_b'].isna().all()
    assert len(transformed_single_row_df) == 1 # Length should remain 1
    pd.testing.assert_index_equal(transformed_single_row_df.index, df_single_row.index)

    # Test with a column containing NaNs that would propagate through transformations
    data_with_nans = {
        'date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']),
        'series_with_nan': [10, np.nan, 30]
    }
    df_with_nans = pd.DataFrame(data_with_nans).set_index('date')
    transform_map_nans = {'series_with_nan': 'percent_change'}

    transformed_df_with_nans = apply_transformations(df_with_nans.copy(), transform_map_nans)

    # Expected: [NaN, NaN, NaN] as percent_change(NaN, previous) = NaN and percent_change(current, NaN) = NaN
    expected_series_nans = pd.Series([np.nan, np.nan, np.nan], index=df_with_nans.index, name='series_with_nan')
    pd.testing.assert_series_equal(transformed_df_with_nans['series_with_nan'], expected_series_nans, check_dtype=False)