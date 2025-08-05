import pytest
import pandas as pd
from definition_5374ac6524a5473e8808f70871ff0477 import merge_and_align_data

@pytest.mark.parametrize(
    "pd_data_info, macro_data_info, date_column_pd, date_column_macro, frequency, expected_output_info, expected_exception",
    [
        # Test Case 1: Standard successful merge and alignment (Quarterly)
        # pd_data dates are quarter-ends. macro_data dates are month-ends, needing alignment.
        (
            {'date': ['2020-03-31', '2020-06-30', '2020-09-30', '2020-12-31'], 'PD': [0.01, 0.015, 0.02, 0.025]},
            {'date': ['2020-01-31', '2020-04-30', '2020-07-31', '2020-10-31'], 'GDP': [100.0, 101.0, 102.0, 103.0], 'CPI': [2.0, 2.1, 2.2, 2.3]},
            'date', 'date', 'Q', # date column names and frequency
            # Expected output assuming macro data aligns to PD data's quarter-end dates (e.g., using merge_asof with 'backward' direction)
            pd.DataFrame({
                'date': pd.to_datetime(['2020-03-31', '2020-06-30', '2020-09-30', '2020-12-31']),
                'PD': [0.01, 0.015, 0.02, 0.025],
                'GDP': [100.0, 101.0, 102.0, 103.0], 
                'CPI': [2.0, 2.1, 2.2, 2.3]
            }),
            None # No exception expected
        ),
        # Test Case 2: Empty PD DataFrame
        # Should result in an empty DataFrame with the expected columns.
        (
            {'date': [], 'PD': []}, # Empty PD data
            {'date': ['2020-03-31', '2020-06-30'], 'GDP': [1.0, 1.2]},
            'date', 'date', 'Q',
            pd.DataFrame(columns=['date', 'PD', 'GDP']).astype({'date': 'datetime64[ns]'}), # Expected empty DataFrame with all potential columns
            None
        ),
        # Test Case 3: No overlapping dates between PD and Macro DataFrames
        # Should result in an empty DataFrame as there's no common temporal period to merge.
        (
            {'date': ['2010-03-31', '2010-06-30'], 'PD': [0.01, 0.015]},
            {'date': ['2020-03-31', '2020-06-30'], 'GDP': [1.0, 1.2]},
            'date', 'date', 'Q',
            pd.DataFrame(columns=['date', 'PD', 'GDP']).astype({'date': 'datetime64[ns]'}),
            None
        ),
        # Test Case 4: Invalid date column name for PD DataFrame
        # Expect a KeyError when trying to access a non-existent column.
        (
            {'date_pd': ['2020-03-31'], 'PD': [0.01]}, # Column name is 'date_pd'
            {'date': ['2020-03-31'], 'GDP': [1.0]},
            'invalid_date_col', 'date', 'Q', # 'invalid_date_col' will not exist in pd_data
            None, # No specific output DataFrame expected
            KeyError # Expecting KeyError
        ),
        # Test Case 5: Invalid frequency string
        # Expect a ValueError if the frequency string is not recognized by pandas or the function's internal logic.
        (
            {'date': ['2020-03-31'], 'PD': [0.01]},
            {'date': ['2020-03-31'], 'GDP': [1.0]},
            'date', 'date', 'ABC', # Invalid frequency string
            None, # No specific output DataFrame expected
            ValueError # Expecting ValueError (e.g., from pandas.to_period or resample)
        )
    ]
)
def test_merge_and_align_data(pd_data_info, macro_data_info, date_column_pd, date_column_macro, frequency, expected_output_info, expected_exception):
    # Convert dictionary information to pandas DataFrames
    pd_data = pd.DataFrame(pd_data_info)
    macro_data = pd.DataFrame(macro_data_info)

    # Ensure date columns are in datetime format for robust testing of time series operations
    # The function itself might handle this conversion, but for testing, ensuring inputs are clean is good.
    if date_column_pd in pd_data.columns:
        pd_data[date_column_pd] = pd.to_datetime(pd_data[date_column_pd])
    if date_column_macro in macro_data.columns:
        macro_data[date_column_macro] = pd.to_datetime(macro_data[date_column_macro])
        
    if expected_exception:
        with pytest.raises(expected_exception):
            merge_and_align_data(pd_data, macro_data, date_column_pd, date_column_macro, frequency)
    else:
        result_df = merge_and_align_data(pd_data, macro_data, date_column_pd, date_column_macro, frequency)
        
        # For DataFrame comparisons, use pandas.testing.assert_frame_equal for robustness.
        # It checks data, index, column names, and dtypes.
        
        if expected_output_info.empty and result_df.empty:
            # For empty DataFrames, ensure columns and dtypes match the expected empty structure.
            pd.testing.assert_frame_equal(result_df, expected_output_info, check_dtype=True)
        else:
            # For non-empty DataFrames, sort by the date column and reset index for consistent comparison.
            # This handles potential differences in row order or index values from the function's internal processing.
            result_df_sorted = result_df.sort_values(by=date_column_pd).reset_index(drop=True)
            expected_output_info_sorted = expected_output_info.sort_values(by=date_column_pd).reset_index(drop=True)
            
            pd.testing.assert_frame_equal(result_df_sorted, expected_output_info_sorted, check_dtype=True)