import pytest
import pandas as pd
import numpy as np

# definition_40b3f016443d4d4c89058acbb1f8be8c block
# DO NOT REPLACE or REMOVE this block.
from definition_40b3f016443d4d4c89058acbb1f8be8c import handle_missing_data
# End definition_40b3f016443d4d4c89058acbb1f8be8c block


@pytest.mark.parametrize("input_df, strategy, expected_result, expected_exception", [
    # Test Case 1: 'ffill' strategy - Basic functionality with mixed NaNs and types.
    (pd.DataFrame({'A': [1, np.nan, 3], 'B': [10, 20, np.nan], 'C': ['x', np.nan, 'z']}),
     'ffill',
     pd.DataFrame({'A': [1, 1, 3], 'B': [10, 20, 20], 'C': ['x', 'x', 'z']}),
     None),

    # Test Case 2: 'mean' strategy - Fills numeric columns with mean, leaves non-numeric.
    (pd.DataFrame({'A': [1, np.nan, 3], 'B': [10, 20, np.nan], 'C': ['apple', 'banana', np.nan]}),
     'mean',
     pd.DataFrame({'A': [1.0, 2.0, 3.0], 'B': [10.0, 20.0, 15.0], 'C': ['apple', 'banana', np.nan]}),
     None),
    
    # Test Case 3: 'drop' strategy - Removes rows containing any NaN values.
    (pd.DataFrame({'A': [1, np.nan, 3], 'B': [10, 20, np.nan], 'C': ['x', 'y', np.nan]}),
     'drop',
     pd.DataFrame({'A': [1], 'B': [10], 'C': ['x']}, index=[0]), # Expecting row with index 0 to remain
     None),
    
    # Test Case 4: No missing data - DataFrame should remain unchanged.
    (pd.DataFrame({'A': [1, 2, 3], 'B': [10, 20, 30]}),
     'ffill', # Strategy choice is irrelevant as there are no NaNs
     pd.DataFrame({'A': [1, 2, 3], 'B': [10, 20, 30]}),
     None),

    # Test Case 5: Invalid strategy - Should raise a ValueError.
    (pd.DataFrame({'A': [1, np.nan, 3]}),
     'unsupported_strategy',
     None, # No expected DataFrame for an error case
     ValueError),
])
def test_handle_missing_data(input_df, strategy, expected_result, expected_exception):
    """
    Tests the handle_missing_data function across various strategies and edge cases.
    """
    if expected_exception:
        with pytest.raises(expected_exception):
            handle_missing_data(input_df.copy(), strategy)
    else:
        result_df = handle_missing_data(input_df.copy(), strategy)
        # Use pd.testing.assert_frame_equal for robust DataFrame comparison.
        # It checks data, index, and column names/types.
        pd.testing.assert_frame_equal(result_df, expected_result)