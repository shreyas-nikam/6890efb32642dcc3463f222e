import pytest
import pandas as pd
import numpy as np
from definition_ba963fccad214de1b4c1891d2dab86bf import perform_stationarity_tests

@pytest.mark.parametrize("series_input, test_type_input, expected_outcome", [
    # Test Case 1: Valid ADF test on a clearly non-stationary series (expected high p-value)
    (pd.Series(np.cumsum(np.random.randn(100)) + 50, name='non_stationary_series'), 'adf',
     {'p_value_threshold': 0.05, 'p_value_comparison': lambda p, t: p > t, 'keys': ['test_statistic', 'p_value', 'critical_values']}),

    # Test Case 2: Valid PP test on a clearly stationary series (expected low p-value)
    (pd.Series(np.random.randn(100), name='stationary_series'), 'pp',
     {'p_value_threshold': 0.05, 'p_value_comparison': lambda p, t: p <= t, 'keys': ['test_statistic', 'p_value', 'critical_values']}),

    # Test Case 3: Invalid 'test_type' string
    (pd.Series(np.random.randn(50)), 'unknown_test', ValueError),

    # Test Case 4: Invalid 'series' type (e.g., list instead of pandas.Series)
    ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'adf', TypeError),

    # Test Case 5: Series with insufficient data for the ADF test (ADF typically requires min 4 observations for default constant regression)
    (pd.Series([1, 2, 3]), 'adf', ValueError),
])
def test_perform_stationarity_tests(series_input, test_type_input, expected_outcome):
    """
    Tests the perform_stationarity_tests function for various valid and invalid inputs,
    covering expected functionality and edge cases.
    """
    if isinstance(expected_outcome, type) and issubclass(expected_outcome, Exception):
        # This branch handles test cases where an exception is expected
        with pytest.raises(expected_outcome):
            perform_stationarity_tests(series_input, test_type_input)
    else:
        # This branch handles test cases where a successful dictionary output is expected
        result = perform_stationarity_tests(series_input, test_type_input)

        # Assert that the result is a dictionary
        assert isinstance(result, dict), f"Expected result to be a dict, but got {type(result)}"

        # Assert that all expected keys are present in the result dictionary
        for key in expected_outcome['keys']:
            assert key in result, f"Expected key '{key}' not found in result"

        # Assert the type of the values for robustness
        assert isinstance(result['test_statistic'], (float, int)), "Test statistic should be a number"
        assert isinstance(result['p_value'], (float, int)), "P-value should be a number"
        assert isinstance(result['critical_values'], dict), "Critical values should be a dictionary"

        # Assert the p-value against the expected threshold and comparison logic
        assert expected_outcome['p_value_comparison'](result['p_value'], expected_outcome['p_value_threshold']), \
            f"P-value {result['p_value']} did not meet expected condition (threshold: {expected_outcome['p_value_threshold']})"