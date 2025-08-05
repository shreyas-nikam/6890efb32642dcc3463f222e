import pytest
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import statsmodels.graphics.tsaplots as sm_plots

# Placeholder for your module import
from definition_a817bcc83ba248dbbee05e6e9c44048d import plot_acf_pacf

@pytest.fixture
def sample_series():
    """Provides a sample pandas Series for testing."""
    return pd.Series(np.random.randn(50), index=pd.date_range(start='2020-01-01', periods=50, freq='D'))

@pytest.mark.parametrize(
    "series_input, lags_input, title_input, output_path_input, expected_outcome_type, expected_match_regex",
    [
        # Test Case 1: Happy Path - Save to file
        (pytest.lazy_fixture('sample_series'), 10, "Test Plot Save", "test_plot.png", "file_saved", None),
        # Test Case 2: Happy Path - Display plot (output_path=None)
        (pytest.lazy_fixture('sample_series'), 10, "Test Plot Display", None, "plot_displayed", None),
        # Test Case 3: Error cases for 'series' argument (Type Errors & Value Errors)
        (None, 10, "Error Test Series", None, TypeError, "Series must be a pandas.Series"),
        ("not a series", 10, "Error Test Series", None, TypeError, "Series must be a pandas.Series"),
        ([1, 2, 3], 10, "Error Test Series", None, TypeError, "Series must be a pandas.Series"),
        (pd.Series([]), 10, "Error Test Series", None, ValueError, "Data must contain at least 2 observations for plotting ACF/PACF"),
        (pd.Series([1.0]), 10, "Error Test Series", None, ValueError, "Data must contain at least 2 observations for plotting ACF/PACF"),
        # Test Case 4: Error cases for 'lags' argument (Type Errors & Value Errors)
        (pytest.lazy_fixture('sample_series'), None, "Error Test Lags", None, TypeError, "Lags must be an integer"),
        (pytest.lazy_fixture('sample_series'), "ten", "Error Test Lags", None, TypeError, "Lags must be an integer"),
        (pytest.lazy_fixture('sample_series'), 10.5, "Error Test Lags", None, TypeError, "Lags must be an integer"),
        (pytest.lazy_fixture('sample_series'), 0, "Error Test Lags", None, ValueError, "Lags must be a positive integer"),
        (pytest.lazy_fixture('sample_series'), -5, "Error Test Lags", None, ValueError, "Lags must be a positive integer"),
        # Test Case 5: Error cases for 'title' and 'output_path' arguments (Type Errors)
        (pytest.lazy_fixture('sample_series'), 10, None, None, TypeError, "Title must be a string"),
        (pytest.lazy_fixture('sample_series'), 10, 123, None, TypeError, "Title must be a string"),
        (pytest.lazy_fixture('sample_series'), 10, "Valid Title", 123, TypeError, "Output path must be a string or None"),
        (pytest.lazy_fixture('sample_series'), 10, "Valid Title", [], TypeError, "Output path must be a string or None"),
    ]
)
def test_plot_acf_pacf(series_input, lags_input, title_input, output_path_input, expected_outcome_type, expected_match_regex, tmp_path, mocker):
    """
    Tests the plot_acf_pacf function for various valid and invalid inputs,
    checking for correct plot generation/saving or appropriate error handling.
    """
    # Mock internal plotting functions from statsmodels and matplotlib functions
    mock_plot_acf = mocker.patch('statsmodels.graphics.tsaplots.plot_acf')
    mock_plot_pacf = mocker.patch('statsmodels.graphics.tsaplots.plot_pacf')
    mock_savefig = mocker.patch('matplotlib.pyplot.savefig')
    mock_show = mocker.patch('matplotlib.pyplot.show')

    # Adjust output_path for file saving test case to use a temporary directory
    resolved_output_path = output_path_input
    if expected_outcome_type == "file_saved":
        resolved_output_path = str(tmp_path / output_path_input)

    try:
        # Call the function under test
        plot_acf_pacf(
            series=series_input,
            lags=lags_input,
            title=title_input,
            output_path=resolved_output_path
        )

        # Assertions for successful execution paths
        if expected_outcome_type == "file_saved":
            mock_savefig.assert_called_once_with(resolved_output_path)
            mock_show.assert_not_called()
            mock_plot_acf.assert_called_once()
            mock_plot_pacf.assert_called_once()
        elif expected_outcome_type == "plot_displayed":
            mock_savefig.assert_not_called()
            mock_show.assert_called_once()
            mock_plot_acf.assert_called_once()
            mock_plot_pacf.assert_called_once()
        else:
            pytest.fail(f"Unexpected successful execution for a case expecting: {expected_outcome_type}")

    except Exception as e:
        # Assertions for expected error paths
        assert isinstance(e, expected_outcome_type)
        if expected_match_regex:
            assert expected_match_regex in str(e)
        
        # Ensure no plotting or saving happened if an error occurred
        mock_savefig.assert_not_called()
        mock_show.assert_not_called()
        mock_plot_acf.assert_not_called()
        mock_plot_pacf.assert_not_called()