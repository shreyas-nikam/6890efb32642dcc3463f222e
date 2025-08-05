import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock

"""
from definition_48a26886a32a4c658e93d6d45373b462 import plot_fan_chart
"""

@pytest.fixture
def sample_forecast_df():
    """Returns a sample DataFrame with expected columns and DatetimeIndex."""
    dates = pd.date_range(start='2023-01-01', periods=5, freq='Q')
    df = pd.DataFrame({
        'forecast': np.array([10.0, 11.0, 12.0, 13.0, 14.0]),
        'lower': np.array([9.5, 10.5, 11.5, 12.5, 13.5]),
        'upper': np.array([10.5, 11.5, 12.5, 13.5, 14.5]),
    }, index=dates)
    return df

@pytest.fixture
def empty_forecast_df():
    """Returns an empty DataFrame with expected columns."""
    dates = pd.DatetimeIndex([])
    df = pd.DataFrame(columns=['forecast', 'lower', 'upper'], index=dates)
    return df

@pytest.fixture
def df_missing_cols():
    """Returns a DataFrame missing required 'lower' and 'upper' columns."""
    dates = pd.date_range(start='2023-01-01', periods=5, freq='Q')
    df = pd.DataFrame({
        'forecast': np.array([10.0, 11.0, 12.0, 13.0, 14.0]),
        'some_other_col': np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
    }, index=dates)
    return df

@pytest.mark.parametrize("forecast_df_fixture_name, title, output_path, expected_exception, expected_mpl_call", [
    # Test Case 1: Valid input, save plot to file
    ("sample_forecast_df", "Quarterly PD Forecast Fan Chart", "test_chart.png", None, "savefig"),
    # Test Case 2: Valid input, display plot (output_path is None)
    ("sample_forecast_df", "Interactive Fan Chart Display", None, None, "show"),
    # Test Case 3: Invalid forecast_df type (not a pandas DataFrame)
    ("not_a_dataframe", "Invalid DataFrame Test", "invalid_df.png", TypeError, None),
    # Test Case 4: forecast_df missing required columns
    ("df_missing_cols", "Missing Columns Test", "missing_cols.png", KeyError, None),
    # Test Case 5: Empty forecast_df (should handle gracefully, producing an empty plot or a plot with no data)
    ("empty_forecast_df", "Empty Data Fan Chart", "empty_data.png", None, "savefig"),
])
def test_plot_fan_chart(mocker, request, forecast_df_fixture_name, title, output_path, expected_exception, expected_mpl_call):
    # Mock matplotlib.pyplot and its common functions used in plotting
    mock_plt = mocker.patch('matplotlib.pyplot')
    mock_plt.figure = MagicMock()
    mock_plt.plot = MagicMock()
    mock_plt.fill_between = MagicMock()
    mock_plt.title = MagicMock()
    mock_plt.xlabel = MagicMock()
    mock_plt.ylabel = MagicMock()
    mock_plt.legend = MagicMock()
    mock_plt.grid = MagicMock()
    mock_plt.tight_layout = MagicMock()
    mock_plt.close = MagicMock()
    mock_savefig = mock_plt.savefig
    mock_show = mock_plt.show

    # Prepare forecast_df based on fixture name or a custom value
    if forecast_df_fixture_name == "not_a_dataframe":
        forecast_df = ["not", "a", "dataframe"] # Invalid type
    else:
        forecast_df = request.getfixturevalue(forecast_df_fixture_name)

    if expected_exception:
        with pytest.raises(expected_exception):
            plot_fan_chart(forecast_df, title, output_path)
        # Assert that no plotting/saving/showing functions were called upon error
        mock_savefig.assert_not_called()
        mock_show.assert_not_called()
    else:
        plot_fan_chart(forecast_df, title, output_path)

        # Assert correct matplotlib call based on output_path
        if expected_mpl_call == "savefig":
            mock_savefig.assert_called_once_with(output_path, bbox_inches='tight')
            mock_show.assert_not_called()
            mock_plt.close.assert_called_once() # Typically close figure after saving
        elif expected_mpl_call == "show":
            mock_show.assert_called_once()
            mock_savefig.assert_not_called()
            mock_plt.close.assert_not_called() # Typically not close figure after show for interactive use

        # Assert common plotting functions were called (assuming a basic plot was attempted)
        mock_plt.figure.assert_called_once()
        mock_plt.plot.assert_called_once()
        mock_plt.fill_between.assert_called_once()
        mock_plt.title.assert_called_once_with(title)
        mock_plt.xlabel.assert_called_once()
        mock_plt.ylabel.assert_called_once()
        mock_plt.legend.assert_called_once()
        mock_plt.grid.assert_called_once()
        mock_plt.tight_layout.assert_called_once()