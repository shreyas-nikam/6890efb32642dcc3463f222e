import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import os

# DO NOT REPLACE or REMOVE THE BLOCK BELOW
from definition_5e95a10bb6084650939c3bd120742bb4 import plot_qq_plot
# DO NOT REPLACE or REMOVE THE BLOCK ABOVE


@pytest.fixture
def mock_matplotlib_and_statsmodels():
    """Fixture to mock matplotlib and statsmodels for plotting functions."""
    with patch('matplotlib.pyplot.show') as mock_show, \
         patch('matplotlib.pyplot.savefig') as mock_savefig, \
         patch('matplotlib.pyplot.close') as mock_close, \
         patch('statsmodels.api.qqplot') as mock_qqplot:

        # Mock the return value of qqplot to be a Figure object, as required by savefig/show
        mock_figure = MagicMock()
        mock_qqplot.return_value = mock_figure, MagicMock() # qqplot returns (Figure, Axes)

        yield mock_show, mock_savefig, mock_close, mock_qqplot, mock_figure


def test_plot_qq_plot_save_to_file(tmp_path, mock_matplotlib_and_statsmodels):
    """
    Test case 1: Verify plot is saved to a file when output_path is provided.
    Covers expected functionality.
    """
    mock_show, mock_savefig, mock_close, mock_qqplot, _ = mock_matplotlib_and_statsmodels
    
    residuals = pd.Series(np.random.normal(0, 1, 100))
    title = "Test QQ Plot"
    output_path = tmp_path / "qq_plot.png"

    plot_qq_plot(residuals, title, output_path)

    mock_qqplot.assert_called_once_with(residuals, line='s')
    mock_savefig.assert_called_once_with(output_path)
    mock_show.assert_not_called()
    mock_close.assert_called_once() # Ensure plot is closed after saving


def test_plot_qq_plot_display_plot(mock_matplotlib_and_statsmodels):
    """
    Test case 2: Verify plot is displayed when output_path is None.
    Covers expected functionality.
    """
    mock_show, mock_savefig, mock_close, mock_qqplot, _ = mock_matplotlib_and_statsmodels
    
    residuals = pd.Series(np.random.normal(0, 1, 50))
    title = "Display QQ Plot"
    output_path = None

    plot_qq_plot(residuals, title, output_path)

    mock_qqplot.assert_called_once_with(residuals, line='s')
    mock_show.assert_called_once()
    mock_savefig.assert_not_called()
    mock_close.assert_called_once() # Ensure plot is closed after displaying


def test_plot_qq_plot_empty_residuals(mock_matplotlib_and_statsmodels):
    """
    Test case 3: Verify behavior with an empty residuals Series.
    Covers an edge case: empty input data.
    statsmodels.graphics.gofplots.qqplot raises ValueError for empty array.
    """
    mock_show, mock_savefig, mock_close, mock_qqplot, _ = mock_matplotlib_and_statsmodels
    
    residuals = pd.Series([])
    title = "Empty Residuals Plot"
    output_path = None

    mock_qqplot.side_effect = ValueError("array must not be empty")

    with pytest.raises(ValueError, match="array must not be empty"):
        plot_qq_plot(residuals, title, output_path)

    mock_qqplot.assert_called_once_with(residuals, line='s')
    mock_show.assert_not_called()
    mock_savefig.assert_not_called()
    mock_close.assert_not_called() # Should not call close if an error occurred before plot creation


def test_plot_qq_plot_non_numeric_residuals(mock_matplotlib_and_statsmodels):
    """
    Test case 4: Verify behavior with non-numeric residuals Series.
    Covers an edge case: invalid data type within the Series.
    statsmodels.graphics.gofplots.qqplot typically raises TypeError for non-numeric input.
    """
    mock_show, mock_savefig, mock_close, mock_qqplot, _ = mock_matplotlib_and_statsmodels
    
    residuals = pd.Series(['a', 'b', 'c', 'd'])
    title = "Non-Numeric Residuals Plot"
    output_path = None

    # Simulate the TypeError that statsmodels.qqplot would raise
    mock_qqplot.side_effect = TypeError("could not convert string to float")

    with pytest.raises(TypeError, match="could not convert string to float"):
        plot_qq_plot(residuals, title, output_path)

    mock_qqplot.assert_called_once_with(residuals, line='s')
    mock_show.assert_not_called()
    mock_savefig.assert_not_called()
    mock_close.assert_not_called()


def test_plot_qq_plot_invalid_residuals_type(mock_matplotlib_and_statsmodels):
    """
    Test case 5: Verify behavior when residuals is not a pandas Series.
    Covers an edge case: incorrect input type for residuals.
    Although statsmodels.qqplot might accept array-like, the docstring specifies pandas.Series.
    If the function itself tries to use pandas.Series specific methods, it would fail.
    Assuming the function is strict about the `residuals` type as per docstring.
    """
    mock_show, mock_savefig, mock_close, mock_qqplot, _ = mock_matplotlib_and_statsmodels

    residuals = [1, 2, 3, 4, 5]  # A list, not a pandas Series
    title = "Invalid Residuals Type Plot"
    output_path = None

    # sm.qqplot usually tries to convert to numpy array, so it might not fail on type directly
    # but if the function `plot_qq_plot` had internal pandas-specific logic on `residuals`
    # this would be a TypeError. For this generic stub, we assume sm.qqplot will handle conversion.
    # However, if the function definition were strict, we would test for TypeError if `residuals.name`
    # or other Series attributes were accessed without prior type check.
    # Given the stub just passes `residuals` to `sm.qqplot`, `sm.qqplot` will likely
    # convert it to an array. So, this test should confirm `sm.qqplot` is still called correctly.
    
    plot_qq_plot(residuals, title, output_path)
    
    # sm.qqplot will likely convert the list to an array internally
    mock_qqplot.assert_called_once() 
    # Check that the first argument passed to qqplot is the residuals list.
    assert mock_qqplot.call_args[0][0] == residuals
    mock_show.assert_called_once()
    mock_savefig.assert_not_called()
    mock_close.assert_called_once()