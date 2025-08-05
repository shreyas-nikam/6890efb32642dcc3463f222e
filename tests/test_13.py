import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

# Block for your_module - DO NOT REMOVE or REPLACE
from definition_9e1b9dd1acea4d0e99f85f561a7ea778 import plot_residual_acf
# End of your_module block

@pytest.mark.parametrize("residuals, lags, title, output_path, expected_exception, expected_mock_calls", [
    # Test Case 1: Basic functionality - display plot (output_path is None)
    (pd.Series(np.random.randn(100)), 20, "ACF Plot Display", None, None,
     {"plot_acf_called": True, "show_called": True, "savefig_called": False}),

    # Test Case 2: Basic functionality - save plot (output_path is provided)
    (pd.Series(np.random.randn(100)), 20, "ACF Plot Save", "output_acf.png", None,
     {"plot_acf_called": True, "show_called": False, "savefig_called": True}),

    # Test Case 3: Edge case - empty residuals. `statsmodels.graphics.tsaplots.plot_acf` will raise ValueError.
    (pd.Series([], dtype=float), 10, "Empty Residuals", None, ValueError,
     {"plot_acf_called": False, "show_called": False, "savefig_called": False}),

    # Test Case 4: Error handling - invalid residuals type (e.g., list instead of pandas.Series).
    # Assuming the function internally expects a pandas Series or uses Series-specific methods,
    # leading to a TypeError or AttributeError. We'll use TypeError as a general expectation for strict typing.
    ([1, 2, 3, 4, 5], 10, "Invalid Residuals Type", None, TypeError,
     {"plot_acf_called": False, "show_called": False, "savefig_called": False}),

    # Test Case 5: Error handling - invalid lags value (e.g., negative integer).
    # `statsmodels.graphics.tsaplots.plot_acf` passes lags to `acf` which requires nlags >= 0.
    (pd.Series(np.random.randn(100)), -5, "Negative Lags", None, ValueError,
     {"plot_acf_called": False, "show_called": False, "savefig_called": False}),
])
@patch('matplotlib.pyplot.show')
@patch('matplotlib.pyplot.savefig')
@patch('statsmodels.graphics.tsaplots.plot_acf')
def test_plot_residual_acf(mock_plot_acf, mock_savefig, mock_show, residuals, lags, title, output_path, expected_exception, expected_mock_calls):
    if expected_exception:
        with pytest.raises(expected_exception):
            plot_residual_acf(residuals, lags, title, output_path)
    else:
        plot_residual_acf(residuals, lags, title, output_path)

        if expected_mock_calls["plot_acf_called"]:
            # For the first two cases, mock_plot_acf needs to be checked
            # The ax argument is typically None by default in plot_acf if not provided.
            mock_plot_acf.assert_called_once_with(residuals, lags=lags, title=title, ax=None)
        else:
            mock_plot_acf.assert_not_called()

        if expected_mock_calls["show_called"]:
            mock_show.assert_called_once()
        else:
            mock_show.assert_not_called()

        if expected_mock_calls["savefig_called"]:
            mock_savefig.assert_called_once_with(output_path)
        else:
            mock_savefig.assert_not_called()