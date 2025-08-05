import pytest
import pandas as pd
import numpy as np
from definition_2162b7eeade546e4b324220e37b87dcd import fit_ardl_model

# Helper data for tests
# Valid series and dataframe with sufficient observations for testing
_endog_valid = pd.Series(np.random.rand(50), index=pd.date_range(start='2000-01-01', periods=50, freq='MS'))
_exog_valid = pd.DataFrame(np.random.rand(50, 2), index=_endog_valid.index, columns=['X1', 'X2'])

# Series/DataFrame with insufficient observations for larger lags
_endog_short = pd.Series(np.random.rand(5), index=pd.date_range(start='2000-01-01', periods=5, freq='MS'))
_exog_short = pd.DataFrame(np.random.rand(5, 2), index=_endog_short.index, columns=['X1', 'X2'])


@pytest.mark.parametrize(
    "endog, exog, order, ardl_order, expected",
    [
        # Test 1: Valid input - for a 'pass' stub, this means it returns None
        (_endog_valid, _exog_valid, (1, 1), (2, 2), None),
        # Test 2: Invalid endog type (list instead of pandas Series)
        ([1, 2, 3], _exog_valid, (1, 1), (2, 2), TypeError),
        # Test 3: Invalid exog type (list of lists instead of pandas DataFrame)
        (_endog_valid, [[1, 2], [3, 4]], (1, 1), (2, 2), TypeError),
        # Test 4: Invalid 'order' value (negative lags) - should raise ValueError
        (_endog_valid, _exog_valid, (-1, 1), (2, 2), ValueError),
        # Test 5: Insufficient data points for specified lags - should raise ValueError
        (_endog_short, _exog_short, (5, 5), (6, 6), ValueError),
    ]
)
def test_fit_ardl_model(endog, exog, order, ardl_order, expected):
    try:
        # Call the function with the given inputs
        result = fit_ardl_model(endog, exog, order, ardl_order)
        # If no exception, assert the result (for a 'pass' stub, this is None)
        assert result == expected
    except Exception as e:
        # If an exception is raised, assert its type matches the expected exception
        assert isinstance(e, expected)