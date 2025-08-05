import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

# Keep the definition_121770fd5ad34878991e9aac0c181702 block as it is. DO NOT REPLACE or REMOVE the block.
from definition_121770fd5ad34878991e9aac0c181702 import fit_arimax_model

# Mock pmdarima for testing purposes
class MockARIMAResultsWrapper:
    """A mock class to simulate the return type of a fitted ARIMAX model."""
    def __init__(self, **kwargs):
        self.summary = MagicMock()
        self.predict = MagicMock(return_value=np.array([1, 2, 3]))
        self.forecast = MagicMock(return_value=np.array([4, 5, 6]))
        self.aic = 100 # Example attribute
        self.bic = 200 # Example attribute

class MockARIMA:
    """A mock class to simulate pmdarima.arima.ARIMA behavior."""
    def __init__(self, order, seasonal_order, suppress_warnings, X=None, **kwargs):
        # Store initialization parameters to verify they are passed correctly
        self.order = order
        self.seasonal_order = seasonal_order
        self.suppress_warnings = suppress_warnings
        self.X_init = X # Exog passed during initialization
        self.kwargs = kwargs
        self._fitted = False

        # Simulate early validation for common pmdarima errors
        if not isinstance(self.order, tuple) or len(self.order) != 3:
            raise TypeError(f"order must be a tuple of 3 integers (p, d, q), but got {type(self.order)}")
        if not isinstance(self.seasonal_order, tuple) or len(self.seasonal_order) != 4:
            raise TypeError(f"seasonal_order must be a tuple of 4 integers (P, D, Q, S), but got {type(self.seasonal_order)}")
        
        # Simulate pmdarima's internal handling of empty X (exog)
        if isinstance(X, pd.DataFrame) and X.empty:
            self.X_init = None # pmdarima treats empty DataFrame as None for exog

    def fit(self, y, X=None):
        """Simulate the model fitting process."""
        if not isinstance(y, (pd.Series, np.ndarray)):
            raise TypeError("endog (y) must be a pandas Series or numpy array.")
        if pd.Series(y).isnull().any():
            # pmdarima often raises ValueError if endog contains NaNs, especially if differencing (d > 0)
            raise ValueError("Input `endog` (y) cannot contain NaN values.")
        if X is not None and not isinstance(X, (pd.DataFrame, np.ndarray)):
            raise TypeError("exog (X) must be a pandas DataFrame or numpy array.")
        
        self._fitted = True
        return MockARIMAResultsWrapper()

@pytest.fixture(autouse=True)
def mock_pmdarima_arima():
    """Fixture to patch pmdarima.arima.ARIMA for all tests in this file."""
    # Ensure the patch applies to where pmdarima.arima.ARIMA is imported within fit_arimax_model
    with patch('pmdarima.arima.ARIMA', new=MockARIMA) as mock_arima_class:
        yield mock_arima_class

# Helper function to create mock data
def create_mock_data(size=10, has_nan_endog=False):
    """Generates mock pandas Series for endog and DataFrame for exog."""
    dates = pd.date_range(start='2020-01-01', periods=size, freq='QS')
    
    endog_data = np.random.rand(size)
    if has_nan_endog:
        if size > 2: # Ensure there's a place to put NaN
            endog_data[2] = np.nan 
        else: # If size is too small, just make the whole array NaN for testing
            endog_data[:] = np.nan 
    endog = pd.Series(endog_data, index=dates, name='PD')

    exog_data = np.random.rand(size, 2)
    exog = pd.DataFrame(exog_data, columns=['exog1', 'exog2'], index=dates)
    return endog, exog

@pytest.mark.parametrize(
    "endog_input, exog_input, order_input, seasonal_order_input, suppress_warnings_input, expected_exception",
    [
        # Test Case 1: Valid Inputs (Standard Scenario)
        (
            create_mock_data()[0],              # endog (pd.Series)
            create_mock_data()[1],              # exog (pd.DataFrame)
            (1, 1, 1),                          # order (tuple)
            (0, 0, 0, 0),                       # seasonal_order (tuple)
            True,                               # suppress_warnings (bool)
            None                                # No expected exception
        ),
        # Test Case 2: No Exogenous Variables (ARIMA only)
        (
            create_mock_data(size=5)[0],        # endog (pd.Series)
            pd.DataFrame(),                     # exog (empty pd.DataFrame)
            (1, 0, 0),                          # order (tuple)
            (0, 0, 0, 0),                       # seasonal_order (tuple)
            False,                              # suppress_warnings (bool)
            None                                # No expected exception
        ),
        # Test Case 3: Invalid `order` Type (List instead of Tuple)
        (
            create_mock_data()[0],              # endog (pd.Series)
            create_mock_data()[1],              # exog (pd.DataFrame)
            [1, 1, 1],                          # Invalid order type (list)
            (0, 0, 0, 0),                       # seasonal_order (tuple)
            True,                               # suppress_warnings (bool)
            TypeError                           # Expected TypeError
        ),
        # Test Case 4: Invalid `endog` Type (List instead of Series)
        (
            [1.0, 2.0, 3.0, 4.0, 5.0],          # Invalid endog type (list)
            create_mock_data(size=5)[1],        # exog (pd.DataFrame)
            (1, 1, 1),                          # order (tuple)
            (0, 0, 0, 0),                       # seasonal_order (tuple)
            True,                               # suppress_warnings (bool)
            TypeError                           # Expected TypeError
        ),
        # Test Case 5: `endog` with NaN values
        (
            create_mock_data(size=7, has_nan_endog=True)[0], # endog with NaN (pd.Series)
            create_mock_data(size=7)[1],                     # exog (pd.DataFrame)
            (1, 0, 0),                                       # order (tuple)
            (0, 0, 0, 0),                                    # seasonal_order (tuple)
            True,                                            # suppress_warnings (bool)
            ValueError                                       # Expected ValueError
        ),
    ]
)
def test_fit_arimax_model(
    mock_pmdarima_arima, # This fixture ensures pmdarima.arima.ARIMA is mocked
    endog_input,
    exog_input,
    order_input,
    seasonal_order_input,
    suppress_warnings_input,
    expected_exception
):
    """
    Tests the fit_arimax_model function with various inputs, including edge cases
    and error handling.
    """
    if expected_exception:
        with pytest.raises(expected_exception):
            fit_arimax_model(endog_input, exog_input, order_input, seasonal_order_input, suppress_warnings_input)
    else:
        # Call the function under test
        result = fit_arimax_model(endog_input, exog_input, order_input, seasonal_order_input, suppress_warnings_input)

        # Assertions for successful cases
        assert isinstance(result, MockARIMAResultsWrapper)

        # Determine the expected exog value for the MockARIMA constructor
        # pmdarima.ARIMA treats empty DataFrames for X (exog) as None
        expected_X_in_init = exog_input if isinstance(exog_input, pd.DataFrame) and not exog_input.empty else None

        # Verify that MockARIMA was initialized with the correct parameters
        mock_pmdarima_arima.assert_called_once_with(
            order=order_input,
            seasonal_order=seasonal_order_input,
            suppress_warnings=suppress_warnings_input,
            X=expected_X_in_init # This confirms exog is passed to the constructor
        )

        # Verify that the .fit() method was called on the instance of MockARIMA
        # mock_pmdarima_arima.return_value is the instance of MockARIMA created
        assert mock_pmdarima_arima.return_value._fitted is True