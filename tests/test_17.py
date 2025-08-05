import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock

# Keep a placeholder definition_e7d130df704343fdb7722a47ed726396 for the import of the module.
# Keep the `your_module` block as it is. DO NOT REPLACE or REMOVE the block.
from definition_e7d130df704343fdb7722a47ed726396 import generate_forecasts

# Mock Model class to simulate a fitted time series model
class MockModel:
    def __init__(self, expects_exog=False):
        self.expects_exog = expects_exog

    def forecast(self, steps=1, exog=None, alpha=None):
        """
        Simulates the forecast method of a time series model.
        Returns a DataFrame with 'forecast', 'lower_ci', 'upper_ci' columns.
        """
        if self.expects_exog and exog is None:
            raise ValueError("Exogenous variables are required for this model.")
        if self.expects_exog and exog is not None:
            if not isinstance(exog, pd.DataFrame) or len(exog) != steps:
                raise ValueError("Exogenous variables must be a pandas DataFrame of the correct length.")

        # Simulate forecast data
        base_forecast = np.linspace(100, 100 + (steps - 1) * 5, steps)
        
        if alpha is not None:
            if not (0 <= alpha <= 1):
                # This check would ideally be done by the main function,
                # but including it here to make mock robust to external calls.
                raise ValueError("Alpha must be between 0 and 1 inclusive.")

            # Simulate confidence intervals
            std_err = base_forecast * 0.02  # Arbitrary standard error
            # Approximate z-score for given alpha (e.g., 1.96 for 95% CI (alpha=0.05))
            # In real statistical libraries, this comes from a distribution.
            z_score = 1.96 # A common value for 95% CI, simplifying for mock
            if alpha == 0.01:
                z_score = 2.58 # For 99% CI

            lower_ci = base_forecast - z_score * std_err
            upper_ci = base_forecast + z_score * std_err
            data = {'forecast': base_forecast, 'lower_ci': lower_ci, 'upper_ci': upper_ci}
        else:
            data = {'forecast': base_forecast}

        # Create a pandas DataFrame with a DatetimeIndex
        index = pd.date_range(start='2025-01-01', periods=steps, freq='Q')
        return pd.DataFrame(data, index=index)

# Fixtures for mock objects
@pytest.fixture
def mock_model_no_exog():
    """Returns a MockModel that does not require exogenous variables."""
    return MockModel(expects_exog=False)

@pytest.fixture
def mock_model_with_exog():
    """Returns a MockModel that requires exogenous variables."""
    return MockModel(expects_exog=True)

@pytest.fixture
def mock_exog_data():
    """Returns a sample pandas DataFrame for exogenous variables."""
    steps = 5
    return pd.DataFrame(
        {'gdp': np.random.rand(steps) * 10, 'cpi': np.random.rand(steps) * 2},
        index=pd.date_range(start='2025-01-01', periods=steps, freq='Q')
    )

# Test cases for generate_forecasts function
def test_generate_forecasts_basic_no_exog(mock_model_no_exog):
    """
    Test basic functionality without exogenous variables and with confidence intervals.
    """
    steps = 3
    alpha = 0.05
    result = generate_forecasts(mock_model_no_exog, steps, exog_forecast=None, alpha=alpha)
    
    assert isinstance(result, pd.DataFrame)
    assert len(result) == steps
    assert 'forecast' in result.columns
    assert 'lower_ci' in result.columns
    assert 'upper_ci' in result.columns
    assert result.index is not None # Check if index is set

def test_generate_forecasts_basic_with_exog(mock_model_with_exog, mock_exog_data):
    """
    Test basic functionality with exogenous variables and different alpha.
    """
    steps = len(mock_exog_data) # Steps should match exog_forecast length
    alpha = 0.01
    result = generate_forecasts(mock_model_with_exog, steps, exog_forecast=mock_exog_data, alpha=alpha)
    
    assert isinstance(result, pd.DataFrame)
    assert len(result) == steps
    assert 'forecast' in result.columns
    assert 'lower_ci' in result.columns
    assert 'upper_ci' in result.columns
    assert result.index is not None

def test_generate_forecasts_invalid_steps(mock_model_no_exog):
    """
    Test edge cases for 'steps' argument (zero or negative).
    Should raise a ValueError.
    """
    with pytest.raises(ValueError, match="Steps must be a positive integer."):
        generate_forecasts(mock_model_no_exog, 0, exog_forecast=None, alpha=0.05)
        
    with pytest.raises(ValueError, match="Steps must be a positive integer."):
        generate_forecasts(mock_model_no_exog, -5, exog_forecast=None, alpha=0.05)

def test_generate_forecasts_invalid_model_object():
    """
    Test case where the 'model' argument is not a valid object
    (e.g., does not have a 'forecast' method).
    Should raise an AttributeError.
    """
    invalid_model = Mock() # A generic mock object without 'forecast' method by default
    steps = 2
    
    with pytest.raises(AttributeError):
        generate_forecasts(invalid_model, steps, exog_forecast=None, alpha=0.05)

def test_generate_forecasts_invalid_alpha_value(mock_model_no_exog):
    """
    Test edge cases for 'alpha' argument (out of valid range 0-1).
    Should raise a ValueError.
    """
    steps = 3
    
    with pytest.raises(ValueError, match="Alpha must be between 0 and 1 inclusive."):
        generate_forecasts(mock_model_no_exog, steps, exog_forecast=None, alpha=1.1)
        
    with pytest.raises(ValueError, match="Alpha must be between 0 and 1 inclusive."):
        generate_forecasts(mock_model_no_exog, steps, exog_forecast=None, alpha=-0.01)

