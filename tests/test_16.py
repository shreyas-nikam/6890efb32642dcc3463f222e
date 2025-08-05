import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

# Keep the definition_8106f7d6d4874ed0a1480a8f0133404a block as it is. DO NOT REPLACE or REMOVE the block.
from definition_8106f7d6d4874ed0a1480a8f0133404a import run_diagnostic_tests

@patch('statsmodels.stats.diagnostic.acorr_ljungbox')
@patch('statsmodels.stats.stattools.jarque_bera')
def test_run_diagnostic_tests_valid_inputs(mock_jarque_bera, mock_acorr_ljungbox):
    """
    Tests the function with valid residuals and model_results, expecting correct diagnostic output.
    """
    residuals = pd.Series(np.random.normal(0, 1, 100))
    mock_model_results = Mock()
    mock_model_results.durbin_watson = 1.95 

    mock_acorr_ljungbox.return_value = (np.array([10.0]), np.array([0.15])) # (statistic, pvalue)
    mock_jarque_bera.return_value = (0.5, 0.77, 0.05, 3.01) # (jb_value, p_value, skew, kurtosis)

    result = run_diagnostic_tests(residuals, mock_model_results)

    assert isinstance(result, dict)
    assert 'durbin_watson' in result
    assert 'ljung_box_p_value' in result 
    assert 'jarque_bera_p_value' in result 
    assert result['durbin_watson'] == 1.95
    assert result['ljung_box_p_value'] == 0.15
    assert result['jarque_bera_p_value'] == 0.77

    mock_acorr_ljungbox.assert_called_once()
    mock_jarque_bera.assert_called_once()

@patch('statsmodels.stats.diagnostic.acorr_ljungbox', side_effect=ValueError("Insufficient observations for Ljung-Box"))
@patch('statsmodels.stats.stattools.jarque_bera', side_effect=ValueError("Insufficient observations for Jarque-Bera"))
def test_run_diagnostic_tests_empty_residuals(mock_jarque_bera, mock_acorr_ljungbox):
    """
    Tests the function with an empty residuals Series, expecting a ValueError due to insufficient data for diagnostics.
    """
    residuals = pd.Series([], dtype=float)
    mock_model_results = Mock()
    mock_model_results.durbin_watson = np.nan 

    with pytest.raises(ValueError) as excinfo:
        run_diagnostic_tests(residuals, mock_model_results)
    assert "Insufficient observations" in str(excinfo.value)

@patch('statsmodels.stats.diagnostic.acorr_ljungbox')
@patch('statsmodels.stats.stattools.jarque_bera')
def test_run_diagnostic_tests_non_numeric_residuals(mock_jarque_bera, mock_acorr_ljungbox):
    """
    Tests the function with non-numeric data in residuals, expecting a TypeError or ValueError.
    """
    residuals = pd.Series(['a', 'b', 'c'])
    mock_model_results = Mock()
    mock_model_results.durbin_watson = 1.0

    mock_acorr_ljungbox.side_effect = TypeError("unsupported operand type(s) for +: 'str' and 'str'")
    mock_jarque_bera.side_effect = ValueError("could not convert string to float")

    with pytest.raises(Exception) as excinfo:
        run_diagnostic_tests(residuals, mock_model_results)
    assert ("unsupported operand type" in str(excinfo.value) or "could not convert" in str(excinfo.value))

@patch('statsmodels.stats.diagnostic.acorr_ljungbox')
@patch('statsmodels.stats.stattools.jarque_bera')
def test_run_diagnostic_tests_missing_durbin_watson_attribute(mock_jarque_bera, mock_acorr_ljungbox):
    """
    Tests the function when model_results lacks the 'durbin_watson' attribute, expecting an AttributeError.
    """
    residuals = pd.Series(np.random.normal(0, 1, 100))
    mock_model_results = Mock() 

    mock_acorr_ljungbox.return_value = (np.array([10.0]), np.array([0.15]))
    mock_jarque_bera.return_value = (0.5, 0.77, 0.05, 3.01)

    with pytest.raises(AttributeError) as excinfo:
        run_diagnostic_tests(residuals, mock_model_results)
    assert "durbin_watson" in str(excinfo.value)

    mock_acorr_ljungbox.assert_not_called()
    mock_jarque_bera.assert_not_called()

@patch('statsmodels.stats.diagnostic.acorr_ljungbox')
@patch('statsmodels.stats.stattools.jarque_bera')
def test_run_diagnostic_tests_partial_failure(mock_jarque_bera, mock_acorr_ljungbox):
    """
    Tests the function with residuals that are sufficient for some diagnostics but not others.
    Expects NaN for tests that cannot be performed.
    """
    residuals = pd.Series(np.random.normal(0, 1, 5)) 
    mock_model_results = Mock()
    mock_model_results.durbin_watson = 1.5

    mock_jarque_bera.return_value = (0.5, 0.77, 0.05, 3.01)

    mock_acorr_ljungbox.return_value = (np.array([np.nan]), np.array([np.nan]))

    result = run_diagnostic_tests(residuals, mock_model_results)

    assert isinstance(result, dict)
    assert result['durbin_watson'] == 1.5
    assert np.isnan(result['ljung_box_p_value'])
    assert result['jarque_bera_p_value'] == 0.77

    mock_acorr_ljungbox.assert_called_once()
    mock_jarque_bera.assert_called_once()