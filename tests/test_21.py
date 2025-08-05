import pytest
import pandas as pd
import numpy as np # Used for potential numeric operations if needed, though not directly in the fixture data creation here.

# Placeholder for your module import
from definition_70041e9343fe45b6a986f8f1f3104eec import plot_scenario_pathways

@pytest.fixture
def mock_matplotlib(mocker):
    """
    Fixture to mock matplotlib.pyplot functions to prevent actual plot rendering
    and allow checking for calls to plot/save functions.
    """
    # Mocking common matplotlib functions that would be called
    mocker.patch('matplotlib.pyplot.figure')
    mocker.patch('matplotlib.pyplot.plot')
    mocker.patch('matplotlib.pyplot.fill_between') # Used for confidence intervals/fan charts
    mocker.patch('matplotlib.pyplot.legend')
    mocker.patch('matplotlib.pyplot.title')
    mocker.patch('matplotlib.pyplot.xlabel')
    mocker.patch('matplotlib.pyplot.ylabel')
    mocker.patch('matplotlib.pyplot.grid')
    mocker.patch('matplotlib.pyplot.tight_layout')
    mocker.patch('matplotlib.pyplot.show')
    mocker.patch('matplotlib.pyplot.savefig')
    
    # Return the mocked pyplot module for easier assertion checks
    return mocker.patch('matplotlib.pyplot')


@pytest.fixture
def sample_data():
    """
    Fixture to provide realistic sample pandas Series and dicts for testing.
    """
    dates = pd.date_range(start='2023-01-01', periods=5, freq='Q')
    
    baseline = pd.Series([0.01, 0.015, 0.02, 0.018, 0.016], index=dates, name='Baseline PD')
    
    stress_scenarios = {
        'Stress 1': pd.Series([0.02, 0.025, 0.035, 0.03, 0.028], index=dates, name='Stress 1 PD'),
        'Stress 2': pd.Series([0.018, 0.022, 0.03, 0.025, 0.023], index=dates, name='Stress 2 PD'),
    }
    
    # Confidence intervals are typically represented as lower/upper bounds
    # For a fan chart, these would often be pandas DataFrames or similar structures.
    confidence_intervals = {
        'Stress 1': pd.DataFrame({
            'lower': [0.015, 0.02, 0.028, 0.025, 0.023],
            'upper': [0.025, 0.03, 0.042, 0.035, 0.033]
        }, index=dates),
        'Stress 2': pd.DataFrame({
            'lower': [0.013, 0.018, 0.025, 0.02, 0.018],
            'upper': [0.023, 0.026, 0.035, 0.03, 0.028]
        }, index=dates),
    }
    return baseline, stress_scenarios, confidence_intervals


def test_plot_scenario_pathways_saves_plot_correctly(mock_matplotlib, sample_data, tmp_path):
    """
    Test case 1: Verify that the function correctly saves the plot to a file
    when an output_path is provided.
    """
    baseline_pd, stress_pd_scenarios, confidence_intervals = sample_data
    output_file = tmp_path / "test_scenario_plot.png"
    
    plot_scenario_pathways(
        baseline_pd, 
        stress_pd_scenarios, 
        confidence_intervals, 
        "My Scenario Plot", 
        str(output_file)
    )
    
    mock_matplotlib.savefig.assert_called_once_with(str(output_file), bbox_inches='tight') # Common savefig arg
    mock_matplotlib.show.assert_not_called()
    mock_matplotlib.plot.assert_called() # Ensure plotting functions were generally called
    mock_matplotlib.fill_between.assert_called() # Ensure CIs were generally called

def test_plot_scenario_pathways_displays_plot_when_output_path_is_none(mock_matplotlib, sample_data):
    """
    Test case 2: Verify that the function displays the plot when output_path is None.
    """
    baseline_pd, stress_pd_scenarios, confidence_intervals = sample_data
    
    plot_scenario_pathways(
        baseline_pd, 
        stress_pd_scenarios, 
        confidence_intervals, 
        "Display Only Plot", 
        None
    )
    
    mock_matplotlib.show.assert_called_once()
    mock_matplotlib.savefig.assert_not_called()
    mock_matplotlib.plot.assert_called()
    mock_matplotlib.fill_between.assert_called()

def test_plot_scenario_pathways_handles_no_stress_scenarios(mock_matplotlib, sample_data):
    """
    Test case 3: Verify that the function handles an empty stress_pd_scenarios dictionary
    gracefully (e.g., plots only the baseline).
    """
    baseline_pd, _, _ = sample_data # Only need baseline_pd
    
    plot_scenario_pathways(
        baseline_pd, 
        {}, # Empty stress scenarios
        None, # No confidence intervals either
        "Baseline Only Forecast", 
        None
    )
    
    mock_matplotlib.show.assert_called_once()
    mock_matplotlib.plot.assert_called() # Baseline should still be plotted
    mock_matplotlib.fill_between.assert_not_called() # No CIs to plot without scenarios

def test_plot_scenario_pathways_handles_none_confidence_intervals(mock_matplotlib, sample_data):
    """
    Test case 4: Verify that the function correctly plots without confidence intervals
    when confidence_intervals is None.
    """
    baseline_pd, stress_pd_scenarios, _ = sample_data
    
    plot_scenario_pathways(
        baseline_pd, 
        stress_pd_scenarios, 
        None, # No confidence intervals
        "Plot Without Confidence Intervals", 
        None
    )
    
    mock_matplotlib.show.assert_called_once()
    mock_matplotlib.plot.assert_called() # Baseline and stress lines should still be plotted
    mock_matplotlib.fill_between.assert_not_called() # Ensure fill_between is not called for CIs

@pytest.mark.parametrize("baseline_pd, stress_pd_scenarios, confidence_intervals, title, output_path, expected_error", [
    # Test case 5: Invalid input types for various arguments
    (123, {}, None, "Title", None, TypeError), # baseline_pd not pandas.Series
    ([1,2,3], {}, None, "Title", None, TypeError), # baseline_pd not pandas.Series
    (pd.Series([1,2,3]), "not_a_dict", None, "Title", None, TypeError), # stress_pd_scenarios not dict
    (pd.Series([1,2,3]), {'s1': 123}, None, "Title", None, TypeError), # stress_pd_scenarios contains non-Series value
    (pd.Series([1,2,3]), {}, None, 123, None, TypeError), # title not str
    (pd.Series([1,2,3]), {}, None, "Title", 123, TypeError), # output_path not str or None
])
def test_plot_scenario_pathways_raises_type_errors_for_invalid_inputs(
    baseline_pd, stress_pd_scenarios, confidence_intervals, title, output_path, expected_error, mocker
):
    """
    Test case 5 (continued): Parametrized test to ensure the function raises TypeError
    for various invalid input types.
    """
    # Mocking matplotlib even for type errors, as the function might internally call it
    # after some initial validation if not strict enough, or to prevent side effects.
    mocker.patch('matplotlib.pyplot.show')
    mocker.patch('matplotlib.pyplot.savefig')

    with pytest.raises(expected_error):
        plot_scenario_pathways(baseline_pd, stress_pd_scenarios, confidence_intervals, title, output_path)