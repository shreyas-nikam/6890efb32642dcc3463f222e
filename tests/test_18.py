import pytest
import pandas as pd
from definition_253b894b749a4f7b9d5552eefef59b35 import incorporate_macro_scenarios

@pytest.fixture
def base_macro_forecast_fixture():
    """Provides a sample base macroeconomic forecast DataFrame."""
    data = {
        'GDP': [100.0, 101.0, 102.0],
        'CPI': [2.0, 2.1, 2.2],
        'Unemployment': [5.0, 5.1, 5.2]
    }
    dates = pd.to_datetime(['2023-01-01', '2023-04-01', '2023-07-01'])
    return pd.DataFrame(data, index=dates)

def test_incorporate_macro_scenarios_basic_application(base_macro_forecast_fixture):
    """
    Test case 1: Verify basic application of stress deviations to the base forecast.
    """
    stress_scenario_data = {
        'GDP': {pd.Timestamp('2023-04-01'): -2.0, pd.Timestamp('2023-07-01'): -3.0},
        'CPI': {pd.Timestamp('2023-01-01'): 0.5}
    }
    variable_map = {} # No mapping needed, names are direct

    expected_df = base_macro_forecast_fixture.copy()
    expected_df.loc[pd.Timestamp('2023-04-01'), 'GDP'] = 101.0 - 2.0 # 99.0
    expected_df.loc[pd.Timestamp('2023-07-01'), 'GDP'] = 102.0 - 3.0 # 99.0
    expected_df.loc[pd.Timestamp('2023-01-01'), 'CPI'] = 2.0 + 0.5 # 2.5

    result_df = incorporate_macro_scenarios(base_macro_forecast_fixture, stress_scenario_data, variable_map)
    pd.testing.assert_frame_equal(result_df, expected_df)

def test_incorporate_macro_scenarios_no_stress_data(base_macro_forecast_fixture):
    """
    Test case 2: Verify the function returns the original DataFrame when no stress data is provided.
    """
    stress_scenario_data = {}
    variable_map = {}

    expected_df = base_macro_forecast_fixture.copy()

    result_df = incorporate_macro_scenarios(base_macro_forecast_fixture, stress_scenario_data, variable_map)
    pd.testing.assert_frame_equal(result_df, expected_df)

def test_incorporate_macro_scenarios_with_variable_mapping():
    """
    Test case 3: Verify correct application of stress data using a variable map.
    """
    base_data_map = {
        'GDP_baseline': [100.0, 101.0],
        'Inflation_rate': [2.0, 2.1]
    }
    dates_map = pd.to_datetime(['2023-01-01', '2023-04-01'])
    base_df_map = pd.DataFrame(base_data_map, index=dates_map)

    stress_data_map = {
        'GDP_stress_impact': {pd.Timestamp('2023-04-01'): -1.5},
        'CPI_deviation': {pd.Timestamp('2023-01-01'): 0.3}
    }
    variable_map = {
        'GDP_stress_impact': 'GDP_baseline',
        'CPI_deviation': 'Inflation_rate'
    }

    expected_df_map = base_df_map.copy()
    expected_df_map.loc[pd.Timestamp('2023-04-01'), 'GDP_baseline'] = 101.0 - 1.5 # 99.5
    expected_df_map.loc[pd.Timestamp('2023-01-01'), 'Inflation_rate'] = 2.0 + 0.3 # 2.3

    result_df_map = incorporate_macro_scenarios(base_df_map, stress_data_map, variable_map)
    pd.testing.assert_frame_equal(result_df_map, expected_df_map)

def test_incorporate_macro_scenarios_extra_variables_in_stress_data(base_macro_forecast_fixture):
    """
    Test case 4: Verify that stress variables not present in the base forecast are ignored.
    """
    stress_scenario_data = {
        'GDP': {pd.Timestamp('2023-04-01'): -2.0},
        'NonExistentVariable': {pd.Timestamp('2023-07-01'): 10.0} # This variable is not in base_macro_forecast_fixture
    }
    variable_map = {}

    expected_df = base_macro_forecast_fixture.copy()
    expected_df.loc[pd.Timestamp('2023-04-01'), 'GDP'] = 101.0 - 2.0 # Only GDP should be adjusted

    result_df = incorporate_macro_scenarios(base_macro_forecast_fixture, stress_scenario_data, variable_map)
    pd.testing.assert_frame_equal(result_df, expected_df)

@pytest.mark.parametrize("base_macro_forecast, stress_scenario_data, variable_map, expected_error", [
    (None, {}, {}, TypeError), # base_macro_forecast is not DataFrame
    (pd.DataFrame(), None, {}, TypeError), # stress_scenario_data is not dict/DataFrame
    (pd.DataFrame(), {}, None, TypeError), # variable_map is not dict
    ([], {}, {}, TypeError), # base_macro_forecast is a list
    (pd.DataFrame(), [], {}, TypeError), # stress_scenario_data is a list
    (pd.DataFrame(), {}, "not_a_dict", TypeError), # variable_map is a string
])
def test_incorporate_macro_scenarios_type_errors(base_macro_forecast, stress_scenario_data, variable_map, expected_error):
    """
    Test case 5: Verify that the function raises TypeError for incorrect input types.
    """
    with pytest.raises(expected_error):
        incorporate_macro_scenarios(base_macro_forecast, stress_scenario_data, variable_map)