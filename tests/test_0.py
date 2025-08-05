import pytest
import pandas as pd
from pytest_mock import MockerFixture
import os

# definition_85331be5a37742aba2df4039d392f3ed block
from definition_85331be5a37742aba2df4039d392f3ed import load_datasets
# End definition_85331be5a37742aba2df4039d392f3ed block

@pytest.fixture
def mock_pd_data() -> pd.DataFrame:
    """Returns a simple DataFrame for Probability of Default (PD) data."""
    return pd.DataFrame({'date': pd.to_datetime(['2023-01-01', '2023-04-01']), 'pd_value': [0.01, 0.015]})

@pytest.fixture
def mock_macro_data() -> pd.DataFrame:
    """Returns a simple DataFrame for Macroeconomic data."""
    return pd.DataFrame({'date': pd.to_datetime(['2023-01-01', '2023-04-01']), 'gdp': [1.0, 1.2], 'cpi': [2.0, 2.1]})

@pytest.mark.parametrize(
    "pd_filepath, macro_filepath, expected_exception, mock_scenario",
    [
        # Test Case 1: Successful loading of two valid CSV files
        ("data/pd_loans.csv", "data/macro_factors.csv", None, "success_csv"),
        # Test Case 2: Successful loading of two valid Parquet files
        ("data/pd_loans.parquet", "data/macro_factors.parquet", None, "success_parquet"),
        # Test Case 3: FileNotFoundError when one or both files do not exist
        ("non_existent_pd.csv", "non_existent_macro.csv", FileNotFoundError, "file_not_found"),
        # Test Case 4: ValueError for an invalid or malformed data file (e.g., empty CSV)
        # Assumes the function, when implemented, handles or re-raises pandas' read errors as ValueError
        ("empty_pd.csv", "data/macro_factors.csv", ValueError, "malformed_pd_csv"),
        # Test Case 5: TypeError for invalid input types (e.g., non-string file paths)
        (123, "data/macro_factors.csv", TypeError, "invalid_type"),
    ]
)
def test_load_datasets(
    mocker: MockerFixture,
    mock_pd_data: pd.DataFrame,
    mock_macro_data: pd.DataFrame,
    pd_filepath: str,
    macro_filepath: str,
    expected_exception: type,
    mock_scenario: str
):
    """
    Tests the load_datasets function for various scenarios including successful loads,
    file not found, invalid file content, and incorrect input types.
    """
    # Mock os.path.exists to control file existence behavior
    mock_os_path_exists = mocker.patch('os.path.exists')
    if mock_scenario == "file_not_found":
        mock_os_path_exists.return_value = False
    else:
        mock_os_path_exists.return_value = True

    # Mock pandas read functions to control their return values or raised exceptions
    mock_read_csv = mocker.patch('pandas.read_csv')
    mock_read_parquet = mocker.patch('pandas.read_parquet')

    if mock_scenario == "success_csv":
        # Simulate pd.read_csv returning mock data for both files
        mock_read_csv.side_effect = [mock_pd_data, mock_macro_data]
        # Ensure pd.read_parquet is not called in this scenario
        mock_read_parquet.side_effect = NotImplementedError("Should not call parquet for CSV scenario")
    elif mock_scenario == "success_parquet":
        # Simulate pd.read_parquet returning mock data for both files
        mock_read_parquet.side_effect = [mock_pd_data, mock_macro_data]
        # Ensure pd.read_csv is not called in this scenario
        mock_read_csv.side_effect = NotImplementedError("Should not call csv for Parquet scenario")
    elif mock_scenario == "malformed_pd_csv":
        # Simulate pd.read_csv raising an EmptyDataError (a subclass of ValueError) for the first file
        # and returning valid data for the second file.
        mock_read_csv.side_effect = [pd.errors.EmptyDataError("Simulated empty or malformed CSV"), mock_macro_data]
        mock_read_parquet.side_effect = NotImplementedError("Should not call parquet in this scenario")
    elif mock_scenario == "invalid_type":
        # For invalid type tests, file reading functions should ideally not be called
        mock_read_csv.side_effect = NotImplementedError("Should not be called for invalid type scenario")
        mock_read_parquet.side_effect = NotImplementedError("Should not be called for invalid type scenario")

    if expected_exception:
        # Assert that the expected exception is raised
        with pytest.raises(expected_exception):
            load_datasets(pd_filepath, macro_filepath)
    else:
        # Call the function and assert its return values for successful scenarios
        pd_df, macro_df = load_datasets(pd_filepath, macro_filepath)
        pd.testing.assert_frame_equal(pd_df, mock_pd_data)
        pd.testing.assert_frame_equal(macro_df, mock_macro_data)

        # Verify that the correct pandas read function was called the correct number of times
        if mock_scenario == "success_csv":
            assert mock_read_csv.call_count == 2
            mock_read_csv.assert_any_call(pd_filepath)
            mock_read_csv.assert_any_call(macro_filepath)
            mock_read_parquet.assert_not_called()
        elif mock_scenario == "success_parquet":
            assert mock_read_parquet.call_count == 2
            mock_read_parquet.assert_any_call(pd_filepath)
            mock_read_parquet.assert_any_call(macro_filepath)
            mock_read_csv.assert_not_called()