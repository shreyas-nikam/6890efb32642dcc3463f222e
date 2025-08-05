import pytest
import pandas as pd
import matplotlib.pyplot as plt
import os

# Keep the your_module block as it is. DO NOT REPLACE or REMOVE the block.
from definition_d33efb169d5141b2a108a0dd19ec7134 import plot_time_series

@pytest.fixture
def sample_dataframe():
    """Provides a sample DataFrame with a DatetimeIndex for testing time-series plots."""
    dates = pd.date_range(start='2020-01-01', periods=5, freq='D')
    data = {
        'column_a': [10, 12, 15, 13, 16],
        'column_b': [5, 6, 4, 7, 8],
        'column_c': [1, 2, 3, 2, 1]
    }
    df = pd.DataFrame(data, index=dates)
    df.index.name = 'Date' # Good practice for time series index
    return df

@pytest.mark.parametrize(
    "columns, output_path_suffix, expected_savefig, expected_show, title",
    [
        (['column_a'], "plot_a.png", True, False, "Single Column Plot"),
        (['column_a', 'column_b'], "plot_ab.png", True, False, "Multi Column Plot"),
        (['column_c'], None, False, True, "Display Plot Test"),
        (['column_a', 'column_c'], None, False, True, "Multi Column Display"),
    ]
)
def test_plot_time_series_successful_plotting(
    mocker, tmp_path, sample_dataframe, columns, output_path_suffix, expected_savefig, expected_show, title
):
    """
    Test that the function successfully generates and either saves or displays plots
    for valid inputs and existing columns.
    """
    mock_savefig = mocker.patch('matplotlib.pyplot.savefig')
    mock_show = mocker.patch('matplotlib.pyplot.show')
    mock_close = mocker.patch('matplotlib.pyplot.close')

    final_output_path = str(tmp_path / output_path_suffix) if output_path_suffix else None

    plot_time_series(sample_dataframe, columns, title, final_output_path)

    if expected_savefig:
        mock_savefig.assert_called_once_with(final_output_path)
    else:
        mock_savefig.assert_not_called()

    if expected_show:
        mock_show.assert_called_once()
    else:
        mock_show.assert_not_called()
    
    # plt.close() should generally be called after savefig or show to free resources.
    # We assume a well-behaved function closes the plot after its operation.
    mock_close.assert_called_once() 

@pytest.mark.parametrize(
    "df_input, columns, title, output_path_suffix, expected_exception",
    [
        # Edge case: Empty DataFrame
        (pd.DataFrame(), [], "Empty DF Test", "empty_df.png", None), 
        # Edge case: Empty list of columns
        (pd.DataFrame(index=pd.date_range(start='2020-01-01', periods=2)), [], "Empty Columns List", "empty_cols.png", None),
        # Edge case: Non-existent column
        (pd.DataFrame({'A': [1], 'Date': pd.to_datetime(['2020-01-01'])}).set_index('Date'), ['non_existent'], "Missing Column", None, KeyError), 
        # Edge case: Partially non-existent columns
        (pd.DataFrame({'A': [1], 'B': [2], 'Date': pd.to_datetime(['2020-01-01'])}).set_index('Date'), ['A', 'non_existent'], "Partial Missing Column", None, KeyError), 
    ]
)
def test_plot_time_series_dataframe_and_column_edge_cases(
    mocker, tmp_path, df_input, columns, title, output_path_suffix, expected_exception
):
    """
    Test various edge cases related to DataFrame content and column existence.
    """
    mock_savefig = mocker.patch('matplotlib.pyplot.savefig')
    mock_show = mocker.patch('matplotlib.pyplot.show')
    mock_close = mocker.patch('matplotlib.pyplot.close')

    final_output_path = str(tmp_path / output_path_suffix) if output_path_suffix else None

    if expected_exception:
        with pytest.raises(expected_exception):
            plot_time_series(df_input, columns, title, final_output_path)
        # No plotting actions should occur if an error is raised
        mock_savefig.assert_not_called()
        mock_show.assert_not_called()
        mock_close.assert_not_called()
    else:
        plot_time_series(df_input, columns, title, final_output_path)
        # For empty dataframes or empty column lists, no meaningful plot can be generated.
        # A robust function should skip actual plotting/showing/saving in these cases.
        mock_savefig.assert_not_called()
        mock_show.assert_not_called()
        mock_close.assert_not_called()


@pytest.mark.parametrize(
    "dataframe_param, columns_param, title_param, output_path_param",
    [
        (None, ['column_a'], "Title", None),                         # Invalid dataframe type (None)
        ("not a dataframe", ['column_a'], "Title", None),           # Invalid dataframe type (str)
        (pd.DataFrame({'A': [1]}), 'not a list', "Title", None),     # Invalid columns type (str)
        (pd.DataFrame({'A': [1]}), ['A'], 123, None),               # Invalid title type (int)
        (pd.DataFrame({'A': [1]}), ['A'], "Title", 123),            # Invalid output_path type (int)
    ]
)
def test_plot_time_series_invalid_input_types(
    dataframe_param, columns_param, title_param, output_path_param
):
    """
    Test that the function raises TypeError for invalid input argument types.
    """
    with pytest.raises(TypeError):
        plot_time_series(dataframe_param, columns_param, title_param, output_path_param)