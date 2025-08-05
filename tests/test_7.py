import pytest
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock

# Keep a placeholder definition_1984e073a5224a6892f1da9774505cde for the import of the module. Keep the `your_module` block as it is. DO NOT REPLACE or REMOVE the block.
from definition_1984e073a5224a6892f1da9774505cde import plot_correlation_heatmap

# Mock pandas, seaborn, and matplotlib components to prevent actual plotting and file I/O during tests.
# These mocks will intercept calls to the actual libraries within the tested function.

@patch('seaborn.heatmap')
@patch('matplotlib.pyplot.savefig')
@patch('matplotlib.pyplot.show')
@patch('matplotlib.pyplot.title')
@patch('matplotlib.pyplot.tight_layout')
@patch('matplotlib.pyplot.figure')
@patch('matplotlib.pyplot.close')
def test_plot_correlation_heatmap_success_save(
    mock_close, mock_figure, mock_tight_layout, mock_title, mock_show, mock_savefig, mock_heatmap
):
    """
    Test case 1: Verify successful plot generation and saving to a file.
    Expects: seaborn.heatmap and matplotlib.pyplot.savefig to be called.
    """
    # Create a mock DataFrame with numerical data and its expected correlation matrix
    mock_df = MagicMock(spec=pd.DataFrame)
    mock_corr_matrix = pd.DataFrame({
        'col_a': [1.0, 0.6],
        'col_b': [0.6, 1.0]
    }, index=['col_a', 'col_b'])
    mock_df.corr.return_value = mock_corr_matrix

    test_title = "Sample Correlation Heatmap"
    test_output_path = "output/test_heatmap.png"

    plot_correlation_heatmap(mock_df, test_title, test_output_path)

    # Assertions for mock calls
    mock_figure.assert_called_once()
    mock_df.corr.assert_called_once()
    mock_heatmap.assert_called_once_with(
        mock_corr_matrix,
        annot=True,
        cmap='coolwarm',
        fmt=".2f",
        linewidths=.5,
        cbar_kws={"shrink": .8}
    )
    mock_title.assert_called_once_with(test_title)
    mock_tight_layout.assert_called_once()
    mock_savefig.assert_called_once_with(test_output_path)
    mock_show.assert_not_called()  # Should not be called when output_path is provided
    mock_close.assert_called_once()


@patch('seaborn.heatmap')
@patch('matplotlib.pyplot.savefig')
@patch('matplotlib.pyplot.show')
@patch('matplotlib.pyplot.title')
@patch('matplotlib.pyplot.tight_layout')
@patch('matplotlib.pyplot.figure')
@patch('matplotlib.pyplot.close')
def test_plot_correlation_heatmap_success_display(
    mock_close, mock_figure, mock_tight_layout, mock_title, mock_show, mock_savefig, mock_heatmap
):
    """
    Test case 2: Verify successful plot generation and display when output_path is None.
    Expects: seaborn.heatmap and matplotlib.pyplot.show to be called.
    """
    # Create a mock DataFrame with numerical data and its expected correlation matrix
    mock_df = MagicMock(spec=pd.DataFrame)
    mock_corr_matrix = pd.DataFrame({
        'feat1': [1.0, -0.3],
        'feat2': [-0.3, 1.0]
    }, index=['feat1', 'feat2'])
    mock_df.corr.return_value = mock_corr_matrix

    test_title = "Display Only Heatmap"
    test_output_path = None

    plot_correlation_heatmap(mock_df, test_title, test_output_path)

    # Assertions for mock calls
    mock_figure.assert_called_once()
    mock_df.corr.assert_called_once()
    mock_heatmap.assert_called_once_with(
        mock_corr_matrix,
        annot=True,
        cmap='coolwarm',
        fmt=".2f",
        linewidths=.5,
        cbar_kws={"shrink": .8}
    )
    mock_title.assert_called_once_with(test_title)
    mock_tight_layout.assert_called_once()
    mock_savefig.assert_not_called()  # Should not be called when output_path is None
    mock_show.assert_called_once()   # Should be called when output_path is None
    mock_close.assert_called_once()


@patch('seaborn.heatmap')
@patch('matplotlib.pyplot.savefig')
@patch('matplotlib.pyplot.show')
@patch('matplotlib.pyplot.title')
@patch('matplotlib.pyplot.tight_layout')
@patch('matplotlib.pyplot.figure')
@patch('matplotlib.pyplot.close')
@pytest.mark.parametrize(
    "dataframe_input",
    [
        pd.DataFrame(),  # Empty DataFrame
        pd.DataFrame({'text_col': ['a', 'b'], 'bool_col': [True, False]}),  # DataFrame with no numerical columns
        pd.DataFrame({'num_col': [1, 2], 'text_col': ['x', 'y']}).select_dtypes(include='object') # df with only non-numeric (after selection)
    ]
)
def test_plot_correlation_heatmap_empty_or_non_numeric_data(
    mock_close, mock_figure, mock_tight_layout, mock_title, mock_show, mock_savefig, mock_heatmap,
    dataframe_input
):
    """
    Test case 3: Handle DataFrames that are empty or contain no numerical columns.
    Expects: The function to run without error, and heatmap to be called with an empty DataFrame (from .corr()).
    """
    test_title = "Non-Numeric Data Heatmap"
    test_output_path = "output/empty_heatmap.png"

    # When .corr() is called on an empty or non-numeric DataFrame, it returns an empty DataFrame.
    expected_corr_output = pd.DataFrame()

    plot_correlation_heatmap(dataframe_input, test_title, test_output_path)

    # Assertions
    mock_figure.assert_called_once()
    # No direct mock for df.corr() as we pass actual pandas objects,
    # and their .corr() method correctly returns an empty DataFrame in these cases.
    mock_heatmap.assert_called_once()
    # Check if the heatmap was called with an empty DataFrame (which df.corr() would return)
    args, kwargs = mock_heatmap.call_args
    assert args[0].empty is True and args[0].equals(expected_corr_output)

    mock_title.assert_called_once_with(test_title)
    mock_tight_layout.assert_called_once()
    mock_savefig.assert_called_once_with(test_output_path)
    mock_show.assert_not_called()
    mock_close.assert_called_once()


@pytest.mark.parametrize("dataframe, title, output_path, expected_exception", [
    (123, "Title", "path.png", TypeError),  # Invalid dataframe type (not a DataFrame)
    ("not_a_df", "Title", "path.png", TypeError), # Invalid dataframe type (string)
    (pd.DataFrame({'A': [1]}), 123, "path.png", TypeError), # Invalid title type (not a string)
    (pd.DataFrame({'A': [1]}), None, "path.png", TypeError), # Invalid title type (None)
])
def test_plot_correlation_heatmap_invalid_input_types(
    dataframe, title, output_path, expected_exception
):
    """
    Test case 4: Verify that the function raises TypeError for invalid input types.
    """
    with pytest.raises(expected_exception):
        plot_correlation_heatmap(dataframe, title, output_path)