import pytest
import pandas as pd
import numpy as np

# definition_66c48bdfe4e94feb900b05a567297e6b block - DO NOT REPLACE or REMOVE
from definition_66c48bdfe4e94feb900b05a567297e6b import generate_stress_multiplier_table
# </your_module>

@pytest.mark.parametrize(
    "baseline_data, stress_data, expected_output, expected_exception",
    [
        # Test Case 1: Basic functionality with positive numeric data
        (
            pd.DataFrame({'PD': [0.01, 0.02], 'LGD': [0.3, 0.4]}),
            pd.DataFrame({'PD': [0.015, 0.03], 'LGD': [0.33, 0.44]}),
            pd.DataFrame({'PD': [1.5, 1.5], 'LGD': [1.1, 1.1]}),
            None,
        ),
        # Test Case 2: Empty DataFrames
        (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            None,
        ),
        # Test Case 3: Baseline with zero values (leading to np.nan for 0/0 and np.inf for x/0)
        (
            pd.DataFrame({'PD': [0.01, 0.00, 0.00], 'LGD': [0.3, 0.4, 0.5]}),
            pd.DataFrame({'PD': [0.015, 0.00, 0.06], 'LGD': [0.33, 0.44, 0.00]}),
            pd.DataFrame({'PD': [1.5, np.nan, np.inf], 'LGD': [1.1, 1.1, 0.0]}),
            None,
        ),
        # Test Case 4: Mismatched columns (Pandas' default behavior: union of columns, NaN for non-matching elements)
        (
            pd.DataFrame({'PD': [0.01, 0.02], 'LGD': [0.3, 0.4]}),
            pd.DataFrame({'PD': [0.015, 0.03], 'Exposure': [100.0, 200.0]}),
            pd.DataFrame({
                'Exposure': [np.nan, np.nan],
                'LGD': [np.nan, np.nan],
                'PD': [1.5, 1.5]
            }, columns=['Exposure', 'LGD', 'PD']), # Columns are sorted alphabetically by default in union
            None,
        ),
        # Test Case 5: Non-DataFrame input for baseline_data (expecting TypeError)
        (
            [1, 2, 3], # Invalid input type
            pd.DataFrame({'PD': [0.015], 'LGD': [0.33]}),
            None,
            TypeError,
        ),
    ]
)
def test_generate_stress_multiplier_table(baseline_data, stress_data, expected_output, expected_exception):
    if expected_exception:
        with pytest.raises(expected_exception):
            generate_stress_multiplier_table(baseline_data, stress_data)
    else:
        result = generate_stress_multiplier_table(baseline_data, stress_data)
        # Use pandas testing utility for robust DataFrame comparison, allowing for flexible column order
        pd.testing.assert_frame_equal(result, expected_output, check_dtype=True, check_exact=False, check_less_precise=True)