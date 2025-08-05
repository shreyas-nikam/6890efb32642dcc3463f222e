import pytest
import yaml
from pathlib import Path

# Placeholder for your_module import
# DO NOT REPLACE or REMOVE this block
# definition_4a28351d92b449c494dcd644ce059611
from definition_4a28351d92b449c494dcd644ce059611 import save_transformation_metadata
# </your_module>


def test_save_transformation_metadata_standard(tmp_path):
    """
    Test saving a typical transformation details dictionary to a YAML file.
    Verifies the file is created and its content matches the input.
    """
    filepath = tmp_path / "test_transform_meta.yaml"
    transformation_details = {
        "column_A": {"type": "log_transform", "params": {"base": "e"}},
        "column_B": {"type": "differencing", "order": 1}
    }
    
    save_transformation_metadata(transformation_details, str(filepath))
    
    assert filepath.exists()
    with open(filepath, 'r') as f:
        loaded_data = yaml.safe_load(f)
    
    assert loaded_data == transformation_details

def test_save_transformation_metadata_empty_details(tmp_path):
    """
    Test saving an empty transformation details dictionary to a YAML file.
    Verifies the file is created and contains an empty YAML dictionary.
    """
    filepath = tmp_path / "empty_transform_meta.yaml"
    transformation_details = {}
    
    save_transformation_metadata(transformation_details, str(filepath))
    
    assert filepath.exists()
    with open(filepath, 'r') as f:
        loaded_data = yaml.safe_load(f)
    
    assert loaded_data == {}

def test_save_transformation_metadata_overwrite_existing_file(tmp_path):
    """
    Test saving transformation details to a filepath that already contains a file.
    Verifies the existing file's content is correctly overwritten.
    """
    filepath = tmp_path / "overwrite_transform_meta.yaml"
    # Create an initial file with old content
    initial_content = {"old_column": {"type": "old_transform_type"}}
    with open(filepath, 'w') as f:
        yaml.dump(initial_content, f)
    
    # New content to save
    new_transformation_details = {
        "new_column_X": {"type": "new_log_diff"},
        "new_column_Y": {"type": "scaling", "factor": 100}
    }
    
    save_transformation_metadata(new_transformation_details, str(filepath))
    
    assert filepath.exists()
    with open(filepath, 'r') as f:
        loaded_data = yaml.safe_load(f)
    
    assert loaded_data == new_transformation_details

@pytest.mark.parametrize("invalid_details", [
    "not_a_dict",
    ["list_of_items"],
    123,
    None,
    (1, 2, 3) # A tuple
])
def test_save_transformation_metadata_invalid_details_type(tmp_path, invalid_details):
    """
    Test saving with invalid types for transformation_details.
    Expects a TypeError as per the function's type hint (dict).
    """
    filepath = tmp_path / "invalid_details_type_meta.yaml"
    with pytest.raises(TypeError):
        save_transformation_metadata(invalid_details, str(filepath))

@pytest.mark.parametrize("invalid_filepath", [
    123,
    None,
    True,
    ["path", "to", "file"], # A list, not a string
    Path("/tmp/test.yaml") # A Path object, if the function strictly expects a string
])
def test_save_transformation_metadata_invalid_filepath_type(invalid_filepath):
    """
    Test saving with invalid types for filepath.
    Expects a TypeError as per the function's type hint (str).
    A dummy transformation_details dict is used.
    """
    transformation_details = {"dummy_col": {"type": "dummy_transform"}}
    with pytest.raises(TypeError):
        save_transformation_metadata(transformation_details, invalid_filepath)