import pytest
import joblib
import os
from pathlib import Path
import pickle

# --- your_module block start ---
from definition_a724cce52a1740ecbdd2dc0425257338 import load_model
# --- your_module block end ---

# Test Case 1: Successful model loading from a valid .pkl file
def test_load_model_success(tmp_path):
    # Create a dummy model object
    class DummyModel:
        def __init__(self, value):
            self.value = value
        def get_value(self):
            return self.value

    dummy_model_instance = DummyModel("Test Data")
    
    # Define a temporary file path for the model
    model_filepath = tmp_path / "test_model.pkl"
    
    # Save the dummy model to the temporary file using joblib
    joblib.dump(dummy_model_instance, model_filepath)
    
    # Call the load_model function with the path to the temporary file
    loaded_model = load_model(str(model_filepath))
    
    # Assert that the loaded object is an instance of DummyModel and retains its data
    assert isinstance(loaded_model, DummyModel)
    assert loaded_model.get_value() == "Test Data"

# Test Case 2: File not found error when the specified path does not exist
def test_load_model_file_not_found(tmp_path):
    # Create a path that definitely does not exist
    non_existent_path = tmp_path / "non_existent_model.pkl"
    
    # Expect FileNotFoundError to be raised when trying to load from a non-existent path
    with pytest.raises(FileNotFoundError):
        load_model(str(non_existent_path))

# Test Case 3: Error when the file exists but is not a valid joblib/pickle file
def test_load_model_invalid_pickle_content(tmp_path):
    corrupted_filepath = tmp_path / "corrupted_model.pkl"
    
    # Write some arbitrary, non-pickle content to the file
    with open(corrupted_filepath, "w") as f:
        f.write("This is definitely not a valid pickle file.")
    
    # Expect an unpickling error (EOFError or _pickle.UnpicklingError) from joblib/pickle
    # EOFError often occurs if the file is empty or too short/malformed to be a valid pickle stream.
    # _pickle.UnpicklingError occurs if the content is not a valid pickle stream.
    with pytest.raises((EOFError, pickle.UnpicklingError)):
        load_model(str(corrupted_filepath))

# Test Case 4: TypeError when filepath is not a string
@pytest.mark.parametrize("invalid_filepath_type", [
    None,
    12345,
    True,
    ['path/to/model.pkl'],
    b'path/to/model.pkl'
])
def test_load_model_invalid_filepath_type(invalid_filepath_type):
    # Expect TypeError to be raised if the filepath argument is not a string
    with pytest.raises(TypeError):
        load_model(invalid_filepath_type)

# Test Case 5: FileNotFoundError when filepath is an empty string
def test_load_model_empty_filepath_string():
    # An empty string as a filepath will typically result in a FileNotFoundError
    # because `open("")` (used internally by joblib) raises this error.
    with pytest.raises(FileNotFoundError):
        load_model("")