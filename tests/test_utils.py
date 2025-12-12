import numpy as np 
import pandas as pd 
import pytest
from lightgbm.utils import ValidateInputData, check_X_y, check_is_fitted, validate_sample_weight

def test_validate_input_accepts_numpy(): 
    X = np.array([[1, 2], [3, 4]])
    result = ValidateInputData(X)
    assert isinstance(result.data, np.ndarray)
    assert result.shape == (2, 2)

def test_validate_input_accepts_pandas(): 
    X = pd.DataFrame([[1, 2], [3, 4]], columns=['column1', 'column2'])
    result = ValidateInputData(X)
    assert isinstance(result.data, pd.DataFrame)
    assert result.shape == (2, 2)

def test_validate_input_rejects_infinite_values(): 
    X = np.array([[1, 2], [np.inf, 4]])
    with pytest.raises(ValueError, match="infinite"):
        ValidateInputData(X)

def test_check_X_y_compatible_shapes():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([1, 0, 1])
    X_checked, y_checked = check_X_y(X, y)
    assert X_checked.shape == (3, 2)
    assert y_checked.shape == (3,)

def test_check_X_y_incompatible_shapes():
    X = np.array([[1, 2], [3, 4]])
    y = np.array([1, 0, 1])
    with pytest.raises(ValueError, match="do not test_check_X_y_compatible_shapes match"):
        check_X_y(X, y)

def test_validate_sample_weight_work(): 
    weights = np.array([1.0, 0.5, 2.0])
    result = validate_sample_weight(weights, expected_length=3)
    assert np.array_equal(result, weights)