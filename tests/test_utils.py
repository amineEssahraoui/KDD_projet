import numpy as np 
import pandas as pd 
import pytest
from lightgbm import LGBMRegressor
from lightgbm.utils import ValidateInputData, check_X_y, check_is_fitted, validate_sample_weight, validate_hyperparameters

def test_validate_input_accepts_numpy(): 
    X = np.array([[1, 2], [3, 4]])
    result = ValidateInputData(X)
    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 2)

def test_validate_input_accepts_pandas(): 
    X = pd.DataFrame([[1, 2], [3, 4]], columns=['column1', 'column2'])
    result = ValidateInputData(X)
    assert isinstance(result, np.ndarray)
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
    with pytest.raises(ValueError, match="do not match"):
        check_X_y(X, y)

def test_validate_sample_weight_work(): 
    weights = np.array([1.0, 0.5, 2.0])
    result = validate_sample_weight(weights, n_samples=3)
    assert np.array_equal(result, weights)

def test_validate_sample_weight_rejects_negative(): 
    weights = np.array([1.0, -0.5, 2.0])
    with pytest.raises(ValueError, match="negative"):
        validate_sample_weight(weights, n_samples=3)

def test_check_is_fitted_raises_if_not_fitted():
    model = LGBMRegressor(num_iterations = 10)
    with pytest.raises(ValueError):
        check_is_fitted(model)

def test_check_is_fitted_passes_when_fitted():
    model = LGBMRegressor(num_iterations = 10)
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([1, 0, 1])
    model.fit(X, y)
    check_is_fitted(model)
    assert True

def test_check_is_fitted_with_empty_tree():
    model = LGBMRegressor(num_iterations = 10)
    model.trees_ = []  
    with pytest.raises(ValueError):
        check_is_fitted(model)

def test_validate_hyperparameters_valid():
   
   validate_hyperparameters (
        num_iterations=100,
        learning_rate=0.1,
        max_depth=5, 
        num_leaves=31,
        min_data_in_leaf=20
   )
   assert True

def test_validate_hyperparameters_negative_num_iterations():
    with pytest.raises(ValueError):
        validate_hyperparameters(num_iterations=-10)

def test_validate_hyperparameters_zero_learning_rate():
    with pytest.raises(ValueError):
        validate_hyperparameters(learning_rate=0)

def test_validate_hyperparameters_negative_learning_rate():
    with pytest.raises(ValueError):
        validate_hyperparameters(learning_rate=-0.1)

def test_validate_hyperparameters_invalid_max_depth():
    with pytest.raises(ValueError):
        validate_hyperparameters(max_depth=0)

def test_validate_hyperparameters_accepts_max_depth_minus_one():
    validate_hyperparameters(max_depth=-1)
    assert True

def test_validate_hyperparameters_low_num_leaves():
    with pytest.raises(ValueError):
        validate_hyperparameters(num_leaves=1)

def test_validate_hyperparameters_zero_min_data_in_leaf():
    with pytest.raises(ValueError):
        validate_hyperparameters(min_data_in_leaf=0)