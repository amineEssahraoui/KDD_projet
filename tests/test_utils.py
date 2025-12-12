import numpy as np 
import pandas as pd 
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