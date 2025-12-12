import numpy as np 
import pandas as pd 
from lightgbm.utils import ValidateInputData, check_X_y, check_is_fitted, validate_sample_weight

def test_validate_input_accepts_numpy(): 
    