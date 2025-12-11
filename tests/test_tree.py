import numpy as np
from lightgbm import DecisionTree

def test_tree_can_fit_sample_data():
    X = np.array([[1,2] , [3,4], [5,6], [7,8]])
    gradients = np.array([0.5, -0.3, 0.2, -0.1])
    hessians = np.ones(4)