import numpy as np
from lightgbm import DecisionTree

def test_tree_can_fit_sample_data():
    X = np.array([[1,2] , [3,4], [5,6], [7,8]])
    gradients = np.array([0.5, -0.3, 0.2, -0.1])
    hessians = np.ones(4)

    tree = DecisionTree (
        max_depth=3,
        num_leaves=10,
        min_data_in_leaf=1,
        lambda_l2=0.1,
        min_gain_to_split=0.0
    )
    tree.fit(X, gradients, hessians)
    assert tree.root_node is not None
    assert tree.n_features_ == 2