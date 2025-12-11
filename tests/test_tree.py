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

def test_tree_leaf_wise_creates_limited_leaves():
    X = np.random.rand(100, 5)
    gradients = np.random.randn(100)
    hessians = np.ones(100)

    num_leaves_target = 5
    tree = DecisionTree (
        max_depth=10,
        num_leaves=num_leaves_target,
        min_data_in_leaf=1,
        main_sum_hessian_in_leaf=0.0,
        lambda_l2=0.1,
        min_gain_to_split=0.0
    )
    tree.fit(X, gradients, hessians)
    def count_leaves(node):
        if node.is_leaf():
            return 1
        return count_leaves(node.left_child) + count_leaves(node.right_child)
    actual_leaves = count_leaves(tree.root)
    assert actual_leaves <= num_leaves_target


def test_tree_predict_returns_correct_shape(): 
    X = np.random.rand(50, 3)
    gradients = np.random.randn(50)
    hessians = np.ones(50)

    tree = DecisionTree (
        max_depth=4,
        num_leaves=15,
        min_data_in_leaf=1,
        lambda_l2=0.1,
        min_gain_to_split=0.0
        min_sum_hessian_in_leaf=0.0
    )
    tree.fit(X, gradients, hessians)
    predictions = tree.predict(X)
    assert predictions.shape == (50,)
    assert np.all(np.isfinite(predictions))

def tree_can_handles_constant_features():
    X = np.array([[1, 5], [1, 6], [1, 7], [1, 8]])
    gradients = np.array([0.5, -0.3, 0.2, -0.1])
    hessians = np.ones(4)

    tree = DecisionTree (
        max_depth=3,
        num_leaves=5,
        min_data_in_leaf=1,
        lambda_l2=0.1,
        min_sum_hessian_in_leaf=0.0,
        min_gain_to_split=0.0
    )
    tree.fit(X, gradients, hessians)
    predictions = tree.predict(X)
    assert predictions.shape == (4,)