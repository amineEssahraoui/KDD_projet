"""
Test suite for DecisionTree implementation.
"""

import numpy as np
import pytest

import sys
from pathlib import Path
_repo_root = Path(__file__).resolve().parents[1]
src_path = str(_repo_root / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from lightgbm import DecisionTree


def test_tree_can_fit_sample_data():
    """Test that tree can fit simple data."""
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    gradients = np.array([0.5, -0.3, 0.2, -0.1])
    hessians = np.ones(4)

    tree = DecisionTree(
        max_depth=3,
        num_leaves=10,
        min_samples_leaf=1,
        lambda_l2=0.1,
        min_gain_to_split=0.0,
    )
    tree.fit(X, gradients, hessians)
    assert tree.root_ is not None
    assert tree.n_features_ == 2


def test_tree_leaf_wise_creates_limited_leaves():
    """Test that num_leaves parameter is respected."""
    np.random.seed(42)
    X = np.random.rand(100, 5)
    gradients = np.random.randn(100)
    hessians = np.ones(100)

    num_leaves_target = 5
    tree = DecisionTree(
        max_depth=10,
        num_leaves=num_leaves_target,
        min_samples_leaf=1,
        lambda_l2=0.1,
        min_gain_to_split=0.0,
    )
    tree.fit(X, gradients, hessians)

    def count_leaves(node):
        if node.is_leaf:
            return 1
        return count_leaves(node.left) + count_leaves(node.right)

    actual_leaves = count_leaves(tree.root_)
    assert actual_leaves <= num_leaves_target


def test_tree_predict_returns_correct_shape():
    """Test that predict returns correct shape."""
    np.random.seed(42)
    X = np.random.rand(50, 3)
    gradients = np.random.randn(50)
    hessians = np.ones(50)

    tree = DecisionTree(
        max_depth=4,
        num_leaves=15,
        min_samples_leaf=1,
        lambda_l2=0.1,
        min_gain_to_split=0.0,
    )
    tree.fit(X, gradients, hessians)
    predictions = tree.predict(X)
    assert predictions.shape == (50,)
    assert np.all(np.isfinite(predictions))


def test_tree_handles_constant_features():
    """Test that tree handles constant features."""
    X = np.array([[1, 5], [1, 6], [1, 7], [1, 8]])
    gradients = np.array([0.5, -0.3, 0.2, -0.1])
    hessians = np.ones(4)

    tree = DecisionTree(
        max_depth=3,
        num_leaves=5,
        min_samples_leaf=1,
        lambda_l2=0.1,
        min_gain_to_split=0.0,
    )
    tree.fit(X, gradients, hessians)
    predictions = tree.predict(X)
    assert predictions.shape == (4,)


def test_tree_with_histogram_mode():
    """Test histogram-based split finding."""
    np.random.seed(42)
    X = np.random.rand(100, 4)
    gradients = np.random.randn(100)
    hessians = np.ones(100)

    tree = DecisionTree(
        max_depth=5,
        num_leaves=20,
        min_samples_leaf=1,
        lambda_l2=1.0,
        min_gain_to_split=0.0,
        use_histogram=True,
    )
    tree.fit(X, gradients, hessians)
    predictions = tree.predict(X)
    assert predictions.shape == (100,)


def test_tree_respects_min_gain_to_split():
    """Test that min_gain_to_split prevents splits with low gain."""
    np.random.seed(42)
    X = np.random.rand(20, 2)
    gradients = np.random.randn(20) * 0.01
    hessians = np.ones(20)

    tree = DecisionTree(
        max_depth=4,
        num_leaves=10,
        min_samples_leaf=1,
        lambda_l2=0.1,
        min_gain_to_split=1.0,  # High threshold
    )
    tree.fit(X, gradients, hessians)

    def count_leaves(node):
        if node.is_leaf:
            return 1
        return count_leaves(node.left) + count_leaves(node.right)

    leaves_count = count_leaves(tree.root_)
    # With high min_gain_to_split, should have few leaves
    assert leaves_count < 5


def test_tree_reproducibility():
    """Test that same random_state gives same results."""
    np.random.seed(42)
    X = np.random.rand(50, 3)
    gradients = np.random.randn(50)
    hessians = np.ones(50)

    tree1 = DecisionTree(
        max_depth=4,
        num_leaves=10,
        min_samples_leaf=1,
        feature_fraction=0.8,
        random_state=42,
    )
    tree1.fit(X, gradients, hessians)
    preds1 = tree1.predict(X)

    tree2 = DecisionTree(
        max_depth=4,
        num_leaves=10,
        min_samples_leaf=1,
        feature_fraction=0.8,
        random_state=42,
    )
    tree2.fit(X, gradients, hessians)
    preds2 = tree2.predict(X)

    assert np.allclose(preds1, preds2)


def test_tree_feature_fraction():
    """Test feature_fraction parameter."""
    np.random.seed(42)
    X = np.random.rand(50, 10)
    gradients = np.random.randn(50)
    hessians = np.ones(50)

    tree = DecisionTree(
        max_depth=4,
        num_leaves=10,
        min_samples_leaf=1,
        feature_fraction=0.5,  # Only use half of features
        random_state=42,
    )
    tree.fit(X, gradients, hessians)
    predictions = tree.predict(X)
    assert predictions.shape == (50,)
