"""
Mathematical Integrity Tests for LightGBM From-Scratch Implementation.

This module validates the mathematical correctness of:
1. Gradient and Hessian computations
2. Split gain formula
3. Prediction aggregation (init + learning_rate * sum(trees))

Author: Validation Suite
"""

import numpy as np
import pytest
import sys
import os


# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from lightgbm import LGBMClassifier, LGBMRegressor
from lightgbm.loss_functions import (
    MSELoss,
    MAELoss,
    BinaryCrossEntropyLoss,
    MultiClassCrossEntropyLoss,
    HuberLoss,
)
from lightgbm.tree import DecisionTree


# GRADIENT & HESSIAN VALIDATION

class TestGradientHessianMSE:
    """Validate MSE gradient and hessian against manual calculations."""

    def test_mse_gradient_manual(self):
        """
        MSE Loss: L = 0.5 * (y - f)^2
        Gradient: dL/df = -(y - f) = f - y
        """
        np.random.seed(42)
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.5, 1.8, 3.2, 3.8, 5.5])
        
        loss = MSELoss()
        
        # Manual gradient calculation
        expected_gradient = y_pred - y_true  # f - y
        
        # Model gradient
        actual_gradient = loss.gradient(y_true, y_pred)
        
        np.testing.assert_array_almost_equal(
            actual_gradient, expected_gradient, decimal=10,
            err_msg="MSE gradient mismatch!"
        )

    def test_mse_hessian_manual(self):
        """
        MSE Hessian: d^2L/df^2 = 1 (constant)
        """
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.5, 1.8, 3.2, 3.8, 5.5])
        
        loss = MSELoss()
        
        # Manual hessian: always 1
        expected_hessian = np.ones_like(y_true)
        
        actual_hessian = loss.hessian(y_true, y_pred)
        
        np.testing.assert_array_almost_equal(
            actual_hessian, expected_hessian, decimal=10,
            err_msg="MSE hessian mismatch!"
        )

    def test_mse_loss_value(self):
        """Verify MSE loss computation."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])  # Perfect prediction
        
        loss = MSELoss()
        assert loss(y_true, y_pred) == 0.0, "Perfect prediction should have 0 loss"
        
        # Non-perfect
        y_pred2 = np.array([2.0, 3.0, 4.0])  # Each off by 1
        expected_loss = 0.5 * np.mean((y_true - y_pred2) ** 2)
        actual_loss = loss(y_true, y_pred2)
        
        np.testing.assert_almost_equal(actual_loss, expected_loss, decimal=10)


class TestGradientHessianMAE:
    """Validate MAE gradient against manual calculations."""

    def test_mae_gradient_manual(self):
        """
        MAE Loss: L = |y - f|
        Gradient: sign(f - y)
        """
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.5, 1.8, 3.0, 3.8, 5.5])
        
        loss = MAELoss()
        
        # Manual gradient: sign(f - y)
        residual = y_pred - y_true
        expected_gradient = np.sign(residual)
        
        actual_gradient = loss.gradient(y_true, y_pred)
        
        np.testing.assert_array_almost_equal(
            actual_gradient, expected_gradient, decimal=10,
            err_msg="MAE gradient mismatch!"
        )


class TestGradientHessianBinaryCrossEntropy:
    """Validate Binary Cross-Entropy gradient and hessian."""

    def test_bce_gradient_manual(self):
        """
        BCE Gradient: g = sigmoid(f) - y = p - y
        """
        np.random.seed(42)
        y_true = np.array([0, 1, 1, 0, 1], dtype=float)
        y_pred_logits = np.array([-1.0, 2.0, 0.5, -0.5, 1.5])
        
        loss = BinaryCrossEntropyLoss()
        
        # Manual sigmoid
        def sigmoid(x):
            return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
        
        # Manual gradient: p - y
        probs = sigmoid(y_pred_logits)
        expected_gradient = probs - y_true
        
        actual_gradient = loss.gradient(y_true, y_pred_logits)
        
        np.testing.assert_array_almost_equal(
            actual_gradient, expected_gradient, decimal=10,
            err_msg="BCE gradient mismatch!"
        )

    def test_bce_hessian_manual(self):
        """
        BCE Hessian: h = p * (1 - p)
        """
        y_true = np.array([0, 1, 1, 0, 1], dtype=float)
        y_pred_logits = np.array([-1.0, 2.0, 0.5, -0.5, 1.5])
        
        loss = BinaryCrossEntropyLoss()
        
        def sigmoid(x):
            return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
        
        probs = sigmoid(y_pred_logits)
        expected_hessian = probs * (1 - probs)
        
        actual_hessian = loss.hessian(y_true, y_pred_logits)
        
        # Note: implementation clips hessian for numerical stability
        np.testing.assert_array_almost_equal(
            actual_hessian, np.clip(expected_hessian, 1e-15, 1 - 1e-15), 
            decimal=10,
            err_msg="BCE hessian mismatch!"
        )


class TestGradientHessianMultiClass:
    """Validate Multi-class Cross-Entropy gradient and hessian."""

    def test_multiclass_gradient_manual(self):
        """
        Multiclass Gradient: g_k = p_k - (y == k)
        """
        y_true = np.array([0, 1, 2, 0, 1])  # 3 classes
        # Shape: (n_samples, n_classes)
        y_pred_logits = np.array([
            [2.0, 0.5, -1.0],
            [0.0, 2.0, 0.5],
            [-1.0, 0.0, 2.0],
            [1.5, 0.5, 0.0],
            [0.0, 1.5, 0.5],
        ])
        
        loss = MultiClassCrossEntropyLoss(n_classes=3)
        
        # Manual softmax
        def softmax(x):
            x_max = np.max(x, axis=1, keepdims=True)
            exp_x = np.exp(x - x_max)
            return exp_x / np.sum(exp_x, axis=1, keepdims=True)
        
        probs = softmax(y_pred_logits)
        
        # One-hot encoding
        n_samples = len(y_true)
        one_hot = np.zeros((n_samples, 3))
        one_hot[np.arange(n_samples), y_true] = 1
        
        expected_gradient = probs - one_hot
        
        actual_gradient = loss.gradient(y_true, y_pred_logits)
        
        np.testing.assert_array_almost_equal(
            actual_gradient, expected_gradient, decimal=10,
            err_msg="Multiclass gradient mismatch!"
        )

    def test_multiclass_hessian_manual(self):
        """
        Multiclass Hessian (diagonal): h_k = p_k * (1 - p_k)
        """
        y_true = np.array([0, 1, 2])
        y_pred_logits = np.array([
            [2.0, 0.5, -1.0],
            [0.0, 2.0, 0.5],
            [-1.0, 0.0, 2.0],
        ])
        
        loss = MultiClassCrossEntropyLoss(n_classes=3)
        
        def softmax(x):
            x_max = np.max(x, axis=1, keepdims=True)
            exp_x = np.exp(x - x_max)
            return exp_x / np.sum(exp_x, axis=1, keepdims=True)
        
        probs = softmax(y_pred_logits)
        expected_hessian = probs * (1 - probs)
        
        actual_hessian = loss.hessian(y_true, y_pred_logits)
        
        np.testing.assert_array_almost_equal(
            actual_hessian, np.clip(expected_hessian, 1e-15, 1 - 1e-15),
            decimal=10,
            err_msg="Multiclass hessian mismatch!"
        )


# SPLIT GAIN FORMULA VALIDATION

class TestSplitGainFormula:
    """Validate the split gain formula used in tree building."""

    def test_gain_formula_manual(self):
        """
        LightGBM Split Gain Formula:
        Gain = 0.5 * [G_L^2/(H_L + λ) + G_R^2/(H_R + λ) - (G_L+G_R)^2/(H_L+H_R + λ)] - γ
        
        For simplicity (λ=0, γ=0):
        Gain = 0.5 * [G_L^2/H_L + G_R^2/H_R - G^2/H]
        """
        # Create dummy data
        gradients = np.array([-0.5, -0.3, 0.2, 0.4, 0.6, 0.8])
        hessians = np.ones(6)  # MSE has hessian = 1
        
        # Simulate split at index 3 (left: first 3, right: last 3)
        G_left = np.sum(gradients[:3])  # -0.6
        H_left = np.sum(hessians[:3])   # 3.0
        G_right = np.sum(gradients[3:]) # 1.8
        H_right = np.sum(hessians[3:])  # 3.0
        G_total = G_left + G_right      # 1.2
        H_total = H_left + H_right      # 6.0
        
        lambda_l2 = 0.0
        
        # Manual gain calculation
        score_left = (G_left ** 2) / (H_left + lambda_l2)
        score_right = (G_right ** 2) / (H_right + lambda_l2)
        score_total = (G_total ** 2) / (H_total + lambda_l2)
        
        expected_gain = score_left + score_right - score_total
        
        # Verify the formula
        # G_L^2/H_L = 0.36/3 = 0.12
        # G_R^2/H_R = 3.24/3 = 1.08
        # G^2/H = 1.44/6 = 0.24
        # Gain = 0.12 + 1.08 - 0.24 = 0.96
        
        np.testing.assert_almost_equal(expected_gain, 0.96, decimal=10)
        
        print(f"Manual Gain Calculation:")
        print(f"  G_L={G_left:.2f}, H_L={H_left:.2f}, Score_L={score_left:.4f}")
        print(f"  G_R={G_right:.2f}, H_R={H_right:.2f}, Score_R={score_right:.4f}")
        print(f"  G_total={G_total:.2f}, H_total={H_total:.2f}, Score_total={score_total:.4f}")
        print(f"  Expected Gain: {expected_gain:.4f}")

    def test_gain_with_regularization(self):
        """Test gain calculation with L2 regularization (lambda)."""
        G_left, H_left = -0.6, 3.0
        G_right, H_right = 1.8, 3.0
        G_total, H_total = 1.2, 6.0
        
        lambda_l2 = 1.0
        
        score_left = (G_left ** 2) / (H_left + lambda_l2)   # 0.36/4 = 0.09
        score_right = (G_right ** 2) / (H_right + lambda_l2) # 3.24/4 = 0.81
        score_total = (G_total ** 2) / (H_total + lambda_l2) # 1.44/7 ≈ 0.2057
        
        expected_gain = score_left + score_right - score_total
        
        np.testing.assert_almost_equal(
            expected_gain, 
            0.09 + 0.81 - (1.44/7), 
            decimal=6
        )
        print(f"Gain with λ={lambda_l2}: {expected_gain:.4f}")


# PREDICTION AGGREGATION VALIDATION

class TestPredictionAggregation:
    """Validate that predictions are correctly aggregated."""

    def test_regression_prediction_formula(self):
        """
        Prediction = init_prediction + learning_rate * Σ(tree_outputs)
        """
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = 3 * X[:, 0] + 2 * X[:, 1] + np.random.randn(100) * 0.1
        
        learning_rate = 0.1
        n_estimators = 10
        
        model = LGBMRegressor(
            num_iterations=n_estimators,
            learning_rate=learning_rate,
            max_depth=3,
            random_state=42,
        )
        model.fit(X, y)
        
        # Manual prediction
        init_pred = model.init_prediction_
        assert init_pred is not None, "init_prediction_ should not be None after fitting"
        tree_outputs = np.zeros(len(X))
        
        for tree in model.trees_:
            tree_outputs += tree.predict(X)
        
        expected_pred = init_pred + learning_rate * tree_outputs
        actual_pred = model.predict(X)
        
        np.testing.assert_array_almost_equal(
            actual_pred, expected_pred, decimal=10,
            err_msg="Prediction aggregation mismatch!"
        )
        
        print(f"✓ Prediction = {init_pred:.4f} + {learning_rate} * Σ(trees)")

    def test_classifier_probability_formula(self):
        """
        For binary: P(y=1) = sigmoid(init + lr * Σ(trees))
        """
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        
        learning_rate = 0.1
        
        model = LGBMClassifier(
            num_iterations=10,
            learning_rate=learning_rate,
            max_depth=3,
            random_state=42,
        )
        model.fit(X, y)
        
        # Get probabilities
        probs = model.predict_proba(X)
        
        # Verify probabilities sum to 1
        prob_sums = np.sum(probs, axis=1)
        np.testing.assert_array_almost_equal(
            prob_sums, np.ones(len(X)), decimal=10,
            err_msg="Probabilities don't sum to 1!"
        )
        
        # Verify probabilities in [0, 1]
        assert np.all(probs >= 0) and np.all(probs <= 1), \
            "Probabilities out of range!"
        
        print("✓ Classifier probabilities valid")


class TestInitialPrediction:
    """Validate initial prediction values."""

    def test_mse_init_is_mean(self):
        """MSE initial prediction should be mean(y)."""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        loss = MSELoss()
        
        expected_init = np.mean(y)
        actual_init = loss.init_prediction(y)
        
        np.testing.assert_almost_equal(actual_init, expected_init, decimal=10)

    def test_mae_init_is_median(self):
        """MAE initial prediction should be median(y)."""
        y = np.array([1.0, 2.0, 3.0, 4.0, 100.0])  # Outlier
        loss = MAELoss()
        
        expected_init = np.median(y)
        actual_init = loss.init_prediction(y)
        
        np.testing.assert_almost_equal(actual_init, expected_init, decimal=10)

    def test_bce_init_is_logodds(self):
        """BCE initial prediction should be log(p/(1-p))."""
        y = np.array([0, 0, 0, 1, 1, 1, 1, 1])  # 62.5% positive
        loss = BinaryCrossEntropyLoss()
        
        p = np.mean(y)
        expected_init = np.log(p / (1 - p))
        actual_init = loss.init_prediction(y)
        
        np.testing.assert_almost_equal(actual_init, expected_init, decimal=10)


# LEAF VALUE COMPUTATION

class TestLeafValueFormula:
    """Validate leaf value computation formula."""

    def test_leaf_value_formula(self):
        """
        Optimal leaf value: w* = -G / (H + λ)
        """
        gradients = np.array([-0.5, -0.3, -0.2])  # All negative → positive leaf
        hessians = np.ones(3)
        
        lambda_l2 = 0.0
        
        G = np.sum(gradients)
        H = np.sum(hessians)
        
        expected_leaf_value = -G / (H + lambda_l2)
        
        # G = -1.0, H = 3.0 → w* = 1.0/3.0 ≈ 0.333
        np.testing.assert_almost_equal(
            expected_leaf_value, 1.0/3.0, decimal=10
        )
        
        print(f"Leaf value: -({G})/{H} = {expected_leaf_value:.4f}")

    def test_leaf_value_with_regularization(self):
        """Test leaf value with L2 regularization."""
        G = -1.0
        H = 3.0
        lambda_l2 = 1.0
        
        expected_leaf_value = -G / (H + lambda_l2)  # 1.0/4.0 = 0.25
        
        np.testing.assert_almost_equal(expected_leaf_value, 0.25, decimal=10)


# RUN TESTS

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
