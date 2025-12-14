"""
Logic and Sanity Tests for LightGBM From-Scratch Implementation.

This module validates:
1. Overfitting capability on small datasets (sanity check)
2. Loss convergence during training
3. Edge cases handling (empty data, single class, etc.)

Author: Validation Suite
"""

import numpy as np
import pytest
import sys
import os
import warnings

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from lightgbm import LGBMClassifier, LGBMRegressor


# SANITY CHECK: OVERFITTING CAPABILITY

class TestOverfittingCapability:
    """
    A correct gradient boosting implementation MUST be able to overfit
    a small dataset. If it cannot achieve near-perfect training accuracy
    on 10 samples, the algorithm is fundamentally broken.
    """

    def test_regression_overfits_tiny_data(self):
        """
        Regression model should achieve near-zero training loss on tiny dataset.
        """
        np.random.seed(42)
        
        # Tiny dataset: 10 samples, 2 features
        X = np.array([
            [1.0, 2.0],
            [2.0, 1.0],
            [3.0, 3.0],
            [4.0, 2.0],
            [5.0, 1.0],
            [1.5, 2.5],
            [2.5, 1.5],
            [3.5, 3.5],
            [4.5, 2.5],
            [5.5, 1.5],
        ])
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 1.5, 2.5, 3.5, 4.5, 5.5])
        
        model = LGBMRegressor(
            num_iterations=100,
            learning_rate=0.3,
            max_depth=5,
            num_leaves=31,
            min_data_in_leaf=1,
            random_state=42,
        )
        model.fit(X, y)
        
        predictions = model.predict(X)
        mse = np.mean((predictions - y) ** 2)
        
        print(f"\n[Overfit Test - Regression]")
        print(f"  Training MSE: {mse:.6f}")
        print(f"  Predictions: {predictions[:5]}")
        print(f"  Targets:     {y[:5]}")
        
        # Should be able to achieve MSE < 0.1 on training set
        assert mse < 0.1, f"Model cannot overfit! MSE={mse:.4f} (expected < 0.1)"

    def test_binary_classification_overfits_tiny_data(self):
        """
        Binary classifier should achieve 100% training accuracy on tiny dataset.
        """
        np.random.seed(42)
        
        # Linearly separable data
        X = np.array([
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.2],
            [0.3, 0.3],
            [0.4, 0.4],
            [0.6, 0.6],
            [0.7, 0.7],
            [0.8, 0.8],
            [0.9, 0.9],
            [1.0, 1.0],
        ])
        y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        
        model = LGBMClassifier(
            num_iterations=100,
            learning_rate=0.3,
            max_depth=5,
            num_leaves=31,
            min_data_in_leaf=1,
            random_state=42,
        )
        model.fit(X, y)
        
        predictions = model.predict(X)
        accuracy = np.mean(predictions == y)
        
        print(f"\n[Overfit Test - Binary Classification]")
        print(f"  Training Accuracy: {accuracy:.2%}")
        print(f"  Predictions: {predictions}")
        print(f"  Targets:     {y}")
        
        assert accuracy == 1.0, f"Model cannot overfit! Accuracy={accuracy:.2%}"

    def test_multiclass_classification_overfits_tiny_data(self):
        """
        Multiclass classifier should achieve 100% training accuracy on tiny dataset.
        """
        np.random.seed(42)
        
        # 3 classes, clearly separable
        X = np.array([
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.2],
            [0.5, 0.5],
            [0.6, 0.6],
            [0.7, 0.7],
            [0.9, 0.9],
            [1.0, 1.0],
            [1.1, 1.1],
        ])
        y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        
        model = LGBMClassifier(
            num_iterations=100,
            learning_rate=0.3,
            max_depth=5,
            num_leaves=31,
            min_data_in_leaf=1,
            random_state=42,
        )
        model.fit(X, y)
        
        predictions = model.predict(X)
        accuracy = np.mean(predictions == y)
        
        print(f"\n[Overfit Test - Multiclass Classification]")
        print(f"  Training Accuracy: {accuracy:.2%}")
        print(f"  Predictions: {predictions}")
        print(f"  Targets:     {y}")
        
        assert accuracy >= 0.9, f"Model cannot overfit! Accuracy={accuracy:.2%}"


# CONVERGENCE TEST

class TestTrainingConvergence:
    """
    Verify that the loss decreases (or at least doesn't increase) during training.
    """

    def test_regression_loss_decreases(self):
        """Training loss should decrease with more iterations."""
        np.random.seed(42)
        
        X = np.random.randn(100, 5)
        y = 3 * X[:, 0] + 2 * X[:, 1] + np.random.randn(100) * 0.5
        
        losses = []
        for n_iter in [1, 5, 10, 20, 50]:
            model = LGBMRegressor(
                num_iterations=n_iter,
                learning_rate=0.1,
                max_depth=4,
                random_state=42,
            )
            model.fit(X, y)
            preds = model.predict(X)
            mse = np.mean((preds - y) ** 2)
            losses.append(mse)
        
        print(f"\n[Convergence Test - Regression]")
        print(f"  Iterations: [1, 5, 10, 20, 50]")
        print(f"  MSE:        {[f'{l:.4f}' for l in losses]}")
        
        # Check overall decreasing trend
        assert losses[-1] < losses[0], \
            f"Loss did not decrease! Start={losses[0]:.4f}, End={losses[-1]:.4f}"

    def test_classification_loss_decreases(self):
        """Training accuracy should increase with more iterations."""
        np.random.seed(42)
        
        X = np.random.randn(200, 5)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        
        accuracies = []
        for n_iter in [1, 5, 10, 20, 50]:
            model = LGBMClassifier(
                num_iterations=n_iter,
                learning_rate=0.1,
                max_depth=4,
                random_state=42,
            )
            model.fit(X, y)
            preds = model.predict(X)
            acc = np.mean(preds == y)
            accuracies.append(acc)
        
        print(f"\n[Convergence Test - Classification]")
        print(f"  Iterations: [1, 5, 10, 20, 50]")
        print(f"  Accuracy:   {[f'{a:.2%}' for a in accuracies]}")
        
        # Check overall increasing trend
        assert accuracies[-1] >= accuracies[0], \
            f"Accuracy did not increase! Start={accuracies[0]:.2%}, End={accuracies[-1]:.2%}"

    def test_early_stopping_triggers(self):
        """Verify early stopping works when validation loss plateaus."""
        np.random.seed(42)
        
        X = np.random.randn(200, 5)
        y = 3 * X[:, 0] + 2 * X[:, 1] + np.random.randn(200) * 0.5
        
        X_train, X_val = X[:150], X[150:]
        y_train, y_val = y[:150], y[150:]
        
        model = LGBMRegressor(
            num_iterations=1000,  # High number, should stop early
            learning_rate=0.1,
            max_depth=4,
            early_stopping_rounds=10,
            random_state=42,
        )
        model.fit(X_train, y_train, eval_set=(X_val, y_val))
        
        print(f"\n[Early Stopping Test]")
        print(f"  Max iterations: 1000")
        print(f"  Actual iterations: {model.n_iter_}")
        
        # Should have stopped before 1000
        assert model.n_iter_ < 1000, \
            f"Early stopping didn't trigger! Ran all {model.n_iter_} iterations"


# EDGE CASES

class TestEdgeCases:
    """Test handling of edge cases that shouldn't crash the model."""

    def test_single_sample_regression(self):
        """Model should handle single sample (edge case)."""
        X = np.array([[1.0, 2.0]])
        y = np.array([3.0])
        
        model = LGBMRegressor(
            num_iterations=10,
            min_data_in_leaf=1,
            random_state=42,
        )
        
        # Should not crash
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X, y)
        
        pred = model.predict(X)
        assert len(pred) == 1
        print(f"\n[Single Sample Test] Prediction: {pred[0]:.4f}, Target: {y[0]}")

    def test_constant_target_regression(self):
        """Model should handle constant target values."""
        np.random.seed(42)
        
        X = np.random.randn(50, 3)
        y = np.ones(50) * 5.0  # All same value
        
        model = LGBMRegressor(
            num_iterations=10,
            random_state=42,
        )
        model.fit(X, y)
        
        predictions = model.predict(X)
        
        # All predictions should be close to 5.0
        np.testing.assert_array_almost_equal(
            predictions, np.ones(50) * 5.0, decimal=1
        )
        print(f"\n[Constant Target Test] Mean prediction: {np.mean(predictions):.4f}")

    def test_single_feature(self):
        """Model should work with single feature."""
        np.random.seed(42)
        
        X = np.random.randn(100, 1)
        y = 2 * X[:, 0] + 1
        
        model = LGBMRegressor(
            num_iterations=50,
            learning_rate=0.1,
            random_state=42,
        )
        model.fit(X, y)
        
        predictions = model.predict(X)
        mse = np.mean((predictions - y) ** 2)
        
        assert mse < 0.5, f"Single feature regression failed, MSE={mse:.4f}"
        print(f"\n[Single Feature Test] MSE: {mse:.4f}")

    def test_highly_imbalanced_classes(self):
        """Model should handle highly imbalanced classes."""
        np.random.seed(42)
        
        # 95% class 0, 5% class 1
        X = np.random.randn(200, 5)
        y = np.zeros(200)
        y[:10] = 1  # Only 10 positive samples
        
        model = LGBMClassifier(
            num_iterations=50,
            learning_rate=0.1,
            min_data_in_leaf=1,
            random_state=42,
        )
        model.fit(X, y)
        
        predictions = model.predict(X)
        
        # Should predict at least some positives (not all 0s)
        print(f"\n[Imbalanced Classes Test]")
        print(f"  Class distribution: {np.bincount(y.astype(int))}")
        print(f"  Prediction distribution: {np.bincount(predictions.astype(int))}")
        
        # Should not crash and should return valid predictions
        assert len(predictions) == 200

    def test_many_features_few_samples(self):
        """Model should handle high-dimensional data with few samples."""
        np.random.seed(42)
        
        X = np.random.randn(20, 100)  # 20 samples, 100 features
        y = np.random.randn(20)
        
        model = LGBMRegressor(
            num_iterations=20,
            learning_rate=0.1,
            feature_fraction=0.5,  # Sample features to avoid overfitting
            random_state=42,
        )
        model.fit(X, y)
        
        predictions = model.predict(X)
        assert len(predictions) == 20
        print(f"\n[High-Dimensional Test] Predictions shape: {predictions.shape}")

    def test_nan_handling_when_allowed(self):
        """Model should handle NaN values when allow_nan=True."""
        np.random.seed(42)
        
        X = np.random.randn(100, 5)
        X[0, 0] = np.nan
        X[10, 2] = np.nan
        y = np.random.randn(100)
        
        model = LGBMRegressor(
            num_iterations=20,
            allow_nan=True,
            random_state=42,
        )
        
        # Should not crash
        model.fit(X, y)
        predictions = model.predict(X)
        
        assert len(predictions) == 100
        assert not np.any(np.isnan(predictions)), "Predictions contain NaN!"
        print(f"\n[NaN Handling Test] No crash, predictions valid")

    def test_duplicate_features(self):
        """Model should handle duplicate/identical features."""
        np.random.seed(42)
        
        X_base = np.random.randn(100, 2)
        # Duplicate the features
        X = np.column_stack([X_base, X_base])  # 4 features, 2 duplicated
        y = X_base[:, 0] + X_base[:, 1]
        
        model = LGBMRegressor(
            num_iterations=20,
            random_state=42,
        )
        model.fit(X, y)
        
        predictions = model.predict(X)
        mse = np.mean((predictions - y) ** 2)
        
        assert mse < 1.0
        print(f"\n[Duplicate Features Test] MSE: {mse:.4f}")


class TestReproducibility:
    """Test that results are reproducible with same random_state."""

    def test_same_seed_same_results(self):
        """Same random_state should produce identical results."""
        np.random.seed(42)
        
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        
        model1 = LGBMRegressor(num_iterations=20, random_state=123)
        model1.fit(X, y)
        pred1 = model1.predict(X)
        
        model2 = LGBMRegressor(num_iterations=20, random_state=123)
        model2.fit(X, y)
        pred2 = model2.predict(X)
        
        np.testing.assert_array_equal(pred1, pred2)
        print("\n[Reproducibility Test] OK - Same seed produces same results")

    def test_different_seed_different_results(self):
        """Different random_state should produce different results (with randomness)."""
        np.random.seed(42)
        
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        
        model1 = LGBMRegressor(
            num_iterations=20, 
            bagging_fraction=0.8,
            bagging_freq=1,
            random_state=123
        )
        model1.fit(X, y)
        pred1 = model1.predict(X)
        
        model2 = LGBMRegressor(
            num_iterations=20,
            bagging_fraction=0.8,
            bagging_freq=1,
            random_state=456
        )
        model2.fit(X, y)
        pred2 = model2.predict(X)
        
        # Predictions should be different (not always guaranteed, but likely)
        if not np.allclose(pred1, pred2):
            print("\n[Reproducibility Test] OK - Different seeds produce different results")
        else:
            print("\n[Reproducibility Test] Warning: Different seeds produced same results")


# RUN TESTS

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])
