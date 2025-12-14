"""
Test suite for LGBMClassifier.

This module validates the custom LightGBM classifier implementation
using synthetic datasets - NO sklearn dependencies.

Framework: pytest
"""

import os
import tempfile

import numpy as np
import pytest

from lightgbm import LGBMClassifier
from lightgbm.utils import accuracy_score, train_test_split


# =============================================================================
# Test Data Generation (No sklearn)
# =============================================================================

def make_binary_classification(n_samples=300, n_features=10, seed=42):
    """Generate synthetic binary classification data."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features))
    
    # Linear decision boundary with some noise
    weights = rng.standard_normal(n_features)
    logits = X @ weights
    probs = 1 / (1 + np.exp(-logits))
    y = (probs > 0.5).astype(int)
    
    return X, y


def make_multiclass_classification(n_samples=300, n_features=10, n_classes=3, seed=42):
    """Generate synthetic multiclass classification data."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features))
    
    # Multiple linear boundaries
    weights = rng.standard_normal((n_features, n_classes))
    logits = X @ weights
    y = np.argmax(logits, axis=1)
    
    return X, y


# =============================================================================
# Binary Classification Tests
# =============================================================================

class TestLGBMClassifierBinary:
    """Test LGBMClassifier on binary classification."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test data."""
        X, y = make_binary_classification(n_samples=300, n_features=10, seed=42)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

    def test_binary_model_fits_without_error(self):
        """Verify that LGBMClassifier fits without raising exceptions."""
        clf = LGBMClassifier(
            num_iterations=50,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        clf.fit(self.X_train, self.y_train)
        assert clf.is_fitted_

    def test_binary_predict_returns_correct_shape(self):
        """Verify predict returns correct shape."""
        clf = LGBMClassifier(num_iterations=30, random_state=42)
        clf.fit(self.X_train, self.y_train)
        preds = clf.predict(self.X_test)
        assert preds.shape == (len(self.y_test),)

    def test_binary_predict_proba_shape(self):
        """Verify predict_proba returns correct shape."""
        clf = LGBMClassifier(num_iterations=30, random_state=42)
        clf.fit(self.X_train, self.y_train)
        probas = clf.predict_proba(self.X_test)
        assert probas.shape == (len(self.y_test), 2)
        
        # Probabilities should sum to 1
        assert np.allclose(probas.sum(axis=1), 1.0)

    def test_binary_predict_proba_range(self):
        """Verify probabilities are in [0, 1]."""
        clf = LGBMClassifier(num_iterations=30, random_state=42)
        clf.fit(self.X_train, self.y_train)
        probas = clf.predict_proba(self.X_test)
        assert probas.min() >= 0.0
        assert probas.max() <= 1.0

    def test_binary_predictions_are_valid_classes(self):
        """Verify predictions are valid class labels."""
        clf = LGBMClassifier(num_iterations=30, random_state=42)
        clf.fit(self.X_train, self.y_train)
        preds = clf.predict(self.X_test)
        assert set(preds).issubset({0, 1})

    def test_binary_accuracy_reasonable(self):
        """Verify accuracy is reasonable (better than random)."""
        clf = LGBMClassifier(
            num_iterations=50,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        clf.fit(self.X_train, self.y_train)
        preds = clf.predict(self.X_test)
        acc = accuracy_score(self.y_test, preds)
        # Should be better than random (50%)
        assert acc > 0.6

    def test_binary_feature_importances(self):
        """Verify feature importances are computed."""
        clf = LGBMClassifier(num_iterations=30, random_state=42)
        clf.fit(self.X_train, self.y_train)
        
        assert clf.feature_importances_ is not None
        assert len(clf.feature_importances_) == self.X_train.shape[1]
        assert np.isclose(clf.feature_importances_.sum(), 1.0)


# =============================================================================
# Multiclass Classification Tests
# =============================================================================

class TestLGBMClassifierMulticlass:
    """Test LGBMClassifier on multiclass classification."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test data."""
        X, y = make_multiclass_classification(
            n_samples=300, n_features=10, n_classes=3, seed=42
        )
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        self.n_classes = 3

    def test_multiclass_model_fits_without_error(self):
        """Verify multiclass fitting works."""
        clf = LGBMClassifier(num_iterations=30, random_state=42)
        clf.fit(self.X_train, self.y_train)
        assert clf.is_fitted_
        assert clf.n_classes_ == self.n_classes

    def test_multiclass_predict_returns_correct_shape(self):
        """Verify predict returns correct shape."""
        clf = LGBMClassifier(num_iterations=30, random_state=42)
        clf.fit(self.X_train, self.y_train)
        preds = clf.predict(self.X_test)
        assert preds.shape == (len(self.y_test),)

    def test_multiclass_predict_proba_shape(self):
        """Verify predict_proba returns correct shape for multiclass."""
        clf = LGBMClassifier(num_iterations=30, random_state=42)
        clf.fit(self.X_train, self.y_train)
        probas = clf.predict_proba(self.X_test)
        assert probas.shape == (len(self.y_test), self.n_classes)
        
        # Probabilities should sum to 1
        assert np.allclose(probas.sum(axis=1), 1.0)

    def test_multiclass_predictions_are_valid_classes(self):
        """Verify predictions are valid class labels."""
        clf = LGBMClassifier(num_iterations=30, random_state=42)
        clf.fit(self.X_train, self.y_train)
        preds = clf.predict(self.X_test)
        assert set(preds).issubset({0, 1, 2})


# =============================================================================
# Early Stopping Tests
# =============================================================================

class TestLGBMClassifierEarlyStopping:
    """Test early stopping functionality."""

    def test_early_stopping_with_eval_set(self):
        """Verify early stopping with validation set works."""
        X, y = make_binary_classification(n_samples=200, seed=42)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        clf = LGBMClassifier(
            num_iterations=100,
            learning_rate=0.1,
            max_depth=3,
            early_stopping_rounds=5,
            random_state=42
        )
        clf.fit(X_train, y_train, eval_set=(X_val, y_val))
        
        # Should be able to make predictions after fitting
        preds = clf.predict(X_val)
        assert len(preds) == len(y_val)

    def test_early_stopping_uses_best_model(self):
        """Verify early stopping uses the best model."""
        X, y = make_binary_classification(n_samples=200, seed=42)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        clf = LGBMClassifier(
            num_iterations=100,
            early_stopping_rounds=5,
            random_state=42
        )
        clf.fit(X_train, y_train, eval_set=(X_val, y_val))
        
        # Should be able to make predictions
        preds = clf.predict(X_val)
        assert len(preds) == len(y_val)


# =============================================================================
# Advanced Features Tests
# =============================================================================

class TestLGBMClassifierAdvanced:
    """Test advanced features."""

    def test_sample_weight_support(self):
        """Verify sample weights are supported."""
        X, y = make_binary_classification(n_samples=100, seed=42)
        weights = np.random.rand(len(y))
        
        clf = LGBMClassifier(num_iterations=20, random_state=42)
        clf.fit(X, y, sample_weight=weights)
        preds = clf.predict(X)
        assert len(preds) == len(y)

    def test_goss_training(self):
        """Verify GOSS training works."""
        X, y = make_binary_classification(n_samples=200, seed=42)
        
        clf = LGBMClassifier(
            num_iterations=30,
            enable_goss=True,
            goss_top_rate=0.2,
            goss_other_rate=0.1,
            random_state=42
        )
        clf.fit(X, y)
        preds = clf.predict(X)
        assert len(preds) == len(y)

    def test_regularization_parameters(self):
        """Verify regularization parameters work."""
        X, y = make_binary_classification(n_samples=100, seed=42)
        
        clf = LGBMClassifier(
            num_iterations=20,
            lambda_l1=0.1,
            lambda_l2=0.1,
            random_state=42
        )
        clf.fit(X, y)
        preds = clf.predict(X)
        assert len(preds) == len(y)

    def test_subsample_and_colsample(self):
        """Verify subsampling parameters work."""
        X, y = make_binary_classification(n_samples=100, seed=42)
        
        clf = LGBMClassifier(
            num_iterations=20,
            bagging_fraction=0.8,
            feature_fraction=0.8,
            bagging_freq=1,
            random_state=42
        )
        clf.fit(X, y)
        preds = clf.predict(X)
        assert len(preds) == len(y)

    def test_learning_rate_decay(self):
        """Verify learning rate decay works."""
        X, y = make_binary_classification(n_samples=100, seed=42)
        
        clf = LGBMClassifier(
            num_iterations=30,
            learning_rate=0.1,
            lr_decay=0.99,
            random_state=42
        )
        clf.fit(X, y)
        preds = clf.predict(X)
        assert len(preds) == len(y)


# =============================================================================
# Serialization Tests
# =============================================================================

class TestLGBMClassifierSerialization:
    """Test model serialization."""

    def test_save_and_load_model(self):
        """Verify model can be saved and loaded."""
        X, y = make_binary_classification(n_samples=100, seed=42)
        
        clf = LGBMClassifier(num_iterations=20, random_state=42)
        clf.fit(X, y)
        preds_original = clf.predict(X)
        
        # Save and load
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
            import pickle
            pickle.dump(clf, f)
            temp_path = f.name
        
        try:
            with open(temp_path, 'rb') as f:
                loaded_clf = pickle.load(f)
            
            preds_loaded = loaded_clf.predict(X)
            assert np.array_equal(preds_original, preds_loaded)
        finally:
            os.unlink(temp_path)


# =============================================================================
# Edge Cases Tests
# =============================================================================

class TestLGBMClassifierEdgeCases:
    """Test edge cases."""

    def test_single_feature(self):
        """Test with single feature."""
        X = np.random.randn(100, 1)
        y = (X[:, 0] > 0).astype(int)
        
        clf = LGBMClassifier(num_iterations=20, random_state=42)
        clf.fit(X, y)
        preds = clf.predict(X)
        assert len(preds) == len(y)

    def test_two_samples_per_class(self):
        """Test with minimal samples."""
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([0, 0, 1, 1])
        
        clf = LGBMClassifier(
            num_iterations=10,
            min_data_in_leaf=1,
            random_state=42
        )
        clf.fit(X, y)
        preds = clf.predict(X)
        assert len(preds) == 4

    def test_reproducibility(self):
        """Verify reproducibility with same random_state."""
        X, y = make_binary_classification(n_samples=100, seed=42)
        
        clf1 = LGBMClassifier(num_iterations=20, random_state=42)
        clf1.fit(X, y)
        preds1 = clf1.predict(X)
        
        clf2 = LGBMClassifier(num_iterations=20, random_state=42)
        clf2.fit(X, y)
        preds2 = clf2.predict(X)
        
        assert np.array_equal(preds1, preds2)

    def test_unfitted_model_raises_error(self):
        """Verify unfitted model raises error on predict."""
        clf = LGBMClassifier(num_iterations=10)
        X = np.random.randn(10, 5)
        
        # Should raise some kind of error
        error_raised = False
        try:
            clf.predict(X)
        except (ValueError, RuntimeError, AttributeError, TypeError):
            error_raised = True
        
        assert error_raised, "Expected error when calling predict on unfitted model"
