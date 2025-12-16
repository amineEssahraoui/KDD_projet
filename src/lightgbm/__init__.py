"""
LightGBM Scratch - A from-scratch implementation of LightGBM.

This package provides a complete, standalone implementation of the
LightGBM gradient boosting algorithm with zero sklearn dependencies.

Features:
- Leaf-wise tree growth (best-first, like LightGBM)
- Gradient-based One-Side Sampling (GOSS)
- Exclusive Feature Bundling (EFB)
- Histogram-based split finding
- L1/L2 regularization
- Early stopping
- Learning rate decay
- Callbacks
- Binary and multi-class classification
- Regression with MSE, MAE, Huber, Quantile losses

Example usage:
    >>> from lightgbm_scratch import LGBMRegressor, LGBMClassifier
    >>> import numpy as np
    >>>
    >>> # Regression
    >>> X = np.random.randn(100, 5)
    >>> y = X[:, 0] + 2 * X[:, 1] + np.random.randn(100) * 0.1
    >>> regressor = LGBMRegressor(num_iterations=50)
    >>> regressor.fit(X, y)
    >>> predictions = regressor.predict(X)
    >>>
    >>> # Classification
    >>> y_class = (X[:, 0] + X[:, 1] > 0).astype(int)
    >>> classifier = LGBMClassifier(num_iterations=50)
    >>> classifier.fit(X, y_class)
    >>> labels = classifier.predict(X)
    >>> probas = classifier.predict_proba(X)
"""

__version__ = "1.0.0"
__author__ = "LightGBM Scratch Contributors"

# Core estimators
from .lgbm_regressor import LGBMRegressor
from .lgbm_classifier import LGBMClassifier

# Tree module
from .tree import DecisionTree, TreeNode, SplitInfo

# Loss functions
from .loss_functions import (
    Loss,
    MSELoss,
    MAELoss,
    HuberLoss,
    QuantileLoss,
    BinaryCrossEntropyLoss,
    MultiClassCrossEntropyLoss,
    get_loss_function,
)

# Base classes
from .base import (
    BaseEstimator,
    BoosterParams,
    Callback,
    EarlyStoppingCallback,
    PrintProgressCallback,
)

# Utility functions
from .utils import (
    check_array,
    check_X_y,
    check_is_fitted,
    check_sample_weight,
    validate_hyperparameters,
    train_test_split,
    accuracy_score,
    log_loss,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    NotFittedError,
    log_message,
    log_training_progress,
)

# Advanced features
from .goss import GOSS, apply_goss
from .efb import FeatureBundler, bundle_features

# Public API
__all__ = [
    # Version
    "__version__",
    # Core estimators
    "LGBMRegressor",
    "LGBMClassifier",
    # Tree
    "DecisionTree",
    "TreeNode",
    "SplitInfo",
    # Loss functions
    "Loss",
    "MSELoss",
    "MAELoss",
    "HuberLoss",
    "QuantileLoss",
    "BinaryCrossEntropyLoss",
    "MultiClassCrossEntropyLoss",
    "get_loss_function",
    # Base classes
    "BaseEstimator",
    "BoosterParams",
    "Callback",
    "EarlyStoppingCallback",
    "PrintProgressCallback",
    # Utilities
    "check_array",
    "check_X_y",
    "check_is_fitted",
    "check_sample_weight",
    "validate_hyperparameters",
    "train_test_split",
    "accuracy_score",
    "log_loss",
    "mean_squared_error",
    "mean_absolute_error",
    "r2_score",
    "log_message",
    "log_training_progress",
    "NotFittedError",
    # GOSS
    "GOSS",
    "apply_goss",
    # EFB
    "FeatureBundler",
    "bundle_features",
]
