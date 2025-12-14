"""
Utility functions for input validation and data handling.

This module provides custom validation functions to replace sklearn utilities.
All validation is done from scratch using only NumPy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union

import numpy as np


# =============================================================================
# Type Aliases
# =============================================================================

ArrayLike = Union[np.ndarray, List[Any], Tuple[Any, ...]]


# =============================================================================
# Input Validation Functions
# =============================================================================

def check_array(
    X: ArrayLike,
    *,
    ensure_2d: bool = True,
    allow_nan: bool = True,
    dtype: type = float,
    copy: bool = False,
) -> np.ndarray:
    """
    Validate and convert input array to numpy array.

    Parameters
    ----------
    X : array-like
        Input data to validate.
    ensure_2d : bool, default=True
        Whether to raise an error if X is not 2D.
    allow_nan : bool, default=True
        Whether to allow NaN values.
    dtype : type, default=float
        Desired dtype of the output array.
    copy : bool, default=False
        Whether to force a copy of the input.

    Returns
    -------
    X_converted : np.ndarray
        Validated and converted array.

    Raises
    ------
    ValueError
        If validation fails.
    TypeError
        If input type is not supported.
    """
    # Convert to numpy array
    if isinstance(X, np.ndarray):
        X_out = X.copy() if copy else X
    elif isinstance(X, (list, tuple)):
        X_out = np.array(X, dtype=dtype)
    else:
        try:
            # Try pandas DataFrame/Series
            if hasattr(X, 'values'):
                X_out = np.asarray(X.values, dtype=dtype)
            else:
                X_out = np.asarray(X, dtype=dtype)
        except Exception as e:
            raise TypeError(
                f"Cannot convert input of type {type(X).__name__} to numpy array: {e}"
            )

    # Ensure correct dtype
    if X_out.dtype != dtype:
        X_out = X_out.astype(dtype)

    # Handle dimensionality
    if X_out.ndim == 1:
        if ensure_2d:
            X_out = X_out.reshape(-1, 1)
    elif X_out.ndim != 2:
        if ensure_2d:
            raise ValueError(
                f"Expected 2D array, got {X_out.ndim}D array instead."
            )

    # Check for empty array
    if X_out.size == 0:
        raise ValueError("Input array cannot be empty.")

    # Check for infinite values
    if np.any(np.isinf(X_out)):
        raise ValueError("Input array contains infinite values.")

    # Check for NaN values
    if not allow_nan and np.any(np.isnan(X_out)):
        raise ValueError("Input array contains NaN values but allow_nan=False.")

    return X_out


def check_X_y(
    X: ArrayLike,
    y: ArrayLike,
    *,
    allow_nan: bool = True,
    multi_output: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Validate X and y arrays for supervised learning.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    y : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Target values.
    allow_nan : bool, default=True
        Whether to allow NaN values in X.
    multi_output : bool, default=False
        Whether y can have multiple outputs.

    Returns
    -------
    X : np.ndarray
        Validated feature matrix.
    y : np.ndarray
        Validated target array.

    Raises
    ------
    ValueError
        If X and y have incompatible shapes.
    """
    X = check_array(X, ensure_2d=True, allow_nan=allow_nan)
    y = check_array(y, ensure_2d=False, allow_nan=False, dtype=float)

    # Flatten y if needed
    if y.ndim == 2 and y.shape[1] == 1:
        y = y.ravel()
    elif y.ndim == 2 and not multi_output:
        raise ValueError(
            f"y has shape {y.shape}, expected 1D array. "
            "Set multi_output=True for multi-output regression."
        )

    # Check consistent number of samples
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"Found input variables with inconsistent numbers of samples: "
            f"X has {X.shape[0]} samples, y has {y.shape[0]} samples."
        )

    return X, y


def check_is_fitted(estimator: Any, attributes: Optional[List[str]] = None) -> None:
    """
    Check if an estimator is fitted by verifying required attributes.

    Parameters
    ----------
    estimator : object
        Estimator instance to check.
    attributes : list of str, optional
        List of attribute names to check. If None, checks for common
        fitted attributes like 'trees_', 'n_features_', etc.

    Raises
    ------
    NotFittedError
        If the estimator is not fitted.
    """
    if attributes is None:
        # Default attributes to check
        attributes = ['trees_', 'n_features_', 'is_fitted_']

    fitted = False
    for attr in attributes:
        if hasattr(estimator, attr):
            val = getattr(estimator, attr)
            if val is not None:
                if isinstance(val, (list, np.ndarray)):
                    if len(val) > 0:
                        fitted = True
                        break
                else:
                    fitted = True
                    break

    if not fitted:
        raise NotFittedError(
            f"This {type(estimator).__name__} instance is not fitted yet. "
            "Call 'fit' with appropriate arguments before using this estimator."
        )


def check_sample_weight(
    sample_weight: Optional[ArrayLike],
    n_samples: int,
) -> Optional[np.ndarray]:
    """
    Validate sample weights.

    Parameters
    ----------
    sample_weight : array-like of shape (n_samples,) or None
        Sample weights.
    n_samples : int
        Expected number of samples.

    Returns
    -------
    sample_weight : np.ndarray or None
        Validated sample weights.

    Raises
    ------
    ValueError
        If sample_weight has incorrect shape or contains invalid values.
    """
    if sample_weight is None:
        return None

    sample_weight = check_array(sample_weight, ensure_2d=False, allow_nan=False)

    if sample_weight.ndim != 1:
        raise ValueError(
            f"sample_weight must be 1D, got shape {sample_weight.shape}"
        )

    if len(sample_weight) != n_samples:
        raise ValueError(
            f"sample_weight has {len(sample_weight)} elements, "
            f"expected {n_samples}."
        )

    if np.any(sample_weight < 0):
        raise ValueError("sample_weight must contain non-negative values.")

    return sample_weight


def validate_hyperparameters(
    *,
    num_iterations: int,
    learning_rate: float,
    max_depth: int,
    num_leaves: int,
    min_data_in_leaf: int,
) -> None:
    """
    Validate common hyperparameters for gradient boosting.

    Parameters
    ----------
    num_iterations : int
        Number of boosting iterations.
    learning_rate : float
        Learning rate / shrinkage.
    max_depth : int
        Maximum tree depth (-1 for unlimited).
    num_leaves : int
        Maximum number of leaves.
    min_data_in_leaf : int
        Minimum samples in a leaf.

    Raises
    ------
    ValueError
        If any hyperparameter is invalid.
    """
    if num_iterations <= 0:
        raise ValueError(f"num_iterations must be positive, got {num_iterations}")

    if learning_rate <= 0:
        raise ValueError(f"learning_rate must be positive, got {learning_rate}")

    if max_depth < -1 or max_depth == 0:
        raise ValueError(
            f"max_depth must be -1 (unlimited) or positive, got {max_depth}"
        )

    if num_leaves < 2:
        raise ValueError(f"num_leaves must be >= 2, got {num_leaves}")

    if min_data_in_leaf < 0:
        raise ValueError(
            f"min_data_in_leaf must be non-negative, got {min_data_in_leaf}"
        )


# =============================================================================
# Custom Exceptions
# =============================================================================

class NotFittedError(ValueError):
    """
    Exception raised when an estimator is used before fitting.

    This exception is raised when calling predict, transform, or similar
    methods before calling fit.
    """
    pass


# =============================================================================
# Data Splitting Functions (sklearn-free)
# =============================================================================

def train_test_split(
    X: ArrayLike,
    y: ArrayLike,
    *,
    test_size: float = 0.2,
    random_state: Optional[int] = None,
    shuffle: bool = True,
    stratify: Optional[ArrayLike] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split arrays into random train and test subsets.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Features to split.
    y : array-like of shape (n_samples,)
        Target to split.
    test_size : float, default=0.2
        Proportion of the dataset to include in the test split.
    random_state : int or None, default=None
        Random seed for reproducibility.
    shuffle : bool, default=True
        Whether to shuffle before splitting.
    stratify : array-like or None, default=None
        If not None, split in a stratified fashion using this as class labels.

    Returns
    -------
    X_train : np.ndarray
        Training features.
    X_test : np.ndarray
        Test features.
    y_train : np.ndarray
        Training targets.
    y_test : np.ndarray
        Test targets.
    """
    X = check_array(X, ensure_2d=True)
    y = check_array(y, ensure_2d=False)

    n_samples = X.shape[0]
    n_test = int(n_samples * test_size)
    n_train = n_samples - n_test

    rng = np.random.default_rng(random_state)

    if stratify is not None:
        # Stratified split
        stratify = np.asarray(stratify)
        classes, y_indices = np.unique(stratify, return_inverse=True)

        train_indices = []
        test_indices = []

        for cls_idx in range(len(classes)):
            cls_mask = y_indices == cls_idx
            cls_indices = np.where(cls_mask)[0]

            if shuffle:
                rng.shuffle(cls_indices)

            n_cls_test = max(1, int(len(cls_indices) * test_size))
            test_indices.extend(cls_indices[:n_cls_test])
            train_indices.extend(cls_indices[n_cls_test:])

        train_indices = np.array(train_indices)
        test_indices = np.array(test_indices)

        if shuffle:
            rng.shuffle(train_indices)
            rng.shuffle(test_indices)
    else:
        # Simple random split
        indices = np.arange(n_samples)
        if shuffle:
            rng.shuffle(indices)

        train_indices = indices[:n_train]
        test_indices = indices[n_train:]

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


# =============================================================================
# Metrics Functions (sklearn-free)
# =============================================================================

def accuracy_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """
    Compute classification accuracy.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth labels.
    y_pred : array-like of shape (n_samples,)
        Predicted labels.

    Returns
    -------
    accuracy : float
        Accuracy score between 0 and 1.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true has shape {y_true.shape}, "
            f"y_pred has shape {y_pred.shape}"
        )

    return float(np.mean(y_true == y_pred))


def log_loss(
    y_true: ArrayLike,
    y_pred_proba: ArrayLike,
    *,
    eps: float = 1e-15,
) -> float:
    """
    Compute log loss (cross-entropy loss).

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True class labels.
    y_pred_proba : array-like of shape (n_samples,) or (n_samples, n_classes)
        Predicted probabilities.
    eps : float, default=1e-15
        Small constant for numerical stability.

    Returns
    -------
    loss : float
        Log loss value.
    """
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)

    # Clip probabilities
    y_pred_proba = np.clip(y_pred_proba, eps, 1 - eps)

    if y_pred_proba.ndim == 1:
        # Binary classification with single probability column
        loss = -np.mean(
            y_true * np.log(y_pred_proba) +
            (1 - y_true) * np.log(1 - y_pred_proba)
        )
    else:
        # Multiclass with probability matrix
        n_samples = len(y_true)
        y_true_int = y_true.astype(int)
        loss = -np.mean(
            np.log(y_pred_proba[np.arange(n_samples), y_true_int])
        )

    return float(loss)


def mean_squared_error(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """
    Compute mean squared error.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth values.
    y_pred : array-like of shape (n_samples,)
        Predicted values.

    Returns
    -------
    mse : float
        Mean squared error.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean((y_true - y_pred) ** 2))


def mean_absolute_error(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """
    Compute mean absolute error.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth values.
    y_pred : array-like of shape (n_samples,)
        Predicted values.

    Returns
    -------
    mae : float
        Mean absolute error.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """
    Compute R-squared (coefficient of determination).

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth values.
    y_pred : array-like of shape (n_samples,)
        Predicted values.

    Returns
    -------
    r2 : float
        R-squared score.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot == 0:
        return 0.0

    return float(1 - ss_res / ss_tot)


# =============================================================================
# Logging Utilities
# =============================================================================

def log_message(message: str, *, verbose: int = 0) -> None:
    """
    Print a log message if verbose level is sufficient.

    Parameters
    ----------
    message : str
        Message to print.
    verbose : int, default=0
        Verbosity level. Message is printed if verbose >= 1.
    """
    if verbose >= 1:
        print(f"[LightGBM] {message}")


def log_training_progress(
    iteration: int,
    total_iterations: int,
    metric_value: float,
    *,
    verbose: int = 0,
    metric_name: str = "loss",
) -> None:
    """
    Log training progress.

    Parameters
    ----------
    iteration : int
        Current iteration number.
    total_iterations : int
        Total number of iterations.
    metric_value : float
        Current metric value.
    verbose : int, default=0
        Verbosity level.
    metric_name : str, default="loss"
        Name of the metric being tracked.
    """
    if verbose >= 1:
        progress = (iteration / total_iterations) * 100
        print(
            f"[LightGBM] Iter {iteration}/{total_iterations} "
            f"({progress:.1f}%) - {metric_name}: {metric_value:.6f}"
        )
