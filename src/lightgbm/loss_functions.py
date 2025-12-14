"""
Loss functions for gradient boosting.

This module provides various loss functions for regression and classification
tasks. Each loss function computes gradients and hessians needed for
gradient boosting optimization.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


# =============================================================================
# Base Loss Class
# =============================================================================

class Loss(ABC):
    """
    Abstract base class for loss functions.

    All loss functions must implement methods to compute:
    - The loss value
    - First-order gradients
    - Second-order hessians
    - Initial prediction (bias)
    """

    @abstractmethod
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the loss value.

        Parameters
        ----------
        y_true : np.ndarray of shape (n_samples,)
            True target values.
        y_pred : np.ndarray of shape (n_samples,)
            Predicted values.

        Returns
        -------
        loss : float
            The computed loss value.
        """
        pass

    @abstractmethod
    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute the first-order gradient.

        Parameters
        ----------
        y_true : np.ndarray of shape (n_samples,)
            True target values.
        y_pred : np.ndarray of shape (n_samples,)
            Predicted values.

        Returns
        -------
        gradient : np.ndarray of shape (n_samples,)
            First-order gradients.
        """
        pass

    @abstractmethod
    def hessian(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute the second-order hessian.

        Parameters
        ----------
        y_true : np.ndarray of shape (n_samples,)
            True target values.
        y_pred : np.ndarray of shape (n_samples,)
            Predicted values.

        Returns
        -------
        hessian : np.ndarray of shape (n_samples,)
            Second-order hessians (diagonal elements).
        """
        pass

    @abstractmethod
    def init_prediction(self, y: np.ndarray) -> float:
        """
        Compute the initial prediction (bias term).

        Parameters
        ----------
        y : np.ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        init_pred : float
            Initial prediction value.
        """
        pass

    def gradient_hessian(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute both gradient and hessian at once.

        Parameters
        ----------
        y_true : np.ndarray of shape (n_samples,)
            True target values.
        y_pred : np.ndarray of shape (n_samples,)
            Predicted values.

        Returns
        -------
        gradient : np.ndarray of shape (n_samples,)
            First-order gradients.
        hessian : np.ndarray of shape (n_samples,)
            Second-order hessians.
        """
        return self.gradient(y_true, y_pred), self.hessian(y_true, y_pred)


# =============================================================================
# Regression Loss Functions
# =============================================================================

class MSELoss(Loss):
    """
    Mean Squared Error loss for regression.

    L(y, f) = 0.5 * (y - f)^2

    This is the standard loss for regression problems.
    The 0.5 factor simplifies the gradient computation.
    """

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute MSE loss."""
        return float(0.5 * np.mean((y_true - y_pred) ** 2))

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute gradient: d/df [0.5*(y-f)^2] = -(y - f) = f - y
        """
        return y_pred - y_true

    def hessian(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute hessian: d^2/df^2 [0.5*(y-f)^2] = 1
        """
        return np.ones_like(y_true)

    def init_prediction(self, y: np.ndarray) -> float:
        """Initial prediction is the mean of targets."""
        return float(np.mean(y))


class MAELoss(Loss):
    """
    Mean Absolute Error loss for regression.

    L(y, f) = |y - f|

    More robust to outliers than MSE.
    """

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute MAE loss."""
        return float(np.mean(np.abs(y_true - y_pred)))

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute gradient: sign(f - y)

        Note: At y == f, gradient is 0.
        """
        residual = y_pred - y_true
        return np.sign(residual)

    def hessian(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute hessian: technically 0 everywhere except at y == f.

        For practical purposes, we return ones to avoid division by zero
        in tree building.
        """
        return np.ones_like(y_true)

    def init_prediction(self, y: np.ndarray) -> float:
        """Initial prediction is the median of targets (minimizes MAE)."""
        return float(np.median(y))


class HuberLoss(Loss):
    """
    Huber loss for regression.

    L(y, f) = 0.5 * (y - f)^2           if |y - f| <= delta
            = delta * |y - f| - 0.5 * delta^2  otherwise

    Combines MSE for small errors and MAE for large errors.

    Parameters
    ----------
    delta : float, default=1.0
        Threshold for switching between MSE and MAE behavior.
    """

    def __init__(self, delta: float = 1.0):
        self.delta = delta

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute Huber loss."""
        residual = np.abs(y_true - y_pred)
        quadratic = np.minimum(residual, self.delta)
        linear = residual - quadratic
        loss = 0.5 * quadratic ** 2 + self.delta * linear
        return float(np.mean(loss))

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Compute Huber gradient."""
        residual = y_pred - y_true
        abs_residual = np.abs(residual)

        # Gradient is residual for small errors, clipped for large errors
        gradient = np.where(
            abs_residual <= self.delta,
            residual,
            self.delta * np.sign(residual)
        )
        return gradient

    def hessian(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Compute Huber hessian."""
        residual = y_pred - y_true
        abs_residual = np.abs(residual)

        # Hessian is 1 for small errors, approaches 0 for large errors
        hessian = np.where(abs_residual <= self.delta, 1.0, 0.0)
        # Add small constant to avoid division by zero
        return hessian + 1e-6

    def init_prediction(self, y: np.ndarray) -> float:
        """Initial prediction is the mean of targets."""
        return float(np.mean(y))


class QuantileLoss(Loss):
    """
    Quantile loss for quantile regression.

    L(y, f) = q * (y - f)     if y >= f
            = (1-q) * (f - y)  if y < f

    Parameters
    ----------
    quantile : float, default=0.5
        Target quantile between 0 and 1.
        0.5 gives median regression (equivalent to MAE).
    """

    def __init__(self, quantile: float = 0.5):
        if not 0 < quantile < 1:
            raise ValueError(f"quantile must be in (0, 1), got {quantile}")
        self.quantile = quantile

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute quantile loss."""
        residual = y_true - y_pred
        loss = np.where(
            residual >= 0,
            self.quantile * residual,
            (self.quantile - 1) * residual
        )
        return float(np.mean(loss))

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Compute quantile gradient."""
        residual = y_true - y_pred
        return np.where(residual >= 0, -self.quantile, 1 - self.quantile)

    def hessian(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Compute quantile hessian (constant for numerical stability)."""
        return np.ones_like(y_true)

    def init_prediction(self, y: np.ndarray) -> float:
        """Initial prediction is the target quantile."""
        return float(np.quantile(y, self.quantile))


# =============================================================================
# Classification Loss Functions
# =============================================================================

class BinaryCrossEntropyLoss(Loss):
    """
    Binary cross-entropy loss for binary classification.

    L(y, f) = -[y * log(sigmoid(f)) + (1-y) * log(1 - sigmoid(f))]

    Works with raw scores (logits), not probabilities.

    Parameters
    ----------
    eps : float, default=1e-15
        Small constant for numerical stability.
    """

    def __init__(self, eps: float = 1e-15):
        self.eps = eps

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid function."""
        # Use clip to prevent overflow
        x = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x))

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute binary cross-entropy loss."""
        prob = self._sigmoid(y_pred)
        prob = np.clip(prob, self.eps, 1 - self.eps)
        loss = -(y_true * np.log(prob) + (1 - y_true) * np.log(1 - prob))
        return float(np.mean(loss))

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute gradient.

        g = sigmoid(f) - y = p - y
        """
        prob = self._sigmoid(y_pred)
        return prob - y_true

    def hessian(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute hessian.

        h = p * (1 - p)
        """
        prob = self._sigmoid(y_pred)
        hess = prob * (1 - prob)
        # Clip for numerical stability
        return np.clip(hess, self.eps, 1 - self.eps)

    def init_prediction(self, y: np.ndarray) -> float:
        """
        Initial prediction is log-odds of positive class.

        f0 = log(p / (1-p)) where p = mean(y)
        """
        p = np.clip(np.mean(y), self.eps, 1 - self.eps)
        return float(np.log(p / (1 - p)))


class MultiClassCrossEntropyLoss(Loss):
    """
    Multi-class cross-entropy loss for multi-class classification.

    L(y, f) = -sum_k [y_k * log(softmax(f)_k)]

    Works with raw scores (logits) for each class.

    Parameters
    ----------
    n_classes : int
        Number of classes.
    eps : float, default=1e-15
        Small constant for numerical stability.
    """

    def __init__(self, n_classes: int, eps: float = 1e-15):
        self.n_classes = n_classes
        self.eps = eps

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """
        Numerically stable softmax.

        Parameters
        ----------
        x : np.ndarray of shape (n_samples, n_classes)
            Raw scores.

        Returns
        -------
        probs : np.ndarray of shape (n_samples, n_classes)
            Probabilities.
        """
        # Subtract max for numerical stability
        x_max = np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute multi-class cross-entropy loss.

        Parameters
        ----------
        y_true : np.ndarray of shape (n_samples,)
            True class labels (integers).
        y_pred : np.ndarray of shape (n_samples, n_classes)
            Raw scores for each class.

        Returns
        -------
        loss : float
            Mean cross-entropy loss.
        """
        probs = self._softmax(y_pred)
        probs = np.clip(probs, self.eps, 1 - self.eps)
        n_samples = len(y_true)
        y_true_int = y_true.astype(int)
        log_probs = np.log(probs[np.arange(n_samples), y_true_int])
        return float(-np.mean(log_probs))

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute gradient.

        For class k: g_k = p_k - (y == k)

        Parameters
        ----------
        y_true : np.ndarray of shape (n_samples,)
            True class labels.
        y_pred : np.ndarray of shape (n_samples, n_classes)
            Raw scores for each class.

        Returns
        -------
        gradient : np.ndarray of shape (n_samples, n_classes)
            Gradients for each sample and class.
        """
        probs = self._softmax(y_pred)
        n_samples = len(y_true)
        y_true_int = y_true.astype(int)

        # Create one-hot encoding
        one_hot = np.zeros((n_samples, self.n_classes))
        one_hot[np.arange(n_samples), y_true_int] = 1

        return probs - one_hot

    def hessian(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute hessian (diagonal approximation).

        For class k: h_k = p_k * (1 - p_k)

        Parameters
        ----------
        y_true : np.ndarray of shape (n_samples,)
            True class labels (unused, but kept for API consistency).
        y_pred : np.ndarray of shape (n_samples, n_classes)
            Raw scores for each class.

        Returns
        -------
        hessian : np.ndarray of shape (n_samples, n_classes)
            Diagonal hessians for each sample and class.
        """
        probs = self._softmax(y_pred)
        hess = probs * (1 - probs)
        return np.clip(hess, self.eps, 1 - self.eps)

    def init_prediction(self, y: np.ndarray) -> np.ndarray:
        """
        Initial prediction is log of class frequencies.

        Parameters
        ----------
        y : np.ndarray of shape (n_samples,)
            Class labels.

        Returns
        -------
        init_pred : np.ndarray of shape (n_classes,)
            Initial raw scores for each class.
        """
        y_int = y.astype(int)
        class_counts = np.bincount(y_int, minlength=self.n_classes)
        class_freqs = class_counts / len(y)
        class_freqs = np.clip(class_freqs, self.eps, 1 - self.eps)
        return np.log(class_freqs)

    def gradient_hessian(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute gradient and hessian efficiently."""
        probs = self._softmax(y_pred)
        n_samples = len(y_true)
        y_true_int = y_true.astype(int)

        # One-hot encoding
        one_hot = np.zeros((n_samples, self.n_classes))
        one_hot[np.arange(n_samples), y_true_int] = 1

        gradient = probs - one_hot
        hessian = probs * (1 - probs)
        hessian = np.clip(hessian, self.eps, 1 - self.eps)

        return gradient, hessian


# =============================================================================
# Loss Function Factory
# =============================================================================

def get_loss_function(
    objective: str,
    *,
    n_classes: int = 2,
    **kwargs
) -> Loss:
    """
    Factory function to create loss objects by name.

    Parameters
    ----------
    objective : str
        Name of the objective/loss function.
        Supported: 'mse', 'mae', 'huber', 'quantile',
                   'binary_crossentropy', 'multiclass_crossentropy'
    n_classes : int, default=2
        Number of classes (for classification objectives).
    **kwargs
        Additional arguments passed to the loss constructor.

    Returns
    -------
    loss : Loss
        Loss function instance.

    Raises
    ------
    ValueError
        If objective name is not recognized.
    """
    objective = objective.lower().replace('-', '_')

    loss_map = {
        'mse': MSELoss,
        'l2': MSELoss,
        'mean_squared_error': MSELoss,
        'mae': MAELoss,
        'l1': MAELoss,
        'mean_absolute_error': MAELoss,
        'huber': HuberLoss,
        'quantile': QuantileLoss,
        'binary_crossentropy': BinaryCrossEntropyLoss,
        'binary': BinaryCrossEntropyLoss,
        'logloss': BinaryCrossEntropyLoss,
    }

    if objective in loss_map:
        return loss_map[objective](**kwargs)
    elif objective in ('multiclass_crossentropy', 'multiclass', 'softmax'):
        return MultiClassCrossEntropyLoss(n_classes=n_classes, **kwargs)
    else:
        raise ValueError(
            f"Unknown objective: '{objective}'. "
            f"Supported objectives: {list(loss_map.keys()) + ['multiclass_crossentropy']}"
        )


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    'Loss',
    'MSELoss',
    'MAELoss',
    'HuberLoss',
    'QuantileLoss',
    'BinaryCrossEntropyLoss',
    'MultiClassCrossEntropyLoss',
    'get_loss_function',
]
