"""
GOSS (Gradient-based One-Side Sampling) implementation.

GOSS is a data sampling technique used in LightGBM to speed up training
by focusing on samples with large gradients while randomly sampling
from samples with small gradients.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


class GOSS:
    """
    Gradient-based One-Side Sampling.

    GOSS keeps all samples with large gradients (top samples) and
    randomly samples from the remaining samples (small gradients).
    This reduces training time while maintaining accuracy.

    Parameters
    ----------
    top_rate : float, default=0.2
        Fraction of samples with the largest gradients to keep.
    other_rate : float, default=0.1
        Fraction of remaining samples to randomly sample.
    random_state : int or None, default=None
        Random seed for reproducibility.

    Attributes
    ----------
    sample_weight_factor_ : float
        Weight amplification factor for sampled small-gradient samples.

    References
    ----------
    Ke, G., et al. "LightGBM: A Highly Efficient Gradient Boosting
    Decision Tree." NeurIPS 2017.
    """

    def __init__(
        self,
        top_rate: float = 0.2,
        other_rate: float = 0.1,
        random_state: Optional[int] = None,
    ):
        if not 0 < top_rate < 1:
            raise ValueError(f"top_rate must be in (0, 1), got {top_rate}")
        if not 0 < other_rate < 1:
            raise ValueError(f"other_rate must be in (0, 1), got {other_rate}")
        if top_rate + other_rate >= 1:
            raise ValueError(
                f"top_rate + other_rate must be < 1, "
                f"got {top_rate + other_rate}"
            )

        self.top_rate = top_rate
        self.other_rate = other_rate
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)

        # Weight factor for small gradient samples
        self.sample_weight_factor_ = (1 - top_rate) / other_rate

    def sample(
        self,
        gradients: np.ndarray,
        hessians: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform GOSS sampling based on gradient magnitudes.

        Parameters
        ----------
        gradients : np.ndarray of shape (n_samples,)
            Gradient values for each sample.
        hessians : np.ndarray of shape (n_samples,) or None
            Hessian values (not used for sampling, but included for API).

        Returns
        -------
        selected_indices : np.ndarray
            Indices of selected samples.
        sample_weights : np.ndarray
            Weights for the selected samples.
        """
        n_samples = len(gradients)

        # Calculate number of samples for each category
        n_top = max(1, int(n_samples * self.top_rate))
        n_other = max(1, int(n_samples * self.other_rate))

        # Sort by absolute gradient (descending)
        abs_gradients = np.abs(gradients)
        sorted_indices = np.argsort(abs_gradients)[::-1]

        # Top samples (large gradients)
        top_indices = sorted_indices[:n_top]

        # Randomly sample from remaining samples
        remaining_indices = sorted_indices[n_top:]
        if len(remaining_indices) > n_other:
            other_indices = self._rng.choice(
                remaining_indices, size=n_other, replace=False
            )
        else:
            other_indices = remaining_indices

        # Combine indices
        selected_indices = np.concatenate([top_indices, other_indices])

        # Compute sample weights
        # Top samples get weight 1, other samples get amplified weight
        sample_weights = np.ones(len(selected_indices))
        sample_weights[n_top:] = self.sample_weight_factor_

        return selected_indices, sample_weights

    def sample_data(
        self,
        X: np.ndarray,
        gradients: np.ndarray,
        hessians: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample data, gradients, and hessians using GOSS.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix.
        gradients : np.ndarray of shape (n_samples,)
            Gradient values.
        hessians : np.ndarray of shape (n_samples,)
            Hessian values.

        Returns
        -------
        X_sampled : np.ndarray
            Sampled feature matrix.
        gradients_sampled : np.ndarray
            Sampled gradients.
        hessians_sampled : np.ndarray
            Sampled hessians.
        sample_weights : np.ndarray
            Sample weights for the sampled data.
        """
        indices, weights = self.sample(gradients)

        return (
            X[indices],
            gradients[indices] * weights,
            hessians[indices] * weights,
            weights,
        )

    def __repr__(self) -> str:
        return (
            f"GOSS(top_rate={self.top_rate}, "
            f"other_rate={self.other_rate}, "
            f"random_state={self.random_state})"
        )


def apply_goss(
    X: np.ndarray,
    gradients: np.ndarray,
    hessians: np.ndarray,
    *,
    top_rate: float = 0.2,
    other_rate: float = 0.1,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convenience function to apply GOSS sampling.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Feature matrix.
    gradients : np.ndarray of shape (n_samples,)
        Gradient values.
    hessians : np.ndarray of shape (n_samples,)
        Hessian values.
    top_rate : float, default=0.2
        Fraction of large gradient samples to keep.
    other_rate : float, default=0.1
        Fraction of small gradient samples to sample.
    random_state : int or None, default=None
        Random seed.

    Returns
    -------
    X_sampled : np.ndarray
        Sampled feature matrix.
    gradients_sampled : np.ndarray
        Sampled and weighted gradients.
    hessians_sampled : np.ndarray
        Sampled and weighted hessians.
    sample_weights : np.ndarray
        Sample weights.
    """
    goss = GOSS(top_rate=top_rate, other_rate=other_rate, random_state=random_state)
    return goss.sample_data(X, gradients, hessians)


__all__ = ['GOSS', 'apply_goss']
