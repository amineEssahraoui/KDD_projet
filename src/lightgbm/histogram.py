"""
Histogram-based split finding for gradient boosting.

This module provides efficient histogram binning and split finding
algorithms that significantly speed up training on large datasets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class HistogramBin:
    """
    Represents a single bin in a histogram.

    Attributes
    ----------
    sum_gradients : float
        Sum of gradients for samples in this bin.
    sum_hessians : float
        Sum of hessians for samples in this bin.
    count : int
        Number of samples in this bin.
    """
    sum_gradients: float = 0.0
    sum_hessians: float = 0.0
    count: int = 0


class HistogramBuilder:
    """
    Builds and manages histograms for efficient split finding.

    The histogram-based approach discretizes continuous features into
    bins, allowing O(#bins) split finding instead of O(#samples).

    Parameters
    ----------
    max_bins : int, default=255
        Maximum number of bins per feature.
    min_data_in_bin : int, default=3
        Minimum number of samples per bin.

    Attributes
    ----------
    bin_edges_ : list of np.ndarray
        Bin edges for each feature.
    n_features_ : int
        Number of features.
    """

    def __init__(
        self,
        max_bins: int = 255,
        min_data_in_bin: int = 3,
    ):
        if max_bins < 2:
            raise ValueError(f"max_bins must be >= 2, got {max_bins}")

        self.max_bins = max_bins
        self.min_data_in_bin = min_data_in_bin

        # State
        self.bin_edges_: Optional[List[np.ndarray]] = None
        self.n_features_: Optional[int] = None

    def fit(self, X: np.ndarray) -> "HistogramBuilder":
        """
        Compute bin edges from the training data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        self : HistogramBuilder
            Fitted builder.
        """
        n_samples, n_features = X.shape
        self.n_features_ = n_features
        self.bin_edges_ = []

        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]

            # Handle NaN values
            valid_mask = ~np.isnan(feature_values)
            valid_values = feature_values[valid_mask]

            if len(valid_values) == 0:
                # All NaN - use dummy bins
                self.bin_edges_.append(np.array([-np.inf, np.inf]))
                continue

            # Determine number of bins based on unique values
            unique_values = np.unique(valid_values)
            n_unique = len(unique_values)

            if n_unique <= self.max_bins:
                # Use midpoints between unique values
                if n_unique == 1:
                    edges = np.array([unique_values[0] - 0.5, unique_values[0] + 0.5])
                else:
                    midpoints = (unique_values[:-1] + unique_values[1:]) / 2
                    edges = np.concatenate([
                        [unique_values[0] - 0.5],
                        midpoints,
                        [unique_values[-1] + 0.5]
                    ])
            else:
                # Use quantile-based binning
                n_bins = self.max_bins
                percentiles = np.linspace(0, 100, n_bins + 1)
                edges = np.percentile(valid_values, percentiles)

                # Remove duplicate edges
                edges = np.unique(edges)

                # Extend edges slightly for numerical stability
                if len(edges) >= 2:
                    eps = (edges[-1] - edges[0]) * 1e-6
                    edges[0] -= eps
                    edges[-1] += eps

            self.bin_edges_.append(edges)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform features to bin indices.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Features to transform.

        Returns
        -------
        X_binned : np.ndarray of shape (n_samples, n_features)
            Bin indices for each feature (dtype=uint8 or uint16).
        """
        if self.bin_edges_ is None:
            raise RuntimeError("HistogramBuilder has not been fitted yet.")

        n_samples, n_features = X.shape
        dtype = np.uint8 if self.max_bins <= 256 else np.uint16
        X_binned = np.zeros((n_samples, n_features), dtype=dtype)

        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
            edges = self.bin_edges_[feature_idx]

            # Digitize (bin 0 is for values <= first edge)
            bins = np.digitize(feature_values, edges[1:-1])

            # Handle NaN - assign to a special bin (last bin + 1)
            nan_mask = np.isnan(feature_values)
            bins[nan_mask] = len(edges) - 1

            X_binned[:, feature_idx] = bins

        return X_binned

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)

    def build_histogram(
        self,
        X_binned: np.ndarray,
        gradients: np.ndarray,
        hessians: np.ndarray,
        indices: np.ndarray,
        feature_idx: int,
    ) -> List[HistogramBin]:
        """
        Build a histogram for a single feature.

        Parameters
        ----------
        X_binned : np.ndarray of shape (n_samples, n_features)
            Binned feature matrix.
        gradients : np.ndarray of shape (n_samples,)
            Gradient values.
        hessians : np.ndarray of shape (n_samples,)
            Hessian values.
        indices : np.ndarray
            Indices of samples to include.
        feature_idx : int
            Feature index to build histogram for.

        Returns
        -------
        histogram : list of HistogramBin
            Histogram bins with aggregated statistics.
        """
        if self.bin_edges_ is None:
            raise RuntimeError("HistogramBuilder has not been fitted yet.")

        n_bins = len(self.bin_edges_[feature_idx])
        histogram = [HistogramBin() for _ in range(n_bins)]

        # Aggregate statistics
        for idx in indices:
            bin_idx = X_binned[idx, feature_idx]
            histogram[bin_idx].sum_gradients += gradients[idx]
            histogram[bin_idx].sum_hessians += hessians[idx]
            histogram[bin_idx].count += 1

        return histogram

    def find_best_split_from_histogram(
        self,
        histogram: List[HistogramBin],
        feature_idx: int,
        lambda_l2: float = 0.0,
        min_samples_leaf: int = 1,
        min_sum_hessian_in_leaf: float = 1e-3,
    ) -> Tuple[float, float, float, float]:
        """
        Find the best split from a histogram.

        Parameters
        ----------
        histogram : list of HistogramBin
            Histogram bins.
        feature_idx : int
            Feature index.
        lambda_l2 : float, default=0.0
            L2 regularization.
        min_samples_leaf : int, default=1
            Minimum samples per leaf.

        Returns
        -------
        best_gain : float
            Best split gain.
        best_threshold : float
            Best split threshold.
        left_value : float
            Predicted value for left child.
        right_value : float
            Predicted value for right child.
        """
        if self.bin_edges_ is None:
            raise RuntimeError("HistogramBuilder has not been fitted yet.")

        edges = self.bin_edges_[feature_idx]
        n_bins = len(histogram)

        # Compute totals
        G_total = sum(h.sum_gradients for h in histogram)
        H_total = sum(h.sum_hessians for h in histogram)
        n_total = sum(h.count for h in histogram)

        # Score for current node
        def compute_score(G: float, H: float) -> float:
            return (G ** 2) / (H + lambda_l2 + 1e-10)

        current_score = compute_score(G_total, H_total)

        best_gain = -np.inf
        best_bin_idx = -1
        best_left_value = 0.0
        best_right_value = 0.0

        # Scan through bins
        G_left = 0.0
        H_left = 0.0
        n_left = 0

        for bin_idx in range(n_bins - 1):
            G_left += histogram[bin_idx].sum_gradients
            H_left += histogram[bin_idx].sum_hessians
            n_left += histogram[bin_idx].count

            G_right = G_total - G_left
            H_right = H_total - H_left
            n_right = n_total - n_left

            # Check constraints (samples and hessian)
            if n_left < min_samples_leaf or n_right < min_samples_leaf:
                continue
            if H_left < min_sum_hessian_in_leaf or H_right < min_sum_hessian_in_leaf:
                continue

            # Compute gain
            left_score = compute_score(G_left, H_left)
            right_score = compute_score(G_right, H_right)
            gain = left_score + right_score - current_score

            if gain > best_gain:
                best_gain = gain
                best_bin_idx = bin_idx
                H_left_safe = max(H_left, min_sum_hessian_in_leaf)
                H_right_safe = max(H_right, min_sum_hessian_in_leaf)
                best_left_value = -G_left / (H_left_safe + lambda_l2 + 1e-10)
                best_right_value = -G_right / (H_right_safe + lambda_l2 + 1e-10)

        # Convert bin index to threshold
        if best_bin_idx >= 0 and best_bin_idx + 1 < len(edges):
            best_threshold = edges[best_bin_idx + 1]
        else:
            best_threshold = 0.0

        return best_gain, best_threshold, best_left_value, best_right_value


def histogram_split(
    X: np.ndarray,
    gradients: np.ndarray,
    hessians: np.ndarray,
    *,
    max_bins: int = 255,
    lambda_l2: float = 0.0,
    min_samples_leaf: int = 1,
    min_sum_hessian_in_leaf: float = 1e-3,
) -> Tuple[int, float, float]:
    """
    Find the best split using histogram-based algorithm.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Feature matrix.
    gradients : np.ndarray of shape (n_samples,)
        Gradients.
    hessians : np.ndarray of shape (n_samples,)
        Hessians.
    max_bins : int, default=255
        Maximum bins per feature.
    lambda_l2 : float, default=0.0
        L2 regularization.
    min_samples_leaf : int, default=1
        Minimum samples per leaf.

    Returns
    -------
    best_feature : int
        Best feature index.
    best_threshold : float
        Best threshold.
    best_gain : float
        Best gain.
    """
    builder = HistogramBuilder(max_bins=max_bins)
    X_binned = builder.fit_transform(X)

    n_features = X.shape[1]
    indices = np.arange(len(X))

    best_gain = -np.inf
    best_feature = 0
    best_threshold = 0.0

    for feature_idx in range(n_features):
        histogram = builder.build_histogram(
            X_binned, gradients, hessians, indices, feature_idx
        )
        gain, threshold, _, _ = builder.find_best_split_from_histogram(
            histogram, feature_idx, lambda_l2, min_samples_leaf, min_sum_hessian_in_leaf
        )

        if gain > best_gain:
            best_gain = gain
            best_feature = feature_idx
            best_threshold = threshold

    return best_feature, best_threshold, best_gain


__all__ = ['HistogramBuilder', 'HistogramBin', 'histogram_split']
