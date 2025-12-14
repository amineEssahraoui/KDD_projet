"""
Exclusive Feature Bundling (EFB) implementation.

EFB is a technique used in LightGBM to bundle mutually exclusive features
together, reducing dimensionality while preserving information.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


class FeatureBundler:
    """
    Exclusive Feature Bundling for dimensionality reduction.

    EFB identifies features that are mutually exclusive (rarely non-zero
    together) and bundles them into a single feature. This reduces the
    number of features while preserving information.

    Parameters
    ----------
    max_conflict_rate : float, default=0.0
        Maximum fraction of samples where bundled features can conflict
        (i.e., both be non-zero). 0.0 means strictly mutually exclusive.
    max_bundles : int or None, default=None
        Maximum number of bundles to create. If None, no limit.

    Attributes
    ----------
    bundles_ : list of list of int
        Each inner list contains feature indices in that bundle.
    bundle_offsets_ : list of np.ndarray
        Offset values used to combine features in each bundle.
    n_original_features_ : int
        Number of features before bundling.
    n_bundled_features_ : int
        Number of features after bundling.

    References
    ----------
    Ke, G., et al. "LightGBM: A Highly Efficient Gradient Boosting
    Decision Tree." NeurIPS 2017.
    """

    def __init__(
        self,
        max_conflict_rate: float = 0.0,
        max_bundles: Optional[int] = None,
    ):
        if not 0 <= max_conflict_rate <= 1:
            raise ValueError(
                f"max_conflict_rate must be in [0, 1], got {max_conflict_rate}"
            )

        self.max_conflict_rate = max_conflict_rate
        self.max_bundles = max_bundles

        # Fitted state
        self.bundles_: Optional[List[List[int]]] = None
        self.bundle_offsets_: Optional[List[np.ndarray]] = None
        self.n_original_features_: Optional[int] = None
        self.n_bundled_features_: Optional[int] = None
        self._feature_to_bundle_: Optional[Dict[int, int]] = None

    def fit(self, X: np.ndarray) -> "FeatureBundler":
        """
        Identify feature bundles from the training data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        self : FeatureBundler
            Fitted bundler.
        """
        n_samples, n_features = X.shape
        self.n_original_features_ = n_features

        # Compute conflict matrix
        # conflict[i, j] = fraction of samples where both features i and j are non-zero
        non_zero = (X != 0).astype(float)
        conflict_counts = non_zero.T @ non_zero  # (n_features, n_features)
        conflict_rate = conflict_counts / n_samples

        # Greedy bundling
        used = np.zeros(n_features, dtype=bool)
        self.bundles_ = []
        self._feature_to_bundle_ = {}

        for i in range(n_features):
            if used[i]:
                continue

            # Start new bundle with feature i
            bundle = [i]
            used[i] = True
            self._feature_to_bundle_[i] = len(self.bundles_)

            # Try to add other features to this bundle
            for j in range(i + 1, n_features):
                if used[j]:
                    continue

                # Check if j conflicts with any feature in the bundle
                can_add = True
                for k in bundle:
                    if conflict_rate[j, k] > self.max_conflict_rate:
                        can_add = False
                        break

                if can_add:
                    bundle.append(j)
                    used[j] = True
                    self._feature_to_bundle_[j] = len(self.bundles_)

            self.bundles_.append(bundle)

            # Check max bundles limit
            if self.max_bundles and len(self.bundles_) >= self.max_bundles:
                # Add remaining features as singleton bundles
                for j in range(n_features):
                    if not used[j]:
                        self.bundles_.append([j])
                        self._feature_to_bundle_[j] = len(self.bundles_) - 1
                        used[j] = True
                break

        # Compute offsets for merging features in each bundle
        self.bundle_offsets_ = []
        for bundle in self.bundles_:
            if len(bundle) == 1:
                self.bundle_offsets_.append(np.array([0.0]))
            else:
                # Each feature in bundle gets an offset based on max value of previous features
                offsets = np.zeros(len(bundle))
                for idx in range(1, len(bundle)):
                    prev_feature = bundle[idx - 1]
                    max_val = np.nanmax(np.abs(X[:, prev_feature]))
                    if np.isnan(max_val) or np.isinf(max_val):
                        max_val = 1.0
                    offsets[idx] = offsets[idx - 1] + max_val + 1

                self.bundle_offsets_.append(offsets)

        self.n_bundled_features_ = len(self.bundles_)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform features by bundling.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Features to transform.

        Returns
        -------
        X_bundled : np.ndarray of shape (n_samples, n_bundles)
            Bundled features.
        """
        if self.bundles_ is None:
            raise RuntimeError("FeatureBundler has not been fitted yet.")

        n_samples = X.shape[0]
        X_bundled = np.zeros((n_samples, len(self.bundles_)))

        for bundle_idx, bundle in enumerate(self.bundles_):
            offsets = self.bundle_offsets_[bundle_idx]

            for feat_idx, feature in enumerate(bundle):
                feature_values = X[:, feature]
                # Add offset to non-zero values
                non_zero_mask = feature_values != 0
                X_bundled[non_zero_mask, bundle_idx] += (
                    feature_values[non_zero_mask] + offsets[feat_idx]
                )

        return X_bundled

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)

    def get_bundle_info(self) -> Dict[str, any]:
        """
        Get information about the bundles.

        Returns
        -------
        info : dict
            Dictionary with bundling information.
        """
        if self.bundles_ is None:
            return {"fitted": False}

        return {
            "fitted": True,
            "n_original_features": self.n_original_features_,
            "n_bundled_features": self.n_bundled_features_,
            "compression_ratio": self.n_original_features_ / self.n_bundled_features_,
            "bundles": self.bundles_,
            "bundle_sizes": [len(b) for b in self.bundles_],
        }

    def __repr__(self) -> str:
        status = "fitted" if self.bundles_ is not None else "not fitted"
        return (
            f"FeatureBundler(max_conflict_rate={self.max_conflict_rate}, "
            f"max_bundles={self.max_bundles}, status={status})"
        )


def bundle_features(
    X: np.ndarray,
    *,
    max_conflict_rate: float = 0.0,
) -> Tuple[np.ndarray, FeatureBundler]:
    """
    Convenience function to bundle features.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Feature matrix.
    max_conflict_rate : float, default=0.0
        Maximum allowed conflict rate for bundling.

    Returns
    -------
    X_bundled : np.ndarray
        Bundled feature matrix.
    bundler : FeatureBundler
        Fitted bundler for transforming new data.
    """
    bundler = FeatureBundler(max_conflict_rate=max_conflict_rate)
    X_bundled = bundler.fit_transform(X)
    return X_bundled, bundler


__all__ = ['FeatureBundler', 'bundle_features']
