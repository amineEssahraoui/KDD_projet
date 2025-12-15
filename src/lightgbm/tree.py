"""
Decision Tree implementation for gradient boosting.

This module provides a custom DecisionTree that supports gradient and hessian
inputs, leaf-wise growth (like LightGBM), histogram binning, and regularization.
Completely sklearn-free.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Node Data Structure

@dataclass
class TreeNode:
    """
    Represents a node in the decision tree.
        G_left_base = G_left_cumsum[valid_idx]
        H_left_base = H_left_cumsum[valid_idx]

        # Consider two policies for NaNs: assign them to left OR to right.
        # Compute both gains and pick the better option per split position.

        # Case A: NaNs go to left
        G_left_a = G_left_base + nan_grad
        H_left_a = H_left_base + nan_hess
        G_right_a = G_total - G_left_a
        H_right_a = H_total - H_left_a

        # Case B: NaNs go to right
        G_left_b = G_left_base
        H_left_b = H_left_base
        G_right_b = G_total - G_left_b
        H_right_b = H_total - H_left_b

        # Check hessian constraints for both assignments
        hessian_ok_a = (H_left_a >= self.min_sum_hessian_in_leaf) & (
            H_right_a >= self.min_sum_hessian_in_leaf
        )
        hessian_ok_b = (H_left_b >= self.min_sum_hessian_in_leaf) & (
            H_right_b >= self.min_sum_hessian_in_leaf
        )

        # Also enforce min_samples_leaf depending on NaN assignment
        n_left_base = n_left_cumsum[valid_idx]
        n_right_base = len(indices) - n_left_base - nan_count
        samples_ok_a = (n_left_base + nan_count >= self.min_samples_leaf) & (
            n_right_base >= self.min_samples_leaf
        )
        samples_ok_b = (n_left_base >= self.min_samples_leaf) & (
            (n_right_base + nan_count) >= self.min_samples_leaf
        )

        # Valid splits are those where at least one assignment meets both constraints
        hessian_ok = (hessian_ok_a & samples_ok_a) | (hessian_ok_b & samples_ok_b)

        if not np.any(valid_splits):
            return best_split

        valid_idx = np.where(valid_splits)[0]

        # Keep only splits where at least one NaN assignment is valid
        valid_idx = valid_idx[hessian_ok]
        if len(valid_idx) == 0:
            return best_split

        # For the remaining valid splits, choose the best assignment per split
        G_left_a = G_left_a[hessian_ok]
        H_left_a = H_left_a[hessian_ok]
        G_right_a = G_right_a[hessian_ok]
        H_right_a = H_right_a[hessian_ok]

        G_left_b = G_left_b[hessian_ok]
        H_left_b = H_left_b[hessian_ok]
        G_right_b = G_right_b[hessian_ok]
        H_right_b = H_right_b[hessian_ok]

        # Compute scores (apply L1 soft-thresholding if needed later)
        if self.lambda_l1 > 0:
            def reg_G(G):
                return np.where(
                    G > self.lambda_l1, G - self.lambda_l1,
                    np.where(G < -self.lambda_l1, G + self.lambda_l1, 0.0)
                )
            G_left_a_reg = reg_G(G_left_a)
            G_right_a_reg = reg_G(G_right_a)
            G_left_b_reg = reg_G(G_left_b)
            G_right_b_reg = reg_G(G_right_b)

            left_scores_a = (G_left_a_reg ** 2) / (np.maximum(H_left_a, self.min_sum_hessian_in_leaf) + self.lambda_l2)
            right_scores_a = (G_right_a_reg ** 2) / (np.maximum(H_right_a, self.min_sum_hessian_in_leaf) + self.lambda_l2)

            left_scores_b = (G_left_b_reg ** 2) / (np.maximum(H_left_b, self.min_sum_hessian_in_leaf) + self.lambda_l2)
            right_scores_b = (G_right_b_reg ** 2) / (np.maximum(H_right_b, self.min_sum_hessian_in_leaf) + self.lambda_l2)
        else:
            left_scores_a = (G_left_a ** 2) / (np.maximum(H_left_a, self.min_sum_hessian_in_leaf) + self.lambda_l2)
            right_scores_a = (G_right_a ** 2) / (np.maximum(H_right_a, self.min_sum_hessian_in_leaf) + self.lambda_l2)

            left_scores_b = (G_left_b ** 2) / (np.maximum(H_left_b, self.min_sum_hessian_in_leaf) + self.lambda_l2)
            right_scores_b = (G_right_b ** 2) / (np.maximum(H_right_b, self.min_sum_hessian_in_leaf) + self.lambda_l2)

        gains_a = left_scores_a + right_scores_a - current_score
        gains_b = left_scores_b + right_scores_b - current_score

        # Choose assignment with larger gain per split
        choose_a = gains_a >= gains_b
        gains = np.where(choose_a, gains_a, gains_b)

        # Keep values for the chosen assignment
        G_left_valid = np.where(choose_a, G_left_a, G_left_b)
        H_left_valid = np.where(choose_a, H_left_a, H_left_b)
        G_right_valid = np.where(choose_a, G_right_a, G_right_b)
        H_right_valid = np.where(choose_a, H_right_a, H_right_b)
        Depth of this node in the tree.
    gain : float
        Information gain from the split (internal nodes only).
    """
    is_leaf: bool = True
    value: float = 0.0
    feature_idx: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["TreeNode"] = None
    right: Optional["TreeNode"] = None
    n_samples: int = 0
    depth: int = 0
    gain: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize node to dictionary."""
        result = {
            'is_leaf': self.is_leaf,
            'value': self.value,
            'feature_idx': self.feature_idx,
            'threshold': self.threshold,
            'n_samples': self.n_samples,
            'depth': self.depth,
            'gain': self.gain,
        }
        if self.left is not None:
            result['left'] = self.left.to_dict()
        if self.right is not None:
            result['right'] = self.right.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TreeNode":
        """Deserialize node from dictionary."""
        node = cls(
            is_leaf=data['is_leaf'],
            value=data['value'],
            feature_idx=data.get('feature_idx'),
            threshold=data.get('threshold'),
            n_samples=data.get('n_samples', 0),
            depth=data.get('depth', 0),
            gain=data.get('gain', 0.0),
        )
        if 'left' in data and data['left'] is not None:
            node.left = cls.from_dict(data['left'])
        if 'right' in data and data['right'] is not None:
            node.right = cls.from_dict(data['right'])
        return node


@dataclass
class SplitInfo:
    """
    Information about a potential split.

    Attributes
    ----------
    gain : float
        Information gain from the split.
    feature_idx : int
        Feature index to split on.
    threshold : float
        Split threshold value.
    left_indices : np.ndarray
        Sample indices going to the left child.
    right_indices : np.ndarray
        Sample indices going to the right child.
    left_value : float
        Prediction value for left child.
    right_value : float
        Prediction value for right child.
    """
    gain: float = -np.inf
    feature_idx: int = 0
    threshold: float = 0.0
    left_indices: Optional[np.ndarray] = None
    right_indices: Optional[np.ndarray] = None
    left_value: float = 0.0
    right_value: float = 0.0

# Decision Tree Clas

class DecisionTree:
    """
    Decision tree for gradient boosting.

    This implementation supports:
    - Leaf-wise (best-first) tree growth like LightGBM
    - Gradient and hessian inputs for split optimization
    - L1 and L2 regularization
    - Feature subsampling
    - Histogram-based split finding

    Parameters
    ----------
    max_depth : int, default=-1
        Maximum depth of the tree. -1 means unlimited.
    min_samples_leaf : int, default=20
        Minimum number of samples required in a leaf.
    num_leaves : int, default=31
        Maximum number of leaves in the tree.
    lambda_l1 : float, default=0.0
        L1 regularization coefficient.
    lambda_l2 : float, default=0.0
        L2 regularization coefficient.
    min_gain_to_split : float, default=0.0
        Minimum gain required to make a split.
    feature_fraction : float, default=1.0
        Fraction of features to consider for each tree.
    use_histogram : bool, default=False
        Whether to use histogram-based split finding.
    max_bins : int, default=255
        Maximum number of bins for histogram.
    random_state : int or None, default=None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        max_depth: int = -1,
        min_samples_leaf: int = 20,
        num_leaves: int = 31,
        lambda_l1: float = 0.0,
        lambda_l2: float = 0.0,
        min_gain_to_split: float = 0.0,
        feature_fraction: float = 1.0,
        use_histogram: bool = False,
        max_bins: int = 255,
        min_sum_hessian_in_leaf: float = 1e-3,
        random_state: Optional[int] = None,
    ):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.num_leaves = num_leaves
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.min_gain_to_split = min_gain_to_split
        self.feature_fraction = feature_fraction
        self.use_histogram = use_histogram
        self.max_bins = max_bins
        self.min_sum_hessian_in_leaf = min_sum_hessian_in_leaf
        self.random_state = random_state

        # Tree state
        self.root_: Optional[TreeNode] = None
        self.n_features_: Optional[int] = None
        self.n_leaves_: int = 0
        self.max_depth_reached_: int = 0

        # Histogram bins (computed during fit)
        self._bin_edges_: Optional[List[np.ndarray]] = None
        self._rng: np.random.Generator = np.random.default_rng(random_state)

    def fit(
        self,
        X: np.ndarray,
        gradients: np.ndarray,
        hessians: np.ndarray,
        *,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "DecisionTree":
        """
        Fit the decision tree to gradients and hessians.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix.
        gradients : np.ndarray of shape (n_samples,)
            First-order gradients.
        hessians : np.ndarray of shape (n_samples,)
            Second-order hessians.
        sample_weight : np.ndarray of shape (n_samples,) or None
            Sample weights.

        Returns
        -------
        self : DecisionTree
            Fitted tree.
        """
        n_samples, n_features = X.shape
        self.n_features_ = n_features

        # Apply sample weights
        if sample_weight is not None:
            gradients = gradients * sample_weight
            hessians = hessians * sample_weight

        # Compute histogram bins if needed
        if self.use_histogram:
            self._compute_bin_edges(X)

        # Select features to consider
        if self.feature_fraction < 1.0:
            n_selected = max(1, int(n_features * self.feature_fraction))
            self._selected_features = self._rng.choice(
                n_features, size=n_selected, replace=False
            )
        else:
            self._selected_features = np.arange(n_features)

        # Initialize indices
        indices = np.arange(n_samples)

        # Build tree using leaf-wise growth
        self._build_tree_leaf_wise(X, gradients, hessians, indices)

        return self

    def _build_tree_leaf_wise(
        self,
        X: np.ndarray,
        gradients: np.ndarray,
        hessians: np.ndarray,
        indices: np.ndarray,
    ) -> None:
        """
        Build tree using leaf-wise (best-first) growth strategy.

        This mimics LightGBM's approach of always splitting the leaf
        with the highest potential gain.
        """
        # Compute initial leaf value
        init_value = self._compute_leaf_value(
            gradients[indices], hessians[indices]
        )

        # Create root node
        self.root_ = TreeNode(
            is_leaf=True,
            value=init_value,
            n_samples=len(indices),
            depth=0,
        )
        self.n_leaves_ = 1
        self.max_depth_reached_ = 0

        # Priority queue: (-gain, node, indices)
        # Using negative gain for max-heap behavior
        split_candidates: List[Tuple[float, int, TreeNode, np.ndarray]] = []
        node_counter = 0

        # Find best split for root
        best_split = self._find_best_split(X, gradients, hessians, indices, depth=0)
        if best_split.gain > self.min_gain_to_split:
            heapq.heappush(
                split_candidates,
                (-best_split.gain, node_counter, self.root_, indices, best_split)
            )
            node_counter += 1

        # Grow tree leaf by leaf
        while split_candidates and self.n_leaves_ < self.num_leaves:
            _, _, node, node_indices, split_info = heapq.heappop(split_candidates)

            # Check depth constraint
            if self.max_depth != -1 and node.depth >= self.max_depth:
                continue

            # Check if we can add two more leaves
            if self.n_leaves_ + 1 > self.num_leaves:
                break

            # Apply the split
            node.is_leaf = False
            node.feature_idx = split_info.feature_idx
            node.threshold = split_info.threshold
            node.gain = split_info.gain

            # Create child nodes
            left_indices = split_info.left_indices
            right_indices = split_info.right_indices

            node.left = TreeNode(
                is_leaf=True,
                value=split_info.left_value,
                n_samples=len(left_indices),
                depth=node.depth + 1,
            )
            node.right = TreeNode(
                is_leaf=True,
                value=split_info.right_value,
                n_samples=len(right_indices),
                depth=node.depth + 1,
            )

            # Update counters
            self.n_leaves_ += 1  # Net gain is 1 (2 new - 1 split)
            self.max_depth_reached_ = max(
                self.max_depth_reached_, node.depth + 1
            )

            # Find best splits for new leaves
            for child, child_indices in [
                (node.left, left_indices),
                (node.right, right_indices),
            ]:
                if len(child_indices) >= 2 * self.min_samples_leaf:
                    child_split = self._find_best_split(
                        X, gradients, hessians, child_indices, depth=child.depth
                    )
                    if child_split.gain > self.min_gain_to_split:
                        heapq.heappush(
                            split_candidates,
                            (-child_split.gain, node_counter, child, child_indices, child_split)
                        )
                        node_counter += 1

    def _find_best_split(
        self,
        X: np.ndarray,
        gradients: np.ndarray,
        hessians: np.ndarray,
        indices: np.ndarray,
        depth: int,
    ) -> SplitInfo:
        """
        Find the best split for a set of samples.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        gradients : np.ndarray
            Gradients for all samples.
        hessians : np.ndarray
            Hessians for all samples.
        indices : np.ndarray
            Indices of samples in this node.
        depth : int
            Current depth.

        Returns
        -------
        split_info : SplitInfo
            Information about the best split found.
        """
        best_split = SplitInfo()

        # Check depth constraint
        if self.max_depth != -1 and depth >= self.max_depth:
            return best_split

        # Get gradients and hessians for this node
        node_gradients = gradients[indices]
        node_hessians = hessians[indices]

        # Sum of gradients and hessians
        G_total = np.sum(node_gradients)
        H_total = np.sum(node_hessians)

        # Current score (for gain calculation)
        current_score = self._compute_score(G_total, H_total)

        # Try each selected feature
        for feature_idx in self._selected_features:
            feature_values = X[indices, feature_idx]

            if self.use_histogram:
                split_info = self._find_best_split_histogram(
                    feature_values, node_gradients, node_hessians,
                    indices, feature_idx, G_total, H_total, current_score
                )
            else:
                split_info = self._find_best_split_exact(
                    feature_values, node_gradients, node_hessians,
                    indices, feature_idx, G_total, H_total, current_score
                )

            if split_info.gain > best_split.gain:
                best_split = split_info

        return best_split

    def _find_best_split_exact(
        self,
        feature_values: np.ndarray,
        gradients: np.ndarray,
        hessians: np.ndarray,
        indices: np.ndarray,
        feature_idx: int,
        G_total: float,
        H_total: float,
        current_score: float,
    ) -> SplitInfo:
        """
        Find best split using exact algorithm (optimized vectorized version).
        """
        best_split = SplitInfo()
        best_split.feature_idx = feature_idx

        n_samples = len(indices)
        if n_samples < 2 * self.min_samples_leaf:
            return best_split

        # Sort by feature value
        sorted_order = np.argsort(feature_values)
        sorted_values = feature_values[sorted_order]
        sorted_gradients = gradients[sorted_order]
        sorted_hessians = hessians[sorted_order]
        sorted_indices = indices[sorted_order]

        # Cumulative sums (vectorized)
        G_left_cumsum = np.cumsum(sorted_gradients)
        H_left_cumsum = np.cumsum(sorted_hessians)

        # Right sums
        G_right = G_total - G_left_cumsum
        H_right = H_total - H_left_cumsum

        # Valid split positions: where value changes AND min_samples_leaf constraint is met
        value_changes = sorted_values[:-1] != sorted_values[1:]
        n_left = np.arange(1, n_samples)
        n_right = n_samples - n_left
        min_samples_ok = (n_left >= self.min_samples_leaf) & (n_right >= self.min_samples_leaf)
        valid_splits = value_changes & min_samples_ok

        if not np.any(valid_splits):
            return best_split

        # Get valid indices
        valid_idx = np.where(valid_splits)[0]

        # Compute scores for valid splits (vectorized)
        G_left_valid = G_left_cumsum[valid_idx]
        H_left_valid = H_left_cumsum[valid_idx]
        G_right_valid = G_right[valid_idx]
        H_right_valid = H_right[valid_idx]

        # Enforce minimum hessian per child
        hessian_ok = (
            (H_left_valid >= self.min_sum_hessian_in_leaf)
            & (H_right_valid >= self.min_sum_hessian_in_leaf)
        )
        if not np.any(hessian_ok):
            return best_split

        G_left_valid = G_left_valid[hessian_ok]
        H_left_valid = H_left_valid[hessian_ok]
        G_right_valid = G_right_valid[hessian_ok]
        H_right_valid = H_right_valid[hessian_ok]
        valid_idx = valid_idx[hessian_ok]

        # Vectorized score computation with L1 regularization
        if self.lambda_l1 > 0:
            G_left_reg = np.where(
                G_left_valid > self.lambda_l1, G_left_valid - self.lambda_l1,
                np.where(G_left_valid < -self.lambda_l1, G_left_valid + self.lambda_l1, 0.0)
            )
            G_right_reg = np.where(
                G_right_valid > self.lambda_l1, G_right_valid - self.lambda_l1,
                np.where(G_right_valid < -self.lambda_l1, G_right_valid + self.lambda_l1, 0.0)
            )
            left_scores = (G_left_reg ** 2) / (np.maximum(H_left_valid, self.min_sum_hessian_in_leaf) + self.lambda_l2)
            right_scores = (G_right_reg ** 2) / (np.maximum(H_right_valid, self.min_sum_hessian_in_leaf) + self.lambda_l2)
        else:
            left_scores = (G_left_valid ** 2) / (np.maximum(H_left_valid, self.min_sum_hessian_in_leaf) + self.lambda_l2)
            right_scores = (G_right_valid ** 2) / (np.maximum(H_right_valid, self.min_sum_hessian_in_leaf) + self.lambda_l2)

        gains = left_scores + right_scores - current_score

        # Find best split
        best_local_idx = np.argmax(gains)
        best_gain = gains[best_local_idx]

        if best_gain > best_split.gain:
            split_pos = valid_idx[best_local_idx]
            n_left = split_pos + 1

            best_split.gain = best_gain
            best_split.threshold = (sorted_values[split_pos] + sorted_values[split_pos + 1]) / 2
            best_split.left_indices = sorted_indices[:n_left]
            best_split.right_indices = sorted_indices[n_left:]
            best_split.left_value = self._compute_leaf_value(
                sorted_gradients[:n_left], sorted_hessians[:n_left]
            )
            best_split.right_value = self._compute_leaf_value(
                sorted_gradients[n_left:], sorted_hessians[n_left:]
            )

        return best_split

        return best_split

    def _find_best_split_histogram(
        self,
        feature_values: np.ndarray,
        gradients: np.ndarray,
        hessians: np.ndarray,
        indices: np.ndarray,
        feature_idx: int,
        G_total: float,
        H_total: float,
        current_score: float,
    ) -> SplitInfo:
        """
        Find best split using histogram-based algorithm (optimized).
        """
        best_split = SplitInfo()
        best_split.feature_idx = feature_idx

        if self._bin_edges_ is None:
            return best_split

        bin_edges = self._bin_edges_[feature_idx]
        n_bins = len(bin_edges) - 1

        if n_bins < 2:
            return best_split

        # Compute histogram (vectorized using bincount)
        valid_mask = ~np.isnan(feature_values)
        valid_values = feature_values[valid_mask]
        if len(valid_values) < 2:
            return best_split

        bin_indices = np.digitize(valid_values, bin_edges[1:-1])
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        # Use numpy bincount for fast histogram on valid (non-NaN) entries
        G_bins = np.bincount(bin_indices, weights=gradients[valid_mask], minlength=n_bins)
        H_bins = np.bincount(bin_indices, weights=hessians[valid_mask], minlength=n_bins)
        # Smooth tiny bin hessians to avoid zero curvature bins
        H_bins = H_bins + 1e-6
        count_bins = np.bincount(bin_indices, minlength=n_bins)

        nan_grad = float(np.sum(gradients[~valid_mask]))
        nan_hess = float(np.sum(hessians[~valid_mask]))
        nan_count = int(np.sum(~valid_mask))

        # Cumulative sums (vectorized)
        G_left_cumsum = np.cumsum(G_bins[:-1])
        H_left_cumsum = np.cumsum(H_bins[:-1])
        n_left_cumsum = np.cumsum(count_bins[:-1])

        # Right sums
        G_right = G_total - G_left_cumsum
        H_right = H_total - H_left_cumsum
        n_right = len(indices) - n_left_cumsum - nan_count

        # Valid splits based on min_samples_leaf
        valid_splits = (
            (n_left_cumsum + nan_count >= self.min_samples_leaf)
            & (n_right >= self.min_samples_leaf)
        )

        if not np.any(valid_splits):
            return best_split

        valid_idx = np.where(valid_splits)[0]

        # Vectorized score computation
        G_left_valid = G_left_cumsum[valid_idx] + nan_grad
        H_left_valid = H_left_cumsum[valid_idx] + nan_hess
        G_right_valid = G_total - G_left_valid
        H_right_valid = H_total - H_left_valid

        hessian_ok = (
            (H_left_valid >= self.min_sum_hessian_in_leaf)
            & (H_right_valid >= self.min_sum_hessian_in_leaf)
        )
        if not np.any(hessian_ok):
            return best_split

        G_left_valid = G_left_valid[hessian_ok]
        H_left_valid = H_left_valid[hessian_ok]
        G_right_valid = G_right_valid[hessian_ok]
        H_right_valid = H_right_valid[hessian_ok]
        valid_idx = valid_idx[hessian_ok]

        if self.lambda_l1 > 0:
            G_left_reg = np.where(
                G_left_valid > self.lambda_l1, G_left_valid - self.lambda_l1,
                np.where(G_left_valid < -self.lambda_l1, G_left_valid + self.lambda_l1, 0.0)
            )
            G_right_reg = np.where(
                G_right_valid > self.lambda_l1, G_right_valid - self.lambda_l1,
                np.where(G_right_valid < -self.lambda_l1, G_right_valid + self.lambda_l1, 0.0)
            )
            left_scores = (G_left_reg ** 2) / (np.maximum(H_left_valid, self.min_sum_hessian_in_leaf) + self.lambda_l2)
            right_scores = (G_right_reg ** 2) / (np.maximum(H_right_valid, self.min_sum_hessian_in_leaf) + self.lambda_l2)
        else:
            left_scores = (G_left_valid ** 2) / (np.maximum(H_left_valid, self.min_sum_hessian_in_leaf) + self.lambda_l2)
            right_scores = (G_right_valid ** 2) / (np.maximum(H_right_valid, self.min_sum_hessian_in_leaf) + self.lambda_l2)

        gains = left_scores + right_scores - current_score

        # Find best split
        best_local_idx = np.argmax(gains)
        best_gain = gains[best_local_idx]

        if best_gain > best_split.gain:
            bin_idx = valid_idx[best_local_idx]
            threshold = bin_edges[bin_idx + 1]
            left_mask = np.isnan(feature_values) | (feature_values <= threshold)
            right_mask = ~left_mask

            best_split.gain = best_gain
            best_split.threshold = threshold
            best_split.left_indices = indices[left_mask]
            best_split.right_indices = indices[right_mask]
            best_split.left_value = self._compute_leaf_value(
                gradients[left_mask], hessians[left_mask]
            )
            best_split.right_value = self._compute_leaf_value(
                gradients[right_mask], hessians[right_mask]
            )

        return best_split

    def _compute_bin_edges(self, X: np.ndarray) -> None:
        """Compute histogram bin edges for each feature."""
        n_features = X.shape[1]
        self._bin_edges_ = []

        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
            # Remove NaN for binning
            valid_values = feature_values[~np.isnan(feature_values)]

            if len(valid_values) == 0:
                # All NaN, use dummy bins
                self._bin_edges_.append(np.array([-np.inf, np.inf]))
            else:
                # Use quantile-based binning
                n_bins = min(self.max_bins, len(np.unique(valid_values)))
                percentiles = np.linspace(0, 100, n_bins + 1)
                edges = np.percentile(valid_values, percentiles)
                # Remove duplicates
                edges = np.unique(edges)
                if len(edges) < 2:
                    edges = np.array([valid_values.min(), valid_values.max()])
                self._bin_edges_.append(edges)

    def _compute_score(self, G: float, H: float) -> float:
        """
        Compute the score for a split.

        Score = G^2 / (H + lambda_l2) with L1 regularization.
        """
        # Apply L1 regularization (soft thresholding)
        if self.lambda_l1 > 0:
            if G > self.lambda_l1:
                G = G - self.lambda_l1
            elif G < -self.lambda_l1:
                G = G + self.lambda_l1
            else:
                return 0.0

        H_safe = np.maximum(H, self.min_sum_hessian_in_leaf)
        return (G ** 2) / (H_safe + self.lambda_l2 + 1e-10)

    def _compute_leaf_value(
        self, gradients: np.ndarray, hessians: np.ndarray
    ) -> float:
        """
        Compute the optimal leaf value.

        value = -G / (H + lambda_l2) with L1 regularization.
        """
        G = np.sum(gradients)
        H = np.sum(hessians)

        H_safe = np.maximum(H, self.min_sum_hessian_in_leaf)

        # Apply L1 regularization
        if self.lambda_l1 > 0:
            if G > self.lambda_l1:
                G = G - self.lambda_l1
            elif G < -self.lambda_l1:
                G = G + self.lambda_l1
            else:
                return 0.0

        value = -G / (H_safe + self.lambda_l2 + 1e-10)
        return value

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions for input samples.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix.

        Returns
        -------
        predictions : np.ndarray of shape (n_samples,)
            Predicted values.
        """
        if self.root_ is None:
            raise RuntimeError("Tree has not been fitted yet.")

        n_samples = X.shape[0]
        predictions = np.zeros(n_samples)

        for i in range(n_samples):
            predictions[i] = self._predict_sample(X[i])

        return predictions

    def _predict_sample(self, x: np.ndarray) -> float:
        """Predict for a single sample."""
        node = self.root_

        while not node.is_leaf:
            value = x[node.feature_idx]

            # Handle NaN (go left by default)
            if np.isnan(value):
                node = node.left
            elif value <= node.threshold:
                node = node.left
            else:
                node = node.right

        return node.value

    def to_dict(self) -> Dict[str, Any]:
        """Serialize tree to dictionary."""
        return {
            'max_depth': self.max_depth,
            'min_samples_leaf': self.min_samples_leaf,
            'num_leaves': self.num_leaves,
            'lambda_l1': self.lambda_l1,
            'lambda_l2': self.lambda_l2,
            'min_gain_to_split': self.min_gain_to_split,
            'feature_fraction': self.feature_fraction,
            'use_histogram': self.use_histogram,
            'max_bins': self.max_bins,
            'min_sum_hessian_in_leaf': self.min_sum_hessian_in_leaf,
            'n_features_': self.n_features_,
            'n_leaves_': self.n_leaves_,
            'max_depth_reached_': self.max_depth_reached_,
            'root_': self.root_.to_dict() if self.root_ else None,
        }

    def from_dict(self, data: Dict[str, Any]) -> "DecisionTree":
        """Deserialize tree from dictionary."""
        self.max_depth = data['max_depth']
        self.min_samples_leaf = data['min_samples_leaf']
        self.num_leaves = data['num_leaves']
        self.lambda_l1 = data['lambda_l1']
        self.lambda_l2 = data['lambda_l2']
        self.min_gain_to_split = data['min_gain_to_split']
        self.feature_fraction = data.get('feature_fraction', 1.0)
        self.use_histogram = data.get('use_histogram', False)
        self.max_bins = data.get('max_bins', 255)
        self.min_sum_hessian_in_leaf = data.get('min_sum_hessian_in_leaf', 1e-3)
        self.n_features_ = data.get('n_features_')
        self.n_leaves_ = data.get('n_leaves_', 0)
        self.max_depth_reached_ = data.get('max_depth_reached_', 0)

        if data.get('root_'):
            self.root_ = TreeNode.from_dict(data['root_'])
        else:
            self.root_ = None

        return self

    @property
    def feature_importances_(self) -> np.ndarray:
        """
        Compute feature importances based on total gain.

        Returns
        -------
        importances : np.ndarray of shape (n_features,)
            Feature importance scores (normalized).
        """
        if self.root_ is None or self.n_features_ is None:
            raise RuntimeError("Tree has not been fitted yet.")

        importances = np.zeros(self.n_features_)
        self._accumulate_importance(self.root_, importances)

        # Normalize
        total = np.sum(importances)
        if total > 0:
            importances /= total

        return importances

    def _accumulate_importance(
        self, node: TreeNode, importances: np.ndarray
    ) -> None:
        """Recursively accumulate feature importances."""
        if node.is_leaf:
            return

        # Add gain for this split
        importances[node.feature_idx] += node.gain

        # Recurse
        if node.left:
            self._accumulate_importance(node.left, importances)
        if node.right:
            self._accumulate_importance(node.right, importances)

# Module Export

__all__ = ['DecisionTree', 'TreeNode', 'SplitInfo']
