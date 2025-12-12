"""Leaf-wise regression tree used by the simplified LightGBM booster."""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class Node:
	feature_index: Optional[int] = None
	threshold: Optional[float] = None
	left: Optional["Node"] = None
	right: Optional["Node"] = None
	value: float = 0.0
	is_leaf: bool = True

	def predict_one(self, x: np.ndarray) -> float:
		if self.is_leaf or self.feature_index is None or self.threshold is None:
			return self.value
		if x[self.feature_index] <= self.threshold:
			return self.left.predict_one(x)  # type: ignore[union-attr]
		return self.right.predict_one(x)  # type: ignore[union-attr]


class DecisionTree:
	"""Regression tree grown leaf-wise using second-order gain."""

	def __init__(
		self,
		max_depth: int,
		num_leaves: int,
		min_data_in_leaf: int,
		min_sum_hessian_in_leaf: float,
		lambda_l2: float,
		min_gain_to_split: float = 0.0,
		lambda_l1: float = 0.0,
		colsample: float = 1.0,
		random_state: Optional[int] = None,
		use_histogram: bool = False,
		monotone_constraints: Optional[np.ndarray] = None,
		categorical_features: Optional[set[int]] = None,
		default_left: bool = True,
	) -> None:
		self.max_depth = max_depth
		self.num_leaves = num_leaves
		self.min_data_in_leaf = min_data_in_leaf
		self.min_sum_hessian_in_leaf = min_sum_hessian_in_leaf
		self.lambda_l2 = lambda_l2
		self.lambda_l1 = lambda_l1
		self.min_gain_to_split = min_gain_to_split
		self.colsample = colsample
		self.random_state = random_state
		self.use_histogram = use_histogram
		self.monotone_constraints = monotone_constraints
		self.categorical_features = categorical_features or set()
		self.default_left = default_left
		self.root = Node()
		self.n_features_ = None
		self.feature_importances_ = None
		self.split_counts_ = None

	# ------------------------------------------------------------------
	def fit(self, X: np.ndarray, gradients: np.ndarray, hessians: np.ndarray) -> "DecisionTree":
		X = np.asarray(X, dtype=float)
		gradients = np.asarray(gradients, dtype=float)
		hessians = np.asarray(hessians, dtype=float)
		self.n_features_ = X.shape[1]

		# Priority queue holds (-gain, depth, Node, indices)
		indices = np.arange(len(X))
		self.root.value = self._leaf_value(gradients, hessians, indices)
		pq = []
		root_gain, root_split = self._best_split(X, gradients, hessians, indices)
		heapq.heappush(pq, (-root_gain, 0, self.root, indices, root_split))
		gain_per_feature = np.zeros(self.n_features_)
		split_counts = np.zeros(self.n_features_)
		leaves = 1

		while pq and leaves < self.num_leaves:
			neg_gain, depth, node, node_indices, split = heapq.heappop(pq)
			gain = -neg_gain
			if gain <= self.min_gain_to_split or split is None or depth >= self.max_depth:
				node.is_leaf = True
				continue

			(feat_idx, threshold, left_idx, right_idx) = split
			node.feature_index = feat_idx
			node.threshold = threshold
			node.is_leaf = False
			gain_per_feature[feat_idx] += gain
			split_counts[feat_idx] += 1

			left_node = Node()
			right_node = Node()
			left_node.value = self._leaf_value(gradients, hessians, left_idx)
			right_node.value = self._leaf_value(gradients, hessians, right_idx)
			node.left = left_node
			node.right = right_node
			leaves += 1

			if depth + 1 < self.max_depth:
				left_gain, left_split = self._best_split(X, gradients, hessians, left_idx)
				if left_split is not None:
					heapq.heappush(pq, (-left_gain, depth + 1, left_node, left_idx, left_split))
				right_gain, right_split = self._best_split(X, gradients, hessians, right_idx)
				if right_split is not None:
					heapq.heappush(pq, (-right_gain, depth + 1, right_node, right_idx, right_split))

		self.feature_importances_ = gain_per_feature
		self.split_counts_ = split_counts
		return self

	# ------------------------------------------------------------------
	def _leaf_value(self, gradients: np.ndarray, hessians: np.ndarray, indices: np.ndarray) -> float:
		G = gradients[indices].sum()
		H = hessians[indices].sum()
		if H < self.min_sum_hessian_in_leaf:
			return 0.0
		# L1 soft-thresholding for leaf value.
		abs_G = abs(G)
		if abs_G <= self.lambda_l1:
			return 0.0
		return -np.sign(G) * (abs_G - self.lambda_l1) / (H + self.lambda_l2)

	def _gain(self, G_left: float, H_left: float, G_right: float, H_right: float, G_total: float, H_total: float) -> float:
		def _score(G: float, H: float) -> float:
			if H < self.min_sum_hessian_in_leaf:
				return -np.inf
			abs_G = abs(G)
			if abs_G <= self.lambda_l1:
				return 0.0
			shrink = abs_G - self.lambda_l1
			return (shrink ** 2) / (H + self.lambda_l2)

		left = _score(G_left, H_left)
		right = _score(G_right, H_right)
		parent = _score(G_total, H_total)
		return left + right - parent

	def _gain_with_nans(
		self,
		G_left: float,
		H_left: float,
		G_right: float,
		H_right: float,
		G_total: float,
		H_total: float,
		nan_G: float,
		nan_H: float,
	) -> Tuple[float, bool]:
		# Evaluate gain when routing NaNs left or right; return best gain and whether NaNs go left.
		left_G_default = G_left + nan_G if self.default_left else G_left
		left_H_default = H_left + nan_H if self.default_left else H_left
		right_G_default = G_total - left_G_default
		right_H_default = H_total - left_H_default

		left_ok = left_H_default >= self.min_sum_hessian_in_leaf and right_H_default >= self.min_sum_hessian_in_leaf
		default_gain = self._gain(left_G_default, left_H_default, right_G_default, right_H_default, G_total, H_total) if left_ok else -np.inf

		left_G_alt = G_left if self.default_left else G_left + nan_G
		left_H_alt = H_left if self.default_left else H_left + nan_H
		right_G_alt = G_total - left_G_alt
		right_H_alt = H_total - left_H_alt
		alt_ok = left_H_alt >= self.min_sum_hessian_in_leaf and right_H_alt >= self.min_sum_hessian_in_leaf
		alt_gain = self._gain(left_G_alt, left_H_alt, right_G_alt, right_H_alt, G_total, H_total) if alt_ok else -np.inf

		if default_gain >= alt_gain:
			return default_gain, self.default_left
		return alt_gain, not self.default_left

	def _best_split(
		self,
		X: np.ndarray,
		gradients: np.ndarray,
		hessians: np.ndarray,
		indices: np.ndarray,
	) -> Tuple[float, Optional[Tuple[int, float, np.ndarray, np.ndarray]]]:
		if len(indices) < 2 * self.min_data_in_leaf:
			return 0.0, None

		n_features = X.shape[1]
		rng = np.random.default_rng(self.random_state)
		feature_indices = np.arange(n_features)
		if self.colsample < 1.0:
			size = max(1, int(n_features * self.colsample))
			feature_indices = np.sort(rng.choice(feature_indices, size=size, replace=False))

		best_gain = 0.0
		best_split = None
		G_total = gradients[indices].sum()
		H_total = hessians[indices].sum()

		for feat_idx in feature_indices:
			if feat_idx in self.categorical_features:
				gain, split = self._best_split_categorical(X, gradients, hessians, indices, feat_idx, G_total, H_total)
			else:
				gain, split = self._best_split_numeric(X, gradients, hessians, indices, feat_idx, G_total, H_total)
			if gain > best_gain:
				best_gain = gain
				best_split = split

		return best_gain, best_split

	# ------------------------------------------------------------------
	def predict(self, X: np.ndarray) -> np.ndarray:
		X = np.asarray(X, dtype=float)
		preds = np.empty(len(X))
		for i, row in enumerate(X):
			preds[i] = self.root.predict_one(row)
		return preds

	# ------------------------------------------------------------------
	def _best_split_numeric(
		self,
		X: np.ndarray,
		gradients: np.ndarray,
		hessians: np.ndarray,
		indices: np.ndarray,
		feat_idx: int,
		G_total: float,
		H_total: float,
	) -> Tuple[float, Optional[Tuple[int, float, np.ndarray, np.ndarray]]]:
		values = X[indices, feat_idx]
		nan_mask = np.isnan(values)
		nan_indices = indices[nan_mask]
		values_nonan = values[~nan_mask]
		grad_nonan = gradients[indices][~nan_mask]
		hess_nonan = hessians[indices][~nan_mask]

		if len(values_nonan) < 2 * self.min_data_in_leaf:
			return 0.0, None

		nan_G = gradients[nan_indices].sum() if len(nan_indices) > 0 else 0.0
		nan_H = hessians[nan_indices].sum() if len(nan_indices) > 0 else 0.0

		if self.use_histogram:
			return self._best_split_hist_numeric(indices[~nan_mask], values_nonan, grad_nonan, hess_nonan, feat_idx, G_total, H_total, nan_G, nan_H, nan_indices)

		order = np.argsort(values_nonan)
		values_sorted = values_nonan[order]
		g_sorted = grad_nonan[order]
		h_sorted = hess_nonan[order]

		# Skip constant feature
		if values_sorted[0] == values_sorted[-1]:
			return 0.0, None

		G_prefix = np.cumsum(g_sorted)
		H_prefix = np.cumsum(h_sorted)

		best_gain = 0.0
		best_split = None
		nan_count = len(nan_indices)
		for i in range(len(values_sorted) - 1):
			left_count = i + 1
			right_count = len(values_sorted) - left_count
			left_count_default = left_count + (nan_count if self.default_left else 0)
			right_count_default = right_count + (0 if self.default_left else nan_count)
			left_count_alt = left_count if self.default_left else left_count + nan_count
			right_count_alt = right_count + nan_count if self.default_left else right_count
			if (left_count_default < self.min_data_in_leaf or right_count_default < self.min_data_in_leaf) and (
				left_count_alt < self.min_data_in_leaf or right_count_alt < self.min_data_in_leaf
			):
				continue

			if values_sorted[i] == values_sorted[i + 1]:
				continue

			G_left = G_prefix[i]
			H_left = H_prefix[i]
			G_right = G_total - G_left - nan_G
			H_right = H_total - H_left - nan_H

			# Try assigning NaNs to the default side and the opposite; pick best.
			gain, assign_left = self._gain_with_nans(G_left, H_left, G_right, H_right, G_total, H_total, nan_G, nan_H)
			if gain > best_gain:
				if assign_left:
					G_left_check = G_left + nan_G
					H_left_check = H_left + nan_H
					G_right_check = G_total - G_left_check
					H_right_check = H_total - H_left_check
				else:
					G_left_check = G_left
					H_left_check = H_left
					G_right_check = G_total - G_left_check
					H_right_check = H_total - H_left_check
				if not self._monotone_ok(feat_idx, G_left_check, H_left_check, G_right_check, H_right_check):
					continue
				threshold = (values_sorted[i] + values_sorted[i + 1]) / 2.0
				left_indices = indices[~nan_mask][order[: left_count]]
				right_indices = indices[~nan_mask][order[left_count:]]
				if assign_left and len(nan_indices) > 0:
					left_indices = np.concatenate([left_indices, nan_indices])
				else:
					right_indices = np.concatenate([right_indices, nan_indices])
				best_gain = gain
				best_split = (feat_idx, threshold, left_indices, right_indices)

		return best_gain, best_split

	def _best_split_hist_numeric(
		self,
		indices: np.ndarray,
		values: np.ndarray,
		gradients: np.ndarray,
		hessians: np.ndarray,
		feat_idx: int,
		G_total: float,
		H_total: float,
		nan_G: float,
		nan_H: float,
		nan_indices: np.ndarray,
	) -> Tuple[float, Optional[Tuple[int, float, np.ndarray, np.ndarray]]]:
		bins = values.astype(int)
		unique_bins = np.unique(bins)
		if len(unique_bins) <= 1:
			return 0.0, None

		G_bin = np.zeros(len(unique_bins))
		H_bin = np.zeros(len(unique_bins))
		count_bin = np.zeros(len(unique_bins), dtype=int)
		for idx, b in enumerate(unique_bins):
			mask = bins == b
			G_bin[idx] = gradients[mask].sum()
			H_bin[idx] = hessians[mask].sum()
			count_bin[idx] = np.sum(mask)

		G_prefix = np.cumsum(G_bin)
		H_prefix = np.cumsum(H_bin)
		count_prefix = np.cumsum(count_bin)

		best_gain = 0.0
		best_split = None
		for i in range(len(unique_bins) - 1):
			nan_count = len(nan_indices)
			left_count = count_prefix[i]
			right_count = len(values) - count_prefix[i]
			left_count_default = left_count + (nan_count if self.default_left else 0)
			right_count_default = right_count + (0 if self.default_left else nan_count)
			left_count_alt = left_count if self.default_left else left_count + nan_count
			right_count_alt = right_count + nan_count if self.default_left else right_count
			if (left_count_default < self.min_data_in_leaf or right_count_default < self.min_data_in_leaf) and (
				left_count_alt < self.min_data_in_leaf or right_count_alt < self.min_data_in_leaf
			):
				continue

			G_left = G_prefix[i]
			H_left = H_prefix[i]
			G_right = G_total - G_left - nan_G
			H_right = H_total - H_left - nan_H
			gain, assign_left = self._gain_with_nans(G_left, H_left, G_right, H_right, G_total, H_total, nan_G, nan_H)
			if gain > best_gain:
				threshold = (unique_bins[i] + unique_bins[i + 1]) / 2.0
				left_indices = indices[bins <= unique_bins[i]]
				right_indices = indices[bins > unique_bins[i]]
				if assign_left and nan_count > 0:
					left_indices = np.concatenate([left_indices, nan_indices])
				elif nan_count > 0:
					right_indices = np.concatenate([right_indices, nan_indices])
				if assign_left:
					G_left_check = G_left + nan_G
					H_left_check = H_left + nan_H
					G_right_check = G_total - G_left_check
					H_right_check = H_total - H_left_check
				else:
					G_left_check = G_left
					H_left_check = H_left
					G_right_check = G_total - G_left_check
					H_right_check = H_total - H_left_check
				if not self._monotone_ok(feat_idx, G_left_check, H_left_check, G_right_check, H_right_check):
					continue
				best_gain = gain
				best_split = (feat_idx, threshold, left_indices, right_indices)

		return best_gain, best_split

	def _best_split_categorical(
		self,
		X: np.ndarray,
		gradients: np.ndarray,
		hessians: np.ndarray,
		indices: np.ndarray,
		feat_idx: int,
		G_total: float,
		H_total: float,
	) -> Tuple[float, Optional[Tuple[int, float, np.ndarray, np.ndarray]]]:
		values = X[indices, feat_idx].astype(int)
		unique_cats, inv = np.unique(values, return_inverse=True)
		if len(unique_cats) <= 1:
			return 0.0, None

		G_cat = np.zeros(len(unique_cats))
		H_cat = np.zeros(len(unique_cats))
		for k in range(len(unique_cats)):
			mask = inv == k
			G_cat[k] = gradients[indices][mask].sum()
			H_cat[k] = hessians[indices][mask].sum()

		# Order categories by leaf value estimate (-G/H)
		leaf_value_est = -G_cat / (H_cat + self.lambda_l2)
		order_cats = np.argsort(leaf_value_est)
		G_sorted = G_cat[order_cats]
		H_sorted = H_cat[order_cats]
		cats_sorted = unique_cats[order_cats]

		G_prefix = np.cumsum(G_sorted)
		H_prefix = np.cumsum(H_sorted)

		best_gain = 0.0
		best_split = None

		# Ordered partition heuristic (LightGBM-style)
		for i in range(len(cats_sorted) - 1):
			left_mask_inv = np.isin(inv, order_cats[: i + 1])
			left_count = left_mask_inv.sum()
			right_count = len(indices) - left_count
			if left_count < self.min_data_in_leaf or right_count < self.min_data_in_leaf:
				continue

			G_left = G_prefix[i]
			H_left = H_prefix[i]
			G_right = G_total - G_left
			H_right = H_total - H_left
			if H_left < self.min_sum_hessian_in_leaf or H_right < self.min_sum_hessian_in_leaf:
				continue

			if not self._monotone_ok(feat_idx, G_left, H_left, G_right, H_right):
				continue

			gain = self._gain(G_left, H_left, G_right, H_right, G_total, H_total)
			if gain > best_gain:
				threshold = (cats_sorted[i] + cats_sorted[i + 1]) / 2.0
				left_mask = np.isin(values, cats_sorted[: i + 1])
				left_indices = indices[left_mask]
				right_indices = indices[~left_mask]
				best_gain = gain
				best_split = (feat_idx, threshold, left_indices, right_indices)

		# One-vs-rest heuristic: try each category alone vs the rest
		for cat in unique_cats:
			left_mask = values == cat
			left_count = left_mask.sum()
			right_count = len(indices) - left_count
			if left_count < self.min_data_in_leaf or right_count < self.min_data_in_leaf:
				continue

			G_left = gradients[indices][left_mask].sum()
			H_left = hessians[indices][left_mask].sum()
			G_right = G_total - G_left
			H_right = H_total - H_left
			if H_left < self.min_sum_hessian_in_leaf or H_right < self.min_sum_hessian_in_leaf:
				continue

			if not self._monotone_ok(feat_idx, G_left, H_left, G_right, H_right):
				continue

			gain = self._gain(G_left, H_left, G_right, H_right, G_total, H_total)
			if gain > best_gain:
				threshold = cat + 0.5  # separates this cat from the rest
				left_indices = indices[left_mask]
				right_indices = indices[~left_mask]
				best_gain = gain
				best_split = (feat_idx, threshold, left_indices, right_indices)

		return best_gain, best_split

	def _monotone_ok(self, feat_idx: int, G_left: float, H_left: float, G_right: float, H_right: float) -> bool:
		if self.monotone_constraints is None:
			return True
		if feat_idx >= len(self.monotone_constraints):
			return True
		constraint = self.monotone_constraints[feat_idx]
		if constraint == 0:
			return True
		v_left = -G_left / (H_left + self.lambda_l2)
		v_right = -G_right / (H_right + self.lambda_l2)
		if constraint > 0 and v_left > v_right:
			return False
		if constraint < 0 and v_left < v_right:
			return False
		return True

