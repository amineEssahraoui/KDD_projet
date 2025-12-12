"""Exclusive Feature Bundling (EFB) to reduce dimensionality.

This is a simplified, CPU-only version of the technique described in the
LightGBM paper (NIPS 2017). The bundler groups mutually exclusive sparse
features into shared "bundles" to shrink the feature space before the
histogram/leaf-wise boosting stage. Only Python-side functionality is
implemented; low-level sparse optimisations from the reference C++ code are
out-of-scope here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import numpy as np


@dataclass
class ExclusiveFeatureBundler:
	"""Greedy Exclusive Feature Bundling (EFB).

	Parameters
	----------
	conflict_rate : float, default 0.0
		Maximum allowed overlap (as a ratio of the feature's non-zero count)
		when adding a feature to an existing bundle. 0.0 enforces strict
		exclusivity; small positive values tolerate limited conflicts.
	"""

	conflict_rate: float = 0.0
	bundles_: List[dict] = field(default_factory=list, init=False)
	feature_map_: List[Tuple[int, float]] | None = field(default=None, init=False)
	original_n_features_: int | None = field(default=None, init=False)

	def fit(self, X: np.ndarray) -> "ExclusiveFeatureBundler":
		X = np.asarray(X)
		if X.ndim != 2:
			raise ValueError("X must be 2-dimensional")
		n_samples, n_features = X.shape
		self.original_n_features_ = n_features
		self.bundles_ = []
		self.feature_map_ = [(-1, 0.0)] * n_features

		# Greedy pack features with lowest density first.
		nnz_per_feature = np.count_nonzero(X, axis=0)
		feature_order = np.argsort(nnz_per_feature)

		for feat_idx in feature_order:
			nnz_positions = set(np.flatnonzero(X[:, feat_idx]))
			feat_nnz = len(nnz_positions)
			feat_values = X[:, feat_idx]
			added = False
			for bundle_idx, bundle in enumerate(self.bundles_):
				conflicts = len(bundle["nnz_positions"].intersection(nnz_positions))
				if feat_nnz == 0:
					conflict_ratio = 0.0
				else:
					conflict_ratio = conflicts / float(feat_nnz)
				if conflict_ratio <= self.conflict_rate:
					offset = bundle["offset_cursor"]
					# Offset separates bundled features so thresholds stay distinct.
					bundle["offset_cursor"] += self._offset_step(feat_values)
					bundle["features"].append(feat_idx)
					bundle["offsets"].append(offset)
					bundle["nnz_positions"].update(nnz_positions)
					self.feature_map_[feat_idx] = (bundle_idx, offset)
					added = True
					break

			if added:
				continue

			offset_cursor = self._offset_step(feat_values)
			self.bundles_.append(
				{
					"features": [feat_idx],
					"offsets": [0.0],
					"offset_cursor": offset_cursor,
					"nnz_positions": set(nnz_positions),
				}
			)
			self.feature_map_[feat_idx] = (len(self.bundles_) - 1, 0.0)

		return self

	def transform(self, X: np.ndarray) -> np.ndarray:
		if self.feature_map_ is None or self.original_n_features_ is None:
			raise RuntimeError("fit must be called before transform")

		X = np.asarray(X)
		if X.ndim != 2:
			raise ValueError("X must be 2-dimensional")
		if X.shape[1] != self.original_n_features_:
			raise ValueError("X has a different number of features than during fit")

		n_samples = X.shape[0]
		bundled = np.zeros((n_samples, len(self.bundles_)), dtype=float)
		for feat_idx, (bundle_idx, offset) in enumerate(self.feature_map_):
			col = X[:, feat_idx]
			mask = (~np.isnan(col)) & (col != 0)
			if not np.any(mask):
				continue
			bundled[mask, bundle_idx] = col[mask] + offset
		return bundled

	def fit_transform(self, X: np.ndarray) -> np.ndarray:
		return self.fit(X).transform(X)

	# ------------------------------------------------------------------
	def bundle_features(self) -> List[Sequence[int]]:
		"""Return the list of feature indices grouped per bundle."""

		if self.bundles_ is None:
			return []
		return [tuple(b["features"]) for b in self.bundles_]

	# ------------------------------------------------------------------
	@staticmethod
	def _offset_step(values: np.ndarray) -> float:
		# Keep bundled thresholds separated even when values are small.
		if values.size == 0:
			return 1.0
		finite_vals = values[np.isfinite(values)]
		if finite_vals.size == 0:
			return 1.0
		max_abs = np.max(np.abs(finite_vals))
		return float(max(1.0, max_abs + 1.0))

