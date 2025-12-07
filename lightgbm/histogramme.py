"""Histogram binning utilities to approximate split search."""

from __future__ import annotations

import numpy as np


class HistogramBinner:
	"""Quantile-based binning (default 255 bins) with per-feature opt-out."""

	def __init__(self, n_bins: int = 255, exclude_features: set[int] | None = None):
		if n_bins < 2:
			raise ValueError("n_bins must be >= 2")
		self.n_bins = n_bins
		self.exclude_features = exclude_features or set()
		self.bin_edges_: list[np.ndarray | None] | None = None

	def fit(self, X: np.ndarray) -> "HistogramBinner":
		X = np.asarray(X, dtype=float)
		n_features = X.shape[1]
		self.bin_edges_ = []
		quantiles = np.linspace(0, 100, num=self.n_bins + 1)
		for j in range(n_features):
			if j in self.exclude_features:
				self.bin_edges_.append(None)
				continue
			edges = np.unique(np.percentile(X[:, j], quantiles))
			if len(edges) == 1:
				# Constant feature
				edges = np.array([edges[0], edges[0]])
			self.bin_edges_.append(edges)
		return self

	def transform(self, X: np.ndarray) -> np.ndarray:
		if self.bin_edges_ is None:
			raise RuntimeError("fit must be called before transform")
		X = np.asarray(X, dtype=float)
		binned = np.zeros_like(X, dtype=int)
		for j, edges in enumerate(self.bin_edges_):
			if edges is None:
				binned[:, j] = X[:, j]
			else:
				binned[:, j] = np.searchsorted(edges[1:-1], X[:, j], side="right")
		return binned

	def fit_transform(self, X: np.ndarray) -> np.ndarray:
		return self.fit(X).transform(X)

