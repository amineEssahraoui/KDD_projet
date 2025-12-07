"""Gradient-based One-Side Sampling (GOSS).

This sampler keeps a proportion of examples with the largest absolute
gradients and randomly samples from the rest, reweighting the sampled
"small-gradient" points to keep the estimator unbiased.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class GOSSSampler:
	top_rate: float = 0.2
	other_rate: float = 0.1
	random_state: int | None = None

	def sample(self, gradients: np.ndarray, hessians: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
		"""Return indices and reweighted gradients/hessians.

		Returns
		-------
		indices : np.ndarray
			Selected row indices.
		grad_scaled : np.ndarray
			Gradients with small-gradient samples upweighted.
		hess_scaled : np.ndarray
			Hessians with small-gradient samples upweighted.
		"""

		g = np.abs(np.asarray(gradients))
		n = g.shape[0]
		if n == 0:
			return np.array([], dtype=int), np.array([]), np.array([])

		rng = np.random.default_rng(self.random_state)
		top_k = max(1, int(self.top_rate * n))
		sorted_idx = np.argsort(-g)
		top_idx = sorted_idx[:top_k]
		rest_idx = sorted_idx[top_k:]

		other_k = max(1, int(self.other_rate * n)) if len(rest_idx) > 0 else 0
		if other_k > 0:
			sampled_rest = rng.choice(rest_idx, size=min(other_k, len(rest_idx)), replace=False)
		else:
			sampled_rest = np.array([], dtype=int)

		indices = np.concatenate([top_idx, sampled_rest])

		grad_scaled = gradients[indices].copy()
		hess_scaled = hessians[indices].copy()

		if len(sampled_rest) > 0:
			weight = (n - top_k) / max(1, len(sampled_rest))
			grad_scaled[len(top_idx) :] *= weight
			hess_scaled[len(top_idx) :] *= weight

		return indices, grad_scaled, hess_scaled

