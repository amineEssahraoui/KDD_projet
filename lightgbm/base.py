"""Base estimator utilities for simplified LightGBM-style models."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


def _rng(seed: Optional[int]) -> np.random.Generator:
	"""Return a numpy Generator configured with the provided seed."""
	return np.random.default_rng(seed)


@dataclass
class BoosterParams:
	"""Hyper-parameters shared by the regressor and classifier."""

	num_iterations: int = 100
	learning_rate: float = 0.1
	lr_decay: float = 1.0
	lr_decay_steps: int = 1
	max_depth: int = 6
	num_leaves: int = 31
	min_data_in_leaf: int = 20
	min_sum_hessian_in_leaf: float = 0.0
	lambda_l2: float = 0.0
	lambda_l1: float = 0.0
	min_gain_to_split: float = 0.0
	subsample: float = 1.0
	colsample: float = 1.0
	random_state: Optional[int] = None


class BaseEstimator(abc.ABC):
	"""Abstract base class exposing the LightGBM-like interface."""

	def __init__(
		self,
		num_iterations: int = 100,
		learning_rate: float = 0.1,
		lr_decay: float = 1.0,
		lr_decay_steps: int = 1,
		max_depth: int = 6,
		num_leaves: int = 31,
		min_data_in_leaf: int = 20,
		min_sum_hessian_in_leaf: float = 0.0,
		lambda_l2: float = 0.0,
		lambda_l1: float = 0.0,
		min_gain_to_split: float = 0.0,
		subsample: float = 1.0,
		colsample: float = 1.0,
		random_state: Optional[int] = None,
	) -> None:
		self.params = BoosterParams(
			num_iterations=num_iterations,
			learning_rate=learning_rate,
			lr_decay=lr_decay,
			lr_decay_steps=lr_decay_steps,
			max_depth=max_depth,
			num_leaves=num_leaves,
			min_data_in_leaf=min_data_in_leaf,
			min_sum_hessian_in_leaf=min_sum_hessian_in_leaf,
			lambda_l2=lambda_l2,
			lambda_l1=lambda_l1,
			min_gain_to_split=min_gain_to_split,
			subsample=subsample,
			colsample=colsample,
			random_state=random_state,
		)


	@abc.abstractmethod
	def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseEstimator":
		raise NotImplementedError

	@abc.abstractmethod
	def predict(self, X: np.ndarray) -> np.ndarray:
		raise NotImplementedError


	def _check_arrays(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
		X = np.asarray(X, dtype=float)
		if X.ndim != 2:
			raise ValueError("X must be 2-dimensional")
		if y is not None:
			y = np.asarray(y, dtype=float).reshape(-1)
			if len(y) != len(X):
				raise ValueError("X and y must have the same length")
			return X, y
		return X

	def _initial_prediction(self, y: np.ndarray) -> float:
		return float(np.mean(y))

	def _row_subsample(self, n_samples: int) -> np.ndarray:
		if self.params.subsample >= 1.0:
			return np.arange(n_samples)
		rng = _rng(self.params.random_state)
		size = max(1, int(n_samples * self.params.subsample))
		return rng.choice(n_samples, size=size, replace=False)

	def _col_subsample(self, n_features: int) -> np.ndarray:
		if self.params.colsample >= 1.0:
			return np.arange(n_features)
		rng = _rng(self.params.random_state)
		size = max(1, int(n_features * self.params.colsample))
		return np.sort(rng.choice(n_features, size=size, replace=False))

	def _aggregate_prediction(self, init_pred: float, trees: List) -> np.ndarray:
		return init_pred + sum(tree.predict for tree in trees)

