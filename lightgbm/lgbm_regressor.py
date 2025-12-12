"""Simplified LightGBM regressor implementation (leaf-wise GBDT)."""

from __future__ import annotations

import pickle
from typing import Callable, List, Optional, Tuple, Sequence

import numpy as np

from .base import BaseEstimator
from .loss_functions import MSELoss
from .tree import DecisionTree
from .goss import GOSSSampler
from .histogramme import HistogramBinner


class LGBMRegressor(BaseEstimator):
	"""Gradient boosting regressor using LightGBM-style trees.

	This is a didactic, lightweight implementation mirroring the core ideas of
	LightGBM: second-order gradients, leaf-wise tree growth, column/row
	sampling, L2 regularisation on leaves, and optional GOSS + early stopping.
	"""

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
		min_gain_to_split: float = 0.0,
		subsample: float = 1.0,
		colsample: float = 1.0,
		random_state: Optional[int] = None,
		use_goss: bool = False,
		top_rate: float = 0.2,
		other_rate: float = 0.1,
		early_stopping_rounds: Optional[int] = None,
		early_stopping_min_delta: float = 1e-3,
		early_stopping_min_delta_rel: float = 1e-3,
		use_histogram: bool = False,
		n_bins: int = 255,
		monotone_constraints: Optional[Sequence[int]] = None,
		categorical_features: Optional[Sequence[int]] = None,
		default_left: bool = True,
		eval_metric: str = "mse",
		verbose_eval: Optional[int] = None,
		callbacks: Optional[Sequence[Callable[[int, dict], None]]] = None,
	) -> None:
		super().__init__(
			num_iterations=num_iterations,
			learning_rate=learning_rate,
			lr_decay=lr_decay,
			lr_decay_steps=lr_decay_steps,
			max_depth=max_depth,
			num_leaves=num_leaves,
			min_data_in_leaf=min_data_in_leaf,
			min_sum_hessian_in_leaf=min_sum_hessian_in_leaf,
			lambda_l2=lambda_l2,
			min_gain_to_split=min_gain_to_split,
			subsample=subsample,
			colsample=colsample,
			random_state=random_state,
		)
		self.loss = MSELoss()
		self.init_prediction_: float = 0.0
		self.trees_: List[DecisionTree] = []
		self.n_features_: Optional[int] = None
		self.feature_importances_: Optional[np.ndarray] = None
		self.use_goss = use_goss
		self.top_rate = top_rate
		self.other_rate = other_rate
		self.early_stopping_rounds = early_stopping_rounds
		self.early_stopping_min_delta = early_stopping_min_delta
		self.early_stopping_min_delta_rel = early_stopping_min_delta_rel
		self.use_histogram = use_histogram
		self.n_bins = n_bins
		self.monotone_constraints = np.array(monotone_constraints) if monotone_constraints is not None else None
		self.categorical_features = set(categorical_features) if categorical_features is not None else set()
		self.default_left = default_left
		self.eval_metric = eval_metric
		self.verbose_eval = verbose_eval
		self.callbacks = list(callbacks) if callbacks is not None else []
		self.eval_history_: list[tuple[int, float]] = []
		self.best_iteration_: Optional[int] = None
		self.split_importances_: Optional[np.ndarray] = None

	# ------------------------------------------------------------------
	def fit(self, X: np.ndarray, y: np.ndarray, eval_set: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> "LGBMRegressor":
		X, y = self._check_arrays(X, y)
		self.n_features_ = X.shape[1]
		n_samples = len(y)
		self.eval_history_ = []

		binner = None
		if self.use_histogram:
			binner = HistogramBinner(self.n_bins, exclude_features=self.categorical_features)
			X_proc = binner.fit_transform(X)
			self._binner = binner
		else:
			X_proc = X
			self._binner = None

		self.init_prediction_ = self._initial_prediction(y)
		y_pred = np.full_like(y, self.init_prediction_, dtype=float)
		rng = np.random.default_rng(self.params.random_state)

		if self.use_goss:
			samplr = GOSSSampler(self.top_rate, self.other_rate, self.params.random_state)
		else:
			samplr = None

		if eval_set is not None:
			# Accept either (X_val, y_val) or [(X_val, y_val)] as provided by sklearn-style interfaces
			if isinstance(eval_set, list) and len(eval_set) > 0:
				X_val_raw, y_val_raw = eval_set[0]
			else:
				X_val_raw, y_val_raw = eval_set
			X_val, y_val = self._check_arrays(X_val_raw, y_val_raw)
			X_val_proc = binner.transform(X_val) if binner is not None else X_val
			val_pred = np.full_like(y_val, self.init_prediction_, dtype=float)
			best_loss = float("inf")
			best_iter = -1
			best_trees: List[DecisionTree] = []
			wait_rounds = 0

		self.trees_ = []
		for iter_idx in range(self.params.num_iterations):
			lr = self.params.learning_rate * (self.params.lr_decay ** (iter_idx // max(1, self.params.lr_decay_steps)))
			grad = self.loss.gradient(y, y_pred)
			hess = self.loss.hessian(y, y_pred)

			if samplr is not None:
				indices, grad_sub, hess_sub = samplr.sample(grad, hess)
				X_sub = X_proc[indices]
			else:
				row_idx = self._row_subsample(n_samples)
				X_sub = X_proc[row_idx]
				grad_sub = grad[row_idx]
				hess_sub = hess[row_idx]

			tree = DecisionTree(
				max_depth=self.params.max_depth,
					num_leaves=self.params.num_leaves,
				min_data_in_leaf=self.params.min_data_in_leaf,
					min_sum_hessian_in_leaf=self.params.min_sum_hessian_in_leaf,
				lambda_l2=self.params.lambda_l2,
				min_gain_to_split=self.params.min_gain_to_split,
				colsample=self.params.colsample,
				random_state=rng.integers(0, 1_000_000),
				use_histogram=self.use_histogram,
				monotone_constraints=self.monotone_constraints,
				categorical_features=self.categorical_features,
					default_left=self.default_left,
			)
			tree.fit(X_sub, grad_sub, hess_sub)
			self.trees_.append(tree)

			y_pred += lr * tree.predict(X_proc)

			if eval_set is not None:
				val_pred += lr * tree.predict(X_val_proc)
				metric_val = self._eval_metric(y_val, val_pred)
				self.eval_history_.append((iter_idx, metric_val))
				if best_iter == -1:
					best_loss = metric_val
					best_iter = iter_idx
					best_trees = list(self.trees_)
					wait_rounds = 0
				else:
					improvement = best_loss - metric_val
					min_required = max(self.early_stopping_min_delta, abs(best_loss) * self.early_stopping_min_delta_rel)
					if improvement > min_required:
						best_loss = metric_val
						best_iter = iter_idx
						best_trees = list(self.trees_)
						wait_rounds = 0
					else:
						wait_rounds += 1
						if self.early_stopping_rounds is not None and wait_rounds >= self.early_stopping_rounds:
							break

				if self.verbose_eval is not None and iter_idx % self.verbose_eval == 0:
					print(f"Iter {iter_idx}: {self.eval_metric}={metric_val:.6f} lr={lr:.4f}")

				for cb in self.callbacks:
					cb(iter_idx, {"metric": metric_val, "lr": lr})

		if eval_set is not None and best_iter >= 0:
			self.trees_ = best_trees
			self.best_iteration_ = best_iter
		else:
			self.best_iteration_ = len(self.trees_) - 1

		self._compute_feature_importances()
		return self

	# ------------------------------------------------------------------
	def predict(self, X: np.ndarray) -> np.ndarray:
		X = self._check_arrays(X)
		if self.use_histogram:
			if self._binner is None:
				raise RuntimeError("Model was not fit with histogram support")
			X_proc = self._binner.transform(X)
		else:
			X_proc = X
		pred = np.full(X.shape[0], self.init_prediction_, dtype=float)
		for tree in self.trees_:
			pred += self.params.learning_rate * tree.predict(X_proc)
		return pred

	def _compute_feature_importances(self) -> None:
		if self.n_features_ is None:
			self.feature_importances_ = None
			self.split_importances_ = None
			return
		imp = np.zeros(self.n_features_)
		imp_split = np.zeros(self.n_features_)
		for tree in self.trees_:
			if tree.feature_importances_ is not None:
				imp += tree.feature_importances_
			if tree.split_counts_ is not None:
				imp_split += tree.split_counts_
		# Normalise to sum to 1
		total = imp.sum()
		if total > 0:
			imp /= total
		self.feature_importances_ = imp
		total_split = imp_split.sum()
		if total_split > 0:
			imp_split /= total_split
		self.split_importances_ = imp_split

	def save_model(self, path: str) -> None:
		state = {
			"params": self.params,
			"trees": self.trees_,
			"init_prediction": self.init_prediction_,
			"n_features": self.n_features_,
			"feature_importances": self.feature_importances_,
			"split_importances": self.split_importances_,
			"use_histogram": self.use_histogram,
			"binner": getattr(self, "_binner", None),
			"categorical_features": self.categorical_features,
			"monotone_constraints": self.monotone_constraints,
		}
		with open(path, "wb") as f:
			pickle.dump(state, f)

	@classmethod
	def load_model(cls, path: str) -> "LGBMRegressor":
		with open(path, "rb") as f:
			state = pickle.load(f)
		model = cls(
			num_iterations=state["params"].num_iterations,
			learning_rate=state["params"].learning_rate,
			max_depth=state["params"].max_depth,
			num_leaves=state["params"].num_leaves,
			min_data_in_leaf=state["params"].min_data_in_leaf,
			min_sum_hessian_in_leaf=state["params"].min_sum_hessian_in_leaf,
			lambda_l2=state["params"].lambda_l2,
			min_gain_to_split=state["params"].min_gain_to_split,
			subsample=state["params"].subsample,
			colsample=state["params"].colsample,
			random_state=state["params"].random_state,
			use_histogram=state["use_histogram"],
			categorical_features=state["categorical_features"],
			monotone_constraints=state["monotone_constraints"],
		)
		model.params = state["params"]
		model.trees_ = state["trees"]
		model.init_prediction_ = state["init_prediction"]
		model.n_features_ = state["n_features"]
		model.feature_importances_ = state["feature_importances"]
		model.split_importances_ = state["split_importances"]
		model._binner = state.get("binner")
		model.best_iteration_ = len(model.trees_) - 1 if model.trees_ else None
		model.eval_history_ = []
		return model

	def _eval_metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
		if self.eval_metric == "mse":
			return float(np.mean((y_true - y_pred) ** 2))
		if self.eval_metric == "rmse":
			return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
		if self.eval_metric == "mae":
			return float(np.mean(np.abs(y_true - y_pred)))
		raise ValueError(f"Unknown eval_metric: {self.eval_metric}")

