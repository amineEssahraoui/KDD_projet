"""Simplified LightGBM regressor implementation (leaf-wise GBDT)."""

from __future__ import annotations

import pickle
from typing import Callable, List, Optional, Tuple, Sequence, Union

import numpy as np

from .base import BaseEstimator
from .efb import ExclusiveFeatureBundler
from .goss import GOSSSampler
from .histogramme import HistogramBinner
from .metrics import mse_score, mae_score, r2_score, rmse_score, mape_score
from .loss_functions import HUBERLoss, MAELoss, MSELoss, QUANTILELoss, RMSELoss, LossFunction
from .utils import ValidateInputData, check_X_y, validate_hyperparameters, check_is_fitted
from .tree import DecisionTree


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
		lambda_l1: float = 0.0,
		min_gain_to_split: float = 0.0,
		subsample: float = 1.0,
		colsample: float = 1.0,
		random_state: Optional[int] = None,
		use_goss: bool = False,
		top_rate: float = 0.2,
		other_rate: float = 0.1,
		use_efb: bool = False,
		efb_conflict_rate: float = 0.0,
		loss: Union[str, LossFunction] = "mse",
		allow_nan: bool = True,
		warm_start: bool = False,
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
			lambda_l1=lambda_l1,
			min_gain_to_split=min_gain_to_split,
			subsample=subsample,
			colsample=colsample,
			random_state=random_state,
		)
		
		if isinstance(loss, str):
			self.loss_name = loss.lower()
			self.loss = self._make_loss(self.loss_name)
		elif isinstance(loss, LossFunction):
			self.loss = loss
			lname = getattr(self.loss, '__class__', type(self.loss)).__name__.lower()
			if "huber" in lname:
				self.loss_name = "huber"
			elif "quantile" in lname:
				self.loss_name = "quantile"
			elif "mse" in lname:
				self.loss_name = "mse"
			elif "mae" in lname:
				self.loss_name = "mae"
			elif "rmse" in lname:
				self.loss_name = "rmse"
			else:
				self.loss_name = lname
		else:
			raise ValueError('loss must be either a string name or a LossFunction instance')
		self.allow_nan = allow_nan
		self.init_prediction_: float = 0.0
		self.trees_: List[DecisionTree] = []
		self.n_features_: Optional[int] = None
		self.bundled_n_features_: Optional[int] = None
		self.feature_importances_: Optional[np.ndarray] = None
		self.use_goss = use_goss
		self.top_rate = top_rate
		self.other_rate = other_rate
		self.use_efb = use_efb
		self.efb_conflict_rate = efb_conflict_rate
		self.warm_start = warm_start
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
		self.tree_learning_rates_: list[float] = []
		self._efb: Optional[ExclusiveFeatureBundler] = None
		self._binner: Optional[HistogramBinner] = None


	def fit(self, X: np.ndarray, y: np.ndarray, eval_set: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> "LGBMRegressor":
		X, y = check_X_y(X, y, allow_nan=self.allow_nan)
		validate_hyperparameters(num_iterations=self.params.num_iterations,
					learning_rate=self.params.learning_rate,
					max_depth=self.params.max_depth,
					num_leaves=self.params.num_leaves,
					min_data_in_leaf=self.params.min_data_in_leaf)
		self.n_features_ = X.shape[1]
		if not isinstance(self.loss, LossFunction) and isinstance(self.loss_name, str):
			self.loss = self._make_loss(self.loss_name)
		self.eval_history_ = []
		rng = np.random.default_rng(self.params.random_state)

		warm_cont = self.warm_start and len(self.trees_) > 0
		if warm_cont and X.shape[1] != self.n_features_:
			raise ValueError("Warm start failed: feature count differs from previous fit")

		# Prepare feature transformations
		if warm_cont:
			X_work = self._efb.transform(X) if self._efb is not None else X
			if self.use_histogram:
				if not hasattr(self, "_binner") or self._binner is None:
					raise RuntimeError("Warm start requires existing histogram binner")
				X_proc = self._binner.transform(X_work)
			else:
				X_proc = X_work
			self.init_prediction_ = getattr(self, "init_prediction_", self._initial_prediction(y))
			y_pred = self.predict(X)
		else:
			self.trees_ = []
			self.tree_learning_rates_ = []
			if self.use_efb:
				if self.monotone_constraints is not None:
					raise ValueError("monotone_constraints are not supported with EFB in this simplified implementation")
				if len(self.categorical_features) > 0:
					raise ValueError("categorical_features are not supported with EFB in this simplified implementation")
				bundler = ExclusiveFeatureBundler(conflict_rate=self.efb_conflict_rate)
				X_work = bundler.fit_transform(X)
				self._efb = bundler
				self.bundled_n_features_ = X_work.shape[1]
			else:
				X_work = X
				self._efb = None
				self.bundled_n_features_ = X_work.shape[1]

			if self.use_histogram:
				binner = HistogramBinner(self.n_bins, exclude_features=self.categorical_features)
				X_proc = binner.fit_transform(X_work)
				self._binner = binner
			else:
				X_proc = X_work
				self._binner = None

			self.init_prediction_ = self._initial_prediction(y)
			y_pred = np.full_like(y, self.init_prediction_, dtype=float)

		base_iter = len(self.trees_)
		n_samples = len(y)

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
			X_val, y_val = check_X_y(X_val_raw, y_val_raw, allow_nan=self.allow_nan)
			if self._efb is not None:
				X_val = self._efb.transform(X_val)
			X_val_proc = self._binner.transform(X_val) if self.use_histogram else X_val
			val_pred = self.predict(X_val_raw) if warm_cont else np.full_like(y_val, self.init_prediction_, dtype=float)
			best_loss = float("inf")
			best_iter = -1
			best_trees: List[DecisionTree] = []
			best_lrs: List[float] = []
			wait_rounds = 0

		self.trees_ = []
		for iter_idx in range(self.params.num_iterations):
			lr = self.params.learning_rate * (self.params.lr_decay ** ((base_iter + iter_idx) // max(1, self.params.lr_decay_steps)))
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
				lambda_l1=self.params.lambda_l1,
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
			self.tree_learning_rates_.append(lr)

			y_pred += lr * tree.predict(X_proc)

			if eval_set is not None:
				val_pred += lr * tree.predict(X_val_proc)
				metric_val = self._eval_metric(y_val, val_pred)
				self.eval_history_.append((base_iter + iter_idx, metric_val))
				if best_iter == -1:
					best_loss = metric_val
					best_iter = base_iter + iter_idx
					best_trees = list(self.trees_)
					best_lrs = list(self.tree_learning_rates_)
					wait_rounds = 0
				else:
					improvement = best_loss - metric_val
					min_required = max(self.early_stopping_min_delta, abs(best_loss) * self.early_stopping_min_delta_rel)
					if improvement > min_required:
						best_loss = metric_val
						best_iter = base_iter + iter_idx
						best_trees = list(self.trees_)
						best_lrs = list(self.tree_learning_rates_)
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
			self.tree_learning_rates_ = best_lrs
			self.best_iteration_ = best_iter
		else:
			self.best_iteration_ = len(self.trees_) - 1

		self._compute_feature_importances()
		return self


	def predict(self, X: np.ndarray) -> np.ndarray:
		# Ensure model is trained
		check_is_fitted(self)
		# Validate input X
		X = ValidateInputData(X, allow_nan=self.allow_nan)
		if self.n_features_ is not None and X.shape[1] != self.n_features_:
			raise ValueError("Input feature dimension does not match training data")
		X_proc = X
		if self._efb is not None:
			X_proc = self._efb.transform(X_proc)
		if self.use_histogram:
			if self._binner is None:
				raise RuntimeError("Model was not fit with histogram support")
			X_proc = self._binner.transform(X_proc)
		pred = np.full(X.shape[0], self.init_prediction_, dtype=float)
		for tree, lr in zip(self.trees_, self.tree_learning_rates_):
			pred += lr * tree.predict(X_proc)
		return pred

	def _compute_feature_importances(self) -> None:
		if self.n_features_ is None:
			self.feature_importances_ = None
			self.split_importances_ = None
			return
		if self._efb is None:
			imp = np.zeros(self.n_features_)
			imp_split = np.zeros(self.n_features_)
			for tree in self.trees_:
				if tree.feature_importances_ is not None:
					imp += tree.feature_importances_
				if tree.split_counts_ is not None:
					imp_split += tree.split_counts_
			self.feature_importances_ = self._normalise_importance(imp)
			self.split_importances_ = self._normalise_importance(imp_split)
			return

		bundle_imp = np.zeros(self.bundled_n_features_ or 0)
		bundle_split_imp = np.zeros(self.bundled_n_features_ or 0)
		for tree in self.trees_:
			if tree.feature_importances_ is not None:
				bundle_imp += tree.feature_importances_
			if tree.split_counts_ is not None:
				bundle_split_imp += tree.split_counts_

		orig_imp = np.zeros(self.n_features_)
		orig_split_imp = np.zeros(self.n_features_)
		for bundle_idx, feats in enumerate(self._efb.bundle_features()):
			if len(feats) == 0:
				continue
			share = 1.0 / len(feats)
			for f in feats:
				orig_imp[f] += bundle_imp[bundle_idx] * share
				orig_split_imp[f] += bundle_split_imp[bundle_idx] * share

		self.feature_importances_ = self._normalise_importance(orig_imp)
		self.split_importances_ = self._normalise_importance(orig_split_imp)

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
			"use_efb": self.use_efb,
			"efb_conflict_rate": self.efb_conflict_rate,
			"efb": getattr(self, "_efb", None),
			"bundled_n_features": self.bundled_n_features_,
			"tree_learning_rates": self.tree_learning_rates_,
			"loss_name": self.loss_name,
			"loss_obj": self.loss,
			"lambda_l1": self.params.lambda_l1,
			"warm_start": self.warm_start,
			"use_goss": self.use_goss,
			"top_rate": self.top_rate,
			"other_rate": self.other_rate,
			"allow_nan": self.allow_nan,
			"eval_metric": self.eval_metric,
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
			lambda_l1=state.get("lambda_l1", 0.0),
			min_gain_to_split=state["params"].min_gain_to_split,
			subsample=state["params"].subsample,
			colsample=state["params"].colsample,
			random_state=state["params"].random_state,
			use_histogram=state["use_histogram"],
			categorical_features=state["categorical_features"],
			monotone_constraints=state["monotone_constraints"],
			use_efb=state.get("use_efb", False),
			efb_conflict_rate=state.get("efb_conflict_rate", 0.0),
			use_goss=state.get("use_goss", False),
			top_rate=state.get("top_rate", 0.2),
			other_rate=state.get("other_rate", 0.1),
			eval_metric=state.get("eval_metric", "mse"),
			loss=state.get("loss_obj", state.get("loss_name", "mse")),
			allow_nan=state.get("allow_nan", True),
			warm_start=state.get("warm_start", False),
		)
		model.params = state["params"]
		model.trees_ = state["trees"]
		model.init_prediction_ = state["init_prediction"]
		model.n_features_ = state["n_features"]
		model.feature_importances_ = state["feature_importances"]
		model.split_importances_ = state["split_importances"]
		model._binner = state.get("binner")
		model._efb = state.get("efb")
		model.bundled_n_features_ = state.get("bundled_n_features")
		model.tree_learning_rates_ = state.get("tree_learning_rates", [])
		model.loss_name = state.get("loss_name", model.loss_name)
		loaded_loss_obj = state.get("loss_obj")
		if loaded_loss_obj is not None:
			model.loss = loaded_loss_obj
		else:
			model.loss = model._make_loss(model.loss_name)
		lname = getattr(model.loss, '__class__', type(model.loss)).__name__.lower()
		if "huber" in lname:
			model.loss_name = "huber"
		elif "quantile" in lname:
			model.loss_name = "quantile"
		elif "mse" in lname:
			model.loss_name = "mse"
		elif "mae" in lname:
			model.loss_name = "mae"
		elif "rmse" in lname:
			model.loss_name = "rmse"
		else:
			model.loss_name = lname
		model.best_iteration_ = len(model.trees_) - 1 if model.trees_ else None
		model.eval_history_ = []
		return model

	def _make_loss(self, name: str):
		name = name.lower()
		if name in ("mse", "mseloss"):
			return MSELoss()
		if name in ("mae", "maeloss"):
			return MAELoss()
		if name in ("rmse", "rmseloss"):
			return RMSELoss()
		if name in ("huber", "huberloss"):
			return HUBERLoss()
		if name in ("quantile", "quantileloss"):
			return QUANTILELoss()
		raise ValueError(f"Unknown loss: {name}")

	def _normalise_importance(self, arr: np.ndarray) -> np.ndarray:
		total = arr.sum()
		if total > 0:
			return arr / total
		return arr

	def _eval_metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
		if self.eval_metric == "mse":
			return mse_score(y_true, y_pred)
		elif self.eval_metric == "mae":
			return mae_score(y_true, y_pred)
		elif self.eval_metric == "r2":
			return r2_score(y_true, y_pred)
		elif self.eval_metric == "rmse":
			return rmse_score(y_true, y_pred)
		elif self.eval_metric == "mape":
			return mape_score(y_true, y_pred)
		else:
			raise ValueError(f"Unknown eval_metric: {self.eval_metric}")
		
