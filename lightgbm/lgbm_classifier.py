"""Simplified LightGBM classifier implementation (leaf-wise GBDT)."""

from __future__ import annotations

import pickle
from typing import List, Optional, Tuple, Sequence

import numpy as np

from .base import BaseEstimator
from .tree import DecisionTree
from .goss import GOSSSampler
from .histogramme import HistogramBinner


# =============================================================================
# Loss Functions for Classification
# =============================================================================

class BinaryCrossEntropyLoss:
    """Binary Cross-Entropy (Log Loss) for binary classification."""

    def __init__(self, eps: float = 1e-15) -> None:
        self.eps = eps

    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        """Compute sigmoid with numerical stability."""
        x = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x))

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute binary cross-entropy loss."""
        prob = self.sigmoid(y_pred)
        prob = np.clip(prob, self.eps, 1 - self.eps)
        return float(-np.mean(y_true * np.log(prob) + (1 - y_true) * np.log(1 - prob)))

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Compute gradient: p - y."""
        return self.sigmoid(y_pred) - y_true

    def hessian(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Compute hessian: p * (1 - p)."""
        prob = self.sigmoid(y_pred)
        return np.maximum(prob * (1 - prob), self.eps)


class MultiClassCrossEntropyLoss:
    """Multi-class Cross-Entropy (Softmax) Loss."""

    def __init__(self, n_classes: int, eps: float = 1e-15) -> None:
        self.n_classes = n_classes
        self.eps = eps

    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        """Compute softmax with numerical stability."""
        x_shifted = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute multi-class cross-entropy loss."""
        n_samples = len(y_true)
        prob = self.softmax(y_pred)
        prob = np.clip(prob, self.eps, 1 - self.eps)
        log_prob = -np.log(prob[np.arange(n_samples), y_true.astype(int)])
        return float(np.mean(log_prob))

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Compute gradient for all classes: p - one_hot(y)."""
        n_samples = len(y_true)
        prob = self.softmax(y_pred)
        y_one_hot = np.zeros((n_samples, self.n_classes))
        y_one_hot[np.arange(n_samples), y_true.astype(int)] = 1
        return prob - y_one_hot

    def hessian(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Compute hessian for all classes: p * (1 - p)."""
        prob = self.softmax(y_pred)
        return np.maximum(prob * (1 - prob), self.eps)


# =============================================================================
# LGBMClassifier Implementation (mirrors LGBMRegressor structure)
# =============================================================================

class LGBMClassifier(BaseEstimator):
    """Gradient boosting classifier using LightGBM-style trees.

    This is a didactic, lightweight implementation mirroring the core ideas of
    LightGBM: second-order gradients, leaf-wise tree growth, column/row
    sampling, L2 regularisation on leaves, and optional GOSS + early stopping.
    """

    def __init__(
        self,
        num_iterations: int = 100,
        learning_rate: float = 0.1,
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
        eval_metric: str = "logloss",
        verbose_eval: Optional[int] = None,
    ) -> None:
        super().__init__(
            num_iterations=num_iterations,
            learning_rate=learning_rate,
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
        # Classification-specific attributes
        self.classes_: Optional[np.ndarray] = None
        self.n_classes_: int = 0
        self.is_binary_: bool = True
        self.loss: Optional[BinaryCrossEntropyLoss | MultiClassCrossEntropyLoss] = None
        self.init_prediction_: float | np.ndarray = 0.0
        self.trees_: List[DecisionTree | List[DecisionTree]] = []
        self.n_features_: Optional[int] = None
        self.feature_importances_: Optional[np.ndarray] = None
        # Extra options (mirrors regressor)
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
        self.eval_history_: list[tuple[int, float]] = []
        self.best_iteration_: Optional[int] = None
        self.split_importances_: Optional[np.ndarray] = None
        self._binner: Optional[HistogramBinner] = None

    # ------------------------------------------------------------------
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eval_set: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> "LGBMClassifier":
        """Fit the gradient boosting classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target labels.
        eval_set : tuple (X_val, y_val), optional
            Validation set for early stopping.

        Returns
        -------
        self : LGBMClassifier
        """
        X, y = self._check_arrays(X, y)
        self.n_features_ = X.shape[1]
        n_samples = len(y)
        self.eval_history_ = []

        # Preprocess labels
        y = y.astype(int)
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        # Remap labels to 0, 1, ..., n_classes-1 if needed
        if not np.array_equal(self.classes_, np.arange(self.n_classes_)):
            label_map = {c: i for i, c in enumerate(self.classes_)}
            y = np.array([label_map[c] for c in y])

        # Determine binary vs multiclass
        self.is_binary_ = self.n_classes_ == 2

        # Histogram binning
        binner = None
        if self.use_histogram:
            binner = HistogramBinner(self.n_bins, exclude_features=self.categorical_features)
            X_proc = binner.fit_transform(X)
            self._binner = binner
        else:
            X_proc = X
            self._binner = None

        # Initialize predictions and loss function
        if self.is_binary_:
            self.loss = BinaryCrossEntropyLoss()
            # Log-odds initialization for binary
            pos_ratio = np.clip(np.mean(y), 1e-7, 1 - 1e-7)
            self.init_prediction_ = float(np.log(pos_ratio / (1 - pos_ratio)))
            y_pred = np.full(n_samples, self.init_prediction_, dtype=float)
        else:
            self.loss = MultiClassCrossEntropyLoss(self.n_classes_)
            # Zero initialization for multiclass (uniform prior)
            self.init_prediction_ = np.zeros(self.n_classes_)
            y_pred = np.tile(self.init_prediction_, (n_samples, 1)).astype(float)

        rng = np.random.default_rng(self.params.random_state)

        # GOSS sampler
        if self.use_goss:
            samplr = GOSSSampler(self.top_rate, self.other_rate, self.params.random_state)
        else:
            samplr = None

        # Early stopping setup
        if eval_set is not None:
            if isinstance(eval_set, list) and len(eval_set) > 0:
                X_val_raw, y_val_raw = eval_set[0]
            else:
                X_val_raw, y_val_raw = eval_set
            X_val, y_val = self._check_arrays(X_val_raw, y_val_raw)
            y_val = y_val.astype(int)
            if not np.array_equal(self.classes_, np.arange(self.n_classes_)):
                label_map = {c: i for i, c in enumerate(self.classes_)}
                y_val = np.array([label_map[c] for c in y_val])
            X_val_proc = binner.transform(X_val) if binner is not None else X_val
            if self.is_binary_:
                val_pred = np.full(len(y_val), self.init_prediction_, dtype=float)
            else:
                val_pred = np.tile(self.init_prediction_, (len(y_val), 1)).astype(float)
            best_loss = float("inf")
            best_iter = -1
            best_trees: List = []
            wait_rounds = 0

        self.trees_ = []

        # ===== Main Boosting Loop =====
        for iter_idx in range(self.params.num_iterations):
            if self.is_binary_:
                # ----- BINARY CLASSIFICATION: 1 tree per iteration -----
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

                y_pred += self.params.learning_rate * tree.predict(X_proc)

                if eval_set is not None:
                    val_pred += self.params.learning_rate * tree.predict(X_val_proc)

            else:
                # ----- MULTICLASS: K trees per iteration (One-vs-Rest) -----
                grad_all = self.loss.gradient(y, y_pred)  # (n_samples, n_classes)
                hess_all = self.loss.hessian(y, y_pred)   # (n_samples, n_classes)

                trees_this_iter = []
                for k in range(self.n_classes_):
                    grad_k = grad_all[:, k]
                    hess_k = hess_all[:, k]

                    if samplr is not None:
                        indices, grad_sub, hess_sub = samplr.sample(grad_k, hess_k)
                        X_sub = X_proc[indices]
                    else:
                        row_idx = self._row_subsample(n_samples)
                        X_sub = X_proc[row_idx]
                        grad_sub = grad_k[row_idx]
                        hess_sub = hess_k[row_idx]

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
                    trees_this_iter.append(tree)

                    y_pred[:, k] += self.params.learning_rate * tree.predict(X_proc)

                    if eval_set is not None:
                        val_pred[:, k] += self.params.learning_rate * tree.predict(X_val_proc)

                self.trees_.append(trees_this_iter)

            # Early stopping evaluation
            if eval_set is not None:
                metric_val = self._eval_metric(y_val, val_pred)
                self.eval_history_.append((iter_idx, metric_val))
                if best_iter == -1:
                    best_loss = metric_val
                    best_iter = iter_idx
                    best_trees = list(self.trees_)
                    wait_rounds = 0
                else:
                    improvement = best_loss - metric_val
                    min_required = max(
                        self.early_stopping_min_delta,
                        abs(best_loss) * self.early_stopping_min_delta_rel,
                    )
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
                    print(f"Iter {iter_idx}: {self.eval_metric}={metric_val:.6f}")

        if eval_set is not None and best_iter >= 0:
            self.trees_ = best_trees
            self.best_iteration_ = best_iter
        else:
            self.best_iteration_ = len(self.trees_) - 1

        self._compute_feature_importances()
        return self

    # ------------------------------------------------------------------
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Probability of each class.
        """
        X = self._check_arrays(X)
        if self.use_histogram:
            if self._binner is None:
                raise RuntimeError("Model was not fit with histogram support")
            X_proc = self._binner.transform(X)
        else:
            X_proc = X

        if self.is_binary_:
            pred = np.full(X.shape[0], self.init_prediction_, dtype=float)
            for tree in self.trees_:
                pred += self.params.learning_rate * tree.predict(X_proc)
            prob_pos = BinaryCrossEntropyLoss.sigmoid(pred)
            return np.column_stack([1 - prob_pos, prob_pos])
        else:
            pred = np.tile(self.init_prediction_, (X.shape[0], 1)).astype(float)
            for trees_iter in self.trees_:
                for k, tree in enumerate(trees_iter):
                    pred[:, k] += self.params.learning_rate * tree.predict(X_proc)
            return MultiClassCrossEntropyLoss.softmax(pred)

    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        probas = self.predict_proba(X)
        class_indices = np.argmax(probas, axis=1)
        return self.classes_[class_indices]

    # ------------------------------------------------------------------
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute classification accuracy."""
        return float(np.mean(self.predict(X) == y))

    # ------------------------------------------------------------------
    def _compute_feature_importances(self) -> None:
        """Aggregate feature importances from all trees."""
        if self.n_features_ is None:
            self.feature_importances_ = None
            self.split_importances_ = None
            return
        imp = np.zeros(self.n_features_)
        imp_split = np.zeros(self.n_features_)

        if self.is_binary_:
            for tree in self.trees_:
                if tree.feature_importances_ is not None:
                    imp += tree.feature_importances_
                if tree.split_counts_ is not None:
                    imp_split += tree.split_counts_
        else:
            for trees_iter in self.trees_:
                for tree in trees_iter:
                    if tree.feature_importances_ is not None:
                        imp += tree.feature_importances_
                    if tree.split_counts_ is not None:
                        imp_split += tree.split_counts_

        total = imp.sum()
        if total > 0:
            imp /= total
        self.feature_importances_ = imp
        total_split = imp_split.sum()
        if total_split > 0:
            imp_split /= total_split
        self.split_importances_ = imp_split

    # ------------------------------------------------------------------
    def _eval_metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute evaluation metric for early stopping."""
        if self.eval_metric == "logloss":
            return self.loss.loss(y_true, y_pred)
        if self.eval_metric == "accuracy":
            if self.is_binary_:
                prob = BinaryCrossEntropyLoss.sigmoid(y_pred)
                preds = (prob >= 0.5).astype(int)
            else:
                preds = np.argmax(y_pred, axis=1)
            return 1.0 - float(np.mean(preds == y_true))  # lower is better
        raise ValueError(f"Unknown eval_metric: {self.eval_metric}")

    # ------------------------------------------------------------------
    def save_model(self, path: str) -> None:
        """Save model to disk."""
        state = {
            "params": self.params,
            "trees": self.trees_,
            "init_prediction": self.init_prediction_,
            "n_features": self.n_features_,
            "classes": self.classes_,
            "n_classes": self.n_classes_,
            "is_binary": self.is_binary_,
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
    def load_model(cls, path: str) -> "LGBMClassifier":
        """Load model from disk."""
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
        model.classes_ = state["classes"]
        model.n_classes_ = state["n_classes"]
        model.is_binary_ = state["is_binary"]
        model.feature_importances_ = state["feature_importances"]
        model.split_importances_ = state["split_importances"]
        model._binner = state.get("binner")
        model.best_iteration_ = len(model.trees_) - 1 if model.trees_ else None
        model.eval_history_ = []
        # Restore loss function
        if model.is_binary_:
            model.loss = BinaryCrossEntropyLoss()
        else:
            model.loss = MultiClassCrossEntropyLoss(model.n_classes_)
        return model
