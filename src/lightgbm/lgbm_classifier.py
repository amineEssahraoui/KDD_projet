"""
LightGBM Classifier implementation.

This module provides a gradient boosting classifier that implements
all LightGBM core features: leaf-wise growth, GOSS, histogram binning,
and EFB. Supports binary and multi-class classification.
Completely sklearn-free.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from .base import BaseEstimator, BoosterParams, Callback
from .efb import FeatureBundler
from .goss import GOSS
from .loss_functions import (
    Loss,
    BinaryCrossEntropyLoss,
    MultiClassCrossEntropyLoss,
)
from .tree import DecisionTree
from .utils import (
    check_array,
    check_X_y,
    check_is_fitted,
    check_sample_weight,
    NotFittedError,
    log_message,
    log_training_progress,
)


class LGBMClassifier(BaseEstimator):
    """
    LightGBM Classifier for gradient boosting classification.

    This implementation includes all core LightGBM features:
    - Leaf-wise tree growth (best-first)
    - Gradient-based One-Side Sampling (GOSS)
    - Histogram-based split finding
    - Exclusive Feature Bundling (EFB)
    - L1/L2 regularization
    - Early stopping
    - Learning rate decay
    - Callbacks
    - Binary and multi-class classification

    Parameters
    ----------
    num_iterations : int, default=100
        Number of boosting iterations (trees).
    learning_rate : float, default=0.1
        Shrinkage rate for each tree's contribution.
    max_depth : int, default=-1
        Maximum tree depth. -1 means unlimited.
    num_leaves : int, default=31
        Maximum number of leaves per tree.
    min_data_in_leaf : int, default=20
        Minimum samples required in a leaf.
    objective : str, default='binary'
        Loss function. Options: 'binary' for binary classification,
        'multiclass' for multi-class classification.
    lambda_l1 : float, default=0.0
        L1 regularization coefficient.
    lambda_l2 : float, default=0.0
        L2 regularization coefficient.
    min_gain_to_split : float, default=0.0
        Minimum gain required for a split.
    feature_fraction : float, default=1.0
        Fraction of features to consider per tree.
    bagging_fraction : float, default=1.0
        Fraction of data to use per tree.
    bagging_freq : int, default=0
        Frequency of bagging (0=disabled).
    early_stopping_rounds : int or None, default=None
        Stop if validation metric doesn't improve for this many rounds.
    enable_goss : bool, default=False
        Enable Gradient-based One-Side Sampling.
    goss_top_rate : float, default=0.2
        Fraction of large gradient samples to keep.
    goss_other_rate : float, default=0.1
        Fraction of small gradient samples to sample.
    use_histogram : bool, default=False
        Use histogram-based split finding.
    max_bins : int, default=255
        Maximum bins for histogram.
    use_efb : bool, default=False
        Enable Exclusive Feature Bundling.
    allow_nan : bool, default=True
        Allow NaN values in features.
    lr_decay : float, default=1.0
        Learning rate decay per iteration.
    warm_start : bool, default=False
        Reuse previous fit's trees.
    verbose : int, default=0
        Verbosity level (0=silent, 1=progress).
    random_state : int or None, default=None
        Random seed for reproducibility.

    Attributes
    ----------
    trees_ : list of list of DecisionTree
        Fitted decision trees. For binary: single list.
        For multiclass: list of lists (one per class).
    classes_ : np.ndarray
        Unique class labels.
    n_classes_ : int
        Number of classes.
    n_features_ : int
        Number of features seen during fit.
    n_iter_ : int
        Actual number of iterations performed.
    is_fitted_ : bool
        Whether the model has been fitted.
    feature_importances_ : np.ndarray
        Feature importance scores.
    training_history_ : dict
        Training and validation loss history.

    Examples
    --------
    >>> from lightgbm_scratch import LGBMClassifier
    >>> import numpy as np
    >>> X = np.random.randn(100, 5)
    >>> y = (X[:, 0] + X[:, 1] > 0).astype(int)
    >>> model = LGBMClassifier(num_iterations=50)
    >>> model.fit(X, y)
    >>> predictions = model.predict(X)
    >>> probabilities = model.predict_proba(X)
    """

    def __init__(
        self,
        num_iterations: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = -1,
        num_leaves: int = 31,
        min_data_in_leaf: int = 20,
        objective: str = 'binary',
        lambda_l1: float = 0.0,
        lambda_l2: float = 0.0,
        min_gain_to_split: float = 0.0,
        feature_fraction: float = 1.0,
        bagging_fraction: float = 1.0,
        bagging_freq: int = 0,
        min_sum_hessian_in_leaf: float = 1e-3,
        early_stopping_rounds: Optional[int] = None,
        enable_goss: bool = False,
        goss_top_rate: float = 0.2,
        goss_other_rate: float = 0.1,
        use_histogram: bool = False,
        max_bins: int = 255,
        use_efb: bool = False,
        allow_nan: bool = True,
        lr_decay: float = 1.0,
        warm_start: bool = False,
        verbose: int = 0,
        random_state: Optional[int] = None,
        # sklearn-compatible aliases
        n_estimators: Optional[int] = None,
        min_samples_leaf: Optional[int] = None,
        reg_alpha: Optional[float] = None,
        reg_lambda: Optional[float] = None,
    ):
        # Apply sklearn-compatible aliases
        if n_estimators is not None:
            num_iterations = n_estimators
        if min_samples_leaf is not None:
            min_data_in_leaf = min_samples_leaf
        if reg_alpha is not None:
            lambda_l1 = reg_alpha
        if reg_lambda is not None:
            lambda_l2 = reg_lambda
            
        super().__init__(
            num_iterations=num_iterations,
            learning_rate=learning_rate,
            max_depth=max_depth,
            num_leaves=num_leaves,
            min_data_in_leaf=min_data_in_leaf,
            lambda_l1=lambda_l1,
            lambda_l2=lambda_l2,
            min_gain_to_split=min_gain_to_split,
            feature_fraction=feature_fraction,
            bagging_fraction=bagging_fraction,
            bagging_freq=bagging_freq,
            min_sum_hessian_in_leaf=min_sum_hessian_in_leaf,
            early_stopping_rounds=early_stopping_rounds,
            enable_goss=enable_goss,
            goss_top_rate=goss_top_rate,
            goss_other_rate=goss_other_rate,
            use_histogram=use_histogram,
            max_bins=max_bins,
            use_efb=use_efb,
            allow_nan=allow_nan,
            lr_decay=lr_decay,
            warm_start=warm_start,
            verbose=verbose,
            random_state=random_state,
        )

        self.objective = objective

        # Classification-specific attributes
        self.classes_: Optional[np.ndarray] = None
        self.n_classes_: int = 0
        self.init_prediction_: Optional[Union[float, np.ndarray]] = None
        self.loss_function_: Optional[Loss] = None
        self._is_binary: bool = True
        self._goss: Optional[GOSS] = None
        self._bundler: Optional[FeatureBundler] = None
        self._rng: np.random.Generator = np.random.default_rng(random_state)

    def _validate_inputs(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Validate and preprocess inputs."""
        X, y = check_X_y(X, y, allow_nan=self.params.allow_nan)
        return X, y

    def _encode_labels(self, y: np.ndarray) -> np.ndarray:
        """Encode labels to 0, 1, ..., n_classes-1."""
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        # Create mapping
        label_map = {label: idx for idx, label in enumerate(self.classes_)}

        # Encode
        y_encoded = np.array([label_map[label] for label in y])
        return y_encoded

    def _get_loss_function(self) -> Loss:
        """Get the appropriate loss function."""
        if self._is_binary:
            return BinaryCrossEntropyLoss()
        else:
            return MultiClassCrossEntropyLoss(n_classes=self.n_classes_)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        eval_set: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        sample_weight: Optional[np.ndarray] = None,
        callbacks: Optional[List[Callback]] = None,
    ) -> "LGBMClassifier":
        """
        Fit the gradient boosting classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Training labels.
        eval_set : tuple of (X_val, y_val) or None, default=None
            Validation set for early stopping.
        sample_weight : array-like of shape (n_samples,) or None, default=None
            Sample weights.
        callbacks : list of Callback or None, default=None
            Training callbacks.

        Returns
        -------
        self : LGBMClassifier
            Fitted classifier.
        """
        # Validate parameters
        self._validate_params()

        # Validate inputs
        X, y = self._validate_inputs(X, y)
        sample_weight = check_sample_weight(sample_weight, len(y))

        # Store callbacks
        self.callbacks_ = callbacks or []

        # Initialize or continue from previous fit
        if not self.params.warm_start or not self.is_fitted_:
            self._initialize_fit(X, y)
        else:
            # Warm start: continue from previous trees
            expected_feats = (
                self._bundler.n_original_features_
                if getattr(self, "_bundler", None) is not None
                else self.n_features_
            )
            if X.shape[1] != expected_feats:
                raise ValueError(
                    f"Number of features mismatch. Expected {expected_feats}, got {X.shape[1]}."
                )

        # Encode labels
        y_encoded = self._encode_labels(y)

        # Apply EFB if enabled
        if self.params.use_efb and self._bundler is None:
            self._bundler = FeatureBundler()
            X = self._bundler.fit_transform(X)
        elif self._bundler is not None:
            X = self._bundler.transform(X)

        # Process validation set
        X_val, y_val_encoded = None, None
        if eval_set is not None:
            X_val, y_val = eval_set
            X_val, y_val = self._validate_inputs(X_val, y_val)

            # Encode validation labels using same mapping
            label_map = {label: idx for idx, label in enumerate(self.classes_)}
            y_val_encoded = np.array([label_map.get(label, 0) for label in y_val])

            if self._bundler is not None:
                X_val = self._bundler.transform(X_val)

        # Initialize GOSS if enabled
        if self.params.enable_goss:
            self._goss = GOSS(
                top_rate=self.params.goss_top_rate,
                other_rate=self.params.goss_other_rate,
                random_state=self.params.random_state,
            )

        # Training loop
        if self._is_binary:
            self._train_binary(X, y_encoded, X_val, y_val_encoded, sample_weight)
        else:
            self._train_multiclass(X, y_encoded, X_val, y_val_encoded, sample_weight)

        self.is_fitted_ = True
        return self

    def _initialize_fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Initialize model state for fitting."""
        self.n_features_ = X.shape[1]
        self.n_iter_ = 0
        self.training_history_ = {"train_loss": [], "val_loss": []}

        # Determine classes and encode labels first
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self._is_binary = self.n_classes_ <= 2

        # Initialize trees structure
        if self._is_binary:
            self.trees_ = []
        else:
            self.trees_ = [[] for _ in range(self.n_classes_)]

        # Get loss function (now n_classes_ is set)
        self.loss_function_ = self._get_loss_function()

        # Create encoded labels for init_prediction
        label_map = {label: idx for idx, label in enumerate(self.classes_)}
        y_encoded_temp = np.array([label_map[label] for label in y])

        # Compute initial prediction with encoded labels
        self.init_prediction_ = self.loss_function_.init_prediction(y_encoded_temp)

        # Normalize init_prediction_ shape: scalar for binary, 1d-array for multiclass
        if self._is_binary:
            # Expect a scalar for binary problems
            if isinstance(self.init_prediction_, np.ndarray):
                if self.init_prediction_.size == 1:
                    self.init_prediction_ = float(self.init_prediction_)
                else:
                    raise ValueError(
                        "Expected scalar init_prediction_ for binary classification, "
                        f"got array of shape {self.init_prediction_.shape}"
                    )
        else:
            # Multiclass: ensure an array of shape (n_classes_,)
            arr = np.asarray(self.init_prediction_)
            if arr.ndim != 1 or arr.shape[0] != self.n_classes_:
                if arr.size == self.n_classes_:
                    arr = arr.reshape(self.n_classes_)
                else:
                    raise ValueError(
                        "init_prediction_ for multiclass must be array-like with length n_classes"
                    )
            self.init_prediction_ = arr

    def _train_binary(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray],
        sample_weight: Optional[np.ndarray],
    ) -> None:
        """Training loop for binary classification."""
        n_samples = X.shape[0]

        # Current raw predictions (logits)
        y_pred = np.full(n_samples, self.init_prediction_)

        # Add predictions from existing trees (warm start)
        for tree in self.trees_:
            y_pred += self.params.learning_rate * tree.predict(X)

        # Validation predictions
        if X_val is not None:
            y_val_pred = np.full(len(y_val), self.init_prediction_)
            for tree in self.trees_:
                y_val_pred += self.params.learning_rate * tree.predict(X_val)

        # Early stopping state
        best_val_loss = np.inf
        rounds_without_improvement = 0

        # Learning rate for this iteration
        current_lr = self.params.learning_rate

        for iteration in range(self.params.num_iterations):
            # Apply learning rate decay
            if self.params.lr_decay < 1.0:
                current_lr = self.params.learning_rate * (
                    self.params.lr_decay ** iteration
                )

            # Compute gradients and hessians
            gradients = self.loss_function_.gradient(y, y_pred)
            hessians = self.loss_function_.hessian(y, y_pred)

            # Apply sample weights
            if sample_weight is not None:
                gradients = gradients * sample_weight
                hessians = hessians * sample_weight

            # Apply GOSS sampling
            if self._goss is not None:
                X_train, gradients_train, hessians_train, _ = self._goss.sample_data(
                    X, gradients, hessians
                )
            else:
                X_train = X
                gradients_train = gradients
                hessians_train = hessians

            # Apply bagging
            if (
                self.params.bagging_fraction < 1.0
                and self.params.bagging_freq > 0
                and iteration % self.params.bagging_freq == 0
            ):
                n_train = len(gradients_train)
                n_sample = max(1, int(n_train * self.params.bagging_fraction))
                indices = self._rng.choice(n_train, size=n_sample, replace=False)
                X_train = X_train[indices]
                gradients_train = gradients_train[indices]
                hessians_train = hessians_train[indices]

            # Build tree
            tree = DecisionTree(
                max_depth=self.params.max_depth,
                min_samples_leaf=self.params.min_data_in_leaf,
                num_leaves=self.params.num_leaves,
                lambda_l1=self.params.lambda_l1,
                lambda_l2=self.params.lambda_l2,
                min_gain_to_split=self.params.min_gain_to_split,
                feature_fraction=self.params.feature_fraction,
                use_histogram=self.params.use_histogram,
                max_bins=self.params.max_bins,
                min_sum_hessian_in_leaf=self.params.min_sum_hessian_in_leaf,
                random_state=self.params.random_state,
            )

            tree.fit(X_train, gradients_train, hessians_train)

            # Update predictions
            tree_predictions = tree.predict(X)
            y_pred += current_lr * tree_predictions

            # Store tree
            self.trees_.append(tree)
            self.n_iter_ += 1

            # Compute training loss
            train_loss = self.loss_function_(y, y_pred)
            self.training_history_["train_loss"].append(train_loss)

            # Validation
            val_loss = None
            if X_val is not None:
                y_val_pred += current_lr * tree.predict(X_val)
                val_loss = self.loss_function_(y_val, y_val_pred)
                self.training_history_["val_loss"].append(val_loss)

            # Verbose output
            if self.params.verbose >= 1:
                # Use logging helpers for consistent formatting
                log_training_progress(
                    self.n_iter_,
                    self.params.num_iterations,
                    train_loss,
                    verbose=self.params.verbose,
                    metric_name="train_loss",
                )
                if val_loss is not None:
                    log_message(f"val_loss: {val_loss:.6f}", verbose=self.params.verbose)

            # Run callbacks
            stop = False
            for callback in self.callbacks_:
                if callback.on_iteration_end(
                    iteration, self, train_loss, val_loss
                ):
                    stop = True

            if stop:
                if self.params.verbose >= 1:
                    log_message(f"Callback requested early stopping at iteration {iteration}", verbose=self.params.verbose)
                break

            # Early stopping check
            if (
                self.params.early_stopping_rounds is not None
                and val_loss is not None
            ):
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    rounds_without_improvement = 0
                else:
                    rounds_without_improvement += 1

                if rounds_without_improvement >= self.params.early_stopping_rounds:
                    if self.params.verbose >= 1:
                        log_message(
                            f"Early stopping at iteration {self.n_iter_} "
                            f"(no improvement for {self.params.early_stopping_rounds} rounds)",
                            verbose=self.params.verbose,
                        )
                    break

    def _train_multiclass(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray],
        sample_weight: Optional[np.ndarray],
    ) -> None:
        """Training loop for multi-class classification."""
        n_samples = X.shape[0]
        n_classes = self.n_classes_

        # Current raw predictions (logits) - one column per class
        y_pred = np.zeros((n_samples, n_classes))
        if isinstance(self.init_prediction_, np.ndarray):
            y_pred += self.init_prediction_

        # Add predictions from existing trees (warm start)
        for class_idx in range(n_classes):
            for tree in self.trees_[class_idx]:
                y_pred[:, class_idx] += self.params.learning_rate * tree.predict(X)

        # Validation predictions
        if X_val is not None:
            n_val = len(y_val)
            y_val_pred = np.zeros((n_val, n_classes))
            if isinstance(self.init_prediction_, np.ndarray):
                y_val_pred += self.init_prediction_
            for class_idx in range(n_classes):
                for tree in self.trees_[class_idx]:
                    y_val_pred[:, class_idx] += (
                        self.params.learning_rate * tree.predict(X_val)
                    )

        # Early stopping state
        best_val_loss = np.inf
        rounds_without_improvement = 0

        # Learning rate for this iteration
        current_lr = self.params.learning_rate

        for iteration in range(self.params.num_iterations):
            # Apply learning rate decay
            if self.params.lr_decay < 1.0:
                current_lr = self.params.learning_rate * (
                    self.params.lr_decay ** iteration
                )

            # Compute gradients and hessians for all classes
            gradients, hessians = self.loss_function_.gradient_hessian(y, y_pred)

            # Apply sample weights
            if sample_weight is not None:
                gradients = gradients * sample_weight[:, np.newaxis]
                hessians = hessians * sample_weight[:, np.newaxis]

            # Build one tree per class
            for class_idx in range(n_classes):
                g_class = gradients[:, class_idx]
                h_class = hessians[:, class_idx]

                # Apply GOSS sampling (per class)
                if self._goss is not None:
                    X_train, g_train, h_train, _ = self._goss.sample_data(
                        X, g_class, h_class
                    )
                else:
                    X_train = X
                    g_train = g_class
                    h_train = h_class

                # Apply bagging
                if (
                    self.params.bagging_fraction < 1.0
                    and self.params.bagging_freq > 0
                    and iteration % self.params.bagging_freq == 0
                ):
                    n_train = len(g_train)
                    n_sample = max(1, int(n_train * self.params.bagging_fraction))
                    indices = self._rng.choice(n_train, size=n_sample, replace=False)
                    X_train = X_train[indices]
                    g_train = g_train[indices]
                    h_train = h_train[indices]

                # Build tree
                tree = DecisionTree(
                    max_depth=self.params.max_depth,
                    min_samples_leaf=self.params.min_data_in_leaf,
                    num_leaves=self.params.num_leaves,
                    lambda_l1=self.params.lambda_l1,
                    lambda_l2=self.params.lambda_l2,
                    min_gain_to_split=self.params.min_gain_to_split,
                    feature_fraction=self.params.feature_fraction,
                    use_histogram=self.params.use_histogram,
                    max_bins=self.params.max_bins,
                    min_sum_hessian_in_leaf=self.params.min_sum_hessian_in_leaf,
                    random_state=self.params.random_state,
                )

                tree.fit(X_train, g_train, h_train)

                # Update predictions for this class
                tree_predictions = tree.predict(X)
                y_pred[:, class_idx] += current_lr * tree_predictions

                # Store tree
                self.trees_[class_idx].append(tree)

            self.n_iter_ += 1

            # Compute training loss
            train_loss = self.loss_function_(y, y_pred)
            self.training_history_["train_loss"].append(train_loss)

            # Validation
            val_loss = None
            if X_val is not None:
                for class_idx in range(n_classes):
                    tree = self.trees_[class_idx][-1]
                    y_val_pred[:, class_idx] += current_lr * tree.predict(X_val)
                val_loss = self.loss_function_(y_val, y_val_pred)
                self.training_history_["val_loss"].append(val_loss)

            # Verbose output
            if self.params.verbose >= 1:
                log_training_progress(
                    self.n_iter_,
                    self.params.num_iterations,
                    train_loss,
                    verbose=self.params.verbose,
                    metric_name="train_loss",
                )
                if val_loss is not None:
                    log_message(f"val_loss: {val_loss:.6f}", verbose=self.params.verbose)

            # Run callbacks
            stop = False
            for callback in self.callbacks_:
                if callback.on_iteration_end(
                    iteration, self, train_loss, val_loss
                ):
                    stop = True

            if stop:
                if self.params.verbose >= 1:
                    log_message(f"Callback requested early stopping at iteration {iteration}", verbose=self.params.verbose)
                break

            # Early stopping check
            if (
                self.params.early_stopping_rounds is not None
                and val_loss is not None
            ):
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    rounds_without_improvement = 0
                else:
                    rounds_without_improvement += 1

                if rounds_without_improvement >= self.params.early_stopping_rounds:
                    if self.params.verbose >= 1:
                        log_message(
                            f"Early stopping at iteration {self.n_iter_} "
                            f"(no improvement for {self.params.early_stopping_rounds} rounds)",
                            verbose=self.params.verbose,
                        )
                    break

    def _predict_raw(self, X: np.ndarray) -> np.ndarray:
        """Predict raw scores (logits)."""
        check_is_fitted(self)

        X = check_array(X, ensure_2d=True, allow_nan=self.params.allow_nan)

        # Apply EFB transformation
        if self._bundler is not None:
            X = self._bundler.transform(X)

        if self._is_binary:
            # Binary: single score per sample
            y_pred = np.full(X.shape[0], self.init_prediction_)
            for tree in self.trees_:
                y_pred += self.params.learning_rate * tree.predict(X)
        else:
            # Multiclass: one score per class
            n_samples = X.shape[0]
            y_pred = np.zeros((n_samples, self.n_classes_))
            if isinstance(self.init_prediction_, np.ndarray):
                y_pred += self.init_prediction_

            for class_idx in range(self.n_classes_):
                for tree in self.trees_[class_idx]:
                    y_pred[:, class_idx] += (
                        self.params.learning_rate * tree.predict(X)
                    )

        return y_pred

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Features to predict.

        Returns
        -------
        proba : np.ndarray of shape (n_samples, n_classes)
            Class probabilities.
        """
        raw_pred = self._predict_raw(X)

        if self._is_binary:
            # Sigmoid for binary
            prob_positive = 1.0 / (1.0 + np.exp(-np.clip(raw_pred, -500, 500)))
            return np.column_stack([1 - prob_positive, prob_positive])
        else:
            # Softmax for multiclass
            exp_pred = np.exp(raw_pred - np.max(raw_pred, axis=1, keepdims=True))
            return exp_pred / np.sum(exp_pred, axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Features to predict.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Predicted class labels.
        """
        proba = self.predict_proba(X)
        class_indices = np.argmax(proba, axis=1)
        return self.classes_[class_indices]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute classification accuracy.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test features.
        y : array-like of shape (n_samples,)
            True labels.

        Returns
        -------
        score : float
            Classification accuracy.
        """
        from .utils import accuracy_score

        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

    @property
    def feature_importances_(self) -> np.ndarray:
        """
        Feature importance based on total gain.

        Returns
        -------
        importances : np.ndarray of shape (n_features,)
            Normalized feature importance scores.
        """
        check_is_fitted(self)

        if self._is_binary:
            trees_to_check = self.trees_
        else:
            # Flatten trees from all classes
            trees_to_check = [
                tree for class_trees in self.trees_ for tree in class_trees
            ]

        if not trees_to_check:
            return np.zeros(self.n_features_)

        # Aggregate importance from all trees
        importances = np.zeros(self.n_features_)
        for tree in trees_to_check:
            if hasattr(tree, 'feature_importances_'):
                importances += tree.feature_importances_

        # Normalize
        total = np.sum(importances)
        if total > 0:
            importances /= total

        return importances

    def _get_extra_save_data(self) -> Dict[str, Any]:
        """Get classifier-specific data for saving."""
        return {
            "objective": self.objective,
            "classes_": self.classes_.tolist() if self.classes_ is not None else None,
            "n_classes_": self.n_classes_,
            "init_prediction_": (
                self.init_prediction_.tolist()
                if isinstance(self.init_prediction_, np.ndarray)
                else self.init_prediction_
            ),
            "_is_binary": self._is_binary,
        }

    def _load_extra_save_data(self, model_data: Dict[str, Any]) -> None:
        """Load classifier-specific data."""
        self.objective = model_data.get("objective", "binary")
        classes = model_data.get("classes_")
        self.classes_ = np.array(classes) if classes else None
        self.n_classes_ = model_data.get("n_classes_", 2)
        init_pred = model_data.get("init_prediction_")
        self.init_prediction_ = (
            np.array(init_pred) if isinstance(init_pred, list) else init_pred
        )
        self._is_binary = model_data.get("_is_binary", True)
        self.loss_function_ = self._get_loss_function()


__all__ = ['LGBMClassifier']
