"""
LightGBM Regressor implementation.

This module provides a gradient boosting regressor that implements
all LightGBM core features: leaf-wise growth, GOSS, histogram binning,
and EFB. Completely sklearn-free.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from .base import BaseEstimator, BoosterParams, Callback
from .efb import FeatureBundler
from .goss import GOSS
from .loss_functions import (
    Loss,
    MSELoss,
    MAELoss,
    HuberLoss,
    QuantileLoss,
    get_loss_function,
)
from .tree import DecisionTree
from .utils import (
    check_array,
    check_X_y,
    check_is_fitted,
    check_sample_weight,
    NotFittedError,
)


class LGBMRegressor(BaseEstimator):
    """
    LightGBM Regressor for gradient boosting regression.

    This implementation includes all core LightGBM features:
    - Leaf-wise tree growth (best-first)
    - Gradient-based One-Side Sampling (GOSS)
    - Histogram-based split finding
    - Exclusive Feature Bundling (EFB)
    - L1/L2 regularization
    - Early stopping
    - Learning rate decay
    - Callbacks

        Notes
        -----
        - `objective` accepts either a string (e.g. 'mse', 'huber') or a `Loss`
            instance (e.g. `HuberLoss(delta=1.0)`) to support custom losses.
        - `allow_nan` (default True) controls whether NaN feature values are
            permitted. Tree histogram splitting has improved NaN handling: NaN
            entries are considered for both left/right assignment and the best
            assignment is chosen when computing split gains.
        - `use_efb` supports Exclusive Feature Bundling. Warm-start feature
            checks compare incoming raw feature count against the bundler's
            `n_original_features_` when a bundler is present.
        - `min_sum_hessian_in_leaf` is enforced uniformly (including
            histogram-based splits) and leaf value computations use a safe
            hessian floor to avoid division-by-zero.
        - Numerical stability improvements were applied (stable sigmoid
            clipping, log-sum-exp for multiclass probabilities in losses).

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
    objective : str, default='mse'
        Loss function. Options: 'mse', 'mae', 'huber', 'quantile'.
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
    trees_ : list of DecisionTree
        Fitted decision trees.
    init_prediction_ : float
        Initial prediction (bias term).
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
    >>> from lightgbm_scratch import LGBMRegressor
    >>> import numpy as np
    >>> X = np.random.randn(100, 5)
    >>> y = X[:, 0] + 2 * X[:, 1] + np.random.randn(100) * 0.1
    >>> model = LGBMRegressor(num_iterations=50)
    >>> model.fit(X, y)
    >>> predictions = model.predict(X)
    """

    def __init__(
        self,
        num_iterations: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = -1,
        num_leaves: int = 31,
        min_data_in_leaf: int = 20,
        objective: str = 'mse',
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

        # Regression-specific attributes
        self.init_prediction_: Optional[float] = None
        self.loss_function_: Optional[Loss] = None
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

    def _get_loss_function(self) -> Loss:
        """Get the appropriate loss function."""
        if isinstance(self.objective, str):
            return get_loss_function(self.objective)
        elif isinstance(self.objective, Loss):
            return self.objective
        else:
            raise ValueError(
                f"objective must be a string or Loss instance, "
                f"got {type(self.objective)}"
            )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        eval_set: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        sample_weight: Optional[np.ndarray] = None,
        callbacks: Optional[List[Callback]] = None,
    ) -> "LGBMRegressor":
        """
        Fit the gradient boosting regressor.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Training targets.
        eval_set : tuple of (X_val, y_val) or None, default=None
            Validation set for early stopping.
        sample_weight : array-like of shape (n_samples,) or None, default=None
            Sample weights.
        callbacks : list of Callback or None, default=None
            Training callbacks.

        Returns
        -------
        self : LGBMRegressor
            Fitted regressor.
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

        # Apply EFB if enabled
        if self.params.use_efb and self._bundler is None:
            self._bundler = FeatureBundler()
            X = self._bundler.fit_transform(X)
        elif self._bundler is not None:
            X = self._bundler.transform(X)

        # Process validation set
        X_val, y_val = None, None
        if eval_set is not None:
            X_val, y_val = eval_set
            X_val, y_val = self._validate_inputs(X_val, y_val)
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
        self._train(X, y, X_val, y_val, sample_weight)

        self.is_fitted_ = True
        return self

    def _initialize_fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Initialize model state for fitting."""
        self.n_features_ = X.shape[1]
        self.trees_ = []
        self.n_iter_ = 0
        self.training_history_ = {"train_loss": [], "val_loss": []}

        # Get loss function
        self.loss_function_ = self._get_loss_function()

        # Compute initial prediction
        self.init_prediction_ = self.loss_function_.init_prediction(y)

    def _train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray],
        sample_weight: Optional[np.ndarray],
    ) -> None:
        """Main training loop."""
        n_samples = X.shape[0]

        # Current predictions
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
                val_str = f", val_loss: {val_loss:.6f}" if val_loss else ""
                print(
                    f"[Iter {self.n_iter_}] train_loss: {train_loss:.6f}{val_str}"
                )

            # Run callbacks
            stop = False
            for callback in self.callbacks_:
                if callback.on_iteration_end(
                    iteration, self, train_loss, val_loss
                ):
                    stop = True

            if stop:
                if self.params.verbose >= 1:
                    print(f"Callback requested early stopping at iteration {iteration}")
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
                        print(
                            f"Early stopping at iteration {self.n_iter_} "
                            f"(no improvement for {self.params.early_stopping_rounds} rounds)"
                        )
                    break

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict target values for samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Features to predict.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Predicted values.
        """
        check_is_fitted(self)

        X = check_array(X, ensure_2d=True, allow_nan=self.params.allow_nan)

        if X.shape[1] != self.n_features_:
            # Check for EFB
            if self._bundler is not None:
                if X.shape[1] != self._bundler.n_original_features_:
                    raise ValueError(
                        f"Expected {self._bundler.n_original_features_} features, "
                        f"got {X.shape[1]}"
                    )
            else:
                raise ValueError(
                    f"Expected {self.n_features_} features, got {X.shape[1]}"
                )

        # Apply EFB transformation
        if self._bundler is not None:
            X = self._bundler.transform(X)

        # Start with initial prediction
        y_pred = np.full(X.shape[0], self.init_prediction_)

        # Add tree predictions
        for tree in self.trees_:
            y_pred += self.params.learning_rate * tree.predict(X)

        return y_pred

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute R² score for predictions.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test features.
        y : array-like of shape (n_samples,)
            True values.

        Returns
        -------
        score : float
            R² score.
        """
        from .utils import r2_score

        y_pred = self.predict(X)
        return r2_score(y, y_pred)

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

        if not self.trees_:
            return np.zeros(self.n_features_)

        # Aggregate importance from all trees
        importances = np.zeros(self.n_features_)
        for tree in self.trees_:
            if hasattr(tree, 'feature_importances_'):
                importances += tree.feature_importances_

        # Normalize
        total = np.sum(importances)
        if total > 0:
            importances /= total

        return importances

    def _get_extra_save_data(self) -> Dict[str, Any]:
        """Get regressor-specific data for saving."""
        return {
            "objective": self.objective,
            "init_prediction_": self.init_prediction_,
        }

    def _load_extra_save_data(self, model_data: Dict[str, Any]) -> None:
        """Load regressor-specific data."""
        self.objective = model_data.get("objective", "mse")
        self.init_prediction_ = model_data.get("init_prediction_", 0.0)
        self.loss_function_ = self._get_loss_function()


__all__ = ['LGBMRegressor']
