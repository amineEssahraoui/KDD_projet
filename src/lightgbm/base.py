"""
Base classes for LightGBM estimators.

This module provides abstract base classes and dataclasses that define
the common interface and parameters for all LightGBM estimators.
No sklearn dependencies.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np


# =============================================================================
# Booster Parameters Dataclass
# =============================================================================

@dataclass
class BoosterParams:
    """
    Dataclass containing all booster hyperparameters.

    This centralizes all configuration options for gradient boosting
    in a single, type-safe structure.

    Parameters
    ----------
    num_iterations : int
        Number of boosting iterations (trees to build).
    learning_rate : float
        Shrinkage rate for each tree's contribution.
    max_depth : int
        Maximum depth of each tree. -1 means unlimited.
    num_leaves : int
        Maximum number of leaves per tree.
    min_data_in_leaf : int
        Minimum number of samples required in a leaf.
    lambda_l1 : float
        L1 regularization coefficient.
    lambda_l2 : float
        L2 regularization coefficient.
    min_gain_to_split : float
        Minimum gain required to make a split.
    feature_fraction : float
        Fraction of features to consider at each split.
    bagging_fraction : float
        Fraction of data to use for each tree.
    bagging_freq : int
        Frequency of bagging (0 = disabled).
    early_stopping_rounds : int or None
        Stop training if validation metric doesn't improve.
    enable_goss : bool
        Whether to use Gradient-based One-Side Sampling.
    goss_top_rate : float
        Fraction of large gradient samples to keep.
    goss_other_rate : float
        Fraction of small gradient samples to keep.
    use_histogram : bool
        Whether to use histogram-based split finding.
    max_bins : int
        Maximum number of bins for histogram.
    use_efb : bool
        Whether to use Exclusive Feature Bundling.
    allow_nan : bool
        Whether to allow NaN values in features.
    lr_decay : float
        Learning rate decay factor per iteration.
    warm_start : bool
        Whether to reuse solution from previous fit.
    verbose : int
        Verbosity level (0=silent, 1=progress, 2=debug).
    random_state : int or None
        Random seed for reproducibility.
    """
    num_iterations: int = 100
    learning_rate: float = 0.1
    max_depth: int = -1
    num_leaves: int = 31
    min_data_in_leaf: int = 20
    lambda_l1: float = 0.0
    lambda_l2: float = 0.0
    min_gain_to_split: float = 0.0
    feature_fraction: float = 1.0
    bagging_fraction: float = 1.0
    bagging_freq: int = 0
    early_stopping_rounds: Optional[int] = None
    enable_goss: bool = False
    goss_top_rate: float = 0.2
    goss_other_rate: float = 0.1
    use_histogram: bool = False
    max_bins: int = 255
    use_efb: bool = False
    allow_nan: bool = True
    lr_decay: float = 1.0
    warm_start: bool = False
    verbose: int = 0
    random_state: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert parameters to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> "BoosterParams":
        """Create BoosterParams from dictionary."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in params.items() if k in valid_keys}
        return cls(**filtered)

    def validate(self) -> None:
        """
        Validate all parameters.

        Raises
        ------
        ValueError
            If any parameter is invalid.
        """
        if self.num_iterations <= 0:
            raise ValueError(
                f"num_iterations must be positive, got {self.num_iterations}"
            )
        if self.learning_rate <= 0:
            raise ValueError(
                f"learning_rate must be positive, got {self.learning_rate}"
            )
        if self.max_depth < -1 or self.max_depth == 0:
            raise ValueError(
                f"max_depth must be -1 or positive, got {self.max_depth}"
            )
        if self.num_leaves < 2:
            raise ValueError(f"num_leaves must be >= 2, got {self.num_leaves}")
        if self.min_data_in_leaf < 0:
            raise ValueError(
                f"min_data_in_leaf must be non-negative, got {self.min_data_in_leaf}"
            )
        if self.lambda_l1 < 0:
            raise ValueError(f"lambda_l1 must be non-negative, got {self.lambda_l1}")
        if self.lambda_l2 < 0:
            raise ValueError(f"lambda_l2 must be non-negative, got {self.lambda_l2}")
        if not 0 < self.feature_fraction <= 1:
            raise ValueError(
                f"feature_fraction must be in (0, 1], got {self.feature_fraction}"
            )
        if not 0 < self.bagging_fraction <= 1:
            raise ValueError(
                f"bagging_fraction must be in (0, 1], got {self.bagging_fraction}"
            )
        if self.max_bins < 2:
            raise ValueError(f"max_bins must be >= 2, got {self.max_bins}")
        if not 0 < self.lr_decay <= 1:
            raise ValueError(f"lr_decay must be in (0, 1], got {self.lr_decay}")
        if not 0 < self.goss_top_rate < 1:
            raise ValueError(
                f"goss_top_rate must be in (0, 1), got {self.goss_top_rate}"
            )
        if not 0 < self.goss_other_rate < 1:
            raise ValueError(
                f"goss_other_rate must be in (0, 1), got {self.goss_other_rate}"
            )
        if self.goss_top_rate + self.goss_other_rate >= 1:
            raise ValueError(
                "goss_top_rate + goss_other_rate must be < 1, "
                f"got {self.goss_top_rate + self.goss_other_rate}"
            )


# =============================================================================
# Callback Base Class
# =============================================================================

class Callback(ABC):
    """
    Abstract base class for training callbacks.

    Callbacks allow custom actions at various points during training.
    """

    @abstractmethod
    def on_iteration_end(
        self,
        iteration: int,
        model: "BaseEstimator",
        train_loss: float,
        val_loss: Optional[float] = None,
    ) -> bool:
        """
        Called at the end of each training iteration.

        Parameters
        ----------
        iteration : int
            Current iteration number (0-indexed).
        model : BaseEstimator
            The model being trained.
        train_loss : float
            Training loss for this iteration.
        val_loss : float or None
            Validation loss if available.

        Returns
        -------
        stop_training : bool
            If True, training will be stopped early.
        """
        pass


class EarlyStoppingCallback(Callback):
    """
    Callback for early stopping based on validation performance.

    Parameters
    ----------
    patience : int
        Number of iterations with no improvement to wait before stopping.
    min_delta : float
        Minimum change to qualify as an improvement.
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss: Optional[float] = None
        self.wait = 0
        self.stopped_iteration: Optional[int] = None

    def on_iteration_end(
        self,
        iteration: int,
        model: "BaseEstimator",
        train_loss: float,
        val_loss: Optional[float] = None,
    ) -> bool:
        """Check for early stopping condition."""
        loss = val_loss if val_loss is not None else train_loss

        if self.best_loss is None or loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.wait = 0
        else:
            self.wait += 1

        if self.wait >= self.patience:
            self.stopped_iteration = iteration
            return True

        return False


class PrintProgressCallback(Callback):
    """
    Callback to print training progress.

    Parameters
    ----------
    print_every : int
        Print progress every N iterations.
    """

    def __init__(self, print_every: int = 10):
        self.print_every = print_every

    def on_iteration_end(
        self,
        iteration: int,
        model: "BaseEstimator",
        train_loss: float,
        val_loss: Optional[float] = None,
    ) -> bool:
        """Print progress if at print interval."""
        if (iteration + 1) % self.print_every == 0:
            val_str = f", val_loss: {val_loss:.6f}" if val_loss is not None else ""
            print(f"[Iter {iteration + 1}] train_loss: {train_loss:.6f}{val_str}")
        return False


# =============================================================================
# Base Estimator Abstract Class
# =============================================================================

class BaseEstimator(ABC):
    """
    Abstract base class for all LightGBM estimators.

    This class defines the common interface that all estimators must implement.
    It is designed to be sklearn-free while maintaining a familiar API.
    """

    def __init__(
        self,
        num_iterations: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = -1,
        num_leaves: int = 31,
        min_data_in_leaf: int = 20,
        lambda_l1: float = 0.0,
        lambda_l2: float = 0.0,
        min_gain_to_split: float = 0.0,
        feature_fraction: float = 1.0,
        bagging_fraction: float = 1.0,
        bagging_freq: int = 0,
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
    ):
        """Initialize the estimator with hyperparameters."""
        self.params = BoosterParams(
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

        # Fitted state
        self.trees_: List[Any] = []
        self.n_features_: Optional[int] = None
        self.n_iter_: int = 0
        self.is_fitted_: bool = False
        self.callbacks_: List[Callback] = []
        self.training_history_: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
        }

    # -------------------------------------------------------------------------
    # Property accessors for common hyperparameters
    # -------------------------------------------------------------------------

    @property
    def num_iterations(self) -> int:
        return self.params.num_iterations

    @property
    def learning_rate(self) -> float:
        return self.params.learning_rate

    @property
    def max_depth(self) -> int:
        return self.params.max_depth

    @property
    def num_leaves(self) -> int:
        return self.params.num_leaves

    @property
    def verbose(self) -> int:
        return self.params.verbose

    @property
    def random_state(self) -> Optional[int]:
        return self.params.random_state

    # -------------------------------------------------------------------------
    # Abstract methods to be implemented by subclasses
    # -------------------------------------------------------------------------

    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        eval_set: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        sample_weight: Optional[np.ndarray] = None,
        callbacks: Optional[List[Callback]] = None,
    ) -> "BaseEstimator":
        """
        Fit the model to training data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training features.
        y : np.ndarray of shape (n_samples,)
            Training targets.
        eval_set : tuple of (X_val, y_val) or None
            Validation set for early stopping.
        sample_weight : np.ndarray or None
            Sample weights.
        callbacks : list of Callback or None
            Callbacks for training.

        Returns
        -------
        self : BaseEstimator
            Fitted estimator.
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Features to predict.

        Returns
        -------
        predictions : np.ndarray
            Predicted values.
        """
        pass

    # -------------------------------------------------------------------------
    # Common methods
    # -------------------------------------------------------------------------

    def get_params(self) -> Dict[str, Any]:
        """
        Get estimator parameters.

        Returns
        -------
        params : dict
            Dictionary of parameter names to values.
        """
        return self.params.to_dict()

    def set_params(self, **params: Any) -> "BaseEstimator":
        """
        Set estimator parameters.

        Parameters
        ----------
        **params : dict
            Parameter names and values.

        Returns
        -------
        self : BaseEstimator
            The estimator instance.
        """
        for key, value in params.items():
            if hasattr(self.params, key):
                setattr(self.params, key, value)
            else:
                raise ValueError(f"Invalid parameter: {key}")
        return self

    def save_model(self, path: str) -> None:
        """
        Save the model to a JSON file.

        Parameters
        ----------
        path : str
            File path to save the model.
        """
        from .utils import check_is_fitted
        check_is_fitted(self)

        model_data = {
            "params": self.params.to_dict(),
            "n_features_": self.n_features_,
            "n_iter_": self.n_iter_,
            "trees_": [self._serialize_tree(tree) for tree in self.trees_],
            "training_history_": self.training_history_,
        }

        # Add subclass-specific data
        model_data.update(self._get_extra_save_data())

        with open(path, 'w') as f:
            json.dump(model_data, f, indent=2)

    def load_model(self, path: str) -> "BaseEstimator":
        """
        Load a model from a JSON file.

        Parameters
        ----------
        path : str
            File path to load the model from.

        Returns
        -------
        self : BaseEstimator
            The loaded model.
        """
        with open(path, 'r') as f:
            model_data = json.load(f)

        self.params = BoosterParams.from_dict(model_data["params"])
        self.n_features_ = model_data["n_features_"]
        self.n_iter_ = model_data["n_iter_"]
        self.trees_ = [
            self._deserialize_tree(tree_data)
            for tree_data in model_data["trees_"]
        ]
        self.training_history_ = model_data.get("training_history_", {})
        self.is_fitted_ = True

        # Load subclass-specific data
        self._load_extra_save_data(model_data)

        return self

    def _serialize_tree(self, tree: Any) -> Dict[str, Any]:
        """Serialize a tree to a dictionary."""
        if hasattr(tree, 'to_dict'):
            return tree.to_dict()
        else:
            return {"tree_data": str(tree)}

    def _deserialize_tree(self, tree_data: Dict[str, Any]) -> Any:
        """Deserialize a tree from a dictionary."""
        from .tree import DecisionTree
        tree = DecisionTree(
            max_depth=self.params.max_depth,
            min_samples_leaf=self.params.min_data_in_leaf,
            num_leaves=self.params.num_leaves,
            lambda_l2=self.params.lambda_l2,
            lambda_l1=self.params.lambda_l1,
        )
        if hasattr(tree, 'from_dict'):
            tree.from_dict(tree_data)
        return tree

    def _get_extra_save_data(self) -> Dict[str, Any]:
        """
        Get subclass-specific data for saving.

        Override in subclasses to add additional data.
        """
        return {}

    def _load_extra_save_data(self, model_data: Dict[str, Any]) -> None:
        """
        Load subclass-specific data.

        Override in subclasses to load additional data.
        """
        pass

    def _validate_params(self) -> None:
        """Validate all hyperparameters."""
        self.params.validate()

    def __repr__(self) -> str:
        """Return string representation of the estimator."""
        class_name = self.__class__.__name__
        params_str = ", ".join(
            f"{k}={v!r}"
            for k, v in self.get_params().items()
            if v != getattr(BoosterParams, k, None)
        )
        return f"{class_name}({params_str})"
