"""
Convergence experiments for the scratch LightGBM implementation.

This script mirrors the experiment spirit from the NeurIPS 2017 LightGBM
paper: we compare baseline boosting against variants that enable
histogram binning, GOSS sampling, and EFB bundling. We track convergence
for binary classification, multiclass classification, and regression.

Metrics captured per boosting iteration:
- Binary and multiclass: ROC-AUC, PR-AUC (macro for multiclass)
- Regression: R^2, RMSE, MAE

Outputs are saved to examples/experiment_outputs.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import (
    make_regression,
    make_classification,
    fetch_california_housing,
    load_breast_cancer,
    load_wine,
)
try:
    import xgboost as xgb
except ImportError:  # pragma: no cover
    xgb = None
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split

# Prefer the local scratch LightGBM over the pip package with the same name.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from lightgbm import LGBMClassifier, LGBMRegressor

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "experiment_outputs")
RANDOM_STATE = 42


@dataclass
class ExperimentConfig:
    name: str
    params: Dict[str, float]


def _ensure_output_dir() -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return OUTPUT_DIR


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def _transform_features(model, X: np.ndarray) -> np.ndarray:
    bundler = getattr(model, "_bundler", None)
    return bundler.transform(X) if bundler is not None else X


def _is_binary_model(model) -> bool:
    flag = getattr(model, "_is_binary", None)
    if flag is None:
        n_classes = getattr(model, "n_classes_", 2)
        return n_classes <= 2
    return bool(flag)


def _staged_raw_predictions_classifier(model: LGBMClassifier, X: np.ndarray) -> Iterable[np.ndarray]:
    X_t = _transform_features(model, X)

    is_binary = _is_binary_model(model)

    if is_binary:
        raw = np.full(X_t.shape[0], model.init_prediction_)
        for tree in model.trees_:
            raw += model.params.learning_rate * tree.predict(X_t)
            yield raw.copy()
    else:
        n_classes = model.n_classes_
        raw = np.zeros((X_t.shape[0], n_classes))
        if isinstance(model.init_prediction_, np.ndarray):
            raw += model.init_prediction_

        n_rounds = len(model.trees_[0]) if model.trees_ else 0
        for idx in range(n_rounds):
            for class_idx in range(n_classes):
                raw[:, class_idx] += model.params.learning_rate * model.trees_[class_idx][idx].predict(X_t)
            yield raw.copy()


def _staged_predictions_regressor(model: LGBMRegressor, X: np.ndarray) -> Iterable[np.ndarray]:
    X_t = _transform_features(model, X)
    preds = np.full(X_t.shape[0], model.init_prediction_)
    for tree in model.trees_:
        preds += model.params.learning_rate * tree.predict(X_t)
        yield preds.copy()


def _binary_metrics(y_true: np.ndarray, raw: np.ndarray) -> Tuple[float, float]:
    probs = _sigmoid(raw)
    roc = roc_auc_score(y_true, probs)
    pr = average_precision_score(y_true, probs)
    return roc, pr


def _multiclass_metrics(y_true: np.ndarray, raw: np.ndarray) -> Tuple[float, float]:
    exp_pred = np.exp(raw - np.max(raw, axis=1, keepdims=True))
    probs = exp_pred / np.sum(exp_pred, axis=1, keepdims=True)
    classes = np.arange(probs.shape[1])
    y_one_hot = label_binarize(y_true, classes=classes)
    roc = roc_auc_score(y_true, probs, multi_class="ovr", average="macro")
    pr = average_precision_score(y_one_hot, probs, average="macro")
    return roc, pr


def _regression_metrics(y_true: np.ndarray, preds: np.ndarray) -> Tuple[float, float, float]:
    r2 = r2_score(y_true, preds)
    rmse = float(np.sqrt(mean_squared_error(y_true, preds)))
    mae = mean_absolute_error(y_true, preds)
    return r2, rmse, mae


def _mask_nan(X, ratio: float = 0.3, seed: int = 42):
    rng = np.random.default_rng(seed)
    Xc = X.copy()
    mask = rng.random(Xc.shape) < ratio
    Xc.values[mask] = np.nan
    return Xc


def _sparse_noise(n_samples: int, n_features: int, zero_ratio: float = 0.8, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n_samples, n_features))
    Z[rng.random((n_samples, n_features)) < zero_ratio] = 0.0
    return Z


def _plot_metric_curves(x: List[int], series: Dict[str, List[float]], title: str, ylabel: str, filename: str) -> None:
    plt.figure(figsize=(8, 5))
    for label, values in series.items():
        plt.plot(x, values, label=label)
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=200)
    plt.close()


def run_binary_classification(configs: List[ExperimentConfig]) -> None:
    data = load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    for cfg in configs:
        print(f"\n[Binary] Config: {cfg.name}")
        model = LGBMClassifier(
            num_iterations=60,
            learning_rate=0.1,
            max_depth=6,
            num_leaves=31,
            **cfg.params,
            random_state=RANDOM_STATE,
        )
        model.fit(X_train, y_train, eval_set=(X_val, y_val))

        iters = []
        roc_train, roc_val, pr_train, pr_val = [], [], [], []

        staged_train = list(_staged_raw_predictions_classifier(model, X_train))
        staged_val = list(_staged_raw_predictions_classifier(model, X_val))

        for idx, raw_train in enumerate(staged_train, start=1):
            raw_val = staged_val[idx - 1]
            if _is_binary_model(model):
                tr_roc, tr_pr = _binary_metrics(y_train, raw_train)
                va_roc, va_pr = _binary_metrics(y_val, raw_val)
            else:
                tr_roc, tr_pr = _multiclass_metrics(y_train, raw_train)
                va_roc, va_pr = _multiclass_metrics(y_val, raw_val)

            iters.append(idx)
            roc_train.append(tr_roc)
            roc_val.append(va_roc)
            pr_train.append(tr_pr)
            pr_val.append(va_pr)

        _plot_metric_curves(
            iters,
            {f"{cfg.name} train": roc_train, f"{cfg.name} val": roc_val},
            title=f"Binary ROC-AUC ({cfg.name})",
            ylabel="ROC-AUC",
            filename=f"binary_roc_auc_{cfg.name}.png",
        )
        _plot_metric_curves(
            iters,
            {f"{cfg.name} train": pr_train, f"{cfg.name} val": pr_val},
            title=f"Binary PR-AUC ({cfg.name})",
            ylabel="PR-AUC",
            filename=f"binary_pr_auc_{cfg.name}.png",
        )

        print(f"  Final ROC-AUC (val): {roc_val[-1]:.4f}, PR-AUC (val): {pr_val[-1]:.4f}")

    # Optional XGBoost baseline
    if xgb is not None:
        print("\n[Binary] Config: xgboost")
        xgb_clf = xgb.XGBClassifier(
            n_estimators=120,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric=["aucpr", "auc"],
            random_state=RANDOM_STATE,
            tree_method="hist",
        )
        xgb_clf.fit(X_train, y_train)
        proba = xgb_clf.predict_proba(X_val)[:, 1]
        roc = roc_auc_score(y_val, proba)
        pr = average_precision_score(y_val, proba)
        print(f"  Final ROC-AUC (val): {roc:.4f}, PR-AUC (val): {pr:.4f}")


def run_multiclass_classification(configs: List[ExperimentConfig]) -> None:
    data = load_wine()
    X, y = data.data, data.target

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    for cfg in configs:
        print(f"\n[Multiclass] Config: {cfg.name}")
        model = LGBMClassifier(
            num_iterations=70,
            learning_rate=0.1,
            max_depth=7,
            num_leaves=35,
            objective="multiclass",
            **cfg.params,
            random_state=RANDOM_STATE,
        )
        model.fit(X_train, y_train, eval_set=(X_val, y_val))

        iters = []
        roc_train, roc_val, pr_train, pr_val = [], [], [], []

        staged_train = list(_staged_raw_predictions_classifier(model, X_train))
        staged_val = list(_staged_raw_predictions_classifier(model, X_val))

        for idx, raw_train in enumerate(staged_train, start=1):
            raw_val = staged_val[idx - 1]
            tr_roc, tr_pr = _multiclass_metrics(y_train, raw_train)
            va_roc, va_pr = _multiclass_metrics(y_val, raw_val)

            iters.append(idx)
            roc_train.append(tr_roc)
            roc_val.append(va_roc)
            pr_train.append(tr_pr)
            pr_val.append(va_pr)

        _plot_metric_curves(
            iters,
            {f"{cfg.name} train": roc_train, f"{cfg.name} val": roc_val},
            title=f"Multiclass ROC-AUC ({cfg.name})",
            ylabel="ROC-AUC (macro)",
            filename=f"multiclass_roc_auc_{cfg.name}.png",
        )
        _plot_metric_curves(
            iters,
            {f"{cfg.name} train": pr_train, f"{cfg.name} val": pr_val},
            title=f"Multiclass PR-AUC ({cfg.name})",
            ylabel="PR-AUC (macro)",
            filename=f"multiclass_pr_auc_{cfg.name}.png",
        )

        print(f"  Final ROC-AUC (val): {roc_val[-1]:.4f}, PR-AUC (val): {pr_val[-1]:.4f}")

    # Optional XGBoost baseline
    if xgb is not None:
        print("\n[Multiclass] Config: xgboost")
        xgb_clf = xgb.XGBClassifier(
            n_estimators=140,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            num_class=3,
            eval_metric=["mlogloss"],
            random_state=RANDOM_STATE,
            tree_method="hist",
        )
        xgb_clf.fit(X_train, y_train)
        proba = xgb_clf.predict_proba(X_val)
        roc = roc_auc_score(y_val, proba, multi_class="ovr", average="macro")
        pr = average_precision_score(label_binarize(y_val, classes=np.arange(3)), proba, average="macro")
        print(f"  Final ROC-AUC (val): {roc:.4f}, PR-AUC (val): {pr:.4f}")


def run_regression(configs: List[ExperimentConfig]) -> None:
    data = fetch_california_housing()
    X, y = data.data, data.target

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE
    )

    for cfg in configs:
        print(f"\n[Regression] Config: {cfg.name}")
        model = LGBMRegressor(
            num_iterations=80,
            learning_rate=0.08,
            max_depth=8,
            num_leaves=45,
            **cfg.params,
            random_state=RANDOM_STATE,
        )
        model.fit(X_train, y_train, eval_set=(X_val, y_val))

        iters = []
        r2_train, r2_val, rmse_train, rmse_val, mae_train, mae_val = [], [], [], [], [], []

        staged_train = list(_staged_predictions_regressor(model, X_train))
        staged_val = list(_staged_predictions_regressor(model, X_val))

        for idx, preds_train in enumerate(staged_train, start=1):
            preds_val = staged_val[idx - 1]
            tr_r2, tr_rmse, tr_mae = _regression_metrics(y_train, preds_train)
            va_r2, va_rmse, va_mae = _regression_metrics(y_val, preds_val)

            iters.append(idx)
            r2_train.append(tr_r2)
            r2_val.append(va_r2)
            rmse_train.append(tr_rmse)
            rmse_val.append(va_rmse)
            mae_train.append(tr_mae)
            mae_val.append(va_mae)

        _plot_metric_curves(
            iters,
            {f"{cfg.name} train": r2_train, f"{cfg.name} val": r2_val},
            title=f"Regression R^2 ({cfg.name})",
            ylabel="R^2",
            filename=f"regression_r2_{cfg.name}.png",
        )
        _plot_metric_curves(
            iters,
            {f"{cfg.name} train": rmse_train, f"{cfg.name} val": rmse_val},
            title=f"Regression RMSE ({cfg.name})",
            ylabel="RMSE",
            filename=f"regression_rmse_{cfg.name}.png",
        )
        _plot_metric_curves(
            iters,
            {f"{cfg.name} train": mae_train, f"{cfg.name} val": mae_val},
            title=f"Regression MAE ({cfg.name})",
            ylabel="MAE",
            filename=f"regression_mae_{cfg.name}.png",
        )

        print(
            f"  Final R^2 (val): {r2_val[-1]:.4f}, RMSE (val): {rmse_val[-1]:.4f}, MAE (val): {mae_val[-1]:.4f}"
        )

    # Optional XGBoost baseline
    if xgb is not None:
        print("\n[Regression] Config: xgboost")
        xgb_reg = xgb.XGBRegressor(
            n_estimators=200,
            learning_rate=0.08,
            max_depth=8,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            tree_method="hist",
        )
        xgb_reg.fit(X_train, y_train)
        preds = xgb_reg.predict(X_val)
        r2, rmse, mae = _regression_metrics(y_val, preds)
        print(f"  Final R^2 (val): {r2:.4f}, RMSE (val): {rmse:.4f}, MAE (val): {mae:.4f}")


def run_regression_noisy(configs: List[ExperimentConfig]) -> None:
    data = fetch_california_housing(as_frame=True)
    X_df, y = data.data, data.target

    X_train_df, X_val_df, y_train, y_val = train_test_split(
        X_df, y, test_size=0.25, random_state=RANDOM_STATE
    )

    # Inject NaNs
    X_train_df = _mask_nan(X_train_df, ratio=0.3, seed=1)
    X_val_df = _mask_nan(X_val_df, ratio=0.3, seed=2)

    # Add sparse noise features
    n_noise = 50
    noise_train = _sparse_noise(len(X_train_df), n_noise, zero_ratio=0.8, seed=3)
    noise_val = _sparse_noise(len(X_val_df), n_noise, zero_ratio=0.8, seed=4)
    noise_cols = [f"noise_{i}" for i in range(n_noise)]

    X_train_df = X_train_df.copy()
    X_val_df = X_val_df.copy()
    X_train_df[noise_cols] = noise_train
    X_val_df[noise_cols] = noise_val

    X_train = X_train_df.values
    X_val = X_val_df.values

    for cfg in configs:
        print(f"\n[Regression-NaN+Noise] Config: {cfg.name}")
        model = LGBMRegressor(
            num_iterations=100,
            learning_rate=0.08,
            max_depth=8,
            num_leaves=45,
            **cfg.params,
            random_state=RANDOM_STATE,
        )
        model.fit(X_train, y_train, eval_set=(X_val, y_val))

        iters = []
        r2_train, r2_val, rmse_train, rmse_val, mae_train, mae_val = [], [], [], [], [], []

        staged_train = list(_staged_predictions_regressor(model, X_train))
        staged_val = list(_staged_predictions_regressor(model, X_val))

        for idx, preds_train in enumerate(staged_train, start=1):
            preds_val = staged_val[idx - 1]
            tr_r2, tr_rmse, tr_mae = _regression_metrics(y_train, preds_train)
            va_r2, va_rmse, va_mae = _regression_metrics(y_val, preds_val)

            iters.append(idx)
            r2_train.append(tr_r2)
            r2_val.append(va_r2)
            rmse_train.append(tr_rmse)
            rmse_val.append(va_rmse)
            mae_train.append(tr_mae)
            mae_val.append(va_mae)

        _plot_metric_curves(
            iters,
            {f"{cfg.name} train": r2_train, f"{cfg.name} val": r2_val},
            title=f"Regression (NaN+Noise) R^2 ({cfg.name})",
            ylabel="R^2",
            filename=f"regression_noise_r2_{cfg.name}.png",
        )
        _plot_metric_curves(
            iters,
            {f"{cfg.name} train": rmse_train, f"{cfg.name} val": rmse_val},
            title=f"Regression (NaN+Noise) RMSE ({cfg.name})",
            ylabel="RMSE",
            filename=f"regression_noise_rmse_{cfg.name}.png",
        )
        _plot_metric_curves(
            iters,
            {f"{cfg.name} train": mae_train, f"{cfg.name} val": mae_val},
            title=f"Regression (NaN+Noise) MAE ({cfg.name})",
            ylabel="MAE",
            filename=f"regression_noise_mae_{cfg.name}.png",
        )

        print(
            f"  Final R^2 (val): {r2_val[-1]:.4f}, RMSE (val): {rmse_val[-1]:.4f}, MAE (val): {mae_val[-1]:.4f}"
        )

    # Optional XGBoost baseline
    if xgb is not None:
        print("\n[Regression-NaN+Noise] Config: xgboost")
        xgb_reg = xgb.XGBRegressor(
            n_estimators=220,
            learning_rate=0.08,
            max_depth=8,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            tree_method="hist",
        )
        xgb_reg.fit(X_train, y_train)
        preds = xgb_reg.predict(X_val)
        r2, rmse, mae = _regression_metrics(y_val, preds)
        print(f"  Final R^2 (val): {r2:.4f}, RMSE (val): {rmse:.4f}, MAE (val): {mae:.4f}")


def run_synthetic_classification(configs: List[ExperimentConfig]) -> None:
    # High-dimensional synthetic binary classification to stress EFB/GOSS
    X, y = make_classification(
        n_samples=5000,
        n_features=400,
        n_informative=60,
        n_redundant=40,
        n_repeated=0,
        n_clusters_per_class=2,
        weights=[0.55, 0.45],
        class_sep=1.0,
        flip_y=0.02,
        random_state=RANDOM_STATE,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    for cfg in configs:
        print(f"\n[Synth-Cls] Config: {cfg.name}")
        model = LGBMClassifier(
            num_iterations=120,
            learning_rate=0.08,
            max_depth=8,
            num_leaves=63,
            **cfg.params,
            random_state=RANDOM_STATE,
        )
        model.fit(X_train, y_train, eval_set=(X_val, y_val))

        iters = []
        roc_train, roc_val, pr_train, pr_val = [], [], [], []

        staged_train = list(_staged_raw_predictions_classifier(model, X_train))
        staged_val = list(_staged_raw_predictions_classifier(model, X_val))

        for idx, raw_train in enumerate(staged_train, start=1):
            raw_val = staged_val[idx - 1]
            tr_roc, tr_pr = _binary_metrics(y_train, raw_train)
            va_roc, va_pr = _binary_metrics(y_val, raw_val)

            iters.append(idx)
            roc_train.append(tr_roc)
            roc_val.append(va_roc)
            pr_train.append(tr_pr)
            pr_val.append(va_pr)

        _plot_metric_curves(
            iters,
            {f"{cfg.name} train": roc_train, f"{cfg.name} val": roc_val},
            title=f"Synthetic Binary ROC-AUC ({cfg.name})",
            ylabel="ROC-AUC",
            filename=f"synth_binary_roc_auc_{cfg.name}.png",
        )
        _plot_metric_curves(
            iters,
            {f"{cfg.name} train": pr_train, f"{cfg.name} val": pr_val},
            title=f"Synthetic Binary PR-AUC ({cfg.name})",
            ylabel="PR-AUC",
            filename=f"synth_binary_pr_auc_{cfg.name}.png",
        )

        print(f"  Final ROC-AUC (val): {roc_val[-1]:.4f}, PR-AUC (val): {pr_val[-1]:.4f}")

    if xgb is not None:
        print("\n[Synth-Cls] Config: xgboost")
        xgb_clf = xgb.XGBClassifier(
            n_estimators=160,
            learning_rate=0.08,
            max_depth=8,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric=["aucpr", "auc"],
            random_state=RANDOM_STATE,
            tree_method="hist",
        )
        xgb_clf.fit(X_train, y_train)
        proba = xgb_clf.predict_proba(X_val)[:, 1]
        roc = roc_auc_score(y_val, proba)
        pr = average_precision_score(y_val, proba)
        print(f"  Final ROC-AUC (val): {roc:.4f}, PR-AUC (val): {pr:.4f}")


def run_synthetic_regression(configs: List[ExperimentConfig]) -> None:
    # High-dimensional synthetic regression to stress histogram/EFB/GOSS
    X, y = make_regression(
        n_samples=5000,
        n_features=300,
        n_informative=80,
        noise=15.0,
        random_state=RANDOM_STATE,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE
    )

    for cfg in configs:
        print(f"\n[Synth-Reg] Config: {cfg.name}")
        model = LGBMRegressor(
            num_iterations=140,
            learning_rate=0.08,
            max_depth=8,
            num_leaves=63,
            **cfg.params,
            random_state=RANDOM_STATE,
        )
        model.fit(X_train, y_train, eval_set=(X_val, y_val))

        iters = []
        r2_train, r2_val, rmse_train, rmse_val, mae_train, mae_val = [], [], [], [], [], []

        staged_train = list(_staged_predictions_regressor(model, X_train))
        staged_val = list(_staged_predictions_regressor(model, X_val))

        for idx, preds_train in enumerate(staged_train, start=1):
            preds_val = staged_val[idx - 1]
            tr_r2, tr_rmse, tr_mae = _regression_metrics(y_train, preds_train)
            va_r2, va_rmse, va_mae = _regression_metrics(y_val, preds_val)

            iters.append(idx)
            r2_train.append(tr_r2)
            r2_val.append(va_r2)
            rmse_train.append(tr_rmse)
            rmse_val.append(va_rmse)
            mae_train.append(tr_mae)
            mae_val.append(va_mae)

        _plot_metric_curves(
            iters,
            {f"{cfg.name} train": r2_train, f"{cfg.name} val": r2_val},
            title=f"Synthetic Regression R^2 ({cfg.name})",
            ylabel="R^2",
            filename=f"synth_regression_r2_{cfg.name}.png",
        )
        _plot_metric_curves(
            iters,
            {f"{cfg.name} train": rmse_train, f"{cfg.name} val": rmse_val},
            title=f"Synthetic Regression RMSE ({cfg.name})",
            ylabel="RMSE",
            filename=f"synth_regression_rmse_{cfg.name}.png",
        )
        _plot_metric_curves(
            iters,
            {f"{cfg.name} train": mae_train, f"{cfg.name} val": mae_val},
            title=f"Synthetic Regression MAE ({cfg.name})",
            ylabel="MAE",
            filename=f"synth_regression_mae_{cfg.name}.png",
        )

        print(
            f"  Final R^2 (val): {r2_val[-1]:.4f}, RMSE (val): {rmse_val[-1]:.4f}, MAE (val): {mae_val[-1]:.4f}"
        )

    if xgb is not None:
        print("\n[Synth-Reg] Config: xgboost")
        xgb_reg = xgb.XGBRegressor(
            n_estimators=220,
            learning_rate=0.08,
            max_depth=9,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            tree_method="hist",
        )
        xgb_reg.fit(X_train, y_train)
        preds = xgb_reg.predict(X_val)
        r2, rmse, mae = _regression_metrics(y_val, preds)
        print(f"  Final R^2 (val): {r2:.4f}, RMSE (val): {rmse:.4f}, MAE (val): {mae:.4f}")


def main() -> None:
    _ensure_output_dir()

    configs = [
        ExperimentConfig("baseline", {"use_histogram": False, "enable_goss": False, "use_efb": False}),
        ExperimentConfig("histogram", {"use_histogram": True, "max_bins": 64}),
        ExperimentConfig("goss_hist", {"use_histogram": True, "enable_goss": True, "goss_top_rate": 0.25, "goss_other_rate": 0.1, "max_bins": 64}),
        ExperimentConfig("efb_hist", {"use_histogram": True, "use_efb": True, "max_bins": 64}),
        ExperimentConfig("histogram_efb_goss", {"use_histogram": True, "use_efb": True, "max_bins":64, "enable_goss": True, "goss_top_rate": 0.25, "goss_other_rate": 0.10})
    ]

    run_binary_classification(configs)
    run_multiclass_classification(configs)
    run_regression(configs)
    run_regression_noisy(configs)
    run_synthetic_classification(configs)
    run_synthetic_regression(configs)

    print(f"\nPlots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
