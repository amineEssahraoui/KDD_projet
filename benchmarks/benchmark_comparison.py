"""
Comprehensive Benchmark: Custom LightGBM vs Scikit-Learn

This script compares the custom LightGBM implementation with sklearn's
GradientBoostingClassifier and GradientBoostingRegressor on real datasets.

Datasets used:
- Regression: California Housing, Diabetes
- Classification: Breast Cancer (binary), Digits (multiclass via binary)
"""

import time
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Sklearn imports for comparison and datasets
from sklearn.datasets import (
    load_breast_cancer,
    load_diabetes,
    fetch_california_housing,
    load_digits,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import (
    accuracy_score as sklearn_accuracy,
    mean_squared_error as sklearn_mse,
    r2_score as sklearn_r2,
    log_loss as sklearn_log_loss,
)
import sys
from pathlib import Path
_repo_root = Path(__file__).resolve().parents[1]
src_path = str(_repo_root / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from lightgbm import LGBMClassifier, LGBMRegressor
from lightgbm.utils import accuracy_score, mean_squared_error, r2_score


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def print_subheader(title: str):
    """Print a formatted subheader."""
    print(f"\n--- {title} ---")


def benchmark_regression(X_train, X_test, y_train, y_test, dataset_name: str):
    """Benchmark regression models on given data."""
    print_subheader(f"Regression: {dataset_name}")
    
    results = {}
    
    # Common hyperparameters
    n_estimators = 100
    learning_rate = 0.1
    max_depth = 5
    
    # 1. Custom LGBMRegressor
    print("  Training Custom LGBMRegressor...", end=" ")
    start = time.time()
    custom_model = LGBMRegressor(
        num_iterations=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_data_in_leaf=10,
        lambda_l2=0.1,
        random_state=42,
    )
    custom_model.fit(X_train, y_train)
    custom_train_time = time.time() - start
    
    start = time.time()
    custom_preds = custom_model.predict(X_test)
    custom_pred_time = time.time() - start
    
    custom_mse = mean_squared_error(y_test, custom_preds)
    custom_r2 = r2_score(y_test, custom_preds)
    print(f"Done ({custom_train_time:.2f}s)")
    
    results['Custom LGBM'] = {
        'MSE': custom_mse,
        'R²': custom_r2,
        'Train Time': custom_train_time,
        'Pred Time': custom_pred_time,
    }
    
    # 2. Sklearn GradientBoostingRegressor
    print("  Training Sklearn GradientBoostingRegressor...", end=" ")
    start = time.time()
    sklearn_model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_samples_leaf=10,
        random_state=42,
    )
    sklearn_model.fit(X_train, y_train)
    sklearn_train_time = time.time() - start
    
    start = time.time()
    sklearn_preds = sklearn_model.predict(X_test)
    sklearn_pred_time = time.time() - start
    
    sklearn_mse_val = sklearn_mse(y_test, sklearn_preds)
    sklearn_r2_val = sklearn_r2(y_test, sklearn_preds)
    print(f"Done ({sklearn_train_time:.2f}s)")
    
    results['Sklearn GB'] = {
        'MSE': sklearn_mse_val,
        'R²': sklearn_r2_val,
        'Train Time': sklearn_train_time,
        'Pred Time': sklearn_pred_time,
    }
    
    # Print comparison
    print(f"\n  {'Model':<20} {'MSE':>12} {'R²':>10} {'Train(s)':>10} {'Pred(s)':>10}")
    print(f"  {'-'*20} {'-'*12} {'-'*10} {'-'*10} {'-'*10}")
    for model_name, metrics in results.items():
        print(f"  {model_name:<20} {metrics['MSE']:>12.4f} {metrics['R²']:>10.4f} "
              f"{metrics['Train Time']:>10.3f} {metrics['Pred Time']:>10.4f}")
    
    # Performance comparison
    mse_ratio = custom_mse / sklearn_mse_val if sklearn_mse_val > 0 else float('inf')
    r2_diff = custom_r2 - sklearn_r2_val
    speed_ratio = sklearn_train_time / custom_train_time if custom_train_time > 0 else 0
    
    print(f"\n  📊 Analysis:")
    if mse_ratio < 1.1:
        print(f"     ✅ Custom MSE is competitive (ratio: {mse_ratio:.2f}x)")
    else:
        print(f"     ⚠️  Custom MSE is higher (ratio: {mse_ratio:.2f}x)")
    
    if r2_diff > -0.05:
        print(f"     ✅ Custom R² is competitive (diff: {r2_diff:+.4f})")
    else:
        print(f"     ⚠️  Custom R² is lower (diff: {r2_diff:+.4f})")
    
    return results


def benchmark_classification(X_train, X_test, y_train, y_test, dataset_name: str):
    """Benchmark classification models on given data."""
    print_subheader(f"Classification: {dataset_name}")
    
    results = {}
    
    # Common hyperparameters
    n_estimators = 100
    learning_rate = 0.1
    max_depth = 5
    
    # 1. Custom LGBMClassifier
    print("  Training Custom LGBMClassifier...", end=" ")
    start = time.time()
    custom_model = LGBMClassifier(
        num_iterations=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_data_in_leaf=10,
        lambda_l2=0.1,
        random_state=42,
    )
    custom_model.fit(X_train, y_train)
    custom_train_time = time.time() - start
    
    start = time.time()
    custom_preds = custom_model.predict(X_test)
    custom_proba = custom_model.predict_proba(X_test)
    custom_pred_time = time.time() - start
    
    custom_acc = accuracy_score(y_test, custom_preds)
    # Log loss for binary
    if custom_proba.shape[1] == 2:
        eps = 1e-15
        proba_clipped = np.clip(custom_proba[:, 1], eps, 1 - eps)
        custom_logloss = -np.mean(y_test * np.log(proba_clipped) + (1 - y_test) * np.log(1 - proba_clipped))
    else:
        custom_logloss = float('nan')
    print(f"Done ({custom_train_time:.2f}s)")
    
    results['Custom LGBM'] = {
        'Accuracy': custom_acc,
        'Log Loss': custom_logloss,
        'Train Time': custom_train_time,
        'Pred Time': custom_pred_time,
    }
    
    # 2. Sklearn GradientBoostingClassifier
    print("  Training Sklearn GradientBoostingClassifier...", end=" ")
    start = time.time()
    sklearn_model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_samples_leaf=10,
        random_state=42,
    )
    sklearn_model.fit(X_train, y_train)
    sklearn_train_time = time.time() - start
    
    start = time.time()
    sklearn_preds = sklearn_model.predict(X_test)
    sklearn_proba = sklearn_model.predict_proba(X_test)
    sklearn_pred_time = time.time() - start
    
    sklearn_acc = sklearn_accuracy(y_test, sklearn_preds)
    if sklearn_proba.shape[1] == 2:
        sklearn_logloss_val = sklearn_log_loss(y_test, sklearn_proba)
    else:
        sklearn_logloss_val = float('nan')
    print(f"Done ({sklearn_train_time:.2f}s)")
    
    results['Sklearn GB'] = {
        'Accuracy': sklearn_acc,
        'Log Loss': sklearn_logloss_val,
        'Train Time': sklearn_train_time,
        'Pred Time': sklearn_pred_time,
    }
    
    # Print comparison
    print(f"\n  {'Model':<20} {'Accuracy':>10} {'Log Loss':>10} {'Train(s)':>10} {'Pred(s)':>10}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for model_name, metrics in results.items():
        logloss_str = f"{metrics['Log Loss']:.4f}" if not np.isnan(metrics['Log Loss']) else "N/A"
        print(f"  {model_name:<20} {metrics['Accuracy']:>10.4f} {logloss_str:>10} "
              f"{metrics['Train Time']:>10.3f} {metrics['Pred Time']:>10.4f}")
    
    # Performance comparison
    acc_diff = custom_acc - sklearn_acc
    
    print(f"\n  📊 Analysis:")
    if acc_diff > -0.02:
        print(f"     ✅ Custom Accuracy is competitive (diff: {acc_diff:+.4f})")
    else:
        print(f"     ⚠️  Custom Accuracy is lower (diff: {acc_diff:+.4f})")
    
    return results


def main():
    """Run all benchmarks."""
    print_header("🚀 LightGBM Custom Implementation vs Sklearn Benchmark")
    print("\nThis benchmark compares our custom LightGBM implementation")
    print("(with NO sklearn dependencies in core) against sklearn's")
    print("GradientBoostingClassifier and GradientBoostingRegressor.")
    
    all_results = {}
    
    # =========================================================================
    # REGRESSION BENCHMARKS
    # =========================================================================
    print_header("📈 REGRESSION BENCHMARKS")
    
    # 1. California Housing Dataset
    print("\nLoading California Housing dataset...")
    housing = fetch_california_housing()
    X, y = np.asarray(housing.data), np.asarray(housing.target)  # type: ignore[union-attr]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    # Scale for better convergence
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"  Samples: {len(X)} | Features: {X.shape[1]} | Train: {len(X_train)} | Test: {len(X_test)}")
    all_results['California Housing'] = benchmark_regression(
        X_train_scaled, X_test_scaled, y_train, y_test, "California Housing"
    )
    
    # 2. Diabetes Dataset
    print("\nLoading Diabetes dataset...")
    diabetes = load_diabetes()
    X, y = np.asarray(diabetes.data), np.asarray(diabetes.target)  # type: ignore[union-attr]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"  Samples: {len(X)} | Features: {X.shape[1]} | Train: {len(X_train)} | Test: {len(X_test)}")
    all_results['Diabetes'] = benchmark_regression(
        X_train_scaled, X_test_scaled, y_train, y_test, "Diabetes"
    )
    
    # =========================================================================
    # CLASSIFICATION BENCHMARKS
    # =========================================================================
    print_header("📊 CLASSIFICATION BENCHMARKS")
    
    # 1. Breast Cancer Dataset (Binary)
    print("\nLoading Breast Cancer dataset...")
    cancer = load_breast_cancer()
    X, y = np.asarray(cancer.data), np.asarray(cancer.target)  # type: ignore[union-attr]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"  Samples: {len(X)} | Features: {X.shape[1]} | Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"  Classes: {np.unique(y)} | Class distribution: {np.bincount(y)}")
    all_results['Breast Cancer'] = benchmark_classification(
        X_train_scaled, X_test_scaled, y_train, y_test, "Breast Cancer (Binary)"
    )
    
    # 2. Digits Dataset - Binary (0 vs 1)
    print("\nLoading Digits dataset (binary: 0 vs 1)...")
    digits = load_digits()
    X, y = np.asarray(digits.data), np.asarray(digits.target)  # type: ignore[union-attr]
    # Filter to binary: 0 vs 1
    mask = (y == 0) | (y == 1)
    X, y = X[mask], y[mask]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"  Samples: {len(X)} | Features: {X.shape[1]} | Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"  Classes: {np.unique(y)} | Class distribution: {np.bincount(y)}")
    all_results['Digits Binary'] = benchmark_classification(
        X_train_scaled, X_test_scaled, y_train, y_test, "Digits (Binary: 0 vs 1)"
    )
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print_header("📋 FINAL SUMMARY")
    
    print("\n┌─────────────────────────────────────────────────────────────────────┐")
    print("│                    PERFORMANCE COMPARISON SUMMARY                    │")
    print("├─────────────────────────────────────────────────────────────────────┤")
    
    # Regression summary
    print("│ REGRESSION                                                          │")
    print("│  Dataset            │ Custom R²  │ Sklearn R² │ Difference         │")
    print("│ ─────────────────── │ ────────── │ ────────── │ ────────────────── │")
    for name in ['California Housing', 'Diabetes']:
        if name in all_results:
            custom_r2 = all_results[name]['Custom LGBM']['R²']
            sklearn_r2 = all_results[name]['Sklearn GB']['R²']
            diff = custom_r2 - sklearn_r2
            status = "✅" if diff > -0.05 else "⚠️"
            print(f"│  {name:<18} │ {custom_r2:>10.4f} │ {sklearn_r2:>10.4f} │ {diff:>+8.4f} {status}        │")
    
    print("├─────────────────────────────────────────────────────────────────────┤")
    
    # Classification summary
    print("│ CLASSIFICATION                                                      │")
    print("│  Dataset            │ Custom Acc │ Sklearn Acc│ Difference         │")
    print("│ ─────────────────── │ ────────── │ ────────── │ ────────────────── │")
    for name in ['Breast Cancer', 'Digits Binary']:
        if name in all_results:
            custom_acc = all_results[name]['Custom LGBM']['Accuracy']
            sklearn_acc = all_results[name]['Sklearn GB']['Accuracy']
            diff = custom_acc - sklearn_acc
            status = "✅" if diff > -0.02 else "⚠️"
            print(f"│  {name:<18} │ {custom_acc:>10.4f} │ {sklearn_acc:>10.4f} │ {diff:>+8.4f} {status}        │")
    
    print("└─────────────────────────────────────────────────────────────────────┘")
    
    print("\n✅ Legend: ✅ = Competitive performance, ⚠️ = Needs improvement")
    print("\n📝 Note: Our custom implementation uses NO sklearn in its core,")
    print("   only numpy. Sklearn is used here only for data loading and comparison.")
    
    print("\n" + "=" * 70)
    print(" BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
