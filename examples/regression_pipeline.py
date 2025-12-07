"""Quick regression model comparison including the local LGBMRegressor.

Usage:
    python examples/regression_pipeline.py --csv examples/data_regression.csv --target Target
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from lightgbm.lgbm_regressor import LGBMRegressor  # type: ignore


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="examples/data_regression.csv", help="Path to CSV file")
    p.add_argument("--target", default="Target", help="Target column name")
    p.add_argument("--test-size", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found at {csv_path}")

    df = pd.read_csv(csv_path)
    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not found in CSV")

    df = df.dropna()
    X = df.drop(columns=[args.target])
    y = df[args.target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.seed)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Use scaled features for linear models; tree/boosting (including our LGBM) on raw for small-feature problems.

    # Use scaled data for linear models, raw for tree/boosting models (including our LGBM)
    models = [
        ("Linear Regression", LinearRegression(), X_train_scaled, X_test_scaled),
        ("Ridge", Ridge(alpha=1.0), X_train_scaled, X_test_scaled),
        ("Lasso", Lasso(alpha=0.1), X_train_scaled, X_test_scaled),
        (
            "Decision Tree",
            DecisionTreeRegressor(max_depth=8, min_samples_leaf=5, random_state=args.seed),
            X_train,
            X_test,
        ),
        (
            "Random Forest",
            RandomForestRegressor(n_estimators=300, max_depth=None, min_samples_leaf=2, random_state=args.seed),
            X_train,
            X_test,
        ),
        (
            "Gradient Boosting",
            GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=3, random_state=args.seed),
            X_train,
            X_test,
        ),
        (
            "Light GBM dialna",
            LGBMRegressor(
                num_iterations=400,
                learning_rate=0.1,
                max_depth=6,
                min_data_in_leaf=2,
                subsample=1.0,
                colsample=1.0,
                lambda_l2=0.0,
                use_histogram=True,
                n_bins=64,
                random_state=args.seed,
            ),
            X_train,
            X_test,
        ),
    ]

    results = []
    for name, model, Xtr, Xte in models:
        model.fit(Xtr, y_train)
        y_pred = model.predict(Xte)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        results.append({'Model': name, 'R²': r2, 'MSE': mse})

    results_df = pd.DataFrame(results).sort_values(by='R²', ascending=False)
    print("\nModel Performance Comparison:")
    print(results_df)


if __name__ == "__main__":
    main()