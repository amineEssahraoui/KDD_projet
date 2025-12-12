from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing, load_diabetes
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.datasets import fetch_openml


def _try_fetch_openml(name: str, version: int | None = None):
    """Fetch an OpenML dataset, returning (data, target) or None on failure."""
    try:
        return fetch_openml(name=name, version=version, as_frame=True)
    except Exception as exc:  # pragma: no cover - defensive for network/version issues
        print(f"[WARN] Skipping dataset '{name}': {exc}")
        return None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from lightgbm.lgbm_regressor import LGBMRegressor  # type: ignore


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--test-size", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-rows", type=int, default=20000, help="Cap rows per dataset for quicker runs (0 = no cap)")
    p.add_argument("--skip-synthetic", action="store_true", help="Skip synthetic stress-test dataset for speed")
    p.add_argument("--plot-dir", type=str, default="artifacts/benchmark_plots", help="Directory to save benchmark plots")
    return p.parse_args()


# -------------------------------------------------------
# BENCHMARK DATASETS
# -------------------------------------------------------

def _maybe_cap(df: pd.DataFrame, target: pd.Series, max_rows: int, seed: int):
    """Optionally downsample rows to speed up benchmarking."""
    if max_rows and len(df) > max_rows:
        sampled = df.sample(n=max_rows, random_state=seed)
        target = target.loc[sampled.index]
        return sampled, target
    return df, target


def load_all_benchmarks(seed=42, max_rows=0, skip_synthetic=False):
    datasets = []

    # 1. California Housing
    cal = fetch_california_housing(as_frame=True)
    Xc, yc = _maybe_cap(cal.data, cal.target, max_rows, seed)
    datasets.append(("CaliforniaHousing", Xc, yc))

    # 2. Diabetes
    diab = load_diabetes(as_frame=True)
    Xd, yd = _maybe_cap(diab.data, diab.target, max_rows, seed)
    datasets.append(("Diabetes", Xd, yd))

    # 3. Energy Efficiency
    energy = _try_fetch_openml(name="energy_efficiency", version=1)
    if energy is not None:
        Xe, ye = _maybe_cap(energy.data, pd.to_numeric(energy.target), max_rows, seed)
        datasets.append(("Energy", Xe, ye))

    # 4. Concrete Strength
    concrete = _try_fetch_openml(name="Concrete_Compressive_Strength", version=1)
    if concrete is None:
        concrete = _try_fetch_openml(name="concrete-compressive-strength", version=1)
    if concrete is not None:
        Xcon, ycon = _maybe_cap(concrete.data, concrete.target.astype(float), max_rows, seed)
        datasets.append(("Concrete", Xcon, ycon))

    # 5. Wine Quality
    wine = _try_fetch_openml(name="wine-quality-red", version=1)
    if wine is not None:
        Xw, yw = _maybe_cap(wine.data, wine.target.astype(float), max_rows, seed)
        datasets.append(("Wine", Xw, yw))

    # 6. House Prices
    house = _try_fetch_openml(name="house_prices", version=1)
    if house is not None:
        h_data = house.data.select_dtypes(include=[np.number])
        h_target = house.target.astype(float)
        h_data, h_target = _maybe_cap(h_data, h_target, max_rows, seed)
        datasets.append(("HousePrices", h_data, h_target))


    return datasets


# -------------------------------------------------------
# MODELS
# -------------------------------------------------------

def make_models(seed=42):

    return [
        ("Linear", LinearRegression()),
        ("Ridge", Ridge(alpha=1.0)),
        ("Lasso", Lasso(alpha=0.05)),
        ("DecisionTree", DecisionTreeRegressor(max_depth=10, min_samples_leaf=5, random_state=seed)),
        ("RandomForest", RandomForestRegressor(n_estimators=150, random_state=seed, n_jobs=-1)),
        ("GradientBoost", GradientBoostingRegressor(n_estimators=200, random_state=seed)),
        ("XGBoost Regressor", XGBRegressor()),
        (
            "LightGBM-local",
            LGBMRegressor(
                num_iterations=120,
                learning_rate=0.2,
                max_depth=6,
                num_leaves=31,
                min_data_in_leaf=15,
                min_sum_hessian_in_leaf=1e-3,
                subsample=0.8,
                colsample=0.8,
                lambda_l2=0.0,
                use_histogram=True,
                n_bins=32,
                use_goss=True,
                top_rate=0.2,
                other_rate=0.1,
                default_left=True,
                random_state=seed,
            ),
        ),
    ]


# -------------------------------------------------------
# BENCHMARK RUNNER
# -------------------------------------------------------

def evaluate(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    mse = mean_squared_error(y_test, pred)
    rmse = float(np.sqrt(mse))

    return {
        "Model": name,
        "R2": round(r2_score(y_test, pred), 4),
        "RMSE": round(rmse, 4),
        "MAE": round(mean_absolute_error(y_test, pred), 4),
    }


def plot_benchmarks(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Mean scores by model
    summary = df.groupby("Model")[["R2", "RMSE", "MAE"]].mean().reset_index()
    summary = summary.sort_values("R2", ascending=False)

    plt.figure(figsize=(8, 5))
    plt.bar(summary["Model"], summary["R2"], color="#4a90e2")
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Mean R2 ↑")
    plt.title("Mean R2 by Model")
    plt.tight_layout()
    plt.savefig(out_dir / "mean_r2.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(summary["Model"], summary["RMSE"], color="#7ed321")
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Mean RMSE ↓")
    plt.title("Mean RMSE by Model")
    plt.tight_layout()
    plt.savefig(out_dir / "mean_rmse.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(summary["Model"], summary["MAE"], color="#f5a623")
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Mean MAE ↓")
    plt.title("Mean MAE by Model")
    plt.tight_layout()
    plt.savefig(out_dir / "mean_mae.png", dpi=150)
    plt.close()

    # Wins per dataset (highest R2)
    winners = df.loc[df.groupby("Dataset")["R2"].idxmax()]
    wins = winners["Model"].value_counts().reset_index()
    wins.columns = ["Model", "Wins"]

    plt.figure(figsize=(8, 5))
    plt.barh(wins["Model"], wins["Wins"], color="#50e3c2")
    plt.xlabel("# Dataset Wins (by R2)")
    plt.title("Per-Model Win Count")
    plt.tight_layout()
    plt.savefig(out_dir / "wins.png", dpi=150)
    plt.close()

    # Per-dataset R2 heatmap
    pivot = df.pivot(index="Dataset", columns="Model", values="R2")
    plt.figure(figsize=(10, max(4, 0.4 * len(pivot))))
    im = plt.imshow(pivot, aspect="auto", cmap="viridis")
    plt.colorbar(im, label="R2")
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=30, ha="right")
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.title("R2 by Dataset and Model")
    plt.tight_layout()
    plt.savefig(out_dir / "r2_heatmap.png", dpi=150)
    plt.close()


def main():

    args = parse_args()
    datasets = load_all_benchmarks(args.seed, max_rows=args.max_rows, skip_synthetic=args.skip_synthetic)
    models = make_models(args.seed)

    leaderboard = []

    for dname, X, y in datasets:

        print(f"\n ===== DATASET: {dname} ===== ")

        # clean
        X = X.dropna()
        y = y.loc[X.index]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=args.seed
        )

        # scaling for linear models
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        dataset_results = []

        for name, model in models:

            # linear = scaled | tree = raw
            if name in ("Linear", "Ridge", "Lasso"):
                res = evaluate(name, model, X_train_scaled, X_test_scaled, y_train, y_test)
            else:
                res = evaluate(name, model, X_train, X_test, y_train, y_test)

            res["Dataset"] = dname
            dataset_results.append(res)

            print(f"{name:<15}  R2={res['R2']} RMSE={res['RMSE']} MAE={res['MAE']}")

        leaderboard.extend(dataset_results)

    # -------------------------------------------------------
    # FINAL SCOREBOARD
    # -------------------------------------------------------

    df = pd.DataFrame(leaderboard)
    print("\n================== LEADERBOARD ==================")
    print(df.sort_values(["Dataset", "R2"], ascending=[True, False]))

    print("\n================== MEAN PERFORMANCE ==================")
    print(df.groupby("Model")[["R2", "RMSE", "MAE"]].mean().sort_values("R2", ascending=False))

    print("\n================== WINS PER DATASET ==================")
    winners = df.loc[df.groupby("Dataset")["R2"].idxmax()]
    print(winners[["Dataset", "Model", "R2"]])

    # --------------------
    # PLOTS
    # --------------------
    plot_benchmarks(df, Path(args.plot_dir))
    print(f"\nSaved plots to {Path(args.plot_dir).resolve()}")


if __name__ == "__main__":
    main()
