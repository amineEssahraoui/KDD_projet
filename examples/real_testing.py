from __future__ import annotations

import numpy as np
import pandas as pd
import time
import sys
from pathlib import Path

from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error, mean_pinball_loss

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from lightgbm.lgbm_regressor import LGBMRegressor  # type: ignore
from lightgbm.loss_functions import HUBERLoss, QUANTILELoss

def evaluate(model_name, model, X_train, X_test, y_train, y_test):
    start = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    if model_name == "GradientBoost":
        num_trees = len(model.estimators_)
    elif model_name == "XGBoost Regressor":
        num_trees = model.get_num_boosting_rounds()
    else:
        num_trees = len(model.trees_)
    
    return r2, rmse, mae, training_time, num_trees, y_pred

    
X, y = fetch_california_housing(as_frame=True, return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

regressors = [
        ("GradientBoost", GradientBoostingRegressor(n_estimators=200, random_state=42)),
        ("XGBoost Regressor", XGBRegressor()),
        (
            "LightGBM-leaf_wise_mse",
            LGBMRegressor(
                loss="mse",
                learning_rate=0.1,
                num_iterations=200,
                max_depth=6,
                num_leaves=31,
                min_data_in_leaf=20,
                min_sum_hessian_in_leaf=1e-3,
                lambda_l2=0.0,
                lambda_l1=0.0,
                subsample=0.8,
                colsample=0.8,
                use_histogram=True,
                n_bins=64,
                use_efb=False,
                use_goss=False
            )
        ),
        (
            "LightGBM-histogram_efb_goss",
            LGBMRegressor(
                loss="mse",
                learning_rate=0.15,
                num_iterations=150,
                max_depth=6,
                num_leaves=31,
                min_data_in_leaf=20,
                min_sum_hessian_in_leaf=1e-3,
                lambda_l2=0.1,
                lambda_l1=0.0,
                subsample=0.8,
                colsample=0.8,
                use_histogram=True,
                n_bins=64,
                use_efb=True,
                efb_conflict_rate=0.0,
                use_goss=True,
                top_rate=0.2,
                other_rate=0.1
            ),
        ),
        (
            "LightGBM-huber_robust",
            LGBMRegressor(
                loss=HUBERLoss(delta=1.0),
                learning_rate=0.1,
                num_iterations=200,
                max_depth=6,
                num_leaves=31,
                min_data_in_leaf=20,
                min_sum_hessian_in_leaf=1e-3,
                lambda_l2=0.1,
                lambda_l1=0.05,
                subsample=0.9,
                colsample=0.9,
                use_histogram=True,
                n_bins=64,
                use_efb=False,
                use_goss=False,
            ),
        ),
        (
            "LightGBM-regularized_shallow",
            LGBMRegressor(
                loss="mse",
                learning_rate=0.05,
                num_iterations=400,
                max_depth=4,
                num_leaves=15,
                min_data_in_leaf=30,
                min_sum_hessian_in_leaf=1e-3,
                lambda_l2=1.0,
                lambda_l1=0.1,
                subsample=0.7,
                colsample=0.7,
                use_histogram=True,
                n_bins=32,
                use_efb=False,
                use_goss=False,
            ),
        ),
        (
            "LightGBM-fast_wide",
            LGBMRegressor(
                loss="mse",
                learning_rate=0.15,
                num_iterations=120,
                max_depth=8,
                num_leaves=63,
                min_data_in_leaf=10,
                min_sum_hessian_in_leaf=1e-3,
                lambda_l2=0.0,
                lambda_l1=0.0,
                subsample=0.9,
                colsample=0.9,
                use_histogram=True,
                n_bins=32,
                use_efb=False,
                use_goss=True,
                top_rate=0.2,
                other_rate=0.1,
                n_jobs=4,
            ),
        ),
        (
            "LightGBM-goss_parallel",
            LGBMRegressor(
                loss="mse",
                learning_rate=0.08,
                num_iterations=250,
                max_depth=7,
                num_leaves=63,
                min_data_in_leaf=15,
                min_sum_hessian_in_leaf=1e-3,
                lambda_l2=0.2,
                lambda_l1=0.05,
                subsample=0.9,
                colsample=0.9,
                use_histogram=True,
                n_bins=64,
                use_efb=False,
                use_goss=True,
                top_rate=0.15,
                other_rate=0.1,
                n_jobs=4,
            ),
        ),
        (
            "LightGBM-efb_parallel",
            LGBMRegressor(
                loss="mse",
                learning_rate=0.1,
                num_iterations=180,
                max_depth=6,
                num_leaves=40,
                min_data_in_leaf=20,
                min_sum_hessian_in_leaf=1e-3,
                lambda_l2=0.1,
                lambda_l1=0.0,
                subsample=0.8,
                colsample=0.9,
                use_histogram=True,
                n_bins=64,
                use_efb=True,
                efb_conflict_rate=0.01,
                use_goss=False,
                n_jobs=4,
            ),
        ),
        (
            "LightGBM-exact_small",
            LGBMRegressor(
                loss="mse",
                learning_rate=0.05,
                num_iterations=120,
                max_depth=5,
                num_leaves=31,
                min_data_in_leaf=25,
                min_sum_hessian_in_leaf=1e-3,
                lambda_l2=0.1,
                lambda_l1=0.05,
                subsample=0.8,
                colsample=0.8,
                use_histogram=False,
                use_efb=False,
                use_goss=False,
                n_jobs=4,
            ),
        ),
        (
            "LightGBM-quantile_p50",
            LGBMRegressor(
                loss=QUANTILELoss(quantile=0.5),
                learning_rate=0.08,
                num_iterations=300,
                max_depth=5,
                num_leaves=25,
                min_data_in_leaf=20,
                min_sum_hessian_in_leaf=1e-3,
                lambda_l2=0.2,
                lambda_l1=0.05,
                subsample=0.8,
                colsample=0.8,
                use_histogram=True,
                n_bins=64,
                use_efb=False,
                use_goss=False,
            ),
        )
    ]

results = []
for model_name, model in regressors:
    r2, rmse, mae, training_time, num_trees, y_pred = evaluate(model_name, model, X_train, X_test, y_train, y_test)
    if model_name != "LightGBM-quantile_p50":
        print(f"{model_name}                       R²={r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}  t={training_time:.4f}  N={num_trees}")
        results.append({
            'Model': model_name,
            'R²': r2,
            'RMSE': rmse,
            'MAE': mae,
            'Training time': training_time,
            'Num Trees Used': num_trees
        })
    else:
        pb_loss = mean_pinball_loss(y_test, y_pred, alpha=0.5)
        cvg_err = np.mean(y_test <= y_pred)
        print(f"{model_name}                       mean_pinball_loss={pb_loss:.4f}  coverage_error={cvg_err:.4f}  t={training_time:.4f}  N={num_trees}")

df_results = pd.DataFrame(results)
print(df_results)