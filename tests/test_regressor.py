"""
Test suite for LGBMRegressor.

Tests the custom LightGBM regressor implementation.
NO sklearn dependencies.
"""

import os
import tempfile

import numpy as np
import pytest

import sys
from pathlib import Path
_repo_root = Path(__file__).resolve().parents[1]
src_path = str(_repo_root / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from lightgbm import LGBMRegressor
from lightgbm.utils import mean_squared_error as mse_score


def _synthetic_regression(seed: int = 42):
    """Generate synthetic regression data."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(300, 5))
    noise = rng.normal(scale=0.05, size=300)
    y = 2.0 * X[:, 0] - 3.0 * X[:, 1] + 1.5 * np.tanh(X[:, 2]) + noise
    return X, y


def test_regressor_beats_mean_baseline():
    """Test that regressor beats mean baseline."""
    X, y = _synthetic_regression()
    baseline = np.full_like(y, y.mean())
    baseline_mse = mse_score(y, baseline)

    model = LGBMRegressor(
        num_iterations=50,
        learning_rate=0.1,
        max_depth=4,
        min_data_in_leaf=10,
        lambda_l2=1.0,
        bagging_fraction=0.9,
        feature_fraction=0.9,
        random_state=0,
    )
    model.fit(X, y)
    preds = model.predict(X)

    model_mse = mse_score(y, preds)
    assert model_mse < baseline_mse * 0.2  # strong improvement over naive mean


def test_predict_shape_matches_input():
    """Test that predict returns correct shape."""
    X, y = _synthetic_regression()
    model = LGBMRegressor(num_iterations=10, learning_rate=0.2, random_state=1)
    model.fit(X, y)
    preds = model.predict(X[:15])
    assert preds.shape == (15,)


def test_early_stopping():
    """Test early stopping functionality."""
    X, y = _synthetic_regression()
    model = LGBMRegressor(
        num_iterations=60,
        learning_rate=0.1,
        max_depth=3,
        min_data_in_leaf=5,
        early_stopping_rounds=5,
        random_state=0,
    )
    model.fit(X, y, eval_set=(X[:80], y[:80]))
    # Should be able to make predictions after fitting
    preds = model.predict(X)
    assert preds.shape == (300,)


def test_goss_training():
    """Test GOSS training."""
    X, y = _synthetic_regression()
    model = LGBMRegressor(
        num_iterations=20,
        learning_rate=0.15,
        enable_goss=True,
        goss_top_rate=0.2,
        goss_other_rate=0.2,
        random_state=123,
    )
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (300,)


def test_histogram_mode():
    """Test histogram-based training."""
    X, y = _synthetic_regression()
    baseline_mse = mse_score(y, np.full_like(y, y.mean()))
    model = LGBMRegressor(
        num_iterations=30,
        learning_rate=0.1,
        use_histogram=True,
        max_bins=32,
        random_state=3,
    )
    model.fit(X, y)
    preds = model.predict(X)
    assert mse_score(y, preds) < baseline_mse * 0.5


def test_regularization():
    """Test L1 and L2 regularization."""
    X, y = _synthetic_regression()
    model = LGBMRegressor(
        num_iterations=20,
        learning_rate=0.1,
        lambda_l1=0.1,
        lambda_l2=0.1,
        random_state=42,
    )
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (300,)


def test_feature_fraction():
    """Test feature subsampling."""
    X, y = _synthetic_regression()
    model = LGBMRegressor(
        num_iterations=20,
        learning_rate=0.1,
        feature_fraction=0.5,
        random_state=42,
    )
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (300,)


def test_bagging():
    """Test row subsampling (bagging)."""
    X, y = _synthetic_regression()
    model = LGBMRegressor(
        num_iterations=20,
        learning_rate=0.1,
        bagging_fraction=0.8,
        bagging_freq=1,
        random_state=42,
    )
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (300,)


def test_learning_rate_decay():
    """Test learning rate decay."""
    X, y = _synthetic_regression()
    model = LGBMRegressor(
        num_iterations=30,
        learning_rate=0.1,
        lr_decay=0.99,
        random_state=42,
    )
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (300,)


def test_reproducibility():
    """Test reproducibility with same random_state."""
    X, y = _synthetic_regression()
    
    model1 = LGBMRegressor(num_iterations=20, random_state=42)
    model1.fit(X, y)
    preds1 = model1.predict(X)
    
    model2 = LGBMRegressor(num_iterations=20, random_state=42)
    model2.fit(X, y)
    preds2 = model2.predict(X)
    
    assert np.allclose(preds1, preds2)


def test_disallow_nan_raises():
    """Test that NaN raises error when allow_nan=False."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(50, 3))
    X[0, 0] = np.nan
    y = rng.normal(size=50)
    model = LGBMRegressor(num_iterations=5, learning_rate=0.1, allow_nan=False)
    
    with pytest.raises(ValueError):
        model.fit(X, y)


def test_allow_nan_works():
    """Test that NaN is handled when allow_nan=True."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(50, 3))
    X[0, 0] = np.nan
    y = rng.normal(size=50)
    model = LGBMRegressor(num_iterations=5, learning_rate=0.1, allow_nan=True)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (50,)


def test_sample_weight():
    """Test sample weight support."""
    X, y = _synthetic_regression()
    weights = np.random.rand(len(y))
    
    model = LGBMRegressor(num_iterations=20, random_state=42)
    model.fit(X, y, sample_weight=weights)
    preds = model.predict(X)
    assert preds.shape == (300,)
