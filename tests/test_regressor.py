import os
import tempfile

import numpy as np

from lightgbm import LGBMRegressor
from lightgbm.metrics import mse_score


def _synthetic_regression(seed: int = 42):
	rng = np.random.default_rng(seed)
	X = rng.normal(size=(300, 5))
	noise = rng.normal(scale=0.05, size=300)
	y = 2.0 * X[:, 0] - 3.0 * X[:, 1] + 1.5 * np.tanh(X[:, 2]) + noise
	return X, y


def test_regressor_beats_mean_baseline():
	X, y = _synthetic_regression()
	baseline = np.full_like(y, y.mean())
	baseline_mse = mse_score(y, baseline)

	model = LGBMRegressor(
		num_iterations=50,
		learning_rate=0.1,
		max_depth=4,
		min_data_in_leaf=10,
		lambda_l2=1.0,
		subsample=0.9,
		colsample=0.9,
		random_state=0,
	)
	model.fit(X, y)
	preds = model.predict(X)

	model_mse = mse_score(y, preds)

	assert model_mse < baseline_mse * 0.2  # strong improvement over naive mean


def test_predict_shape_matches_input():
	X, y = _synthetic_regression()
	model = LGBMRegressor(num_iterations=10, learning_rate=0.2, random_state=1)
	model.fit(X, y)
	preds = model.predict(X[:15])
	assert preds.shape == (15,)


def test_early_stopping_reduces_iterations():
	X, y = _synthetic_regression()
	# Add a validation set; with patience 5 the model should stop before 60 in this simple task
	model = LGBMRegressor(
		num_iterations=60,
		learning_rate=0.1,
		max_depth=3,
		min_data_in_leaf=5,
		early_stopping_rounds=5,
		random_state=0,
	)
	model.fit(X, y, eval_set=(X[:80], y[:80]))
	assert model.best_iteration_ is not None
	assert model.best_iteration_ < 59


def test_goss_training_runs_and_importances_sum_to_one():
	X, y = _synthetic_regression()
	model = LGBMRegressor(
		num_iterations=20,
		learning_rate=0.15,
		use_goss=True,
		top_rate=0.2,
		other_rate=0.2,
		random_state=123,
	)
	model.fit(X, y)
	assert model.feature_importances_ is not None
	assert np.isclose(model.feature_importances_.sum(), 1.0)
	assert model.split_importances_ is not None
	assert np.isclose(model.split_importances_.sum(), 1.0)


def test_eval_metric_mae_and_history():
	X, y = _synthetic_regression()
	model = LGBMRegressor(
		num_iterations=15,
		learning_rate=0.15,
		eval_metric="mae",
		early_stopping_rounds=3,
		verbose_eval=None,
		random_state=7,
	)
	model.fit(X, y, eval_set=(X[:100], y[:100]))
	assert len(model.eval_history_) > 0
	assert model.best_iteration_ is not None


def test_histogram_mode_improves_baseline():
	X, y = _synthetic_regression()
	baseline_mse = mse_score(y, np.full_like(y, y.mean()))
	model = LGBMRegressor(
		num_iterations=30,
		learning_rate=0.1,
		use_histogram=True,
		n_bins=32,
		random_state=3,
	)
	model.fit(X, y)
	preds = model.predict(X)
	assert mse_score(y, preds) < baseline_mse * 0.5


def test_monotonic_constraint_enforces_order():
	rng = np.random.default_rng(0)
	X = rng.uniform(-2, 2, size=(200, 1))
	y = 2 * X[:, 0] + rng.normal(scale=0.1, size=200)
	model = LGBMRegressor(
		num_iterations=40,
		learning_rate=0.1,
		monotone_constraints=[1],
		random_state=5,
	)
	model.fit(X, y)
	X_sorted = np.sort(X, axis=0)
	preds = model.predict(X_sorted)
	assert np.all(np.diff(preds) >= -1e-6)


def test_categorical_feature_supports_splits():
	rng = np.random.default_rng(1)
	n = 240
	cat = rng.integers(0, 3, size=n)
	noise_feat = rng.normal(size=n)
	y = np.where(cat == 0, 0.5, np.where(cat == 1, 2.0, 4.0)) + rng.normal(scale=0.1, size=n)
	X = np.column_stack([cat, noise_feat])
	model = LGBMRegressor(
		num_iterations=30,
		learning_rate=0.1,
		categorical_features=[0],
		use_histogram=True,
		random_state=9,
	)
	model.fit(X, y)
	probes = np.array([[0, 0.0], [1, 0.0], [2, 0.0]])
	preds = model.predict(probes)
	assert preds[0] < preds[1] < preds[2]


def test_model_save_and_load_roundtrip():
	X, y = _synthetic_regression()
	model = LGBMRegressor(num_iterations=15, learning_rate=0.15, random_state=11, use_histogram=True)
	model.fit(X, y)
	with tempfile.NamedTemporaryFile(delete=False) as tmp:
		path = tmp.name
	try:
		model.save_model(path)
		loaded = LGBMRegressor.load_model(path)
		preds_orig = model.predict(X[:20])
		preds_loaded = loaded.predict(X[:20])
		assert np.allclose(preds_orig, preds_loaded)
	finally:
		if os.path.exists(path):
			os.remove(path)


def test_nan_routing_and_min_hessian():
	rng = np.random.default_rng(4)
	X = rng.normal(size=(120, 2))
	X[:30, 0] = np.nan
	y = 1.5 * np.nan_to_num(X[:, 0], nan=0.2) - 0.7 * X[:, 1] + rng.normal(scale=0.05, size=120)
	model = LGBMRegressor(
		num_iterations=25,
		learning_rate=0.1,
		min_sum_hessian_in_leaf=1e-3,
		default_left=True,
		random_state=8,
	)
	model.fit(X, y)
	preds = model.predict(X[:10])
	assert np.isfinite(preds).all()


def test_disallow_nan_raises():
	rng = np.random.default_rng(42)
	X = rng.normal(size=(50, 3))
	X[0, 0] = np.nan
	y = rng.normal(size=50)
	model = LGBMRegressor(num_iterations=5, learning_rate=0.1, allow_nan=False)
	try:
		model.fit(X, y)
	except ValueError:
		# expected because NaNs are not allowed
		return
	raise AssertionError("Expected ValueError when fitting with NaN and allow_nan=False")


def test_predict_disallow_nan_raises():
	rng = np.random.default_rng(123)
	X = rng.normal(size=(40, 2))
	y = rng.normal(size=40)
	model = LGBMRegressor(num_iterations=5, learning_rate=0.1)
	model.fit(X, y)
	X_new = X.copy()
	X_new[0, 0] = np.nan
	model_no_nan = LGBMRegressor(num_iterations=2, learning_rate=0.1, allow_nan=False)
	model_no_nan.fit(X, y)
	try:
		model_no_nan.predict(X_new)
	except ValueError:
		return
	raise AssertionError("Expected ValueError when predicting with NaN and allow_nan=False")
