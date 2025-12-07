"""Basic metrics used in tests."""

import numpy as np


def mse_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
	return float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))

