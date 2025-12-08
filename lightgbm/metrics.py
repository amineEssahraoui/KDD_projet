"""Basic metrics used in tests."""

import numpy as np


def mse_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
	return float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))

def mae_score(y_true: np.ndarray, y_pred: np.ndarray) -> float: 
	y_true = np.asarray (y_true)
	y_pred = np.asarray(y_pred)
	return float(np.sum (np.abs (y_true - y_pred)))

def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
	y_true = np.asarray(y_true)
	y_pred = np.asarray(y_pred)
	ss_res = np.sum((y_true - y_pred) ** 2)
	ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
	eps = 1e-15
	return float(1 - ss_res / (ss_tot + eps))