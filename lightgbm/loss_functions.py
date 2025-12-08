"""Loss functions with first and second order derivatives."""

import abc
import numpy as np


class LossFunction(abc.ABC):
	@abc.abstractmethod
	def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
		raise NotImplementedError

	@abc.abstractmethod
	def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
		raise NotImplementedError

	@abc.abstractmethod
	def hessian(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
		raise NotImplementedError


class MSELoss(LossFunction):
	"""Mean squared error with constant Hessian used by LightGBM for regression."""

	def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
		return float(0.5 * np.sum((y_true - y_pred) ** 2))

	def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
		# First derivative wrt predictions
		return y_pred - y_true

	def hessian(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
		# Second derivative is constant for squared error.
		return np.ones_like(y_pred)

class MAELoss(LossFunction):
	"""Mean absolute error with constant Hessian used by LightGBM for regression."""
	def loss (self , y_true : np.ndarray , y_pred : np.ndarray ) -> float :
		return float (np.sum (np.abs (y_true - y_pred)))	
	
	def gradient ( self , y_true : np.ndarray , y_pred : np.ndarray ) -> np.ndarray :
		return np.sign(y_pred - y_true)
	
	def hessian ( self , y_true : np.ndarray , y_pred : np.ndarray ) -> np.ndarray :
		return np.zeros_like(y_pred)
	
class RMSELoss(LossFunction): 
	"""Root mean squared error with constant Hessian used by LightGBM for regression."""
	
	def loss(self , y_true: np.ndarray, y_pred: np.ndarray) -> float:
		return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
	
	def gradient(self , y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
		eps = 1e-15
		rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
		n = len(y_true)
		return (y_pred - y_true) / ((rmse+eps)  * n)
	
	def hessian(self , y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
		n = len(y_true)
		eps = 1e-15
		rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
		return  1 / ((rmse+eps) * n) * (1 - ((y_pred - y_true) ** 2) / ((rmse+eps) ** 2 * n))
	
class R2Loss(LossFunction):
	"""R-squared loss function for regression."""
	
	def loss(self , y_true: np.ndarray, y_pred: np.ndarray) -> float:
		eps = 1e-15
		ss_res = np.sum((y_true - y_pred) ** 2)
		ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
		return float(1 - ss_res / (ss_tot + eps))
	