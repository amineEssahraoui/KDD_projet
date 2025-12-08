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
		return 1 / ((rmse+eps) * n) * (1 - ((y_pred - y_true) ** 2) / ((rmse+eps) ** 2 * n))
	
class HUBERLoss(LossFunction):
	"""Huber loss with constant Hessian used by LightGBM for regression."""
	
	def __init__(self, delta: float = 1.0):
		self.delta = delta
	
	def loss(self , y_true: np.ndarray, y_pred: np.ndarray) -> float:
		residual = y_true - y_pred
		is_small_error = np.abs(residual) <= self.delta
		squared_loss = 0.5 * (residual ** 2)
		linear_loss = self.delta * (np.abs(residual) - 0.5 * self.delta)
		return float(np.sum(np.where(is_small_error, squared_loss, linear_loss)))
	
	def gradient(self , y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
		residual = y_pred - y_true
		is_small_error = np.abs(residual) <= self.delta
		grad_small_error = residual
		grad_large_error = self.delta * np.sign(residual)
		return np.where(is_small_error, grad_small_error, grad_large_error)
	
	def hessian(self , y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
		residual = y_pred - y_true
		is_small_error = np.abs(residual) <= self.delta
		return np.where(is_small_error, 1.0, 0.0)
	
class QUANTILELoss(LossFunction):
	"""Quantile loss with constant Hessian used by LightGBM for regression."""
	
	def __init__(self, quantile: float = 0.5):
		self.quantile = quantile
	