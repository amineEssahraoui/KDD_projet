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
		return float(0.5 * np.mean((y_true - y_pred) ** 2))

	def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
		# First derivative wrt predictions
		return y_pred - y_true

	def hessian(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
		# Second derivative is constant for squared error.
		return np.ones_like(y_pred)

class MAELoss(LossFunction):
	"""Mean absolute error with constant Hessian used by LightGBM for regression."""
	def loss (self , y_true : np.ndarray , y_pred : np.ndarray ) -> float :
		return float ( np . mean ( np . abs ( y_true - y_pred ) ) )	
	