"""
Fonctions utilitaires pour LightGBM
"""
import numpy as np
import pandas as pd
from datetime import datetime

def ValidateInputData(X , allow_nan: bool = True) -> np.ndarray: 
    """
    Valide et convertit X en numpy array
    
    Args:
        X : array-like, shape (n_samples, n_features)
        
    Returns:
        X : numpy array validé
        
    Raises:
        ValueError si X invalide
    """

    if isinstance(X, pd.DataFrame):
        X = X.values
    elif isinstance(X, pd.Series):
        X = X.to_numpy().reshape(-1, 1)
    elif not isinstance(X, np.ndarray):
        raise ValueError("Input data must be a numpy array or pandas DataFrame/Series.")
    
    if X.ndim == 1: 
        X = X.reshape(-1,1)
    elif X.ndim != 2: 
        raise ValueError("Input data must be 1D or 2D.")
    
    if np.any(np.isinf(X)):
        raise ValueError("Input data contains infinite values.")
    
    if not allow_nan and np.any(np.isnan(X)):
        raise ValueError("Input data contains NaN values.")
    
    if X.shape[0] == 0:
        raise ValueError("Input data cannot be empty.")

    return X

def check_X_y(X, y):
    """
    Vérifie que X et y ont des dimensions compatibles
    
    Args:
        X : array-like, shape (n_samples, n_features)
        y : array-like, shape (n_samples,)
        
    Returns:
        None
        
    Raises:
        ValueError si les dimensions ne sont pas compatibles
    """
    
    X = ValidateInputData(X)
    if isinstance(y, (pd.Series, pd.DataFrame)):
        y = y.values
    y = np.asarray(y)

    if y.ndim != 1:
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.ravel()
        else:  
            raise ValueError("y must be a 1D array or a 2D array with a single column.")
    
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"Number of samples in X ({X.shape[0]}) and y ({y.shape[0]}) do not match.")

    if np.any(np.isinf(y)):
        raise ValueError("y contains infinite values.")
    
    if np.any(np.isnan(y)):
        raise ValueError("y contains NaN values.")
    
    return X, y

def validate_sample_weight(sample_weight, n_samples):
    """
    Valide le vecteur sample_weight
    
    Args:
        sample_weight : array-like, shape (n_samples,)
        n_samples : int, nombre d'échantillons
        
    Returns:
        sample_weight : numpy array validé
        
    Raises:
        ValueError si sample_weight invalide
    """
    
    if sample_weight is None:
        return None
    
    sample_weight = np.asarray(sample_weight)
    if sample_weight.ndim != 1:
        raise ValueError("sample_weight must be a 1D array.")
    
    if sample_weight.shape[0] != n_samples:
        raise ValueError(f"sample_weight length ({sample_weight.shape[0]}) does not match number of samples ({n_samples}).")
    
    if np.any(sample_weight < 0):
        raise ValueError("sample_weight contains negative values.")
    
    return sample_weight

def check_is_fitted(estimator):
    """
    Vérifie qu'un modèle a été entraîné
    
    Args:
        estimator : objet du modèle
        
    Raises:
        ValueError si le modèle n'est pas entraîné
    """
    if not hasattr(estimator, 'trees_') or estimator.trees_ is None:
        raise ValueError("This model is not fitted yet. Call 'fit' first.")
    
    if len(estimator.trees_) == 0:
        raise ValueError("Model has no trees. Training might have failed.")

def validate_hyperparameters(num_iterations=None, learning_rate=None, 
                            max_depth=None, num_leaves=None, 
                            min_data_in_leaf=None):
    """
    Valide les hyperparamètres du modèle
    
    Args:
        num_iterations : int
        learning_rate : float
        max_depth : int
        num_leaves : int
        min_data_in_leaf : int
    """
    if num_iterations is not None:
        if not isinstance(num_iterations, int) or num_iterations <= 0:
            raise ValueError("num_iterations must be a positive integer.")
    
    if learning_rate is not None:
        if not isinstance(learning_rate, (int, float)) or learning_rate <= 0:
            raise ValueError("learning_rate must be a positive number.")
    
    if max_depth is not None:
        if max_depth != -1 and (not isinstance(max_depth, int) or max_depth <= 0):
            raise ValueError("max_depth must be -1 or a positive integer.")
    
    if num_leaves is not None:
        if not isinstance(num_leaves, int) or num_leaves < 2:
            raise ValueError("num_leaves must be an integer >= 2.")
    
    if min_data_in_leaf is not None:
        if not isinstance(min_data_in_leaf, int) or min_data_in_leaf <= 0:
            raise ValueError("min_data_in_leaf must be a positive integer.")
        

def log_message(message, verbose=0):
    """
    Affiche un message si verbose activé
    """
    if verbose > 0:
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}")


def log_training_progress(iteration, total_iterations, loss, verbose=0):
    """
    Affiche la progression de l'entraînement
    """
    if verbose > 0 and (iteration % max(1, total_iterations // 10) == 0 or iteration == total_iterations):
        print(f"[LightGBM] Iteration {iteration}/{total_iterations} - Loss: {loss:.6f}")