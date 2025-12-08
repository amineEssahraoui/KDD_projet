"""
Fonctions utilitaires pour LightGBM
"""
import numpy as np
import pandas as pd
from datetime import datetime

def ValidateInputData(X): 
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
    
    if np.any(np.isnan(X)):
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