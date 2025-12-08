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
    
    if X.ndim != 2:
        raise ValueError("Input data must be 2-dimensional.")
    
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
    y = np.asarray(y)
    
    if X.shape[0] != y.shape[0]:
        raise ValueError("Number of samples in X and y must be equal.")
    
    if X.shape[0] == 0 or y.shape[0] == 0:
        raise ValueError("X and y cannot be empty.")
    
    if y.shape == ():
        raise ValueError("y must be a 1-dimensional array.")
    
    return X, y

