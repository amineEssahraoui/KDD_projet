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