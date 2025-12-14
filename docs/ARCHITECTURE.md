# Architecture LightGBM

## Vue d'ensemble

Ce document décrit l'architecture technique de l'implémentation LightGBM from scratch.

## Structure des modules

### `base.py` - Classe de base

- `BaseEstimator` : Classe abstraite définissant l'interface commune
  - Initialisation des hyperparamètres
  - Méthodes abstraites `fit()` et `predict()`
  - Calcul des gradients et hessians

### `tree.py` - Arbres de décision

- Implémentation de Decision Trees
- Support du regression et classification
- Stratégie de construction `leaf_wise`

### `lgbm_classifier.py` - Classifier

- Classification multi-classe et binaire
- Utilise des arbres de décision
- Fonction de perte adaptée à la classification

### `lgbm_regressor.py` - Regressor

- Régression linéaire multi-variable
- Fonction de perte MSE
- Prédictions continues

## Optimisations

### `goss.py` - Gradient-based One-Side Sampling

- Réduit le nombre d'instances
- Garde les gradients larges
- Améliore l'efficacité

### `efb.py` - Exclusive Feature Bundling

- Combine les features mutellement exclusives
- Réduit la dimensionalité
- Accélère l'apprentissage

### `histogramme.py` - Binning des features

- Discrétise les features continues
- Réduit la consommation mémoire
- Accélère la construction des splits

## Loss Functions

Module `loss_functions.py` :

- Cross-entropy pour la classification
- MSE pour la régression
- Calcul des gradients et hessians

## Métriques

Module `metrics.py` :

- Accuracy, Precision, Recall, F1 (classification)
- RMSE, MAE, R² (régression)
- Courves ROC et matrices de confusion

## Utilities

Module `utils.py` :

- Fonctions helpers
- Chargement de données
- Normalisation et preprocessing
