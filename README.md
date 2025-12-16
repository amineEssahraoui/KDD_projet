# LightGBM From Scratch

Une implémentation complète de LightGBM (Light Gradient Boosting Machine) en pur Python/NumPy, développée comme projet académique sur les algorithmes d'arbres de décision.

**🎯 Zéro dépendance sklearn** - Tous les algorithmes implémentés from scratch avec NumPy uniquement !

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Only-green)](https://numpy.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Table des matières

- [Features](#-features)
- [Installation](#-installation)
- [Démarrage rapide](#-démarrage-rapide)
- [Structure du projet](#-structure-du-projet)
- [Documentation](#-documentation)
- [Tests et benchmarks](#-tests-et-benchmarks)
- [Exemples d'utilisation](#-exemples-dutilisation)
- [Architecture](#️-architecture)
- [Contributions](#-contributions)
- [Auteurs](#-auteurs)
- [Licence](#-licence)
- [Références](#-références)

---

## 🚀 Features

### Algorithmes implémentés

- ✅ **Classification binaire & multiclasse** - Support complet des deux
- ✅ **Régression** - MSE, MAE, Huber, Quantile loss
- ✅ **Croissance leaf-wise** - Optimisation clé de LightGBM
- ✅ **GOSS** (Gradient-based One-Side Sampling) - Entraînement ~2-3x plus rapide
- ✅ **Histogram Binning** - Recherche efficace de splits
- ✅ **EFB** (Exclusive Feature Bundling) - Pour données haute dimension
- ✅ **Early Stopping** - Prévient l'overfitting
- ✅ **Régularisation L1/L2** - Contrôle de la complexité
- ✅ **Feature Subsampling** - Sélection aléatoire de features
- ✅ **Sample Weighting** - Support des poids d'échantillons
- ✅ **API compatible sklearn** - Interface familière

### Fonctions de perte disponibles

**Régression** :
- `MSELoss` : Mean Squared Error (L2)
- `MAELoss` : Mean Absolute Error (L1)
- `HuberLoss` : Robuste aux outliers
- `QuantileLoss` : Régression quantile

**Classification** :
- `BinaryCrossEntropyLoss` : Classification binaire
- `MultiClassCrossEntropyLoss` : Classification multiclasse

---

## 📦 Installation

### Depuis source

```bash
# Cloner le dépôt
git clone https://github.com/amineEssahraoui/KDD_projet.git
cd KDD_projet

# Installation en mode développement
pip install -e .

# Ou installer seulement les dépendances
pip install -r requirements.txt
```

### Dépendances

**Runtime** (obligatoire) :
```
numpy>=1.24.0
```

**Development** (optionnel, pour tests/benchmarks) :
```
pytest>=7.0.0
pandas>=2.0.0
scikit-learn>=1.2.0
scipy>=1.11.0
matplotlib>=3.8.0
seaborn>=0.13.0
```

---

## 🎯 Démarrage rapide

### Classification

```python
from lightgbm import LGBMClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Générer données
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Entraîner modèle
clf = LGBMClassifier(num_iterations=100, learning_rate=0.1, max_depth=6)
clf.fit(X_train, y_train)

# Prédictions
predictions = clf.predict(X_test)
probabilities = clf.predict_proba(X_test)

print(f"Accuracy: {(predictions == y_test).mean():.4f}")
```

### Régression

```python
from lightgbm import LGBMRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# Générer données
X, y = make_regression(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Entraîner modèle
reg = LGBMRegressor(num_iterations=100, learning_rate=0.1, max_depth=6)
reg.fit(X_train, y_train)

# Prédictions
predictions = reg.predict(X_test)
```

### Early Stopping

```python
clf = LGBMClassifier(num_iterations=1000, early_stopping_rounds=10)
clf.fit(X_train, y_train, eval_set=(X_val, y_val))
print(f"Arrêté à l'itération: {clf.n_iter_}")
```

### GOSS (accélération)

```python
clf = LGBMClassifier(
    num_iterations=100, 
    enable_goss=True, 
    goss_top_rate=0.2
)
clf.fit(X_train, y_train)
```

---

## 📁 Structure du projet

```
KDD_projet/
│
├── src/lightgbm/              # 📦 Package principal
│   ├── __init__.py            # Exports publics
│   ├── base.py                # Classes de base (BaseEstimator, BoosterParams, Callback)
│   ├── lgbm_classifier.py     # LGBMClassifier
│   ├── lgbm_regressor.py      # LGBMRegressor
│   ├── tree.py                # DecisionTree avec croissance leaf-wise
│   ├── histogram.py           # Histogram binning (intégré dans tree.py)
│   ├── goss.py                # GOSS sampling (classe GOSS, apply_goss)
│   ├── efb.py                 # Exclusive Feature Bundling (FeatureBundler, bundle_features)
│   ├── loss_functions.py      # Fonctions de perte + gradients/hessians
│   └── utils.py               # Validation et utilitaires
│
├── tests/                     # ✅ Suite de tests
│   ├── test_classifier.py     # Tests LGBMClassifier
│   ├── test_regressor.py      # Tests LGBMRegressor
│   ├── test_tree.py           # Tests DecisionTree
│   ├── test_goss.py           # Tests GOSS
│   ├── test_utils.py          # Tests utilitaires
│   ├── test_math_integrity.py # Validation mathématique
│   └── test_logic_sanity.py   # Tests de sanité
│
├── benchmarks/                # 📊 Comparaisons de performance
│   └── benchmark_comparison.py # Compare avec sklearn GradientBoosting
│
├── examples/                  # 📖 Exemples d'utilisation
│   ├── complete_testing.ipynb # Notebook complet avec exemples
│   └── regression_pipeline.py # Pipeline de régression
│
├── docs/                      # 📚 Documentation
│   ├── ARCHITECTURE.md        # Architecture détaillée
│   └── IMPLEMENTATION_GUIDE.md # Guide d'utilisation
│
├── .github/workflows/         # 🔄 CI/CD
│   └── ci.yml                 # GitHub Actions
│
├── pyproject.toml             # Configuration du projet
├── requirements.txt           # Dépendances
├── LICENSE                    # Licence MIT
└── README.md                  # Ce fichier
```

---

## 📚 Documentation

### Fichiers principaux

#### Package source (`src/lightgbm/`)

| Fichier | Description | Classes/Fonctions principales |
|---------|-------------|-------------------------------|
| `__init__.py` | Point d'entrée du package | Exports publics |
| `base.py` | Classes abstraites de base | `BaseEstimator`, `BoosterParams`, `Callback`, `EarlyStoppingCallback` |
| `lgbm_classifier.py` | Classificateur gradient boosting | `LGBMClassifier` |
| `lgbm_regressor.py` | Régresseur gradient boosting | `LGBMRegressor` |
| `tree.py` | Arbre de décision | `DecisionTree`, `TreeNode`, `SplitInfo` |
| `loss_functions.py` | Fonctions de perte | `MSELoss`, `MAELoss`, `HuberLoss`, `QuantileLoss`, `BinaryCrossEntropyLoss`, `MultiClassCrossEntropyLoss`, `get_loss_function()` |
| `goss.py` | GOSS sampling | `GOSS`, `apply_goss()` |
| `efb.py` | Feature bundling | `FeatureBundler`, `bundle_features()` |
| `utils.py` | Utilitaires | `check_array()`, `check_X_y()`, `train_test_split()`, `accuracy_score()`, `mean_squared_error()`, etc. |

### Imports courants

```python
# Estimateurs
from lightgbm import LGBMClassifier, LGBMRegressor

# Arbres et structures
from lightgbm import DecisionTree, TreeNode, SplitInfo

# Fonctions de perte
from lightgbm.loss_functions import (
    MSELoss, MAELoss, HuberLoss, QuantileLoss,
    BinaryCrossEntropyLoss, MultiClassCrossEntropyLoss,
    get_loss_function
)

# Features avancées
from lightgbm import GOSS, FeatureBundler

# Utilitaires
from lightgbm.utils import (
    train_test_split, accuracy_score, mean_squared_error,
    mean_absolute_error, r2_score
)

# Callbacks
from lightgbm.base import EarlyStoppingCallback
```

### Guides détaillés

- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** : Architecture complète du système
  - Vue d'ensemble des modules
  - Diagrammes de classes et séquences
  - Formules mathématiques
  - Flux de données

- **[IMPLEMENTATION_GUIDE.md](docs/IMPLEMENTATION_GUIDE.md)** : Guide d'utilisation pratique
  - Exemples détaillés
  - Tuning des hyperparamètres
  - Features avancées (GOSS, EFB, callbacks)
  - Troubleshooting

---

## ✅ Tests et benchmarks

### Exécuter les tests

```bash
# Tous les tests
python -m pytest tests/ -v

# Tests spécifiques
python -m pytest tests/test_classifier.py -v
python -m pytest tests/test_regressor.py -v

# Avec couverture
python -m pytest tests/ --cov=src/lightgbm --cov-report=html
```

### Tests disponibles

| Fichier de test | Description |
|-----------------|-------------|
| `test_classifier.py` | Classification binaire et multiclasse |
| `test_regressor.py` | Régression avec différentes loss |
| `test_tree.py` | Arbres de décision leaf-wise |
| `test_goss.py` | GOSS sampling |
| `test_utils.py` | Fonctions utilitaires |
| `test_math_integrity.py` | Validation mathématique (gradients, hessians, gains) |
| `test_logic_sanity.py` | Tests de sanité (overfitting, convergence) |

### Benchmarks

```bash
# Comparer avec sklearn
python benchmarks/benchmark_comparison.py
```

**Résultats typiques** (n=2000, 50 arbres) :

| Tâche | Notre LightGBM | sklearn | Rapport vitesse |
|-------|---------------|---------|-----------------|
| Classification binaire | 89.8% acc | 89.6% acc | ~2x plus lent |
| Régression | 90.9% R² | 89.1% R² | ~2x plus lent |
| Multiclasse (3 classes) | 95.0% acc | - | Fonctionnel ! |

Notre implémentation atteint une précision comparable à sklearn tout en étant seulement ~2x plus lente (Python pur vs Cython).

---

## 📖 Exemples d'utilisation

### 1. Régression basique

```python
from lightgbm import LGBMRegressor
import numpy as np

# Données
X = np.random.randn(1000, 10)
y = 3*X[:, 0] + 2*X[:, 1] + np.random.randn(1000)*0.5

# Modèle
model = LGBMRegressor(
    num_iterations=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42
)

model.fit(X, y)
predictions = model.predict(X)

print(f"MSE: {np.mean((y - predictions)**2):.4f}")
```

### 2. Classification avec validation

```python
from lightgbm import LGBMClassifier
from lightgbm.utils import train_test_split, accuracy_score
import numpy as np

# Données
X = np.random.randn(1000, 15)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Modèle
clf = LGBMClassifier(num_iterations=100, random_state=42)
clf.fit(X_train, y_train)

# Évaluation
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}")
```

### 3. Fonction de perte personnalisée (Huber)

```python
from lightgbm import LGBMRegressor
from lightgbm.loss_functions import HuberLoss
import numpy as np

# Données avec outliers
X = np.random.randn(500, 5)
y = X[:, 0] + 2*X[:, 1] + np.random.randn(500)*0.5
y[::50] += 10  # Ajouter outliers

# Huber loss (robuste)
model = LGBMRegressor(
    objective=HuberLoss(delta=1.0),
    num_iterations=100,
    learning_rate=0.1
)
model.fit(X, y)
```

### 4. GOSS pour grandes données

```python
from lightgbm import LGBMRegressor
import numpy as np

# Grandes données
X = np.random.randn(50000, 30)
y = X[:, 0] + 2*X[:, 1] + np.random.randn(50000)*0.5

# Avec GOSS (plus rapide)
model = LGBMRegressor(
    num_iterations=100,
    enable_goss=True,
    goss_top_rate=0.2,
    goss_other_rate=0.1,
    use_histogram=True,
    max_bins=128
)
model.fit(X, y)
```

### 5. Early stopping avec validation

```python
from lightgbm import LGBMClassifier
from lightgbm.utils import train_test_split
import numpy as np

X = np.random.randn(1000, 10)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

clf = LGBMClassifier(
    num_iterations=1000,
    early_stopping_rounds=20,
    learning_rate=0.1,
    verbose=1
)

clf.fit(X_train, y_train, eval_set=(X_val, y_val))
print(f"Arrêté à: {clf.n_iter_}")
```

### Plus d'exemples

Consultez le notebook `examples/complete_testing.ipynb` pour des exemples complets avec :
- Régression California Housing
- Régression avec NaN et features sparse
- Classification crédit avec classes déséquilibrées
- Comparaisons de performances

---

## 🏗️ Architecture

### Principes de conception

Notre implémentation suit fidèlement le papier LightGBM original avec ces différences clés par rapport au gradient boosting standard :

1. **Croissance leaf-wise** vs level-wise
   - Sélectionne et split la feuille avec le gain maximal
   - Plus efficace que croissance par niveau (XGBoost)

2. **GOSS** (Gradient-based One-Side Sampling)
   - Garde tous les échantillons avec grands gradients
   - Échantillonne les petits gradients
   - Réduit données de ~70% sans perte de précision

3. **EFB** (Exclusive Feature Bundling)
   - Combine features mutuellement exclusives
   - Réduit dimensionnalité pour données sparse

4. **Histogram Binning**
   - Discrétise features continues en bins
   - Complexité O(max_bins) au lieu de O(n_samples)

### Formules mathématiques clés

**Gain de split** :
```
Gain = [G²_L/(H_L+λ) + G²_R/(H_R+λ) - G²/(H+λ)] / 2 - γ

où:
  G = Σ gradients
  H = Σ hessians
  λ = lambda_l2 (régularisation L2)
  γ = min_gain_to_split
```

**Valeur de feuille optimale** :
```
w* = -G / (H + λ)
```

**Prédiction finale** :
```
ŷ = init_prediction + learning_rate × Σ tree_k(x)
```

Pour plus de détails, voir [ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## 🎛️ Hyperparamètres

| Paramètre | Default | Description |
|-----------|---------|-------------|
| `num_iterations` / `n_estimators` | 100 | Nombre d'arbres |
| `learning_rate` | 0.1 | Taux d'apprentissage |
| `max_depth` | -1 | Profondeur max (-1 = illimité) |
| `num_leaves` | 31 | Nombre max de feuilles par arbre |
| `min_data_in_leaf` / `min_samples_leaf` | 20 | Échantillons min par feuille |
| `lambda_l1` / `reg_alpha` | 0.0 | Régularisation L1 |
| `lambda_l2` / `reg_lambda` | 0.0 | Régularisation L2 |
| `feature_fraction` | 1.0 | Fraction de features par arbre |
| `bagging_fraction` | 1.0 | Fraction d'échantillons par arbre |
| `enable_goss` | False | Activer GOSS |
| `use_histogram` | False | Activer histogram binning |
| `early_stopping_rounds` | None | Patience pour early stopping |

Voir [IMPLEMENTATION_GUIDE.md](docs/IMPLEMENTATION_GUIDE.md) pour guide complet de tuning.

---

## 🤝 Contributions

Les contributions sont les bienvenues ! Pour contribuer :

1. Fork le dépôt
2. Créer une branche feature (`git checkout -b feature/amazing-feature`)
3. Commit les changements (`git commit -m 'Add amazing feature'`)
4. Push vers la branche (`git push origin feature/amazing-feature`)
5. Ouvrir une Pull Request

### Guidelines

- Suivre le style de code existant
- Ajouter des tests pour nouvelles features
- Mettre à jour la documentation
- S'assurer que tous les tests passent

---

## 👥 Auteurs

- **Amine Essahraoui** 
- **Mohammed Amine Zbida**
- **Abderrarak Khall**

---

## 📄 Licence

Ce projet est sous licence MIT - voir le fichier [LICENSE](LICENSE) pour détails.

---

## 📚 Références

### Papiers scientifiques

1. **Ke, G., et al.** (2017). "LightGBM: A Highly Efficient Gradient Boosting Decision Tree." 
   *NeurIPS 2017*. 
   [Lien](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree)

2. **Chen, T., & Guestrin, C.** (2016). "XGBoost: A Scalable Tree Boosting System." 
   *KDD 2016*. 
   [Lien](https://arxiv.org/abs/1603.02754)

3. **Friedman, J. H.** (2001). "Greedy function approximation: A gradient boosting machine." 
   *Annals of statistics*.

### Ressources en ligne

- [LightGBM Documentation officielle](https://lightgbm.readthedocs.io/)
- [Gradient Boosting Explained](https://explained.ai/gradient-boosting/)
- [Understanding LightGBM Parameters](https://lightgbm.readthedocs.io/en/latest/Parameters.html)

---

**Dernière mise à jour** : Décembre 2025  
**Version** : 1.0.0  
**Status** : ✅ Production Ready