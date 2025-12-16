# Guide d'implémentation LightGBM From Scratch

## Table des matières
- [Introduction](#introduction)
- [Installation et configuration](#installation-et-configuration)
- [Guide d'utilisation](#guide-dutilisation)
- [Paramètres et tuning](#paramètres-et-tuning)
- [Features avancées](#features-avancées)
- [Benchmarking et comparaisons](#benchmarking-et-comparaisons)
- [Debugging et troubleshooting](#debugging-et-troubleshooting)
- [Exemples complets](#exemples-complets)

---

## Introduction

Ce guide explique comment utiliser l'implémentation LightGBM from scratch. L'API est conçue pour être compatible avec scikit-learn tout en n'ayant **aucune dépendance sklearn** dans le code source.

**Points clés** :
- ✅ API familière (fit, predict, predict_proba)
- ✅ Zéro dépendance sklearn dans src/
- ✅ Support régression et classification (binaire/multiclasse)
- ✅ Features avancées (GOSS, EFB, histogram binning)
- ✅ Early stopping et callbacks

---

## Installation et configuration

### Installation depuis source

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

**Runtime** (uniquement NumPy) :
```
numpy>=1.24.0
```

**Development** :
```
pytest>=7.0.0
pandas>=2.0.0
scikit-learn>=1.2.0  # Pour tests/benchmarks uniquement
scipy>=1.11.0
matplotlib>=3.8.0
seaborn>=0.13.0
```

### Vérification de l'installation

```python
import lightgbm
print(lightgbm.__version__)  # 1.0.0

# Test rapide
from lightgbm import LGBMRegressor
import numpy as np

X = np.random.randn(100, 5)
y = X[:, 0] + 2*X[:, 1] + np.random.randn(100)*0.1

model = LGBMRegressor(num_iterations=10)
model.fit(X, y)
print(f"Prédictions : {model.predict(X[:5])}")
```

---

## Guide d'utilisation

### 1. Régression de base

```python
from lightgbm import LGBMRegressor
import numpy as np

# Données
X_train = np.random.randn(1000, 10)
y_train = 3*X_train[:, 0] + 2*X_train[:, 1] + np.random.randn(1000)*0.5

# Modèle
model = LGBMRegressor(
    num_iterations=100,      # Nombre d'arbres
    learning_rate=0.1,       # Taux d'apprentissage
    max_depth=6,             # Profondeur max
    random_state=42          # Reproductibilité
)

# Entraînement
model.fit(X_train, y_train)

# Prédiction
X_test = np.random.randn(100, 10)
predictions = model.predict(X_test)
```

### 2. Classification binaire

```python
from lightgbm import LGBMClassifier
import numpy as np

# Données binaires
X = np.random.randn(500, 8)
y = (X[:, 0] + X[:, 1] > 0).astype(int)  # Classe 0 ou 1

# Modèle
clf = LGBMClassifier(
    num_iterations=50,
    learning_rate=0.1,
    max_depth=5
)

clf.fit(X, y)

# Prédictions
labels = clf.predict(X)           # Classes : 0 ou 1
probas = clf.predict_proba(X)     # Probabilités : shape (n, 2)

print(f"Accuracy : {(labels == y).mean():.3f}")
```

### 3. Classification multiclasse

```python
from lightgbm import LGBMClassifier
import numpy as np

# Données 3 classes
X = np.random.randn(600, 10)
y = np.random.randint(0, 3, size=600)  # Classes : 0, 1, 2

# Modèle
clf = LGBMClassifier(
    num_iterations=80,
    learning_rate=0.1,
    max_depth=6
)

clf.fit(X, y)

# Prédictions
labels = clf.predict(X)           # Classes : 0, 1 ou 2
probas = clf.predict_proba(X)     # Probabilités : shape (n, 3)

print(f"Classes : {clf.classes_}")
print(f"Proba pour premier échantillon : {probas[0]}")
```

### 4. Utilisation avec train/test split

```python
from lightgbm import LGBMRegressor
from lightgbm.utils import train_test_split, mean_squared_error
import numpy as np

# Données
X = np.random.randn(1000, 15)
y = X[:, 0] + 2*X[:, 1] - X[:, 2] + np.random.randn(1000)*0.3

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42
)

# Entraînement
model = LGBMRegressor(num_iterations=100, learning_rate=0.1)
model.fit(X_train, y_train)

# Évaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"MSE : {mse:.4f}")
```

---

## Paramètres et tuning

### Paramètres principaux

#### Contrôle de l'entraînement

**num_iterations** (ou n_estimators)
- Type : int
- Default : 100
- Description : Nombre d'arbres à construire
- Conseil : Commencer avec 100-200, augmenter si underfitting

**learning_rate**
- Type : float
- Default : 0.1
- Description : Shrinkage appliqué à chaque arbre
- Conseil : Valeurs typiques 0.01-0.3. Plus petit = meilleur généralement mais plus lent

**max_depth**
- Type : int
- Default : -1 (illimité)
- Description : Profondeur maximale des arbres
- Conseil : 3-10 généralement. -1 pour utiliser num_leaves uniquement

**num_leaves**
- Type : int
- Default : 31
- Description : Nombre maximal de feuilles par arbre (clé de LightGBM)
- Conseil : 2^max_depth ou moins. Valeurs typiques : 15-63

**min_data_in_leaf** (ou min_samples_leaf)
- Type : int
- Default : 20
- Description : Échantillons minimum dans une feuille
- Conseil : Augmenter pour éviter overfitting (20-100)

#### Régularisation

**lambda_l1** (ou reg_alpha)
- Type : float
- Default : 0.0
- Description : Régularisation L1 sur les poids de feuilles
- Conseil : 0-10, essayer 0.1, 1.0

**lambda_l2** (ou reg_lambda)
- Type : float
- Default : 0.0
- Description : Régularisation L2 sur les poids de feuilles
- Conseil : 0-10, essayer 0.1, 1.0

**min_gain_to_split**
- Type : float
- Default : 0.0
- Description : Gain minimum requis pour effectuer un split
- Conseil : 0-1, essayer 0.01, 0.1

#### Échantillonnage

**bagging_fraction**
- Type : float
- Default : 1.0
- Description : Fraction de données à utiliser par arbre
- Conseil : 0.5-1.0, essayer 0.8

**feature_fraction**
- Type : float
- Default : 1.0
- Description : Fraction de features à considérer par arbre
- Conseil : 0.5-1.0, essayer 0.8

**bagging_freq**
- Type : int
- Default : 0
- Description : Fréquence du bagging (0 = désactivé)
- Conseil : 1-10 si bagging_fraction < 1.0

#### Features avancées

**enable_goss**
- Type : bool
- Default : False
- Description : Activer GOSS (Gradient-based One-Side Sampling)
- Conseil : True pour datasets > 10k samples

**goss_top_rate**
- Type : float
- Default : 0.2
- Description : Fraction de gradients importants à garder
- Conseil : 0.1-0.3

**goss_other_rate**
- Type : float
- Default : 0.1
- Description : Fraction de petits gradients à échantillonner
- Conseil : 0.05-0.2

**use_histogram**
- Type : bool
- Default : False
- Description : Utiliser binning histogramme pour splits
- Conseil : True pour accélérer sur grandes données

**max_bins**
- Type : int
- Default : 255
- Description : Nombre max de bins pour histogram
- Conseil : 32-255, plus petit = plus rapide

**use_efb**
- Type : bool
- Default : False
- Description : Exclusive Feature Bundling pour données sparse
- Conseil : True si données très sparse

#### Autres

**early_stopping_rounds**
- Type : int or None
- Default : None
- Description : Arrêter si pas d'amélioration pendant N rounds
- Conseil : 10-50, nécessite eval_set

**lr_decay**
- Type : float
- Default : 1.0
- Description : Décroissance du learning rate par itération
- Conseil : 0.95-1.0, 1.0 = pas de decay

**verbose**
- Type : int
- Default : 0
- Description : Niveau de verbosité (0=silencieux, 1=progrès)
- Conseil : 1 pour suivre l'entraînement

**random_state**
- Type : int or None
- Default : None
- Description : Seed pour reproductibilité
- Conseil : Fixer pour résultats reproductibles

### Configurations recommandées

#### Configuration rapide (prototypage)

```python
model = LGBMRegressor(
    num_iterations=50,
    learning_rate=0.2,
    max_depth=4,
    min_data_in_leaf=50,
    verbose=1
)
```

#### Configuration équilibrée (production)

```python
model = LGBMRegressor(
    num_iterations=200,
    learning_rate=0.05,
    max_depth=6,
    num_leaves=31,
    min_data_in_leaf=20,
    lambda_l1=0.1,
    lambda_l2=0.1,
    bagging_fraction=0.8,
    feature_fraction=0.8,
    bagging_freq=5,
    early_stopping_rounds=20,
    random_state=42,
    verbose=1
)
```

#### Configuration grandes données (avec optimisations)

```python
model = LGBMRegressor(
    num_iterations=300,
    learning_rate=0.05,
    max_depth=8,
    num_leaves=63,
    min_data_in_leaf=50,
    lambda_l2=1.0,
    enable_goss=True,
    goss_top_rate=0.2,
    goss_other_rate=0.1,
    use_histogram=True,
    max_bins=128,
    use_efb=True,  # Si sparse
    bagging_fraction=0.9,
    feature_fraction=0.9,
    bagging_freq=5,
    early_stopping_rounds=30,
    random_state=42,
    verbose=1
)
```

#### Configuration haute précision (compromis vitesse)

```python
model = LGBMRegressor(
    num_iterations=500,
    learning_rate=0.01,
    max_depth=10,
    num_leaves=127,
    min_data_in_leaf=10,
    lambda_l1=0.05,
    lambda_l2=0.5,
    min_gain_to_split=0.01,
    bagging_fraction=0.7,
    feature_fraction=0.7,
    bagging_freq=1,
    early_stopping_rounds=50,
    random_state=42,
    verbose=1
)
```

---

## Features avancées

### 1. Early Stopping

Arrête l'entraînement si la métrique de validation ne s'améliore pas.

```python
from lightgbm import LGBMRegressor
from lightgbm.utils import train_test_split
import numpy as np

# Données
X = np.random.randn(1000, 10)
y = X[:, 0] + 2*X[:, 1] + np.random.randn(1000)*0.3

# Split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Modèle avec early stopping
model = LGBMRegressor(
    num_iterations=1000,  # Maximum
    early_stopping_rounds=20,  # Patience
    learning_rate=0.1,
    verbose=1
)

# Entraînement avec validation
model.fit(
    X_train, y_train,
    eval_set=(X_val, y_val)  # Données de validation
)

print(f"Arrêté à l'itération : {model.n_iter_}")
print(f"Historique : {model.training_history_}")
```

### 2. Sample Weights

Pondérer l'importance des échantillons (utile pour données déséquilibrées).

```python
from lightgbm import LGBMClassifier
import numpy as np

# Données déséquilibrées
X = np.random.randn(1000, 10)
y = np.zeros(1000, dtype=int)
y[:50] = 1  # Seulement 5% de classe 1

# Calculer poids
pos_weight = len(y) / (2 * np.sum(y))
sample_weight = np.where(y == 1, pos_weight, 1.0)

print(f"Poids classe positive : {pos_weight:.2f}")

# Entraînement avec poids
clf = LGBMClassifier(num_iterations=100)
clf.fit(X, y, sample_weight=sample_weight)

# Prédictions
preds = clf.predict(X)
print(f"Classe 1 prédites : {np.sum(preds)} / {np.sum(y)}")
```

### 3. Fonctions de perte personnalisées

```python
from lightgbm import LGBMRegressor
from lightgbm.loss_functions import HuberLoss, QuantileLoss
import numpy as np

# Données avec outliers
X = np.random.randn(500, 5)
y = X[:, 0] + 2*X[:, 1] + np.random.randn(500)*0.5
y[::50] += 10  # Ajouter outliers

# Huber loss (robuste aux outliers)
model_huber = LGBMRegressor(
    objective=HuberLoss(delta=1.0),
    num_iterations=100,
    learning_rate=0.1
)
model_huber.fit(X, y)

# Quantile regression (médiane)
model_quantile = LGBMRegressor(
    objective=QuantileLoss(quantile=0.5),
    num_iterations=100,
    learning_rate=0.1
)
model_quantile.fit(X, y)

print("Prédictions Huber :", model_huber.predict(X[:5]))
print("Prédictions Quantile :", model_quantile.predict(X[:5]))
```

### 4. Feature Importance

```python
from lightgbm import LGBMRegressor
import numpy as np

X = np.random.randn(500, 10)
y = 3*X[:, 0] + 2*X[:, 5] + X[:, 8] + np.random.randn(500)*0.1

model = LGBMRegressor(num_iterations=100)
model.fit(X, y)

# Importances (basées sur gains)
importances = model.feature_importances_

print("Feature importances :")
for idx, imp in enumerate(importances):
    if imp > 0.01:  # Afficher seulement features importantes
        print(f"  Feature {idx} : {imp:.4f}")
```

### 5. GOSS (Gradient-based One-Side Sampling)

Accélère l'entraînement en échantillonnant intelligemment les données.

```python
from lightgbm import LGBMRegressor
import numpy as np
import time

# Grandes données
X = np.random.randn(10000, 20)
y = X[:, 0] + 2*X[:, 1] + np.random.randn(10000)*0.5

# Sans GOSS
start = time.time()
model_normal = LGBMRegressor(
    num_iterations=100,
    enable_goss=False
)
model_normal.fit(X, y)
time_normal = time.time() - start

# Avec GOSS
start = time.time()
model_goss = LGBMRegressor(
    num_iterations=100,
    enable_goss=True,
    goss_top_rate=0.2,
    goss_other_rate=0.1
)
model_goss.fit(X, y)
time_goss = time.time() - start

print(f"Temps sans GOSS : {time_normal:.2f}s")
print(f"Temps avec GOSS : {time_goss:.2f}s")
print(f"Accélération : {time_normal/time_goss:.2f}x")
```

### 6. Callbacks personnalisés

```python
from lightgbm import LGBMRegressor
from lightgbm.base import Callback
import numpy as np

# Callback personnalisé
class PrintLossCallback(Callback):
    def on_iteration_end(self, iteration, model, train_loss, val_loss=None):
        if iteration % 10 == 0:
            print(f"Iteration {iteration}: train_loss={train_loss:.4f}")
        return False  # False = continue, True = stop

# Utilisation
X = np.random.randn(500, 5)
y = X[:, 0] + 2*X[:, 1] + np.random.randn(500)*0.1

callback = PrintLossCallback()
model = LGBMRegressor(num_iterations=50)
model.fit(X, y, callbacks=[callback])
```

---

## Benchmarking et comparaisons

### Comparer avec sklearn

```bash
# Exécuter benchmark complet
python benchmarks/benchmark_comparison.py
```

### Benchmark custom

```python
from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import make_regression
import time

# Données
X, y = make_regression(n_samples=5000, n_features=20, random_state=42)

# Notre implémentation
start = time.time()
our_model = LGBMRegressor(num_iterations=100, learning_rate=0.1)
our_model.fit(X, y)
our_time = time.time() - start
our_pred = our_model.predict(X)
our_mse = ((y - our_pred)**2).mean()

# Sklearn
start = time.time()
sk_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1)
sk_model.fit(X, y)
sk_time = time.time() - start
sk_pred = sk_model.predict(X)
sk_mse = ((y - sk_pred)**2).mean()

print(f"Notre LightGBM : MSE={our_mse:.4f}, Time={our_time:.2f}s")
print(f"Sklearn GB     : MSE={sk_mse:.4f}, Time={sk_time:.2f}s")
print(f"Ratio vitesse  : {sk_time/our_time:.2f}x")
```

---

## Debugging et troubleshooting

### Problèmes courants

#### 1. Overfitting

**Symptômes** : Excellent score training, mauvais score test

**Solutions** :
```python
# Augmenter régularisation
model = LGBMRegressor(
    lambda_l1=1.0,
    lambda_l2=1.0,
    min_data_in_leaf=50,
    min_gain_to_split=0.1
)

# Réduire complexité
model = LGBMRegressor(
    num_leaves=15,
    max_depth=4
)

# Augmenter subsampling
model = LGBMRegressor(
    bagging_fraction=0.7,
    feature_fraction=0.7,
    bagging_freq=5
)
```

#### 2. Underfitting

**Symptômes** : Scores training et test tous deux médiocres

**Solutions** :
```python
# Augmenter nombre d'arbres
model = LGBMRegressor(num_iterations=500)

# Augmenter complexité
model = LGBMRegressor(
    num_leaves=63,
    max_depth=8,
    min_data_in_leaf=10
)

# Réduire régularisation
model = LGBMRegressor(
    lambda_l1=0.0,
    lambda_l2=0.0
)
```

#### 3. Entraînement trop lent

**Solutions** :
```python
# Activer optimisations
model = LGBMRegressor(
    enable_goss=True,
    use_histogram=True,
    max_bins=64,
    num_iterations=100  # Réduire si nécessaire
)
```

#### 4. Erreur "NotFittedError"

**Cause** : Appel predict() avant fit()

**Solution** :
```python
model = LGBMRegressor()
model.fit(X_train, y_train)  # Ne pas oublier !
predictions = model.predict(X_test)
```

#### 5. NaN dans prédictions

**Causes possibles** :
- NaN dans features avec allow_nan=False
- Hessians trop petits

**Solutions** :
```python
# Autoriser NaN
model = LGBMRegressor(allow_nan=True)

# Augmenter min_sum_hessian_in_leaf
model = LGBMRegressor(min_sum_hessian_in_leaf=1e-2)
```

### Activer verbose

```python
model = LGBMRegressor(
    num_iterations=100,
    verbose=1  # Afficher progrès
)
model.fit(X, y)
```

Sortie :
```
[LightGBM] Iter 1/100 (1.0%) - train_loss: 2.456123
[LightGBM] Iter 10/100 (10.0%) - train_loss: 1.234567
...
```

### Inspecter l'historique

```python
model.fit(X_train, y_train, eval_set=(X_val, y_val))

print("Training loss :", model.training_history_['train_loss'])
print("Validation loss :", model.training_history_['val_loss'])

# Visualiser
import matplotlib.pyplot as plt
plt.plot(model.training_history_['train_loss'], label='Train')
plt.plot(model.training_history_['val_loss'], label='Val')
plt.legend()
plt.show()
```

---

## Exemples complets

### Exemple 1 : Pipeline complet de régression

Voir `examples/complete_testing.ipynb` section 1.

### Exemple 2 : Classification avec validation croisée

```python
from lightgbm import LGBMClassifier
from lightgbm.utils import accuracy_score
import numpy as np

# Données
np.random.seed(42)
X = np.random.randn(1000, 15)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

# Validation croisée manuelle (5-fold)
n_folds = 5
fold_size = len(X) // n_folds
scores = []

for fold in range(n_folds):
    # Split
    val_start = fold * fold_size
    val_end = (fold + 1) * fold_size
    
    X_val = X[val_start:val_end]
    y_val = y[val_start:val_end]
    X_train = np.vstack([X[:val_start], X[val_end:]])
    y_train = np.hstack([y[:val_start], y[val_end:]])
    
    # Entraînement
    clf = LGBMClassifier(
        num_iterations=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    clf.fit(X_train, y_train)
    
    # Évaluation
    y_pred = clf.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    scores.append(acc)
    print(f"Fold {fold+1}: Accuracy = {acc:.4f}")

print(f"\nMoyenne : {np.mean(scores):.4f} ± {np.std(scores):.4f}")
```

### Exemple 3 : Comparaison plusieurs objectifs

```python
from lightgbm import LGBMRegressor
from lightgbm.loss_functions import MSELoss, MAELoss, HuberLoss
from lightgbm.utils import mean_squared_error, mean_absolute_error
import numpy as np

# Données avec outliers
X = np.random.randn(500, 5)
y = X[:, 0] + 2*X[:, 1] + np.random.randn(500)*0.5
y[::50] += 10  # Outliers

# Test plusieurs loss functions
losses = [
    ('MSE', 'mse'),
    ('MAE', 'mae'),
    ('Huber', HuberLoss(delta=1.0))
]

for name, objective in losses:
    model = LGBMRegressor(
        objective=objective,
        num_iterations=100,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X, y)
    
    pred = model.predict(X)
    mse = mean_squared_error(y, pred)
    mae = mean_absolute_error(y, pred)
    
    print(f"{name:10} - MSE: {mse:.4f}, MAE: {mae:.4f}")
```

---

## Ressources supplémentaires

### Documentation

- [README.md](README.md) : Vue d'ensemble du projet
- [ARCHITECTURE.md](ARCHITECTURE.md) : Architecture détaillée
- Tests : `tests/` pour exemples d'utilisation

### Papiers de référence

- [LightGBM: A Highly Efficient Gradient Boosting Decision Tree](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree) (NeurIPS 2017)
- [XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754) (KDD 2016)

### Tutoriels externes

- [LightGBM Documentation officielle](https://lightgbm.readthedocs.io/)
- [Gradient Boosting explained](https://explained.ai/gradient-boosting/)

---

**Dernière mise à jour** : Décembre 2024  
**Version** : 1.0.0  
**Auteurs** : LightGBM Scratch Contributors