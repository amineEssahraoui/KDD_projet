# Guide d'Implémentation

## Installation

### Prérequis

- Python >= 3.8
- pip ou conda

### Étapes d'installation

```bash
# Cloner le repository
git clone <repository-url>
cd lightgbm_package

# Créer un environnement virtuel (optionnel)
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt

# Installer le package en mode développement
pip install -e .
```

## Utilisation

### Classification

```python
from lightgbm import LGBMClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Charger les données
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Créer et entraîner le modèle
clf = LGBMClassifier(num_iterations=100, learning_rate=0.1)
clf.fit(X_train, y_train)

# Faire des prédictions
y_pred = clf.predict(X_test)
```

### Régression

```python
from lightgbm import LGBMRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

# Charger les données
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Créer et entraîner le modèle
reg = LGBMRegressor(num_iterations=100, learning_rate=0.1)
reg.fit(X_train, y_train)

# Faire des prédictions
y_pred = reg.predict(X_test)
```

## Hyperparamètres principaux

- `num_iterations` : Nombre de boosting iterations (défaut: 100)
- `learning_rate` : Taux d'apprentissage (défaut: 0.1)
- `max_depth` : Profondeur maximale des arbres (défaut: 5)
- `num_leaves` : Nombre maximal de feuilles (défaut: 31)
- `min_data_in_leaf` : Nombre minimum d'instances par feuille (défaut: 20)
- `lambda_l1` : Régularisation L1 (défaut: 0.0)
- `lambda_l2` : Régularisation L2 (défaut: 0.0)

## Exécution des tests

```bash
# Exécuter tous les tests
pytest tests/

# Exécuter un fichier de test spécifique
pytest tests/test_classifier.py

# Avec verbosité
pytest tests/ -v

# Avec couverture de code
pytest tests/ --cov=lightgbm
```

## Structure des notebooks

Les notebooks dans `examples/` fournissent des démonstrations pratiques :

- Données de régression
- Classification multi-classe
- Pipelines de preprocessing
