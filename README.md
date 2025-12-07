# 🌳 LightGBM Package - Implémentation from Scratch

Implementation complète de LightGBM (Light Gradient Boosting Machine) dans le cadre d'un projet académique sur les algorithmes d'arbres de décision.

---

## 📁 Structure du Projet

```
lightgbm_package/
│
├── lightgbm/                      # Package principal
├── tests/                         # Tests unitaires
├── examples/                      # Notebooks de démonstration
├── docs/                          # Documentation technique
├── README.md                      # Ce fichier
├── setup.py                       # Configuration d'installation
└── requirements.txt               # Dépendances
```

---

## 📦 Description des Fichiers

### **1. Package Principal : `lightgbm/`**

#### **`__init__.py`**
Fichier d'initialisation du package Python. Expose les classes principales pour faciliter l'import :
```python
from lightgbm import LGBMClassifier, LGBMRegressor
```

---

#### **`base.py`** - Classes de Base
**Rôle :** Contient la classe abstraite `BaseEstimator` dont héritent tous les modèles LightGBM.

**Contenu :**
- `BaseEstimator` : Classe abstraite définissant l'interface commune
  - Initialisation des hyperparamètres (learning_rate, num_iterations, max_depth, etc.)
  - Méthodes abstraites `fit()` et `predict()`
  - Méthodes utilitaires pour initialisation des prédictions
  - Calcul des gradients et hessians

**Pourquoi :** Évite la duplication de code entre Classifier et Regressor, assure une interface cohérente.

---

#### **`lgbm_classifier.py`** - Classifier Principal
**Rôle :** Implémentation du modèle de classification LightGBM.

**Contenu :**
- `LGBMClassifier` : Classe héritant de `BaseEstimator`
  - Classification binaire et multiclasse
  - Méthode `fit(X, y)` : entraînement du modèle
  - Méthode `predict(X)` : prédiction des classes
  - Méthode `predict_proba(X)` : prédiction des probabilités
  - Gestion des loss functions appropriées (Binary/Multiclass Cross-Entropy)

**Utilisation typique :**
```python
clf = LGBMClassifier(num_iterations=100, learning_rate=0.1)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
```

---

#### **`lgbm_regressor.py`** - Regressor Principal
**Rôle :** Implémentation du modèle de régression LightGBM.

**Contenu :**
- `LGBMRegressor` : Classe héritant de `BaseEstimator`
  - Régression pour valeurs continues
  - Méthode `fit(X, y)` : entraînement
  - Méthode `predict(X)` : prédiction des valeurs
  - Utilise MSE (Mean Squared Error) comme fonction de perte

**Utilisation typique :**
```python
reg = LGBMRegressor(num_iterations=100, learning_rate=0.1)
reg.fit(X_train, y_train)
predictions = reg.predict(X_test)
```

---

#### **`tree.py`** - Structure d'Arbre
**Rôle :** Implémentation de l'arbre de décision avec croissance leaf-wise.

**Contenu :**
- `Node` : Classe représentant un nœud de l'arbre
  - Attributs : feature_index, threshold, left, right, value, is_leaf
  - Méthodes pour la navigation dans l'arbre
  
- `DecisionTree` : Classe de l'arbre de décision
  - Construction de l'arbre selon stratégie leaf-wise
  - Méthode `fit(X, gradients, hessians)` : construction de l'arbre
  - Méthode `_find_best_split()` : recherche du meilleur split
  - Méthode `_compute_leaf_value()` : calcul de la valeur optimale d'une feuille
  - Méthode `predict(X)` : prédiction via traversée de l'arbre

**Particularité :** Implémente la croissance **leaf-wise** (feuille par feuille) au lieu de **level-wise** (niveau par niveau), ce qui est une caractéristique clé de LightGBM.

---

#### **`histogram.py`** - Gestion des Histogrammes
**Rôle :** Construction d'histogrammes (binning) pour accélérer la recherche de splits.

**Contenu :**
- `HistogramBuilder` : Classe pour la construction d'histogrammes
  - Méthode `fit(X)` : création des bins pour chaque feature
  - Méthode `transform(X)` : conversion des features en indices de bins
  - Méthode `build_histogram()` : construction de l'histogramme pour une feature
  - Méthode `compute_split_gain()` : calcul du gain pour chaque position de split

**Principe :** Au lieu d'évaluer tous les points possibles pour un split (O(n)), on regroupe les valeurs en bins (par défaut 255) et on évalue uniquement les frontières de bins (O(bins)), ce qui accélère considérablement l'entraînement.

---

#### **`goss.py`** - Gradient-based One-Side Sampling
**Rôle :** Implémentation de GOSS, une technique d'échantillonnage intelligent.

**Contenu :**
- `GOSSSampler` : Classe pour l'échantillonnage GOSS
  - Méthode `sample(gradients)` : sélection des échantillons importants
  - Retourne les indices sélectionnés et leurs poids

**Principe :** 
- Garde tous les échantillons avec les plus grands gradients (top_rate, ex: 20%)
- Échantillonne aléatoirement parmi les petits gradients (other_rate, ex: 10%)
- Pondère les petits gradients pour compenser l'échantillonnage
- **Avantage :** Réduit le nombre d'échantillons sans perdre d'information critique, accélère l'entraînement.

---

#### **`efb.py`** - Exclusive Feature Bundling
**Rôle :** Regroupement de features mutuellement exclusives.

**Contenu :**
- `EFBBundler` : Classe pour le bundling de features
  - Méthode `fit(X)` : identification des bundles
  - Méthode `transform(X)` : transformation de X avec features fusionnées
  - Méthode `_build_conflict_graph()` : construction du graphe de conflits
  - Méthode `_greedy_bundling()` : algorithme de bundling glouton

**Principe :** Deux features sont "exclusives" si elles sont rarement non-nulles simultanément (typique des features one-hot). On peut les fusionner en ajoutant un offset, réduisant ainsi la dimensionnalité sans perte d'information.

---

#### **`leaf_wise.py`** - Croissance Leaf-Wise
**Rôle :** Implémentation spécifique de la stratégie de croissance leaf-wise.

**Contenu :**
- Fonctions utilitaires pour la croissance leaf-wise
- Gestion de la priority queue des feuilles
- Algorithmes de sélection de la meilleure feuille à développer

**Différence avec level-wise :**
- **Level-wise** (XGBoost) : Développe tous les nœuds d'un niveau avant de passer au suivant
- **Leaf-wise** (LightGBM) : Choisit la feuille avec le gain maximum et la développe en priorité
- **Avantage :** Convergence plus rapide, arbres plus profonds mais plus efficaces

---

#### **`loss_functions.py`** - Fonctions de Perte
**Rôle :** Définition des fonctions de perte pour différents types de problèmes.

**Contenu :**
- `LossFunction` : Classe abstraite de base
  - Méthode `loss()` : calcul de la perte
  - Méthode `gradient()` : calcul du gradient (dérivée première)
  - Méthode `hessian()` : calcul du hessian (dérivée seconde)

- `MSELoss` : Mean Squared Error (régression)
- `BinaryCrossEntropy` : Cross-entropy binaire (classification binaire)
- `MulticlassCrossEntropy` : Cross-entropy multiclasse (classification multiclasse)

**Pourquoi gradient et hessian :** LightGBM utilise l'approximation de second ordre (Taylor), d'où le besoin des dérivées première et seconde.

---

#### **`metrics.py`** - Métriques d'Évaluation
**Rôle :** Fonctions pour évaluer la performance des modèles.

**Contenu :**
- `accuracy_score()` : Exactitude pour classification
- `mse_score()` : Mean Squared Error
- `mae_score()` : Mean Absolute Error
- `logloss()` : Logarithmic Loss
- `auc_roc()` : Area Under ROC Curve

**Utilisation :** Permet d'évaluer les modèles pendant et après l'entraînement.

---

#### **`utils.py`** - Fonctions Utilitaires
**Rôle :** Fonctions auxiliaires utilisées dans tout le package.

**Contenu :**
- Validation des données d'entrée
- Conversion de formats
- Gestion des valeurs manquantes
- Fonctions mathématiques (sigmoid, softmax)
- Outils de logging et de verbosité

---

### **2. Tests : `tests/`**

#### **`__init__.py`**
Fichier d'initialisation du module de tests.

---

#### **`test_classifier.py`**
Tests unitaires pour `LGBMClassifier` :
- Test d'entraînement sur données synthétiques
- Test de prédiction
- Test de predict_proba
- Comparaison avec sklearn
- Tests de cas limites

---

#### **`test_regressor.py`**
Tests unitaires pour `LGBMRegressor` :
- Test d'entraînement
- Test de prédiction
- Validation des performances (MSE, MAE)
- Comparaison avec sklearn

---

#### **`test_tree.py`**
Tests pour la structure d'arbre :
- Test de construction de nœuds
- Test de croissance leaf-wise
- Test de find_best_split
- Test de prédiction

---

#### **`test_histogram.py`**
Tests pour les histogrammes :
- Test de création de bins
- Test de transformation
- Test de calcul de gain
- Validation de la vitesse d'exécution

---

#### **`test_goss.py`**
Tests pour GOSS :
- Test de sampling
- Validation des poids
- Vérification de la réduction d'échantillons

---

### **3. Examples : `examples/`**

#### **`classification_example.ipynb`**
Notebook de démonstration pour la classification :
- Chargement d'un dataset (Iris, Titanic)
- Entraînement du LGBMClassifier
- Évaluation des performances
- Visualisation des résultats
- Comparaison avec autres algorithmes

---

#### **`regression_example.ipynb`**
Notebook de démonstration pour la régression :
- Chargement d'un dataset (Boston Housing, California Housing)
- Entraînement du LGBMRegressor
- Évaluation (MSE, MAE, R²)
- Visualisation des prédictions vs valeurs réelles

---

#### **`comparison_with_sklearn.ipynb`**
Notebook de comparaison :
- Benchmark LightGBM vs Sklearn GradientBoosting
- Comparaison de vitesse d'entraînement
- Comparaison de précision
- Analyse des hyperparamètres

---

### **4. Documentation : `docs/`**

#### **`lightgbm_theory.md`**
Documentation théorique complète :
- Principe du Gradient Boosting
- Innovations de LightGBM (leaf-wise, histograms, GOSS, EFB)
- Formules mathématiques
- Références bibliographiques

---

#### **`api_reference.md`**
Référence complète de l'API :
- Liste de toutes les classes et méthodes
- Paramètres détaillés
- Valeurs de retour
- Exemples d'utilisation

---

#### **`user_guide.md`**
Guide utilisateur :
- Installation
- Premiers pas
- Tutoriels pas à pas
- Bonnes pratiques
- FAQ

---

### **5. Fichiers Racine**

#### **`README.md`**
Ce fichier - Vue d'ensemble du projet et description de tous les fichiers.

---

#### **`setup.py`**
Script d'installation du package :
- Configuration pour pip install
- Métadonnées du package (nom, version, auteur)
- Dépendances
- Point d'entrée

**Utilisation :**
```bash
pip install -e .
```

---

#### **`requirements.txt`**
Liste des dépendances Python nécessaires :
```
numpy>=1.19.0
pandas>=1.1.0
scikit-learn>=0.23.0
matplotlib>=3.3.0
jupyter>=1.0.0
pytest>=6.0.0
```

**Installation :**
```bash
pip install -r requirements.txt
```

---

## 🚀 Installation

```bash
# Cloner le repository
git clone https://github.com/votre-repo/lightgbm_package.git
cd lightgbm_package

# Installer les dépendances
pip install -r requirements.txt

# Installer le package en mode développement
pip install -e .
```

---

## 📖 Utilisation Rapide

```python
from lightgbm import LGBMClassifier, LGBMRegressor

# Classification
clf = LGBMClassifier(num_iterations=100, learning_rate=0.1)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

# Régression
reg = LGBMRegressor(num_iterations=100, learning_rate=0.1)
reg.fit(X_train, y_train)
predictions = reg.predict(X_test)
```

---

## 👥 Auteurs

Projet réalisé dans le cadre du cours [Nom du Cours] par :
- [Nom Étudiant 1]
- [Nom Étudiant 2]
- [Nom Étudiant 3]

---

## 📚 Références

- [LightGBM: A Highly Efficient Gradient Boosting Decision Tree](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree.pdf)
- [Documentation officielle LightGBM](https://lightgbm.readthedocs.io/)
- [Gradient Boosting Machines - Friedman](https://statweb.stanford.edu/~jhf/ftp/trebst.pdf)

---

## 📄 Licence

Ce projet est réalisé à des fins éducatives dans le cadre d'un projet académique.
