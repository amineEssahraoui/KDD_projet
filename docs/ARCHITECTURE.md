# Architecture Documentation

## Table des matières
- [Vue d'ensemble](#vue-densemble)
- [Structure du projet](#structure-du-projet)
- [Architecture des modules](#architecture-des-modules)
- [Flux de données](#flux-de-données)
- [Diagrammes](#diagrammes)

---

## Vue d'ensemble

Ce projet implémente LightGBM (Light Gradient Boosting Machine) from scratch en Python pur avec NumPy uniquement. L'architecture suit les principes du gradient boosting avec des optimisations spécifiques à LightGBM :

- **Croissance leaf-wise** : Contrairement à XGBoost qui croît par niveau, LightGBM sélectionne la feuille avec le gain maximal
- **GOSS** : Gradient-based One-Side Sampling pour accélérer l'entraînement
- **EFB** : Exclusive Feature Bundling pour réduire la dimensionnalité
- **Histogram Binning** : Discrétisation des features pour des splits plus rapides

---

## Structure du projet

```
KDD_projet/
│
├── src/lightgbm/              # Package principal
│   ├── __init__.py            # Exports publics
│   ├── base.py                # Classes de base abstraites
│   ├── lgbm_classifier.py     # Classificateur
│   ├── lgbm_regressor.py      # Régresseur
│   ├── tree.py                # Arbre de décision
│   ├── loss_functions.py      # Fonctions de perte
│   ├── histogram.py           # Binning histogramme
│   ├── goss.py                # GOSS sampling
│   ├── efb.py                 # Feature bundling
│   └── utils.py               # Utilitaires
│
├── tests/                     # Suite de tests
│   ├── test_classifier.py     # Tests classificateur
│   ├── test_regressor.py      # Tests régresseur
│   ├── test_tree.py           # Tests arbre
│   ├── test_goss.py           # Tests GOSS
│   ├── test_utils.py          # Tests utilitaires
│   ├── test_math_integrity.py # Tests mathématiques
│   └── test_logic_sanity.py   # Tests de logique
│
├── benchmarks/                # Comparaisons de performance
│   └── benchmark_comparison.py
│
├── examples/                  # Exemples d'utilisation
│   ├── complete_testing.ipynb
│   └── regression_pipeline.py
│
├── docs/                      # Documentation
│
├── .github/workflows/         # CI/CD
│   └── ci.yml
│
├── pyproject.toml             # Configuration du projet
├── requirements.txt           # Dépendances
└── README.md                  # Documentation principale
```

---

## Architecture des modules

### 1. **Module Base** (`base.py`)

**Rôle** : Fournit les classes abstraites et structures de données communes.

**Composants principaux** :

- **BoosterParams** : Dataclass qui centralise tous les hyperparamètres (num_iterations, learning_rate, max_depth, num_leaves, etc.)
- **BaseEstimator** : Classe abstraite avec méthodes `fit()` et `predict()` que tous les estimateurs doivent implémenter
- **Callback** : Interface abstraite pour les callbacks d'entraînement
- **EarlyStoppingCallback** : Implémentation concrète pour l'arrêt précoce basé sur la validation

**Relations** :
- Hérité par `LGBMClassifier` et `LGBMRegressor`
- Utilise `BoosterParams` pour centraliser la configuration
- Implémente le pattern Strategy via `Callback`

---

### 2. **Module Loss Functions** (`loss_functions.py`)

**Rôle** : Implémente les fonctions de perte avec gradient et hessian.

**Architecture** :

**Classe abstraite Loss** : Définit l'interface avec les méthodes :
- `__call__()` : Calcule la valeur de perte
- `gradient()` : Premier ordre ∂L/∂f
- `hessian()` : Second ordre ∂²L/∂f²
- `init_prediction()` : Prédiction initiale

**Implémentations** :

**Régression** (dans `loss_functions.py`) :
- **MSELoss** : Perte quadratique moyenne
- **MAELoss** : Erreur absolue moyenne
- **HuberLoss** : Combinaison MSE/MAE avec seuil delta
- **QuantileLoss** : Pour régression quantile

**Classification** (dans `loss_functions.py`) :
- **BinaryCrossEntropyLoss** : Entropie croisée binaire
- **MultiClassCrossEntropyLoss** : Softmax cross-entropy

**Fonction utilitaire** :
- `get_loss_function()` : Factory pour créer une fonction de perte par nom

**Formules mathématiques clés** :

```
MSE:
  Loss:     L(y,f) = 0.5 * (y - f)²
  Gradient: g = f - y
  Hessian:  h = 1

Binary CE:
  Loss:     L(y,f) = -[y·log(σ(f)) + (1-y)·log(1-σ(f))]
  Gradient: g = σ(f) - y
  Hessian:  h = σ(f)·(1-σ(f))
  
  où σ(f) = 1/(1+e^(-f))

Multiclass CE:
  Loss:     L(y,f) = -Σ[y_k · log(softmax(f)_k)]
  Gradient: g_k = p_k - (y==k)
  Hessian:  h_k = p_k·(1-p_k)
```

---

### 3. **Module Tree** (`tree.py`)

**Rôle** : Implémente l'arbre de décision avec croissance leaf-wise.

**Structures de données** (dans `tree.py`) :

- **TreeNode** : Représente un nœud avec attributs is_leaf, value, feature_idx, threshold, left, right, n_samples, depth, gain
- **SplitInfo** : Contient les informations d'un split (gain, feature_idx, threshold, indices gauche/droite, valeurs)

**Classe DecisionTree** (dans `tree.py`) :

Méthodes principales :
- `fit()` : Entraîne l'arbre sur gradients et hessians
- `_build_tree_leaf_wise()` : Construit l'arbre en sélectionnant itérativement la feuille avec le gain maximal (file de priorité)
- `_find_best_split()` : Trouve le meilleur split pour un nœud
- `_find_best_split_exact()` : Recherche exhaustive (O(n log n))
- `_find_best_split_histogram()` : Recherche sur bins (O(max_bins))
- `_compute_leaf_value()` : Calcule w* = -G/(H+λ)
- `predict()` : Prédictions pour nouveaux échantillons

**Formule du gain de split** :

```
Gain = Score_left + Score_right - Score_parent - γ

où Score = G²/(H + λ)

G = Σ gradients
H = Σ hessians
λ = lambda_l2 (régularisation L2)
γ = min_gain_to_split
```

**Avec régularisation L1** :
```python
if lambda_l1 > 0:
    G_reg = soft_threshold(G, lambda_l1)
    # G_reg = G - λ₁  si G > λ₁
    #       = G + λ₁  si G < -λ₁
    #       = 0       sinon
```

**Valeur de feuille optimale** :
```
w* = -G / (H + λ)
```

---

### 4. **Module GOSS** (`goss.py`)

**Rôle** : Gradient-based One-Side Sampling pour accélérer l'entraînement.

**Principe** :
1. Trier les échantillons par magnitude absolue du gradient
2. Garder top_rate% des plus grands gradients
3. Échantillonner aléatoirement other_rate% des petits gradients
4. Amplifier les poids des petits gradients échantillonnés par facteur (1-top_rate)/other_rate

**Classe GOSS** (dans `goss.py`) :

Méthodes principales :
- `__init__()` : Initialise avec top_rate et other_rate
- `sample()` : Sélectionne les échantillons et retourne indices + poids
- `sample_data()` : Version complète qui retourne X, gradients et hessians échantillonnés

**Fonction utilitaire** :
- `apply_goss()` : Fonction convenience pour appliquer GOSS en une ligne

**Avantages** :
- Réduit le nombre d'échantillons de ~70%
- Préserve la précision en gardant les gradients importants
- Accélération typique : 2-3x

---

### 5. **Module EFB** (`efb.py`)

**Rôle** : Exclusive Feature Bundling pour réduire la dimensionnalité.

**Principe** :

Deux features sont mutuellement exclusives si elles sont rarement non-nulles ensemble. Le bundling permet de combiner plusieurs features en une seule.

**Exemple** :
- Feature A: [0, 0, 5, 0, 0, 3] - Sparse
- Feature B: [2, 0, 0, 1, 0, 0] - Sparse
- Conflit : 0% → Peuvent être bundlées

**Classe FeatureBundler** (dans `efb.py`) :

Méthodes principales :
- `fit()` : Identifie les bundles en calculant matrice de conflits et applique algorithme greedy
- `transform()` : Transforme les features originales en features bundlées avec offsets
- `fit_transform()` : Combine fit et transform
- `get_bundle_info()` : Retourne informations sur les bundles créés

**Fonction utilitaire** :
- `bundle_features()` : Fonction convenience pour bundler en une ligne

**Algorithme greedy** :
1. Calculer matrice de conflits (fréquence où features sont non-nulles ensemble)
2. Pour chaque feature non utilisée, créer un nouveau bundle
3. Ajouter features compatibles (conflit < max_conflict_rate)
4. Calculer offsets pour éviter collisions lors du merge

---

### 6. **Module Histogram** (intégré dans `tree.py`)

**Rôle** : Binning des features pour des splits plus rapides.

**Principe** :

Au lieu de tester tous les seuils possibles (exact split), discrétiser les features en max_bins bins (généralement 255). Cela réduit considérablement la complexité.

**Complexité** :
- Exact : O(n_samples × n_features × log(n_samples))
- Histogram : O(n_samples × n_features + max_bins × n_features)

**Implémentation dans DecisionTree** (dans `tree.py`) :

Méthodes principales :
- `_compute_bin_edges()` : Calcule les limites des bins par feature (quantile-based)
- `_find_best_split_histogram()` : Trouve meilleur split en utilisant histogrammes
  - Assigne échantillons aux bins avec np.digitize()
  - Agrège gradients/hessians par bin avec np.bincount()
  - Calcule cumsum pour tester splits
  - Retourne meilleur split basé sur gains

**Processus** :
1. Binning initial : créer bins pour chaque feature (une fois)
2. Pour chaque split : assigner valeurs aux bins
3. Agréger statistiques (G, H) par bin
4. Tester splits entre bins successifs
5. Sélectionner meilleur bin

---

### 7. **Modules Estimateurs**

#### **LGBMRegressor** (`lgbm_regressor.py`)

**Classe principale pour la régression**

Méthodes clés :
- `fit()` : Entraîne le modèle
  - Initialise loss function via `get_loss_function(objective)`
  - Calcule prédiction initiale
  - Applique EFB si activé
  - Initialise GOSS si activé
  - Boucle d'entraînement : calcule gradients/hessians → applique GOSS → construit arbre → met à jour prédictions
  - Gère early stopping et callbacks

- `predict()` : Prédit valeurs
  - Applique transformation EFB si nécessaire
  - Somme : init_prediction + learning_rate * Σ tree.predict(X)

- `score()` : Calcule R² score
- `feature_importances_` : Retourne importances basées sur gains

**Paramètre objective** : Accepte string ('mse', 'mae', 'huber', 'quantile') ou instance Loss custom

#### **LGBMClassifier** (`lgbm_classifier.py`)

**Classe principale pour la classification (binaire et multiclasse)**

Méthodes clés :
- `fit()` : Entraîne le modèle
  - Encode labels en indices 0,1,...,K-1
  - Détermine si binaire (n_classes ≤ 2) ou multiclasse
  - Pour binaire : un ensemble d'arbres, logits
  - Pour multiclasse : K ensembles d'arbres (un par classe)
  - Calcule gradients/hessians de cross-entropy
  - Construit arbres et met à jour logits

- `predict_proba()` : Prédit probabilités
  - Calcule logits raw : init + lr * Σ trees
  - Binaire : applique sigmoid
  - Multiclasse : applique softmax

- `predict()` : Prédit labels
  - Appelle predict_proba()
  - Retourne classe avec probabilité maximale

- `score()` : Calcule accuracy
- `feature_importances_` : Importance agrégée sur toutes les classes

**Différences binaire/multiclasse** :
- Binaire : self.trees_ = [tree1, tree2, ...] (liste simple)
- Multiclasse : self.trees_ = [[trees_class0], [trees_class1], ...] (liste de listes)

---

## Flux de données

### Entraînement (Régression)

```
1. Préparation
   X, y → check_X_y() → X_valid, y_valid
                      ↓
                  EFB (optionnel)
                      ↓
                  X_bundled

2. Initialisation
   y → loss.init_prediction() → init_pred
   init_pred → np.full(n_samples) → y_pred

3. Itération de boosting
   ┌─────────────────────────────────┐
   │                                 │
   │  y, y_pred                      │
   │      ↓                          │
   │  loss.gradient()                │
   │  loss.hessian()                 │
   │      ↓                          │
   │  g, h                           │
   │      ↓                          │
   │  GOSS (optionnel)               │
   │      ↓                          │
   │  X_sample, g_sample, h_sample   │
   │      ↓                          │
   │  DecisionTree.fit()             │
   │      ↓                          │
   │  tree                           │
   │      ↓                          │
   │  tree.predict(X)                │
   │      ↓                          │
   │  tree_pred                      │
   │      ↓                          │
   │  y_pred += lr * tree_pred       │
   │      │                          │
   └──────┘                          │
          ↓                          │
      Valider convergence            │
          ↓                          │
      Retour à l'itération ou fin    │
```

### Prédiction

```
X_new
  ↓
EFB.transform() (si utilisé)
  ↓
X_bundled
  ↓
init_prediction_
  ↓
for tree in trees_:
    pred += learning_rate * tree.predict(X_bundled)
  ↓
pred (raw)
  ↓
Pour classification: sigmoid/softmax
  ↓
Prédictions finales
```

---

## Diagrammes

### Diagramme de classes (simplifié)

```
              ┌─────────────────┐
              │  BaseEstimator  │
              │   (abstract)    │
              └────────┬────────┘
                       │
          ┌────────────┴────────────┐
          │                         │
  ┌───────▼────────┐       ┌───────▼────────┐
  │ LGBMRegressor  │       │ LGBMClassifier │
  └────────────────┘       └────────────────┘
          │                         │
          └─────────┬───────────────┘
                    │
              uses  ▼
          ┌─────────────────┐
          │  DecisionTree   │
          └────────┬────────┘
                   │
         ┌─────────┼─────────┐
         │         │         │
    uses ▼    uses ▼    uses ▼
  ┌──────┐  ┌─────┐  ┌─────┐
  │ GOSS │  │ EFB │  │Loss │
  └──────┘  └─────┘  └─────┘
```

### Diagramme de séquence (Training)

```
User         Estimator        Tree          Loss
 │              │              │             │
 │─fit(X, y)───→│              │             │
 │              │              │             │
 │              │─init_pred───→│────────────→│
 │              │←─────────────│←────────────│
 │              │              │             │
 │              │ LOOP [num_iterations]     │
 │              │              │             │
 │              │─gradients───→│────────────→│
 │              │─hessians────→│────────────→│
 │              │←─────────────│←────────────│
 │              │              │             │
 │              │─fit(X,g,h)──→│             │
 │              │              │             │
 │              │              │ [build tree]│
 │              │              │             │
 │              │←─tree────────│             │
 │              │              │             │
 │              │─predict(X)──→│             │
 │              │←─pred────────│             │
 │              │              │             │
 │              │ [update y_pred]           │
 │              │              │             │
 │              │ END LOOP     │             │
 │              │              │             │
 │←─model───────│              │             │
 │              │              │             │
```

---

## Considérations de performance

### Complexité algorithmique

**Exact split finding** :
- Tri : O(n log n) par feature
- Scan : O(n) par feature
- Total par arbre : O(n_features × n_samples × log(n_samples))

**Histogram split finding** :
- Binning : O(n_samples × n_features) (une fois)
- Histogram : O(n_samples × n_features)
- Split : O(max_bins × n_features)
- Total par arbre : O(n_samples × n_features + max_bins × n_features)

**GOSS** :
- Réduction de n_samples : ~70% en moins
- Accélération : ~2-3x

**EFB** :
- Réduction de n_features : dépend de la sparsité
- Exemple : 1000 features → 200 bundles (5x réduction)

### Compromis mémoire

**Stockage des arbres** :
```
Taille arbre ≈ num_leaves × (
    1 float (value) + 
    1 int (feature_idx) + 
    1 float (threshold)
) ≈ num_leaves × 16 bytes

Pour 100 arbres × 31 leaves : ~50 KB
```

**Histogram bins** :
```
Mémoire ≈ n_features × max