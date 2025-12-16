# ALGORITHMS.md

## Vue d'ensemble des algorithmes implémentés

Ce document détaille les algorithmes et techniques mathématiques implémentés dans ce projet LightGBM from scratch.

---

## 1. Gradient Boosting

### Principe général

Le gradient boosting construit un ensemble de modèles faibles (arbres de décision) de manière séquentielle, où chaque nouveau modèle corrige les erreurs des modèles précédents.

**Formule de prédiction:**
```
F_m(x) = F_{m-1}(x) + η · h_m(x)
```

où:
- `F_m(x)` est la prédiction après m itérations
- `η` est le learning rate (taux d'apprentissage)
- `h_m(x)` est le m-ième arbre

### Algorithme complet

```
Initialisation: F_0(x) = argmin_γ Σ L(y_i, γ)

Pour m = 1 à M:
    1. Calculer les gradients: g_i = -∂L(y_i, F_{m-1}(x_i))/∂F
    2. Calculer les hessiens: h_i = ∂²L(y_i, F_{m-1}(x_i))/∂F²
    3. Construire arbre h_m(x) en minimisant les gradients
    4. Mettre à jour: F_m(x) = F_{m-1}(x) + η · h_m(x)

Prédiction finale: F_M(x)
```

---

## 2. Leaf-wise Tree Growth (Croissance par feuilles)

### Différence avec Level-wise

**Level-wise (niveau par niveau):**
- Divise tous les nœuds d'un même niveau
- Créé un arbre équilibré
- Peut être sous-optimal

**Leaf-wise (feuille par feuille - LightGBM):**
- Divise la feuille avec le gain maximal
- Créé un arbre déséquilibré mais plus efficace
- Meilleure précision avec moins de nœuds

### Algorithme

```python
# File: src/lightgbm/tree.py
# Méthode: _build_tree_leaf_wise

1. Créer racine avec tous les échantillons
2. Calculer la valeur de feuille initiale
3. file_priorité = []

4. Tant que nombre_feuilles < num_leaves:
    a. Pour chaque feuille candidate:
       - Trouver meilleur split
       - Calculer gain du split
       - Ajouter à file_priorité (triée par gain)
    
    b. Extraire feuille avec gain maximal
    c. Appliquer le split
    d. Créer deux enfants
    e. nombre_feuilles += 1

5. Retourner arbre
```

**Gain de split:**
```
Gain = 0.5 * [G_L²/(H_L + λ) + G_R²/(H_R + λ) - G²/(H + λ)] - γ
```

où:
- `G_L, G_R` = somme des gradients gauche/droite
- `H_L, H_R` = somme des hessiens gauche/droite
- `λ` = régularisation L2
- `γ` = gain minimum pour split

---

## 3. GOSS (Gradient-based One-Side Sampling)

### Principe

GOSS accélère l'entraînement en échantillonnant intelligemment les données:
1. Garder tous les échantillons avec **grands gradients** (top_rate)
2. Échantillonner aléatoirement parmi les **petits gradients** (other_rate)
3. Ré-pondérer les petits gradients pour compenser

### Algorithme

```python
# File: src/lightgbm/goss.py

def GOSS(gradients, top_rate=0.2, other_rate=0.1):
    n = len(gradients)
    n_top = int(n * top_rate)
    n_other = int(n * other_rate)
    
    # 1. Trier par gradient absolu
    sorted_indices = argsort(|gradients|, descending=True)
    
    # 2. Sélectionner top gradients
    top_indices = sorted_indices[:n_top]
    
    # 3. Échantillonner autres
    remaining = sorted_indices[n_top:]
    other_indices = random.choice(remaining, n_other)
    
    # 4. Combiner
    selected = concatenate([top_indices, other_indices])
    
    # 5. Ré-pondération
    weight_factor = (1 - top_rate) / other_rate
    weights = ones(len(selected))
    weights[n_top:] = weight_factor
    
    return selected, weights
```

### Justification mathématique

Le ré-pondération assure que:
```
E[Σ weighted_gradients] ≈ Σ all_gradients
```

**Facteur de pondération:**
```
w = (1 - a) / b
```
où `a = top_rate`, `b = other_rate`

---

## 4. Histogram Binning

### Principe

Au lieu de tester tous les seuils possibles, on discrétise les features en bins:
- Réduit la complexité de O(n) à O(bins)
- Accélère la recherche de splits
- Réduit l'utilisation mémoire

### Algorithme

```python
# File: src/lightgbm/tree.py
# Méthode: _find_best_split_histogram

1. Pré-calcul des bins (lors du fit):
   bins = quantile(feature_values, percentiles)

2. Pour chaque feature lors du split:
   a. Assigner chaque valeur à un bin
      bin_idx = digitize(values, bins)
   
   b. Construire histogramme de gradients/hessians
      G_bins = bincount(bin_idx, weights=gradients)
      H_bins = bincount(bin_idx, weights=hessians)
   
   c. Cumulative sums
      G_left_cumsum = cumsum(G_bins)
      H_left_cumsum = cumsum(H_bins)
   
   d. Trouver meilleur split parmi les bins
      Pour chaque position de split i:
          G_L = G_left_cumsum[i]
          H_L = H_left_cumsum[i]
          G_R = G_total - G_L
          H_R = H_total - H_L
          
          gain[i] = compute_gain(G_L, H_L, G_R, H_R)
      
      best_split = argmax(gain)

3. Retourner seuil correspondant au meilleur bin
```

---

## 5. EFB (Exclusive Feature Bundling)

### Principe

Regrouper les features mutuellement exclusives (rarement non-nulles ensemble) pour réduire la dimensionnalité.

**Exemple:**
```
Feature A: [1, 0, 0, 2, 0]
Feature B: [0, 3, 0, 0, 4]
→ Regrouper en: [1, 3, 0, 2, 4] avec offsets
```

### Algorithme

```python
# File: src/lightgbm/efb.py

1. Calculer matrice de conflits:
   conflict[i,j] = fraction d'échantillons où features i et j sont non-nulles

2. Greedy bundling:
   bundles = []
   used = []
   
   Pour chaque feature i non utilisée:
       bundle = [i]
       Pour chaque feature j > i non utilisée:
           Si conflict(j, bundle) < max_conflict_rate:
               Ajouter j au bundle
               Marquer j comme utilisé
       
       bundles.append(bundle)

3. Calculer offsets pour merger:
   Pour chaque bundle [f1, f2, ..., fk]:
       offset[0] = 0
       offset[i] = offset[i-1] + max(|feature[i-1]|) + 1

4. Créer features bundlées:
   bundled_feature = Σ (feature[i] + offset[i]) pour i dans bundle
```

### Transformation

```python
def transform(X, bundles, offsets):
    X_bundled = zeros((n_samples, n_bundles))
    
    pour bundle_idx, bundle in enumerate(bundles):
        pour feat_idx, feature in enumerate(bundle):
            values = X[:, feature]
            non_zero = values != 0
            
            X_bundled[non_zero, bundle_idx] += (
                values[non_zero] + offsets[bundle_idx][feat_idx]
            )
    
    return X_bundled
```

---

## 6. Loss Functions (Fonctions de perte)

### 6.1 MSE (Mean Squared Error)

**Fonction de perte:**
```
L(y, f) = 0.5 * (y - f)²
```

**Gradient:**
```
∂L/∂f = f - y
```

**Hessian:**
```
∂²L/∂f² = 1
```

**Prédiction initiale:**
```
f_0 = mean(y)
```

---

### 6.2 MAE (Mean Absolute Error)

**Fonction de perte:**
```
L(y, f) = |y - f|
```

**Gradient:**
```
∂L/∂f = sign(f - y)
```

**Hessian:**
```
∂²L/∂f² = 1  (pour stabilité numérique)
```

**Prédiction initiale:**
```
f_0 = median(y)
```

---

### 6.3 Huber Loss

**Fonction de perte:**
```
L(y, f) = {
    0.5 * (y - f)²           si |y - f| ≤ δ
    δ * |y - f| - 0.5 * δ²   sinon
}
```

**Gradient:**
```
∂L/∂f = {
    f - y           si |f - y| ≤ δ
    δ * sign(f - y) sinon
}
```

**Hessian:**
```
∂²L/∂f² = {
    1               si |f - y| ≤ δ
    δ / |f - y|     sinon
}
```

---

### 6.4 Binary Cross-Entropy

**Fonction de perte:**
```
L(y, f) = -[y * log(σ(f)) + (1-y) * log(1 - σ(f))]

où σ(f) = 1 / (1 + e^(-f))  (sigmoid)
```

**Gradient:**
```
∂L/∂f = σ(f) - y = p - y
```

**Hessian:**
```
∂²L/∂f² = p * (1 - p)

où p = σ(f)
```

**Prédiction initiale:**
```
f_0 = log(p / (1-p))  où p = mean(y)
```

**Prédiction finale:**
```
P(y=1|x) = σ(f(x))
```

---

### 6.5 Multi-class Cross-Entropy

**Fonction de perte:**
```
L(y, f) = -Σ_k [1(y=k) * log(p_k)]

où p_k = softmax(f)_k = e^(f_k) / Σ_j e^(f_j)
```

**Gradient pour classe k:**
```
∂L/∂f_k = p_k - 1(y=k)
```

**Hessian pour classe k:**
```
∂²L/∂f_k² = p_k * (1 - p_k)
```

**Prédiction initiale:**
```
f_0,k = log(freq(y=k))
```

---

## 7. Régularisation

### 7.1 L1 Regularization (Lasso)

**Soft thresholding sur les gradients:**

```python
def regularize_gradient(G, lambda_l1):
    if G > lambda_l1:
        return G - lambda_l1
    elif G < -lambda_l1:
        return G + lambda_l1
    else:
        return 0
```

**Effet:** Force certains poids à zéro → sélection de features

---

### 7.2 L2 Regularization (Ridge)

**Ajout au dénominateur dans le calcul du score:**

```
score = G² / (H + λ₂)

leaf_value = -G / (H + λ₂)
```

**Effet:** Réduit l'amplitude des poids → prévient overfitting

---

### 7.3 Min Gain to Split

**Condition de split:**
```
if gain < min_gain_to_split:
    ne pas splitter
```

**Effet:** Empêche les splits inutiles → arbre plus simple

---

## 8. Feature Importance

### Calcul basé sur le gain

```python
def compute_feature_importances(trees):
    importances = zeros(n_features)
    
    pour tree in trees:
        pour node in tree.all_nodes():
            if not node.is_leaf:
                feature = node.feature_idx
                gain = node.gain
                importances[feature] += gain
    
    # Normalisation
    importances /= sum(importances)
    
    return importances
```

**Interprétation:**
- Valeur élevée → feature importante pour les prédictions
- Basé sur la réduction totale de la perte

---

## 9. Early Stopping

### Principe

Arrêter l'entraînement quand la performance sur validation ne s'améliore plus.

### Algorithme

```python
best_loss = inf
patience_counter = 0

pour iteration in range(num_iterations):
    train_tree()
    val_loss = evaluate_on_validation()
    
    if val_loss < best_loss - min_delta:
        best_loss = val_loss
        patience_counter = 0
    else:
        patience_counter += 1
    
    if patience_counter >= early_stopping_rounds:
        print(f"Early stopping at iteration {iteration}")
        break
```

---

## 10. Complexité algorithmique

### Sans optimisations

**Par itération:**
- Recherche de split: O(n * d * n)
  - n échantillons
  - d features
  - n valeurs possibles par feature

**Total:** O(M * n² * d) pour M itérations

---

### Avec optimisations LightGBM

**Histogram binning:**
- Recherche de split: O(n * d * b)
  - b bins (b << n)

**GOSS:**
- Échantillons utilisés: a*n + b*n (a, b << 1)

**EFB:**
- Features bundlées: d' << d

**Total optimisé:** O(M * n * d' * b)

**Gain:** Réduction significative, typiquement 5-10x plus rapide

---

## 11. Gestion des valeurs manquantes (NaN)

### Stratégie

Pour chaque split candidat, tester deux assignations des NaN:
1. NaN → gauche
2. NaN → droite

Choisir l'assignation qui maximise le gain.

### Implémentation

```python
# Dans _find_best_split_histogram

# Séparer NaN des valeurs valides
valid_mask = ~isnan(feature_values)
nan_mask = isnan(feature_values)

# Calculer gradients/hessians pour NaN
G_nan = sum(gradients[nan_mask])
H_nan = sum(hessians[nan_mask])

# Tester les deux assignations
# Option A: NaN à gauche
G_left_A = G_left_base + G_nan
H_left_A = H_left_base + H_nan

# Option B: NaN à droite
G_left_B = G_left_base
H_left_B = H_left_base

# Calculer gains pour les deux options
gain_A = compute_gain(G_left_A, H_left_A, ...)
gain_B = compute_gain(G_left_B, H_left_B, ...)

# Choisir la meilleure
if gain_A > gain_B:
    use_option_A()
else:
    use_option_B()
```

---

## Références

1. **LightGBM Paper**: Ke et al., "LightGBM: A Highly Efficient Gradient Boosting Decision Tree", NeurIPS 2017
2. **Gradient Boosting**: Friedman, "Greedy Function Approximation: A Gradient Boosting Machine", 2001
3. **XGBoost Paper**: Chen & Guestrin, "XGBoost: A Scalable Tree Boosting System", KDD 2016

---

## Notes d'implémentation

- **Stabilité numérique:** Clipping des valeurs extrêmes (sigmoid, exp, log)
- **Hessian floor:** Minimum de 1e-3 pour éviter division par zéro
- **Leaf-wise growth:** Priority queue (heap) pour efficacité
- **Vectorisation:** Opérations NumPy pour performance
