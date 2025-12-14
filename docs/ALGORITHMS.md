# Algorithmes Implémentés

## Gradient Boosting Decision Trees

L'implémentation suit l'algorithme original de LightGBM avec les optimisations principales :

### Phase 1 : Initialisation

1. Prédiction initiale (moyenne pour régression, log odds pour classification)
2. Calcul des résidus (gradients)

### Phase 2 : Boosting Itératif

Pour chaque itération :

1. **Calcul des gradients et hessians**

   - Classification : Cross-entropy
   - Régression : MSE

2. **Construction de l'arbre**

   - Stratégie `leaf_wise` (croissance en profondeur)
   - Split optimal basé sur la réduction de la perte

3. **Mise à jour des prédictions**
   - Apprentissage graduel (learning rate)
   - Accumulation des prédictions

## GOSS (Gradient-based One-Side Sampling)

Réduit le nombre d'instances tout en conservant le pouvoir prédictif :

1. Trier les instances par gradient absolu
2. Garder les instances avec gradients larges (top_a%)
3. Sampler aléatoirement les instances avec gradients petits (bottom_b%)
4. Pondérer les instances pour ajuster la perte

## EFB (Exclusive Feature Bundling)

Combine les features non-corrélées :

1. Construire un graphe de conflits entre features
2. Identifier les ensembles de features mutellement exclusives
3. Combiner les features dans chaque ensemble
4. Réduire la dimensionalité

## Binning / Histogramme

Discrétise les features continues :

1. Diviser chaque feature en `max_bins` intervalles
2. Assigner chaque valeur à son bin
3. Construire des histogrammes pour les splits
4. Réduit la consommation mémoire de O(n*d) à O(k*d)

## Leaf-wise Tree Growth

Stratégie de croissance optimale pour boosting :

1. Commencer par une racine
2. À chaque itération, sélectionner la feuille avec meilleure réduction de perte
3. Splitter cette feuille en deux
4. Répéter jusqu'à max_depth

Avantage : Converge plus rapidement que level-wise

## Régularisation

- **L1 (Lasso)** : Encourage la sparsité
- **L2 (Ridge)** : Pénalise les grands coefficients
- **Early Stopping** : Arrête l'entraînement si pas d'amélioration

## Fonction de perte

### Classification (Binary Cross-Entropy)

```
Loss = -[y*log(p) + (1-y)*log(1-p)]
Gradient = p - y
Hessian = p*(1-p)
```

### Régression (Mean Squared Error)

```
Loss = (y - y_pred)²
Gradient = 2*(y_pred - y)
Hessian = 2
```
