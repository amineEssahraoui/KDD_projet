# LightGBM From Scratch

A complete implementation of LightGBM (Light Gradient Boosting Machine) in pure Python/NumPy, developed as an academic project on decision tree algorithms.

**Zero sklearn dependency** — All algorithms implemented from scratch with NumPy only!

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Only-green)](https://numpy.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Table of Contents

- [Features](#-features)
- [Mathematical Foundations](#-mathematical-foundations)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Documentation](#-documentation)
- [Tests & Benchmarks](#-tests--benchmarks)
- [Usage Examples](#-usage-examples)
- [Architecture](#️-architecture)
- [Contributions](#-contributions)
- [Authors](#-authors)
- [License](#-license)
- [References](#-references)

---

## Features

### Implemented Algorithms

- **Binary & Multiclass Classification** — Full support for both
- **Regression** — MSE, MAE, Huber, Quantile loss
- **Leaf-wise tree growth** — LightGBM's key optimization
- **GOSS** (Gradient-based One-Side Sampling) — Training ~2-3x faster
- **Histogram Binning** — Efficient split search
- **EFB** (Exclusive Feature Bundling) — For high-dimensional data
- **Early Stopping** — Prevents overfitting
- **L1/L2 Regularization** — Complexity control
- **Feature Subsampling** — Random feature selection
- **Sample Weighting** — Sample weight support
- **sklearn-compatible API** — Familiar interface

### Available Loss Functions

**Regression:**
- `MSELoss` : Mean Squared Error (L2)
- `MAELoss` : Mean Absolute Error (L1)
- `HuberLoss` : Robust to outliers
- `QuantileLoss` : Quantile regression

**Classification:**
- `BinaryCrossEntropyLoss` : Binary classification
- `MultiClassCrossEntropyLoss` : Multiclass classification

---

## Mathematical Foundations

### 1. Additive Model

$$F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x)$$

- $F_{m-1}(x)$ — prediction from all previous trees
- $h_m(x)$ — new decision tree trained on residuals
- $\eta$ — learning rate $(0 < \eta \leq 1)$

### 2. Objective Function (2nd-order Taylor Expansion)

$$\mathcal{L}^{(m)} \approx \sum_{i=1}^{n} \left[ g_i \cdot h_m(x_i) + \frac{1}{2} h_i \cdot h_m(x_i)^2 \right] + \Omega(h_m)$$

- $g_i = \frac{\partial \mathcal{L}}{\partial \hat{y}_i}$ — gradient (slope of the loss)
- $h_i = \frac{\partial^2 \mathcal{L}}{\partial \hat{y}_i^2}$ — hessian (curvature of the loss)
- $\Omega(h_m)$ — regularization term

### 3. Optimal Leaf Weight

$$w_j^* = -\frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda}$$

- $I_j$ — set of samples in leaf $j$
- $\lambda$ — L2 regularization (`lambda_l2`)

### 4. Regularized Split Gain

$$\text{Gain} = \frac{1}{2} \left[ \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda} \right] - \gamma$$

- $G_L, G_R$ — gradient sums in left/right children
- $H_L, H_R$ — hessian sums in left/right children
- $\gamma$ — minimum gain threshold (`min_gain_to_split`)

### 5. Final Prediction

$$\hat{y} = F_0 + \eta \sum_{m=1}^{M} h_m(x)$$

### 6. Binary Classification — Sigmoid

$$p = \sigma(F(x)) = \frac{1}{1 + e^{-F(x)}}$$

| | Formula |
|---|---|
| Gradient | $g_i = p_i - y_i$ |
| Hessian | $h_i = p_i(1 - p_i)$ |

### 7. Multiclass — Softmax

$$p_k(x) = \frac{e^{F_k(x)}}{\sum_{j=1}^{K} e^{F_j(x)}}$$

| | Formula |
|---|---|
| Gradient | $g_{ik} = p_{ik} - y_{ik}$ |
| Hessian (diagonal approx.) | $h_{ik} \approx p_{ik}(1 - p_{ik})$ |

### 8. Histogram Binning — Complexity

| Operation | Exact Greedy | Histogram-Based |
|---|---|---|
| Split finding per node | $O(N \cdot D)$ | $O(B \cdot D)$ |
| Memory | $O(N \cdot D)$ | $O(B \cdot D)$ |
| Typical magnitude | $N = 10^6$ | $B = 255$ |

### 9. Loss Functions Reference

| Loss | $\mathcal{L}(y, \hat{y})$ | Gradient $g_i$ | Hessian $h_i$ |
|---|---|---|---|
| MSE | $\frac{1}{2}(y-\hat{y})^2$ | $\hat{y}-y$ | $1$ |
| MAE | $|y-\hat{y}|$ | $\text{sign}(\hat{y}-y)$ | $1$ (approx.) |
| Huber | $\frac{1}{2}r^2$ or $\delta|r|-\frac{1}{2}\delta^2$ | $r$ or $\delta\cdot\text{sign}(r)$ | $1$ or $\delta/|r|$ |
| Quantile | $q(y-\hat{y})$ or $(1-q)(\hat{y}-y)$ | $-q$ or $1-q$ | $1$ (approx.) |
| Binary CE | $-[y\log p+(1-y)\log(1-p)]$ | $p_i-y_i$ | $p_i(1-p_i)$ |
| Multiclass CE | $-\sum_k y_k\log(p_k)$ | $p_{ik}-y_{ik}$ | $p_{ik}(1-p_{ik})$ |

---

---

## Quick Start

### Classification
```python
from lightgbm import LGBMClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

clf = LGBMClassifier(num_iterations=100, learning_rate=0.1, max_depth=6)
clf.fit(X_train, y_train)

predictions = clf.predict(X_test)
probabilities = clf.predict_proba(X_test)
print(f"Accuracy: {(predictions == y_test).mean():.4f}")
```

### Regression
```python
from lightgbm import LGBMRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

X, y = make_regression(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

reg = LGBMRegressor(num_iterations=100, learning_rate=0.1, max_depth=6)
reg.fit(X_train, y_train)
predictions = reg.predict(X_test)
```

### Early Stopping
```python
clf = LGBMClassifier(num_iterations=1000, early_stopping_rounds=10)
clf.fit(X_train, y_train, eval_set=(X_val, y_val))
print(f"Stopped at iteration: {clf.n_iter_}")
```

### GOSS (Acceleration)
```python
clf = LGBMClassifier(
    num_iterations=100,
    enable_goss=True,
    goss_top_rate=0.2
)
clf.fit(X_train, y_train)
```

---

## Project Structure
```
KDD_projet/
│
├── src/lightgbm/              # Main package
│   ├── __init__.py            # Public exports
│   ├── base.py                # Base classes (BaseEstimator, BoosterParams, Callback)
│   ├── lgbm_classifier.py     # LGBMClassifier
│   ├── lgbm_regressor.py      # LGBMRegressor
│   ├── tree.py                # DecisionTree with leaf-wise growth
│   ├── histogram.py           # Histogram binning (integrated in tree.py)
│   ├── goss.py                # GOSS sampling (GOSS class, apply_goss)
│   ├── efb.py                 # Exclusive Feature Bundling
│   ├── loss_functions.py      # Loss functions + gradients/hessians
│   └── utils.py               # Validation and utilities
│
├── tests/                     # Test suite
│   ├── test_classifier.py
│   ├── test_regressor.py
│   ├── test_tree.py
│   ├── test_goss.py
│   ├── test_utils.py
│   ├── test_math_integrity.py
│   └── test_logic_sanity.py
│
├── benchmarks/
│   └── benchmark_comparison.py
│
├── examples/
│   ├── complete_testing.ipynb
│   └── regression_pipeline.py
│
├── docs/
│   ├── ARCHITECTURE.md
│   └── IMPLEMENTATION_GUIDE.md
│
├── pyproject.toml
├── requirements.txt
├── LICENSE
└── README.md
```
