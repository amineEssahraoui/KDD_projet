# LightGBM From Scratch

A complete implementation of LightGBM (Light Gradient Boosting Machine) in pure Python/NumPy, developed as an academic project on decision tree algorithms.

**Zero sklearn dependencies** - All algorithms implemented from scratch!

---

## Features

- **Binary & Multiclass Classification** - Full support for both
- **Regression** - MSE, MAE, Huber, Quantile loss functions
- **Leaf-wise Tree Growth** - Key LightGBM optimization
- **GOSS Sampling** - Gradient-based One-Side Sampling for faster training
- **Histogram Binning** - Efficient split finding
- **EFB** - Exclusive Feature Bundling for high-dimensional data
- **Early Stopping** - Prevent overfitting
- **L1/L2 Regularization** - Prevent overfitting
- **Feature Subsampling** - Random feature selection
- **Sample Weighting** - Support for weighted samples
- **sklearn-compatible API** - Familiar interface (`n_estimators`, `max_depth`, etc.)

---

## Installation

```bash
# Clone the repository
git clone https://github.com/amineEssahraoui/KDD_projet.git
cd KDD_projet

# Install in development mode
pip install -e .

# Or install dependencies only
pip install -r requirements.txt
```

---

## Quick Start

### Classification

```python
from lightgbm import LGBMClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate data
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Train model
clf = LGBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=6)
clf.fit(X_train, y_train)

# Predict
predictions = clf.predict(X_test)
probabilities = clf.predict_proba(X_test)

print(f"Accuracy: {(predictions == y_test).mean():.4f}")
```

### Regression

```python
from lightgbm import LGBMRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# Generate data
X, y = make_regression(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Train model
reg = LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=6)
reg.fit(X_train, y_train)

# Predict
predictions = reg.predict(X_test)
```

### Early Stopping

```python
clf = LGBMClassifier(n_estimators=1000, early_stopping_rounds=10)
clf.fit(X_train, y_train, eval_set=(X_val, y_val))
print(f"Stopped at iteration: {clf.n_iter_}")
```

### GOSS (Gradient-based One-Side Sampling)

```python
clf = LGBMClassifier(n_estimators=100, enable_goss=True, goss_top_rate=0.2)
clf.fit(X_train, y_train)
```

---

## Project Structure

```
KDD_projet/
|
+-- src/lightgbm/           # Main package
|   +-- __init__.py         # Package exports
|   +-- base.py             # Base estimator classes
|   +-- lgbm_classifier.py  # LGBMClassifier
|   +-- lgbm_regressor.py   # LGBMRegressor
|   +-- tree.py             # Decision tree with leaf-wise growth
|   +-- histogram.py        # Histogram binning for splits
|   +-- goss.py             # GOSS sampling
|   +-- efb.py              # Exclusive Feature Bundling
|   +-- loss_functions.py   # MSE, MAE, CrossEntropy, etc.
|   +-- utils.py            # Validation and utilities
|
+-- tests/                  # Test suite
|   +-- test_classifier.py  # Classifier tests
|   +-- test_regressor.py   # Regressor tests
|   +-- test_tree.py        # Tree tests
|   +-- test_histogram.py   # Histogram tests
|   +-- test_goss.py        # GOSS tests
|   +-- test_utils.py       # Utility tests
|   +-- test_math_integrity.py   # Math validation tests
|   +-- test_logic_sanity.py     # Sanity checks
|
+-- benchmarks/             # Performance benchmarks
|   +-- benchmark_comparison.py  # Compare with sklearn
|
+-- examples/               # Example notebooks
|   +-- Seance_1_Regression.ipynb
|   +-- regression_pipeline.py
|
+-- docs/                   # Documentation
+-- pyproject.toml          # Package configuration
+-- requirements.txt        # Dependencies
+-- README.md               # This file
```

---

## Module Descriptions

### `base.py` - Base Classes
- `BoosterParams`: Dataclass for all hyperparameters
- `BaseEstimator`: Abstract base class with common functionality
- `Callback`: Training callback interface

### `lgbm_classifier.py` - Classification
- `LGBMClassifier`: Gradient boosting classifier
- Supports binary and multiclass classification
- `predict()`, `predict_proba()`, `score()` methods

### `lgbm_regressor.py` - Regression
- `LGBMRegressor`: Gradient boosting regressor
- Multiple loss functions: MSE, MAE, Huber, Quantile
- `predict()`, `score()` methods

### `tree.py` - Decision Tree
- `TreeNode`: Node structure
- `SplitInfo`: Split information container
- `DecisionTree`: Leaf-wise tree with histogram support
- Key LightGBM optimization: grows deepest leaf first

### `histogram.py` - Histogram Binning
- `HistogramBuilder`: Creates bins for efficient split finding
- Reduces O(n) to O(bins) complexity per split

### `goss.py` - GOSS Sampling
- `GOSS`: Gradient-based One-Side Sampling
- Keeps top gradients, samples small gradients
- Speeds up training without losing accuracy

### `efb.py` - Exclusive Feature Bundling
- `FeatureBundler`: Bundles mutually exclusive features
- Reduces dimensionality for sparse data

### `loss_functions.py` - Loss Functions
- `MSELoss`: Mean Squared Error
- `MAELoss`: Mean Absolute Error
- `HuberLoss`: Robust regression
- `QuantileLoss`: Quantile regression
- `BinaryCrossEntropyLoss`: Binary classification
- `MultiClassCrossEntropyLoss`: Multiclass classification

### `utils.py` - Utilities
- Input validation functions
- Metrics: accuracy, MSE, MAE, R2
- Train/test split utility

---

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_estimators` / `num_iterations` | 100 | Number of boosting iterations |
| `learning_rate` | 0.1 | Shrinkage rate |
| `max_depth` | -1 | Max tree depth (-1 = unlimited) |
| `num_leaves` | 31 | Max leaves per tree |
| `min_data_in_leaf` / `min_samples_leaf` | 20 | Min samples per leaf |
| `lambda_l1` / `reg_alpha` | 0.0 | L1 regularization |
| `lambda_l2` / `reg_lambda` | 0.0 | L2 regularization |
| `feature_fraction` | 1.0 | Feature subsampling ratio |
| `bagging_fraction` | 1.0 | Sample subsampling ratio |
| `enable_goss` | False | Use GOSS sampling |
| `use_histogram` | False | Use histogram binning |
| `early_stopping_rounds` | None | Early stopping patience |

---

## Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_classifier.py -v

# Run with coverage
python -m pytest tests/ --cov=src/lightgbm --cov-report=html
```

---

## Benchmarks

Compare with sklearn's GradientBoosting:

```bash
python benchmarks/benchmark_comparison.py
```

### Performance Results (n=2000, 50 trees)

| Task | Our LightGBM | sklearn | Speed Ratio |
|------|-------------|---------|-------------|
| Binary Classification | 89.8% acc | 89.6% acc | ~2x slower |
| Regression | 90.9% R2 | 89.1% R2 | ~2x slower |
| Multiclass (3 classes) | 95.0% acc | - | Working! |

Our implementation achieves comparable accuracy to sklearn while being only ~2x slower (pure Python vs Cython).

---

## Key Differences from Standard GBDT

1. **Leaf-wise Growth**: Grows the leaf with maximum gain first (vs level-wise)
2. **GOSS**: Samples based on gradient magnitude
3. **EFB**: Bundles mutually exclusive features
4. **Histogram Binning**: Discretizes features for faster split finding

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## Authors

- LightGBM Scratch Contributors (KDD Project)

---

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

## References

- [LightGBM Paper](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree)
- [XGBoost Paper](https://arxiv.org/abs/1603.02754)
- [Gradient Boosting Explained](https://explained.ai/gradient-boosting/)
