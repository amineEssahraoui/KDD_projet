import sys
import os
import numpy as np

# Ensure local `src/` package is importable when running tests without installation
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from lightgbm.lgbm_classifier import LGBMClassifier
from lightgbm.lgbm_regressor import LGBMRegressor


def test_regressor_uses_logging(capsys):
    X = np.random.RandomState(0).rand(20, 3)
    y = X[:, 0] * 2.0 + 0.1 * np.random.RandomState(1).randn(20)

    model = LGBMRegressor(num_iterations=3, num_leaves=3, verbose=1)
    model.fit(X, y)

    captured = capsys.readouterr()
    assert "[LightGBM]" in captured.out
    assert "Iter" in captured.out


def test_classifier_uses_logging(capsys):
    rng = np.random.RandomState(2)
    X = rng.rand(30, 4)
    y = (X[:, 0] + X[:, 1] > 1.0).astype(int)

    model = LGBMClassifier(num_iterations=3, num_leaves=3, verbose=1)
    model.fit(X, y)

    captured = capsys.readouterr()
    assert "[LightGBM]" in captured.out
    assert "Iter" in captured.out

