import numpy as np

from lightgbm import LGBMRegressor, LGBMClassifier


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
