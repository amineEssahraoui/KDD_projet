"""
Test suite for LGBMClassifier using real-world datasets.

This module validates the custom LightGBM classifier implementation against
sklearn's GradientBoostingClassifier using real datasets:
- Binary classification: Breast Cancer dataset
- Multiclass classification: Wine dataset

Framework: pytest / unittest compatible
"""

import unittest
import numpy as np
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss

from lightgbm import LGBMClassifier


class TestLGBMClassifierBinary(unittest.TestCase):
    """Test LGBMClassifier on binary classification (Breast Cancer dataset)."""

    @classmethod
    def setUpClass(cls):
        """Load and split the breast cancer dataset once for all tests."""
        data = load_breast_cancer()
        cls.X = data.data
        cls.y = data.target
        cls.feature_names = data.feature_names
        cls.target_names = data.target_names

        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(
            cls.X, cls.y, test_size=0.3, random_state=42, stratify=cls.y
        )

        # Common hyperparameters for fair comparison
        cls.n_estimators = 50
        cls.learning_rate = 0.1
        cls.max_depth = 5
        cls.random_state = 42

    def test_01_model_fits_without_error(self):
        """Verify that LGBMClassifier fits without raising exceptions."""
        clf = LGBMClassifier(
            num_iterations=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=self.random_state,
        )
        try:
            clf.fit(self.X_train, self.y_train)
        except Exception as e:
            self.fail(f"LGBMClassifier.fit() raised an exception: {e}")

        # Verify model attributes are set
        self.assertIsNotNone(clf.classes_)
        self.assertEqual(len(clf.classes_), 2)
        self.assertEqual(len(clf.trees_), self.n_estimators)
        self.assertTrue(clf.is_binary_)

    def test_02_predict_returns_valid_labels(self):
        """Verify predict() returns labels from the correct set."""
        clf = LGBMClassifier(
            num_iterations=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=self.random_state,
        )
        clf.fit(self.X_train, self.y_train)
        y_pred = clf.predict(self.X_test)

        # Check output shape
        self.assertEqual(y_pred.shape, self.y_test.shape)

        # Check all predictions are valid class labels
        unique_preds = set(np.unique(y_pred))
        valid_classes = set(clf.classes_)
        self.assertTrue(unique_preds.issubset(valid_classes))

    def test_03_predict_proba_returns_valid_probabilities(self):
        """Verify predict_proba() returns valid probability distributions."""
        clf = LGBMClassifier(
            num_iterations=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=self.random_state,
        )
        clf.fit(self.X_train, self.y_train)
        y_proba = clf.predict_proba(self.X_test)

        # Check shape: (n_samples, n_classes)
        self.assertEqual(y_proba.shape, (len(self.X_test), 2))

        # Check probabilities are in [0, 1]
        self.assertTrue(np.all(y_proba >= 0))
        self.assertTrue(np.all(y_proba <= 1))

        # Check probabilities sum to 1
        row_sums = y_proba.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(len(self.X_test)))

    def test_04_accuracy_comparable_to_sklearn(self):
        """Compare accuracy with sklearn's GradientBoostingClassifier."""
        # Train custom model
        custom_clf = LGBMClassifier(
            num_iterations=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_data_in_leaf=5,
            random_state=self.random_state,
        )
        custom_clf.fit(self.X_train, self.y_train)
        custom_pred = custom_clf.predict(self.X_test)
        custom_acc = accuracy_score(self.y_test, custom_pred)

        # Train sklearn model
        sklearn_clf = GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=self.random_state,
        )
        sklearn_clf.fit(self.X_train, self.y_train)
        sklearn_pred = sklearn_clf.predict(self.X_test)
        sklearn_acc = accuracy_score(self.y_test, sklearn_pred)

        print(f"\n[Binary - Breast Cancer]")
        print(f"  Custom LGBMClassifier Accuracy:  {custom_acc:.4f}")
        print(f"  Sklearn GradientBoosting Accuracy: {sklearn_acc:.4f}")
        print(f"  Difference: {abs(custom_acc - sklearn_acc):.4f}")

        # Assert custom model achieves reasonable accuracy
        self.assertGreaterEqual(custom_acc, 0.90, "Custom model accuracy should be >= 90%")

        # Assert custom model is within 10% of sklearn
        margin = 0.10
        self.assertGreaterEqual(
            custom_acc,
            sklearn_acc - margin,
            f"Custom accuracy {custom_acc:.4f} should be within {margin} of sklearn {sklearn_acc:.4f}",
        )

    def test_05_logloss_comparable_to_sklearn(self):
        """Compare log loss with sklearn's GradientBoostingClassifier."""
        # Train custom model
        custom_clf = LGBMClassifier(
            num_iterations=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_data_in_leaf=5,
            random_state=self.random_state,
        )
        custom_clf.fit(self.X_train, self.y_train)
        custom_proba = custom_clf.predict_proba(self.X_test)
        custom_logloss = log_loss(self.y_test, custom_proba)

        # Train sklearn model
        sklearn_clf = GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=self.random_state,
        )
        sklearn_clf.fit(self.X_train, self.y_train)
        sklearn_proba = sklearn_clf.predict_proba(self.X_test)
        sklearn_logloss = log_loss(self.y_test, sklearn_proba)

        print(f"\n[Binary - Breast Cancer]")
        print(f"  Custom LGBMClassifier Log Loss:  {custom_logloss:.4f}")
        print(f"  Sklearn GradientBoosting Log Loss: {sklearn_logloss:.4f}")

        # Assert custom model achieves reasonable log loss (lower is better)
        self.assertLessEqual(custom_logloss, 0.30, "Custom model log loss should be <= 0.30")

    def test_06_score_method(self):
        """Test the score() convenience method."""
        clf = LGBMClassifier(
            num_iterations=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=self.random_state,
        )
        clf.fit(self.X_train, self.y_train)

        train_score = clf.score(self.X_train, self.y_train)
        test_score = clf.score(self.X_test, self.y_test)

        self.assertIsInstance(train_score, float)
        self.assertIsInstance(test_score, float)
        self.assertGreaterEqual(train_score, 0.0)
        self.assertLessEqual(train_score, 1.0)


class TestLGBMClassifierMulticlass(unittest.TestCase):
    """Test LGBMClassifier on multiclass classification (Wine dataset)."""

    @classmethod
    def setUpClass(cls):
        """Load and split the wine dataset once for all tests."""
        data = load_wine()
        cls.X = data.data
        cls.y = data.target
        cls.feature_names = data.feature_names
        cls.target_names = data.target_names
        cls.n_classes = len(cls.target_names)

        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(
            cls.X, cls.y, test_size=0.3, random_state=42, stratify=cls.y
        )

        # Common hyperparameters
        cls.n_estimators = 50
        cls.learning_rate = 0.1
        cls.max_depth = 4
        cls.random_state = 42

    def test_01_model_fits_without_error(self):
        """Verify that LGBMClassifier fits on multiclass data without exceptions."""
        clf = LGBMClassifier(
            num_iterations=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=self.random_state,
        )
        try:
            clf.fit(self.X_train, self.y_train)
        except Exception as e:
            self.fail(f"LGBMClassifier.fit() raised an exception: {e}")

        # Verify model attributes
        self.assertIsNotNone(clf.classes_)
        self.assertEqual(len(clf.classes_), self.n_classes)
        self.assertEqual(len(clf.trees_), self.n_estimators)
        self.assertFalse(clf.is_binary_)

        # Each iteration should have n_classes trees (One-vs-Rest)
        self.assertEqual(len(clf.trees_[0]), self.n_classes)

    def test_02_predict_returns_valid_labels(self):
        """Verify predict() returns labels from the correct set."""
        clf = LGBMClassifier(
            num_iterations=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=self.random_state,
        )
        clf.fit(self.X_train, self.y_train)
        y_pred = clf.predict(self.X_test)

        # Check output shape
        self.assertEqual(y_pred.shape, self.y_test.shape)

        # Check all predictions are valid class labels
        unique_preds = set(np.unique(y_pred))
        valid_classes = set(clf.classes_)
        self.assertTrue(unique_preds.issubset(valid_classes))

    def test_03_predict_proba_returns_valid_probabilities(self):
        """Verify predict_proba() returns valid probability distributions."""
        clf = LGBMClassifier(
            num_iterations=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=self.random_state,
        )
        clf.fit(self.X_train, self.y_train)
        y_proba = clf.predict_proba(self.X_test)

        # Check shape: (n_samples, n_classes)
        self.assertEqual(y_proba.shape, (len(self.X_test), self.n_classes))

        # Check probabilities are in [0, 1]
        self.assertTrue(np.all(y_proba >= 0))
        self.assertTrue(np.all(y_proba <= 1))

        # Check probabilities sum to 1
        row_sums = y_proba.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(len(self.X_test)))

    def test_04_accuracy_comparable_to_sklearn(self):
        """Compare multiclass accuracy with sklearn's GradientBoostingClassifier."""
        # Train custom model
        custom_clf = LGBMClassifier(
            num_iterations=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_data_in_leaf=3,
            random_state=self.random_state,
        )
        custom_clf.fit(self.X_train, self.y_train)
        custom_pred = custom_clf.predict(self.X_test)
        custom_acc = accuracy_score(self.y_test, custom_pred)

        # Train sklearn model
        sklearn_clf = GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=self.random_state,
        )
        sklearn_clf.fit(self.X_train, self.y_train)
        sklearn_pred = sklearn_clf.predict(self.X_test)
        sklearn_acc = accuracy_score(self.y_test, sklearn_pred)

        print(f"\n[Multiclass - Wine]")
        print(f"  Custom LGBMClassifier Accuracy:  {custom_acc:.4f}")
        print(f"  Sklearn GradientBoosting Accuracy: {sklearn_acc:.4f}")
        print(f"  Difference: {abs(custom_acc - sklearn_acc):.4f}")

        # Assert custom model achieves reasonable accuracy
        self.assertGreaterEqual(custom_acc, 0.85, "Custom model accuracy should be >= 85%")

        # Assert custom model is within 15% of sklearn
        margin = 0.15
        self.assertGreaterEqual(
            custom_acc,
            sklearn_acc - margin,
            f"Custom accuracy {custom_acc:.4f} should be within {margin} of sklearn {sklearn_acc:.4f}",
        )

    def test_05_logloss_comparable_to_sklearn(self):
        """Compare multiclass log loss with sklearn's GradientBoostingClassifier."""
        # Train custom model
        custom_clf = LGBMClassifier(
            num_iterations=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_data_in_leaf=3,
            random_state=self.random_state,
        )
        custom_clf.fit(self.X_train, self.y_train)
        custom_proba = custom_clf.predict_proba(self.X_test)
        custom_logloss = log_loss(self.y_test, custom_proba)

        # Train sklearn model
        sklearn_clf = GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=self.random_state,
        )
        sklearn_clf.fit(self.X_train, self.y_train)
        sklearn_proba = sklearn_clf.predict_proba(self.X_test)
        sklearn_logloss = log_loss(self.y_test, sklearn_proba)

        print(f"\n[Multiclass - Wine]")
        print(f"  Custom LGBMClassifier Log Loss:  {custom_logloss:.4f}")
        print(f"  Sklearn GradientBoosting Log Loss: {sklearn_logloss:.4f}")

        # Assert custom model achieves reasonable log loss
        self.assertLessEqual(custom_logloss, 0.50, "Custom model log loss should be <= 0.50")


class TestLGBMClassifierEdgeCases(unittest.TestCase):
    """Test edge cases and robustness of LGBMClassifier."""

    def test_01_non_contiguous_labels(self):
        """Test handling of non-contiguous class labels."""
        X = np.random.randn(100, 5)
        y = np.array([10, 20, 30] * 33 + [10])  # Labels: 10, 20, 30

        clf = LGBMClassifier(num_iterations=20, random_state=42)
        clf.fit(X, y)
        y_pred = clf.predict(X)

        # Check predictions are in original label set
        unique_preds = set(np.unique(y_pred))
        valid_classes = {10, 20, 30}
        self.assertTrue(unique_preds.issubset(valid_classes))

    def test_02_small_dataset(self):
        """Test with small dataset (few samples and features)."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        clf = LGBMClassifier(num_iterations=20, max_depth=3, min_data_in_leaf=2, random_state=42)
        clf.fit(X, y)
        acc = clf.score(X, y)

        self.assertGreater(acc, 0.7, "Should learn simple threshold pattern")

    def test_03_feature_importances(self):
        """Test feature importances are computed correctly."""
        data = load_breast_cancer()
        X, y = data.data, data.target

        clf = LGBMClassifier(num_iterations=30, max_depth=4, random_state=42)
        clf.fit(X, y)

        self.assertIsNotNone(clf.feature_importances_)
        self.assertEqual(len(clf.feature_importances_), X.shape[1])
        self.assertAlmostEqual(clf.feature_importances_.sum(), 1.0, places=5)

    def test_04_subsample_and_colsample(self):
        """Test with row and column subsampling."""
        data = load_breast_cancer()
        X, y = data.data, data.target
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        clf = LGBMClassifier(
            num_iterations=30,
            max_depth=4,
            subsample=0.8,
            colsample=0.8,
            random_state=42,
        )
        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)

        self.assertGreater(acc, 0.85, "Subsampled model should still perform well")


class TestLGBMClassifierBenchmarkSummary(unittest.TestCase):
    """Summary benchmark comparing custom LGBMClassifier vs sklearn."""

    def test_full_benchmark(self):
        """Run complete benchmark on both datasets and summarize."""
        print("\n" + "=" * 70)
        print("BENCHMARK SUMMARY: Custom LGBMClassifier vs Sklearn GradientBoosting")
        print("=" * 70)

        results = []

        # Binary: Breast Cancer
        data_bc = load_breast_cancer()
        X_train, X_test, y_train, y_test = train_test_split(
            data_bc.data, data_bc.target, test_size=0.3, random_state=42, stratify=data_bc.target
        )

        custom_clf = LGBMClassifier(
            num_iterations=50, learning_rate=0.1, max_depth=5, min_data_in_leaf=5, random_state=42
        )
        custom_clf.fit(X_train, y_train)
        custom_acc_bc = accuracy_score(y_test, custom_clf.predict(X_test))
        custom_ll_bc = log_loss(y_test, custom_clf.predict_proba(X_test))

        sklearn_clf = GradientBoostingClassifier(
            n_estimators=50, learning_rate=0.1, max_depth=5, random_state=42
        )
        sklearn_clf.fit(X_train, y_train)
        sklearn_acc_bc = accuracy_score(y_test, sklearn_clf.predict(X_test))
        sklearn_ll_bc = log_loss(y_test, sklearn_clf.predict_proba(X_test))

        results.append(("Breast Cancer (Binary)", custom_acc_bc, sklearn_acc_bc, custom_ll_bc, sklearn_ll_bc))

        # Multiclass: Wine
        data_wine = load_wine()
        X_train, X_test, y_train, y_test = train_test_split(
            data_wine.data, data_wine.target, test_size=0.3, random_state=42, stratify=data_wine.target
        )

        custom_clf = LGBMClassifier(
            num_iterations=50, learning_rate=0.1, max_depth=4, min_data_in_leaf=3, random_state=42
        )
        custom_clf.fit(X_train, y_train)
        custom_acc_wine = accuracy_score(y_test, custom_clf.predict(X_test))
        custom_ll_wine = log_loss(y_test, custom_clf.predict_proba(X_test))

        sklearn_clf = GradientBoostingClassifier(
            n_estimators=50, learning_rate=0.1, max_depth=4, random_state=42
        )
        sklearn_clf.fit(X_train, y_train)
        sklearn_acc_wine = accuracy_score(y_test, sklearn_clf.predict(X_test))
        sklearn_ll_wine = log_loss(y_test, sklearn_clf.predict_proba(X_test))

        results.append(("Wine (Multiclass)", custom_acc_wine, sklearn_acc_wine, custom_ll_wine, sklearn_ll_wine))

        # Print summary table
        print("\n{:<25} {:>12} {:>12} {:>12} {:>12}".format(
            "Dataset", "Custom Acc", "Sklearn Acc", "Custom LL", "Sklearn LL"
        ))
        print("-" * 70)
        for name, c_acc, s_acc, c_ll, s_ll in results:
            print("{:<25} {:>12.4f} {:>12.4f} {:>12.4f} {:>12.4f}".format(name, c_acc, s_acc, c_ll, s_ll))
        print("-" * 70)

        # All tests should pass - models should achieve reasonable performance
        self.assertGreaterEqual(custom_acc_bc, 0.90)
        self.assertGreaterEqual(custom_acc_wine, 0.85)


if __name__ == "__main__":
    unittest.main(verbosity=2)
