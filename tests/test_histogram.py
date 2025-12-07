import numpy as np

from lightgbm.histogramme import HistogramBinner


def test_histogram_bins_cover_range():
	binner = HistogramBinner(n_bins=8)
	X = np.array([[0.0, -1.0], [0.5, 0.0], [1.0, 2.0]])
	binner.fit(X)
	binned = binner.transform(X)
	assert binned.shape == X.shape
	assert binned.max() < 8
	assert binned.min() >= 0


def test_histogram_fit_transform_matches_transform():
	binner = HistogramBinner(n_bins=4)
	X = np.random.rand(10, 3)
	ft = binner.fit_transform(X)
	tr = binner.transform(X)
	assert np.array_equal(ft, tr)
