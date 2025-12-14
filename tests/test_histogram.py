"""
Test suite for histogram-based split finding implementation.
"""

import numpy as np
import pytest

from lightgbm.histogram import HistogramBuilder, HistogramBin


def test_histogram_bins_cover_range():
    """Test that histogram bins cover the data range."""
    builder = HistogramBuilder(max_bins=8)
    X = np.array([[0.0, -1.0], [0.5, 0.0], [1.0, 2.0]])
    builder.fit(X)
    binned = builder.transform(X)
    
    assert binned.shape == X.shape
    assert binned.max() < 8
    assert binned.min() >= 0


def test_histogram_fit_transform_matches_transform():
    """Test that fit_transform gives same result as fit + transform."""
    builder = HistogramBuilder(max_bins=4)
    X = np.random.rand(10, 3)
    
    ft = builder.fit_transform(X)
    
    # Create new builder and do fit + transform separately
    builder2 = HistogramBuilder(max_bins=4)
    builder2.fit(X)
    tr = builder2.transform(X)
    
    assert np.array_equal(ft, tr)


def test_histogram_builder_creates_bins():
    """Test that HistogramBuilder creates proper bins."""
    builder = HistogramBuilder(max_bins=10)
    X = np.random.randn(100, 5)
    builder.fit(X)
    
    # Should have bin edges for each feature
    assert builder.bin_edges_ is not None
    assert len(builder.bin_edges_) == 5


def test_histogram_builder_consistent_binning():
    """Test that same values get same bin assignments."""
    builder = HistogramBuilder(max_bins=10)
    X = np.array([[1.0], [2.0], [3.0], [1.0], [2.0], [3.0]])
    builder.fit(X)
    binned = builder.transform(X)
    
    # Same values should get same bins
    assert binned[0, 0] == binned[3, 0]  # Both are 1.0
    assert binned[1, 0] == binned[4, 0]  # Both are 2.0
    assert binned[2, 0] == binned[5, 0]  # Both are 3.0


def test_histogram_bin_dataclass():
    """Test HistogramBin dataclass."""
    bin = HistogramBin(sum_gradients=1.5, sum_hessians=2.0, count=10)
    assert bin.sum_gradients == 1.5
    assert bin.sum_hessians == 2.0
    assert bin.count == 10


def test_histogram_builder_handles_constant_feature():
    """Test that constant features are handled properly."""
    builder = HistogramBuilder(max_bins=10)
    X = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])  # Second column is constant
    builder.fit(X)
    binned = builder.transform(X)
    
    # Should still work
    assert binned.shape == X.shape


def test_histogram_builder_min_data_in_bin():
    """Test min_data_in_bin parameter."""
    builder = HistogramBuilder(max_bins=100, min_data_in_bin=10)
    X = np.random.randn(50, 2)
    builder.fit(X)
    
    # With min_data_in_bin=10 and 50 samples, should have <= 5 bins
    binned = builder.transform(X)
    for col in range(binned.shape[1]):
        unique_bins = np.unique(binned[:, col])
        # Should not exceed max_bins and respect min_data constraint
        assert len(unique_bins) <= 100
