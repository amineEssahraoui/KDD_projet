"""
Test suite for GOSS (Gradient-based One-Side Sampling) implementation.
"""

import numpy as np
import pytest

from lightgbm.goss import GOSS


def test_goss_selects_top_and_samples_rest():
    """Test that GOSS selects top gradients and samples rest."""
    gradients = np.array([0.9, 0.8, 0.1, 0.05, 0.02, 0.5])
    
    goss = GOSS(top_rate=0.3, other_rate=0.5, random_state=0)
    selected_indices, sample_weights = goss.sample(gradients)
    
    # Should select some samples
    assert len(selected_indices) >= 2
    
    # Weights should be same length as indices
    assert len(sample_weights) == len(selected_indices)
    
    # The largest gradient (index 0, value 0.9) should be in selected
    assert 0 in selected_indices


def test_goss_reweights_small_gradients():
    """Test that GOSS applies weight factor to sampled small gradients."""
    gradients = np.array([1.0, 0.1, 0.05, 0.02, 0.01])
    
    goss = GOSS(top_rate=0.2, other_rate=0.4, random_state=1)
    selected_indices, sample_weights = goss.sample(gradients)
    
    # Top samples should have weight 1
    # The first sample(s) in selected_indices are top samples
    n_top = max(1, int(len(gradients) * 0.2))
    
    # First n_top weights should be 1.0
    assert np.allclose(sample_weights[:n_top], 1.0)
    
    # Remaining weights should be the amplification factor
    if len(sample_weights) > n_top:
        expected_factor = (1 - 0.2) / 0.4  # (1 - top_rate) / other_rate = 2.0
        assert np.allclose(sample_weights[n_top:], expected_factor)


def test_goss_sample_data():
    """Test GOSS sample_data method."""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    gradients = np.random.randn(100)
    hessians = np.ones(100)
    
    goss = GOSS(top_rate=0.2, other_rate=0.1, random_state=42)
    X_sampled, grad_sampled, hess_sampled, weights = goss.sample_data(X, gradients, hessians)
    
    # Should have sampled fewer than original
    assert len(X_sampled) < len(X)
    assert len(grad_sampled) == len(X_sampled)
    assert len(hess_sampled) == len(X_sampled)
    assert len(weights) == len(X_sampled)


def test_goss_validation():
    """Test GOSS parameter validation."""
    # top_rate must be between 0 and 1
    with pytest.raises(ValueError):
        GOSS(top_rate=0, other_rate=0.1)
    
    with pytest.raises(ValueError):
        GOSS(top_rate=1.0, other_rate=0.1)
    
    # other_rate must be between 0 and 1
    with pytest.raises(ValueError):
        GOSS(top_rate=0.2, other_rate=0)
    
    with pytest.raises(ValueError):
        GOSS(top_rate=0.2, other_rate=1.0)


def test_goss_reproducibility():
    """Test that GOSS produces reproducible results with same random_state."""
    gradients = np.array([0.9, 0.8, 0.1, 0.05, 0.02, 0.5, 0.3, 0.2, 0.15, 0.12])
    
    goss1 = GOSS(top_rate=0.2, other_rate=0.3, random_state=42)
    goss2 = GOSS(top_rate=0.2, other_rate=0.3, random_state=42)
    
    idx1, w1 = goss1.sample(gradients)
    idx2, w2 = goss2.sample(gradients)
    
    assert np.array_equal(idx1, idx2)
    assert np.array_equal(w1, w2)
