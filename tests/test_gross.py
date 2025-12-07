import numpy as np

from lightgbm.goss import GOSSSampler


def test_goss_selects_top_and_samples_rest():
	grad = np.array([0.9, 0.8, 0.1, 0.05, 0.02, 0.5])
	hess = np.ones_like(grad)
	samplr = GOSSSampler(top_rate=0.3, other_rate=0.5, random_state=0)
	idx, g_scaled, h_scaled = samplr.sample(grad, hess)
	assert len(idx) >= 2  # top element + some sampled
	assert g_scaled.shape == h_scaled.shape == idx.shape
	# Ensure the largest gradient is included
	assert 0 in idx


def test_goss_reweights_small_gradients():
	grad = np.array([1.0, 0.1, 0.05, 0.02, 0.01])
	hess = np.ones_like(grad)
	samplr = GOSSSampler(top_rate=0.2, other_rate=0.4, random_state=1)
	idx, g_scaled, h_scaled = samplr.sample(grad, hess)
	# All non-top indices should share same scaling factor
	if len(idx) > 1:
		top_g = g_scaled[0]
		scaled_section = g_scaled[1:]
		assert np.allclose(scaled_section / grad[idx[1:]], scaled_section[0] / grad[idx[1]])
