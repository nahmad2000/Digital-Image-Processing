"""Tests for denoising filter functions."""

import numpy as np
from skimage import img_as_float
import pytest

# Placeholder imports (assuming refactoring or direct access)
from notebooks.noise_analysis_demo import (
    apply_gaussian_blur,
    apply_median_filter,
    apply_nl_means,
)
from .helpers import create_dummy_image


@pytest.fixture
def noisy_image_float() -> np.ndarray:
    """Provides a sample noisy float image."""
    img = img_as_float(create_dummy_image(32, 32, 3))
    # Add some noise for realism
    noisy = img + np.random.normal(0, 0.1, img.shape)
    return np.clip(noisy, 0, 1)


def test_apply_gaussian_blur(noisy_image_float: np.ndarray) -> None:
    """Test Gaussian blur filter."""
    denoised = apply_gaussian_blur(noisy_image_float, ksize=(3, 3))
    assert denoised.shape == noisy_image_float.shape
    assert denoised.dtype == np.float64
    assert np.min(denoised) >= 0.0
    assert np.max(denoised) <= 1.0


def test_apply_median_filter(noisy_image_float: np.ndarray) -> None:
    """Test Median filter."""
    denoised = apply_median_filter(noisy_image_float, ksize=3)
    assert denoised.shape == noisy_image_float.shape
    assert denoised.dtype == np.float64
    assert np.min(denoised) >= 0.0
    assert np.max(denoised) <= 1.0


@pytest.mark.slow  # Mark NLM test as potentially slow
def test_apply_nl_means(noisy_image_float: np.ndarray) -> None:
    """Test Non-Local Means filter."""
    # Reduce image size for faster testing if needed
    small_image = noisy_image_float[:16, :16, :]
    denoised = apply_nl_means(small_image, h_factor=0.8)
    assert denoised.shape == small_image.shape
    assert denoised.dtype == np.float64
    assert np.min(denoised) >= 0.0
    assert np.max(denoised) <= 1.0