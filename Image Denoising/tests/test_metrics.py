"""Tests for evaluation metric functions."""

import numpy as np
from skimage import img_as_float
import pytest

# Placeholder import
from notebooks.noise_analysis_demo import calculate_metrics
from .helpers import create_dummy_image


@pytest.fixture
def sample_images() -> tuple[np.ndarray, np.ndarray]:
    """Provides two slightly different sample float images."""
    img1 = img_as_float(create_dummy_image(32, 32, 1))  # Grayscale
    img2 = img1.copy()
    img2[10:15, 10:15] += 0.1  # Introduce a small difference
    img2 = np.clip(img2, 0, 1)
    return img1, img2


def test_calculate_metrics_identical(sample_images: tuple[np.ndarray, np.ndarray]) -> None:
    """Test metrics for identical images."""
    img1, _ = sample_images
    metrics = calculate_metrics(img1, img1)
    # PSNR should be infinity for identical images
    assert metrics["PSNR"] == np.inf
    # SSIM should be 1.0 for identical images
    assert np.isclose(metrics["SSIM"], 1.0)


def test_calculate_metrics_different(sample_images: tuple[np.ndarray, np.ndarray]) -> None:
    """Test metrics for different images."""
    img1, img2 = sample_images
    metrics = calculate_metrics(img1, img2)
    # PSNR should be a finite positive number
    assert 0 < metrics["PSNR"] < np.inf
    # SSIM should be between -1 and 1 (typically > 0 for similar images)
    assert -1.0 <= metrics["SSIM"] < 1.0


def test_calculate_metrics_completely_different() -> None:
    """Test metrics for completely different images."""
    img1 = np.zeros((32, 32), dtype=np.float64)
    img2 = np.ones((32, 32), dtype=np.float64)
    metrics = calculate_metrics(img1, img2)
    assert metrics["PSNR"] == 0.0  # Or very close to 0 depending on implementation details
    # SSIM should be low, potentially negative or zero
    assert metrics["SSIM"] < 0.1 # Expect very low similarity