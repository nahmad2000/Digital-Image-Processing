"""Helper functions for tests."""

import numpy as np


def create_dummy_image(
    height: int = 32, width: int = 32, channels: int = 3
) -> np.ndarray:
    """Creates a simple dummy image (e.g., gradient)."""
    if channels == 1:
        img = np.linspace(0, 255, width * height, dtype=np.uint8).reshape(
            (height, width)
        )
    elif channels == 3:
        img = np.zeros((height, width, channels), dtype=np.uint8)
        base = np.linspace(0, 255, width * height, dtype=np.uint8).reshape(
            (height, width)
        )
        img[:, :, 0] = base  # Red channel gradient
        img[:, :, 1] = np.flipud(base)  # Green channel gradient
        img[:, :, 2] = np.fliplr(base)  # Blue channel gradient
    else:
        raise ValueError("Unsupported number of channels")
    return img