import numpy as np

# ITU-R BT.601 irreversible RGB⇔YCbCr conversion constants [cite: 1]

_FWD = np.array([[ 0.299    ,  0.587    ,  0.114   ],
                 [-0.168736 , -0.331264 ,  0.5     ],
                 [ 0.5      , -0.418688 , -0.081312]], dtype=np.float64) # [cite: 1]

_INV = np.linalg.inv(_FWD) # [cite: 1]

def rgb_to_ycbcr(rgb: np.ndarray) -> np.ndarray:
    """Converts RGB image (H, W, 3) uint8 to YCbCr float64."""
    if rgb.shape[-1] != 3:
        raise ValueError("Input image must be RGB (3 channels)")
    if rgb.dtype != np.uint8:
         print(f"Warning: Input image dtype is {rgb.dtype}, expected uint8. Converting.")
         rgb = rgb.astype(np.uint8)

    out = rgb.astype(np.float64) @ _FWD.T
    # Level shift Y [0, 255] -> [0, 255], Cb/Cr [+/- 127.5] -> [+/- 127.5] + 128 -> [0.5, 255.5]
    out[...,1:] += 128.0 # [cite: 1]
    return out

def ycbcr_to_rgb(ycbcr: np.ndarray) -> np.ndarray:
    """Converts YCbCr image (H, W, 3) float64 to RGB uint8."""
    if ycbcr.shape[-1] != 3:
        raise ValueError("Input image must be YCbCr (3 channels)")

    arr = ycbcr.copy()
    # Remove level shift from Cb/Cr before matrix multiplication
    arr[...,1:] -= 128.0 # [cite: 1]

    rgb = arr @ _INV.T

    # Clip to valid [0, 255] range and convert to uint8
    return np.clip(np.rint(rgb), 0, 255).astype(np.uint8) # [cite: 1]