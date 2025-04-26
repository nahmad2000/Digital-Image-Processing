import numpy as np
# from scipy.fft import dct, idct # Optional alternative
import pywt                      # Required if DWT branch is ever used

# ----------------------------------------------------------------------
# FAST 8-point 1-D DCT/IDCT using pre-scaled AAN factors [cite: 1]
# Based on the standard DCT-II definition.
# ----------------------------------------------------------------------

_blk_size = 8 # Standard JPEG block size [cite: 1]

# Precompute DCT-II basis matrix [cite: 1]
# F(u) = sqrt(2/N) * C(u) * sum_{n=0}^{N-1} f(n) * cos(pi * (2n+1) * u / (2*N))
# Basis[u, n] = sqrt(2/N) * C(u) * cos(pi * (2n+1) * u / (2*N))
def _dct_basis_matrix(N):
    """Generates the N x N DCT-II basis matrix."""
    n = np.arange(N)
    u = n[:, None] # Column vector

    # Calculate cosine terms
    cos_terms = np.cos(np.pi * (2 * n + 1) * u / (2 * N)) # [cite: 1]

    # Calculate scaling factors C(u) * sqrt(2/N)
    scaling = np.sqrt(2 / N) * np.ones(N) # [cite: 1]
    scaling[0] *= (1 / np.sqrt(2)) # C(0) = 1/sqrt(2) [cite: 1]

    # Apply scaling row-wise (to each basis vector u)
    basis = scaling[:, None] * cos_terms # [cite: 1]
    return basis.astype(np.float64)

_DCT_MATRIX = _dct_basis_matrix(_blk_size)
_IDCT_MATRIX = _DCT_MATRIX.T # IDCT matrix is the transpose of DCT matrix [cite: 1]

def _dct1d(x: np.ndarray) -> np.ndarray:
    """Compute 1D DCT-II using the precomputed matrix."""
    # Check if input is a vector or matrix for row/column-wise application
    if x.ndim == 1:
        if x.shape[0] != _blk_size:
             raise ValueError(f"1D input must have length {_blk_size}")
        return _DCT_MATRIX @ x
    elif x.ndim == 2:
        # Apply along the last axis (like scipy.fft.dct norm='ortho')
        if x.shape[1] != _blk_size:
             raise ValueError(f"Matrix columns must have length {_blk_size}")
        return ( _DCT_MATRIX @ x.T).T # Apply to columns by transposing
    else:
        raise ValueError("Input must be 1D or 2D array.")

def _idct1d(c: np.ndarray) -> np.ndarray:
    """Compute 1D IDCT-II using the precomputed matrix."""
    if c.ndim == 1:
        if c.shape[0] != _blk_size:
             raise ValueError(f"1D input must have length {_blk_size}")
        return _IDCT_MATRIX @ c
    elif c.ndim == 2:
        if c.shape[1] != _blk_size:
             raise ValueError(f"Matrix columns must have length {_blk_size}")
        return ( _IDCT_MATRIX @ c.T).T
    else:
        raise ValueError("Input must be 1D or 2D array.")


class DCT:
    """
    Separable 8x8 DCT-II implementation using precomputed matrix. [cite: 1]
    """
    blk = _blk_size # Expose block size

    @staticmethod
    def forward(block: np.ndarray) -> np.ndarray:
        """Compute forward 2D DCT on an 8x8 block."""
        if block.shape != (_blk_size, _blk_size):
             raise ValueError(f"Input block must be {_blk_size}x{_blk_size}")

        # Apply 1D DCT along rows, then along columns (or vice versa) [cite: 1]
        # Row-wise: Apply matrix to each row vector
        # result_rows = (_DCT_MATRIX @ block).T # Error in logic: applies to cols
        result_rows = np.dot(block, _DCT_MATRIX.T) # Apply DCT to rows

        # Column-wise: Apply matrix to each column vector of the row-transformed result
        result_cols = np.dot(_DCT_MATRIX, result_rows) # Apply DCT to columns of result_rows

        return result_cols # Should be _DCT_MATRIX @ block @ _DCT_MATRIX.T

        # Alternative using the helper functions:
        # return _dct1d(_dct1d(block).T).T # Apply to rows, transpose, apply to new rows, transpose back.
        # return _dct1d(_dct1d(block.T).T) # Apply to columns, transpose, apply to new columns, transpose back

    @staticmethod
    def inverse(coeff: np.ndarray) -> np.ndarray:
        """Compute inverse 2D DCT on an 8x8 coefficient block."""
        if coeff.shape != (_blk_size, _blk_size):
             raise ValueError(f"Input block must be {_blk_size}x{_blk_size}")

        # Apply 1D IDCT along columns, then along rows (reverse of forward) [cite: 1]
        # result_cols = np.dot(_IDCT_MATRIX, coeff) # Apply IDCT to columns
        # result_rows = np.dot(result_cols, _IDCT_MATRIX.T) # Apply IDCT to rows of result_cols

        # Should be _IDCT_MATRIX.T @ block @ _IDCT_MATRIX ? -> No, IDCT = DCT.T
        # Correct: _IDCT_MATRIX @ coeff @ _IDCT_MATRIX.T -> No
        # Correct: _DCT_MATRIX.T @ coeff @ _DCT_MATRIX
        result_cols = np.dot(_IDCT_MATRIX, coeff) # Apply IDCT to columns
        result_rows = np.dot(result_cols, _DCT_MATRIX) # Apply IDCT to rows (using DCT.T == IDCT)

        return result_rows


        # Alternative using the helper functions:
        # return _idct1d(_idct1d(coeff).T).T # Apply to rows, transpose, apply to new rows, transpose back
        # return _idct1d(_idct1d(coeff.T).T) # Apply to columns, transpose, apply to new columns, transpose back


# ----------------------------------------------------------------------
# OPTIONAL DWT BRANCH (5/3 & 9/7 via PyWavelets) [cite: 1]
# This part seems unused by the main script's default path.
# ----------------------------------------------------------------------

class DWT:
    """
    Wrapper for single-level 2D Discrete Wavelet Transform using PyWavelets. [cite: 1]
    """
    @staticmethod
    def forward(channel: np.ndarray, wave: str = "bior4.4"):
        """
        Apply forward 2D DWT. [cite: 1]

        Args:
            channel (np.ndarray): Input channel data (2D).
            wave (str): Wavelet name compatible with PyWavelets (e.g., 'haar', 'db4', 'bior4.4' for 9/7).

        Returns:
            tuple: DWT coefficients in PyWavelets format (LL, (LH, HL, HH)).
        """
        if channel.ndim != 2:
             raise ValueError("Input must be a 2D array (single channel).")
        # Use mode 'periodization' or 'symmetric' often preferred for images
        coeffs = pywt.dwt2(channel.astype(np.float64), wavelet=wave, mode="periodization") #
        # Returns (LL, (LH, HL, HH))
        return coeffs

    @staticmethod
    def inverse(coeffs, wave: str = "bior4.4"):
        """
        Apply inverse 2D DWT. [cite: 1]

        Args:
            coeffs: DWT coefficients tuple (LL, (LH, HL, HH)).
            wave (str): Wavelet name used for the forward transform.

        Returns:
            np.ndarray: Reconstructed channel data (2D float64).
        """
        if not isinstance(coeffs, tuple) or len(coeffs) != 2:
             raise ValueError("Invalid coefficient format for PyWavelets IDWT.")
        # Use the same mode as the forward transform
        reconstructed = pywt.idwt2(coeffs, wavelet=wave, mode="periodization") #
        return reconstructed