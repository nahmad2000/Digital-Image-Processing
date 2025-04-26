import math
import os
import struct
from pathlib import Path
from typing import Tuple, Dict, Optional

import numpy as np
from PIL import Image # Requires Pillow library
# Requires scikit-image library for SSIM
try:
    from skimage.metrics import structural_similarity as _ssim #
    from skimage.metrics import peak_signal_noise_ratio as _psnr # Use skimage's PSNR too
    SKIMAGE_AVAILABLE = True
except ImportError:
    print("Warning: scikit-image not found. SSIM and PSNR metrics will not be available.")
    _ssim = None
    _psnr = None
    SKIMAGE_AVAILABLE = False


# ======================================================================
# I/O HELPERS
# ======================================================================

def imread(path: str) -> Optional[np.ndarray]:
    """Load an image, ensuring it's RGB, return as np.uint8 array (H,W,3)."""
    try:
        img = Image.open(path)
        # Convert to RGB if it's not (e.g., Grayscale, RGBA, Palette)
        if img.mode != 'RGB':
             print(f"Converting image from {img.mode} to RGB.")
             img = img.convert("RGB")
        return np.asarray(img, dtype=np.uint8) #
    except FileNotFoundError:
        print(f"Error: Image file not found at {path}")
        return None
    except Exception as e:
        print(f"Error reading image {path}: {e}")
        return None


def imsave(path: str, arr: np.ndarray) -> bool:
    """Save an RGB np.uint8 array (H,W,3) as an image file (e.g., PNG)."""
    try:
        if arr.dtype != np.uint8:
             print(f"Warning: Array dtype is {arr.dtype}, converting to uint8 for saving.")
             arr = arr.astype(np.uint8)
        if arr.ndim != 3 or arr.shape[2] != 3:
             raise ValueError("Array must be 3D with 3 channels (RGB) to save.")

        img = Image.fromarray(arr, mode="RGB") #
        # Ensure output directory exists
        Path(path).parent.mkdir(parents=True, exist_ok=True) #
        # Save as PNG by default, which is lossless
        img.save(path, optimize=True) #
        return True
    except Exception as e:
        print(f"Error saving image to {path}: {e}")
        return False


def file_bytes(path: str) -> int:
    """Get the size of a file in bytes."""
    try:
        return os.path.getsize(path) #
    except FileNotFoundError:
        print(f"Error: File not found at {path} for size check.")
        return 0
    except Exception as e:
        print(f"Error getting size of file {path}: {e}")
        return 0


# ======================================================================
# IMAGE PADDING
# ======================================================================

def pad_image(img: np.ndarray, blk_size: int = 8) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Pad image dimensions to be multiples of blk_size using replication."""
    h, w = img.shape[:2]

    pad_h = (blk_size - h % blk_size) % blk_size #
    pad_w = (blk_size - w % blk_size) % blk_size #

    if pad_h == 0 and pad_w == 0:
        return img, (0, 0) # No padding needed

    # Pad using edge replication (more common than zero padding for DCT)
    # np.pad mode='edge' replicates the boundary values
    padded_img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge") #

    # Zero padding alternative:
    # padded_img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)),
    #                      mode="constant", constant_values=0)

    return padded_img, (pad_h, pad_w)


def unpad_image(img: np.ndarray, pads: Tuple[int, int]) -> np.ndarray:
    """Remove padding added by pad_image."""
    pad_h, pad_w = pads
    h, w = img.shape[:2]

    # Calculate original dimensions
    # Ensure padded dimensions are not smaller than pads (can happen with 0 pads)
    orig_h = max(0, h - pad_h)
    orig_w = max(0, w - pad_w)


    # Slice the array to remove padding
    # Handle case where pad_h or pad_w might be 0
    # Slicing [:orig_h] works even if orig_h == h (pad_h == 0)
    return img[:orig_h, :orig_w, :] #


# ======================================================================
# METRICS
# ======================================================================

def psnr(ref: np.ndarray, test: np.ndarray, max_val: float = 255.0) -> float:
    """Calculate Peak Signal-to-Noise Ratio (PSNR) in dB."""
    if not SKIMAGE_AVAILABLE:
         print("Skipping PSNR calculation: scikit-image not available.")
         return 0.0

    if ref.shape != test.shape:
        # Automatically handle potential off-by-one errors from unpadding?
        # Or just raise error. Let's raise for now.
        raise ValueError(f"Reference shape {ref.shape} and test shape {test.shape} must match for PSNR.")
    if ref.dtype != test.dtype:
         print(f"Warning: dtypes differ ({ref.dtype} vs {test.dtype}). Casting to float64.")
         ref = ref.astype(np.float64)
         test = test.astype(np.float64)

    try:
         # Use scikit-image's PSNR which handles potential MSE=0
         psnr_val = _psnr(ref, test, data_range=max_val) #
         return psnr_val
    except Exception as e:
         print(f"Error calculating PSNR: {e}")
         return 0.0

    # Manual calculation (less robust):
    # mse = np.mean((ref.astype(np.float64) - test.astype(np.float64)) ** 2)
    # if mse <= 1e-10: # Handle near-zero MSE (identical images)
    #     return float('inf')
    # return 20 * math.log10(max_val) - 10 * math.log10(mse)


def ssim(ref: np.ndarray, test: np.ndarray) -> float:
    """Calculate Structural Similarity Index (SSIM)."""
    if not SKIMAGE_AVAILABLE:
        print("Skipping SSIM calculation: scikit-image not available.")
        return 0.0

    if ref.shape != test.shape:
        raise ValueError(f"Reference shape {ref.shape} and test shape {test.shape} must match for SSIM.")
    if ref.ndim != 3 or ref.shape[2] != 3:
         raise ValueError("SSIM calculation expects RGB images (3 channels).")

    try:
        # Use scikit-image's SSIM, specify multichannel and data range
        # Default parameters for gaussian_weights, sigma, use_sample_covariance often work well.
        ssim_val = _ssim(ref, test, channel_axis=-1, data_range=255.0) # Use channel_axis if available
        return ssim_val
    except TypeError:
         # Fallback for older scikit-image versions potentially missing channel_axis
         try:
              print("Warning: scikit-image version might be old. Using SSIM without channel_axis.")
              ssim_val = _ssim(ref, test, multichannel=True, data_range=255.0)
              return ssim_val
         except Exception as e:
              print(f"Error calculating SSIM (fallback): {e}")
              return 0.0
    except Exception as e:
        print(f"Error calculating SSIM: {e}")
        return 0.0


# ======================================================================
# BIT I/O CLASSES
# ======================================================================

class BitWriter:
    """Writes bits sequentially to an internal buffer."""
    __slots__ = ("_buffer", "_current_byte", "_bit_count")

    def __init__(self) -> None:
        self._buffer = bytearray()
        self._current_byte = 0 # Holds bits being assembled for the next byte
        self._bit_count = 0    # Number of bits currently in _current_byte (0-7)

    def write(self, bits: int, num_bits: int) -> None:
        """Write num_bits (0 to 32) from the LSB of 'bits'."""
        if num_bits < 0 or num_bits > 32:
             raise ValueError("Number of bits must be between 0 and 32")
        if num_bits == 0:
            return

        # Process bits from MSB to LSB of the input 'bits' value
        for i in range(num_bits - 1, -1, -1):
            bit = (bits >> i) & 1
            # Add the bit to the current byte
            self._current_byte = (self._current_byte << 1) | bit
            self._bit_count += 1

            # If the current byte is full, append it to the buffer
            if self._bit_count == 8:
                self._buffer.append(self._current_byte)
                self._current_byte = 0
                self._bit_count = 0

    def flush(self) -> bytes:
        """Write any remaining bits (padded with 0s) and return the byte buffer."""
        if self._bit_count > 0:
            # Pad the last byte with zeros on the right (LSB side)
            padded_byte = self._current_byte << (8 - self._bit_count)
            self._buffer.append(padded_byte & 0xFF)
            self._current_byte = 0
            self._bit_count = 0
        return bytes(self._buffer)


class BitReader:
    """Reads bits sequentially from a byte buffer."""
    __slots__ = ("_data", "_byte_pos", "_current_byte", "_bit_pos")

    def __init__(self, data: bytes) -> None:
        if not isinstance(data, bytes):
             raise TypeError("Input data must be bytes.")
        self._data = data
        self._byte_pos = 0     # Index of the current byte being read from _data
        self._current_byte = 0 # Holds the value of the current byte
        self._bit_pos = 8      # Position of the next bit to read (8 = need new byte, 0-7 = bits left)

    def read(self, num_bits: int) -> int:
        """Read and return the next num_bits (1 to 32) as an integer."""
        if num_bits <= 0 or num_bits > 32:
             raise ValueError("Number of bits to read must be between 1 and 32")

        value = 0
        for _ in range(num_bits):
            # If all bits from the current byte are read, load the next byte
            if self._bit_pos == 8:
                if self._byte_pos >= len(self._data):
                    raise EOFError("Attempted to read past the end of the bitstream.")
                self._current_byte = self._data[self._byte_pos]
                self._byte_pos += 1
                self._bit_pos = 0 # Start reading from MSB (bit 7)

            # Extract the next bit (from MSB side)
            bit = (self._current_byte >> (7 - self._bit_pos)) & 1
            self._bit_pos += 1

            # Append the bit to the value being built
            value = (value << 1) | bit

        return value

    def align_to_byte(self) -> None:
         """Advance the reader position to the start of the next byte."""
         self._bit_pos = 8 # Force reading a new byte on the next read() call


# ======================================================================
# SIMPLE HEADER (Example - Not JPEG Standard)
# Format: MAGIC | W (H) | H (H) | Q (B) | DC Table Len (H) | DC Table | AC Table Len (H) | AC Table
# ======================================================================

_MAGIC = b"HUFF" # Changed magic slightly to distinguish from original 'HUF0'

def _dump_huff_table_lengths(bw: BitWriter, table: Dict[int, Tuple[int, int]]) -> None:
    """Write Huffman table symbol lengths to the bitstream."""
    # Filter out zero-length codes if any (shouldn't be generated by canonical builder)
    valid_entries = {sym: ln for sym, (_, ln) in table.items() if ln > 0}
    num_entries = len(valid_entries)

    # Write number of entries (e.g., 16 bits)
    bw.write(num_entries, 16) # Max 65535 symbols

    # Write each symbol (e.g., 8 bits for DC/AC) and its length (e.g., 8 bits)
    # Sort for deterministic output (optional but good practice)
    for sym, ln in sorted(valid_entries.items()):
        # Ensure symbol and length fit in allocated bits
        if not (0 <= sym <= 255): raise ValueError(f"Symbol {sym} out of 8-bit range.")
        # --- MODIFIED CHECK ---
        # Allow lengths up to 255 since we write length with 8 bits
        # Relaxed from JPEG's strict 1-16 limit for this example code.
        if not (1 <= ln <= 255): raise ValueError(f"Huffman length {ln} out of supported 1-255 range.")
        bw.write(sym, 8) # Write symbol value (0-255)
        bw.write(ln, 8)  # Write code length (1-255)

def _load_huff_table_lengths(br: BitReader) -> Dict[int, int]:
    """Read Huffman table symbol lengths from the bitstream."""
    lengths = {}
    try:
        num_entries = br.read(16)
    except EOFError:
         raise EOFError("Bitstream ended while reading number of Huffman table entries.")

    for _ in range(num_entries):
         try:
              sym = br.read(8)
              ln = br.read(8)
              if ln == 0: # Skip entries with length 0 if somehow written
                   print("Warning: Read Huffman table entry with length 0, skipping.")
                   continue
              if ln > 255: # Add check corresponding to writing logic
                   raise ValueError(f"Read Huffman length {ln} > 255, which is unsupported.")
              lengths[sym] = ln
         except EOFError:
              raise EOFError("Bitstream ended while reading Huffman table entries.")
    return lengths

def write_header(bw: BitWriter, w: int, h: int, q: int,
                 huff_dc: Dict[int, Tuple[int, int]],
                 huff_ac: Dict[int, Tuple[int, int]]) -> None:
    """Write the custom header information using BitWriter."""
    # Write Magic bytes directly to buffer before bit writing starts
    # This assumes the BitWriter buffer is empty initially or handles prepending
    if not bw._buffer and bw._bit_count == 0:
         bw._buffer.extend(_MAGIC)
    else:
         # If buffer not empty, this approach might corrupt bit alignment
         # A safer way: handle magic outside BitWriter or have BitWriter support byte writing
         # For this script, assume it's called first.
         print("Warning: write_header called on non-empty BitWriter buffer.")
         bw._buffer = bytearray(_MAGIC) + bw._buffer # Prepend magic (potential issue if bits were pending)
         if bw._bit_count > 0: print("Warning: Pending bits might be lost/corrupted by header write.")


    bw.write(w, 16) # Image Width (uint16)
    bw.write(h, 16) # Image Height (uint16)
    bw.write(q, 8)  # Quality factor (uint8)
    _dump_huff_table_lengths(bw, huff_dc) # DC Huffman table lengths
    _dump_huff_table_lengths(bw, huff_ac) # AC Huffman table lengths

def read_header(br: BitReader) -> Tuple[int, int, int, Dict[int, int], Dict[int, int]]:
    """Read the custom header information using BitReader."""
    # Read and verify Magic bytes
    magic_bytes = bytes([br.read(8) for _ in range(len(_MAGIC))])
    if magic_bytes != _MAGIC:
        raise ValueError(f"Invalid magic number. Expected {_MAGIC}, got {magic_bytes}")

    w = br.read(16) # Width
    h = br.read(16) # Height
    q = br.read(8)  # Quality
    dc_lengths = _load_huff_table_lengths(br) # DC lengths
    ac_lengths = _load_huff_table_lengths(br) # AC lengths

    # Optional: Align to byte boundary after header if needed
    # br.align_to_byte()

    return w, h, q, dc_lengths, ac_lengths


# ======================================================================
# CSV METRICS LOGGING
# ======================================================================

def append_metrics_csv(csv_path: str, filename: str, quality: int,
                       psnr_val: float, ssim_val: float, bpp_val: float,
                       ratio_val: float, orig_size_b: int, comp_size_b: int) -> None:
    """Append compression metrics for one run to a CSV file."""
    file_exists = Path(csv_path).exists()
    # Define header columns
    header = ("Filename", "Quality", "PSNR(dB)", "SSIM", "BPP",
              "CompressionRatio", "OriginalBytes", "CompressedBytes")

    try:
        with open(csv_path, "a", newline="") as fh:
            # Write header only if file doesn't exist
            if not file_exists or os.path.getsize(csv_path) == 0: # Check size too
                fh.write(",".join(header) + "\n")

            # Format data row
            # Use 'inf' for infinite PSNR
            psnr_str = f"{psnr_val:.4f}" if np.isfinite(psnr_val) else "inf"
            row = (
                f'"{Path(filename).name}"', # Use Path to get basename, quote if needed
                f"{quality}",
                psnr_str,
                f"{ssim_val:.6f}",
                f"{bpp_val:.6f}",
                f"{ratio_val:.4f}",
                f"{orig_size_b}",
                f"{comp_size_b}"
            )
            fh.write(",".join(row) + "\n")
    except Exception as e:
        print(f"Error writing metrics to CSV {csv_path}: {e}")