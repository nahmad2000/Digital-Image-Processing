# -*- coding: utf-8 -*-
"""
Enhanced JPEG-like Image Compression Implementation
This script implements the core steps of JPEG compression for color images with
additional features for better visualization, performance, and usability:
Features:
- Quality factor adjustment for controlling compression strength
- RGB to YCbCr conversion with optional chroma subsampling
- Block-based DCT compression with optimized block handling
- Advanced visualizations including coefficient heatmaps and frequency domain views
- Progress tracking for longer operations
- Batch processing support for multiple images
- Optional GUI interface for easier use
- Zigzag coding and basic entropy estimation
- Performance optimizations including multithreading
- Grayscale image support
"""
import numpy as np
import numpy.typing as npt
from PIL import Image, ImageTk # Added ImageTk for GUI
from scipy.fftpack import dct, idct
from skimage import metrics
import os
import time
import math # Added for ceiling in progress bar
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm # For heatmap visualization
from tqdm import tqdm
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple, List, Dict, Optional, Union, Callable
import tkinter as tk
from tkinter import filedialog, ttk, messagebox # Added messagebox
import io
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# --- Constants ---
# Standard JPEG Quantization Matrices
Y_QUANT_MATRIX = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
], dtype=np.float32)

CBCR_QUANT_MATRIX = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99]
], dtype=np.float32)

# RGB to YCbCr Conversion Matrix (BT.601 standard)
RGB_TO_YCBCR_MATRIX = np.array([
    [0.299, 0.587, 0.114],
    [-0.168736, -0.331264, 0.5],
    [0.5, -0.418688, -0.081312]
], dtype=np.float32)

# YCbCr to RGB Conversion Matrix (Inverse)
YCBCR_TO_RGB_MATRIX = np.array([
    [1., 0., 1.402],
    [1., -0.344136, -0.714136],
    [1., 1.772, 0.]
], dtype=np.float32)


BLOCK_SIZE = 8

# Zigzag pattern for coefficient ordering (8x8)
# Generated using a simple helper function (not included for brevity)
ZIGZAG_INDICES = np.array([
     0,  1,  8, 16,  9,  2,  3, 10,
    17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63
], dtype=np.int32)

# Inverse Zigzag pattern (Mapping zigzag index back to 2D index)
INVERSE_ZIGZAG_INDICES = np.argsort(ZIGZAG_INDICES)


# --- Helper Functions ---
def load_image(image_path: str) -> Tuple[Optional[npt.NDArray[np.uint8]], bool]:
    """Loads an image, checks if grayscale, returns array and grayscale flag."""
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return None, False
    try:
        image = Image.open(image_path)
        is_grayscale = False
        # Convert grayscale or other modes to RGB for consistent processing
        if image.mode == 'L':
            print("Input image is grayscale.")
            is_grayscale = True
            image = image.convert('RGB') # Convert to RGB for unified pipeline
        elif image.mode != 'RGB':
            print(f"Converting image from {image.mode} to RGB.")
            image = image.convert('RGB')

        return np.array(image, dtype=np.uint8), is_grayscale
    except Exception as e:
        print(f"Error loading image: {e}")
        return None, False

def save_image(image_array: npt.NDArray[np.uint8], output_path: str, quality: int = 95) -> bool:
    """Saves the image array to the specified path using Pillow."""
    try:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")

        # Handle potential single-channel grayscale array before saving
        if image_array.ndim == 3 and image_array.shape[2] == 1:
            image_array = image_array.squeeze(axis=2) # Remove channel dim
            img = Image.fromarray(image_array, 'L') # Save as grayscale
        elif image_array.ndim == 2: # Already grayscale
            img = Image.fromarray(image_array, 'L')
        else: # Assumed RGB
             img = Image.fromarray(image_array, 'RGB')

        # Quality setting affects formats like JPEG, WEBP. Ignored by PNG.
        img.save(output_path, quality=quality)
        print(f"Compressed image saved to: {output_path}")
        return True
    except Exception as e:
        print(f"Error saving image: {e}")
        return False

def adjust_quantization_matrices(quality: int) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Adjusts quantization matrices based on quality factor (1-100)."""
    if not 1 <= quality <= 100:
        raise ValueError("Quality factor must be between 1 and 100")

    if quality < 50:
        scale_factor = 5000 / quality
    else:
        scale_factor = 200 - 2 * quality

    # Calculate scaled matrices
    y_matrix = (Y_QUANT_MATRIX * scale_factor + 50) / 100
    cbcr_matrix = (CBCR_QUANT_MATRIX * scale_factor + 50) / 100

    # Clip values and ensure minimum of 1
    y_matrix = np.clip(np.floor(y_matrix), 1, 255)
    cbcr_matrix = np.clip(np.floor(cbcr_matrix), 1, 255)

    return y_matrix.astype(np.float32), cbcr_matrix.astype(np.float32)


def rgb_to_ycbcr(rgb_array: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Converts an RGB image array (0-255) to YCbCr format."""
    ycbcr = np.dot(rgb_array, RGB_TO_YCBCR_MATRIX.T)
    ycbcr[:, :, 1:] += 128.0 # Add offset to Cb and Cr
    return ycbcr

def ycbcr_to_rgb(ycbcr_array: npt.NDArray[np.float32]) -> npt.NDArray[np.uint8]:
    """Converts a YCbCr image array back to RGB format."""
    ycbcr_shifted = ycbcr_array.copy()
    ycbcr_shifted[:, :, 1:] -= 128.0 # Remove offset from Cb and Cr
    rgb = np.dot(ycbcr_shifted, YCBCR_TO_RGB_MATRIX.T)
    # Clip values to valid range [0, 255] and convert to uint8
    return np.clip(rgb, 0, 255).astype(np.uint8)


def apply_chroma_subsampling(cbcr_channels: npt.NDArray[np.float32], mode: str = "4:2:0") -> npt.NDArray[np.float32]:
    """
    Applies chroma subsampling ONLY to CbCr channels.
    Assumes CbCr channels are passed as shape (height, width, 2).
    Returns subsampled CbCr channels, potentially with reduced dimensions.
    """
    height, width, _ = cbcr_channels.shape
    cb = cbcr_channels[:, :, 0]
    cr = cbcr_channels[:, :, 1]

    if mode == "4:4:4":
        return cbcr_channels # No subsampling
    elif mode == "4:2:2":
        # Subsample horizontally by averaging pairs of columns
        cb_sub = (cb[:, 0::2] + cb[:, 1::2]) / 2
        cr_sub = (cr[:, 0::2] + cr[:, 1::2]) / 2
    elif mode == "4:2:0":
        # Subsample horizontally and vertically
        # Average 2x2 blocks
        cb_sub = (cb[0::2, 0::2] + cb[1::2, 0::2] + cb[0::2, 1::2] + cb[1::2, 1::2]) / 4
        cr_sub = (cr[0::2, 0::2] + cr[1::2, 0::2] + cr[0::2, 1::2] + cr[1::2, 1::2]) / 4
    else:
        raise ValueError(f"Unsupported subsampling mode: {mode}")

    # Stack the subsampled channels back together
    return np.stack((cb_sub, cr_sub), axis=-1)

def upscale_chroma(cbcr_subsampled: npt.NDArray[np.float32], target_dims: Tuple[int, int]) -> npt.NDArray[np.float32]:
    """Upscales subsampled CbCr channels to target dimensions using nearest neighbor."""
    target_h, target_w = target_dims
    current_h, current_w, _ = cbcr_subsampled.shape

    # Use PIL for efficient resizing (nearest neighbor)
    cb_img = Image.fromarray(cbcr_subsampled[:, :, 0])
    cr_img = Image.fromarray(cbcr_subsampled[:, :, 1])

    cb_resized = cb_img.resize((target_w, target_h), Image.NEAREST)
    cr_resized = cr_img.resize((target_w, target_h), Image.NEAREST)

    return np.stack((np.array(cb_resized), np.array(cr_resized)), axis=-1)

def level_shift(ycbcr_array: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Shifts all channels by -128.0 for DCT."""
    return ycbcr_array - 128.0

def inverse_level_shift(dct_coeffs: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Shifts all channels back by +128.0 after IDCT."""
    return dct_coeffs + 128.0

def efficient_block_splitter(image_channel: npt.NDArray[np.float32]) -> Tuple[npt.NDArray[np.float32], int, int, Tuple[int, int]]:
    """Efficiently divides a single image channel into blocks using stride tricks."""
    height, width = image_channel.shape
    original_dims = (height, width)

    # Pad image if dimensions are not divisible by BLOCK_SIZE
    pad_h = (BLOCK_SIZE - height % BLOCK_SIZE) % BLOCK_SIZE
    pad_w = (BLOCK_SIZE - width % BLOCK_SIZE) % BLOCK_SIZE
    if pad_h > 0 or pad_w > 0:
        padded_channel = np.pad(image_channel, ((0, pad_h), (0, pad_w)), mode='symmetric')
    else:
        padded_channel = image_channel

    padded_height, padded_width = padded_channel.shape
    num_blocks_h = padded_height // BLOCK_SIZE
    num_blocks_w = padded_width // BLOCK_SIZE

    # Use stride tricks to get blocks without copying data
    shape = (num_blocks_h, num_blocks_w, BLOCK_SIZE, BLOCK_SIZE)
    strides = (padded_channel.strides[0] * BLOCK_SIZE,
               padded_channel.strides[1] * BLOCK_SIZE,
               padded_channel.strides[0],
               padded_channel.strides[1])
    blocks = np.lib.stride_tricks.as_strided(padded_channel, shape=shape, strides=strides)

    # Reshape to (num_blocks, BLOCK_SIZE, BLOCK_SIZE)
    blocks_reshaped = blocks.reshape(-1, BLOCK_SIZE, BLOCK_SIZE)

    return blocks_reshaped, num_blocks_h, num_blocks_w, original_dims


def process_channel_blocks(blocks: npt.NDArray[np.float32],
                           quant_matrix: npt.NDArray[np.float32],
                           operation: str = 'compress', # 'compress' or 'decompress'
                           num_threads: int = 4,
                           use_tqdm: bool = True,
                           desc: str = "Processing") -> npt.NDArray[np.float32]:
    """Applies DCT+Quantization or Dequantization+IDCT to blocks of a single channel."""
    total_blocks = blocks.shape[0]
    processed_blocks = np.zeros_like(blocks, dtype=np.float32)

    # --- Define worker functions ---
    def compress_block(block: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        dct_block = dct(dct(block, axis=0, norm='ortho'), axis=1, norm='ortho')
        quantized_block = np.round(dct_block / quant_matrix)
        return quantized_block

    def decompress_block(quantized_block: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        dequantized_block = quantized_block * quant_matrix
        idct_block = idct(idct(dequantized_block, axis=0, norm='ortho'), axis=1, norm='ortho')
        return idct_block

    worker_func = compress_block if operation == 'compress' else decompress_block

    # --- Execute processing ---
    # Use ThreadPoolExecutor for better resource management
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Use map to apply function to all blocks and collect results
        # Wrap with tqdm for progress bar if enabled
        if use_tqdm:
            results_iterator = tqdm(executor.map(worker_func, [blocks[i] for i in range(total_blocks)]),
                                    total=total_blocks, desc=f"{desc} ({operation})", unit="block", leave=False)
        else:
            results_iterator = executor.map(worker_func, [blocks[i] for i in range(total_blocks)])

        # Assign results back to the output array
        for i, result in enumerate(results_iterator):
            processed_blocks[i] = result

    return processed_blocks


def apply_zigzag_coding(quantized_blocks: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Applies zigzag ordering to quantized blocks (shape: num_blocks, 8, 8)."""
    # Use advanced indexing with the precomputed ZIGZAG_INDICES
    # Flatten each block first, then index using zigzag pattern
    return quantized_blocks.reshape(quantized_blocks.shape[0], -1)[:, ZIGZAG_INDICES]

def inverse_zigzag(zigzag_data: npt.NDArray[np.float32],
                   num_blocks: int) -> npt.NDArray[np.float32]:
    """Converts zigzag arrays back to block format (num_blocks, 8, 8)."""
    # Use advanced indexing with the precomputed INVERSE_ZIGZAG_INDICES
    # This places elements back into their original 2D block order after flattening
    flat_blocks = zigzag_data[:, INVERSE_ZIGZAG_INDICES]
    # Reshape back to (num_blocks, BLOCK_SIZE, BLOCK_SIZE)
    return flat_blocks.reshape(num_blocks, BLOCK_SIZE, BLOCK_SIZE)


def estimate_entropy(zigzag_data: npt.NDArray[np.float32]) -> Dict[str, float]:
    """Estimates entropy based on coefficient distribution (for one channel)."""
    metrics_dict = {}
    total_coeffs = zigzag_data.size
    if total_coeffs == 0:
        return {"entropy_bits_per_coeff": 0, "zero_ratio": 1.0}

    # Calculate Zero Ratio
    nonzero_coeffs = np.count_nonzero(zigzag_data)
    metrics_dict["zero_ratio"] = 1.0 - (nonzero_coeffs / total_coeffs)

    # Estimate Entropy (bits per coefficient)
    # Use unique values and their counts for probability calculation
    unique_values, counts = np.unique(zigzag_data, return_counts=True)
    probabilities = counts / total_coeffs
    # Shannon entropy formula: -sum(p * log2(p))
    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-12)) # Epsilon for log(0)
    metrics_dict["entropy_bits_per_coeff"] = entropy

    # Estimate bits needed (simplified)
    # Real JPEG uses RLE + Huffman/Arithmetic coding
    # This is a coarse estimate: bits = entropy * total coefficients
    metrics_dict["estimated_bits"] = entropy * total_coeffs

    return metrics_dict


def efficient_block_merger(processed_blocks: npt.NDArray[np.float32],
                          num_blocks_h: int,
                          num_blocks_w: int,
                          original_dims: Tuple[int, int]) -> npt.NDArray[np.float32]:
    """Efficiently merges blocks back into a single image channel."""
    original_height, original_width = original_dims
    padded_height = num_blocks_h * BLOCK_SIZE
    padded_width = num_blocks_w * BLOCK_SIZE

    # Reshape the blocks array into the padded image structure
    padded_channel = processed_blocks.reshape(num_blocks_h, num_blocks_w, BLOCK_SIZE, BLOCK_SIZE)
    # Transpose axes to bring block rows/cols together
    padded_channel = padded_channel.transpose(0, 2, 1, 3).reshape(padded_height, padded_width)

    # Crop back to original dimensions
    return padded_channel[:original_height, :original_width]


def calculate_metrics(original_path: str, compressed_path: str,
                      original_img_arr: npt.NDArray[np.uint8],
                      compressed_img_arr: npt.NDArray[np.uint8],
                      entropy_results_y: Dict[str, float] = None,
                      entropy_results_cb: Dict[str, float] = None,
                      entropy_results_cr: Dict[str, float] = None) -> Dict[str, Union[float, str]]:
    """Calculates and returns image quality and compression metrics."""
    metrics_dict = {}
    try:
        # File size metrics
        original_size = os.path.getsize(original_path)
        compressed_size = os.path.getsize(compressed_path)
        metrics_dict["original_size_kb"] = f"{original_size / 1024:.2f}"
        metrics_dict["compressed_size_kb"] = f"{compressed_size / 1024:.2f}"
        if compressed_size == 0:
            metrics_dict["compression_ratio_file"] = float('inf')
        else:
            metrics_dict["compression_ratio_file"] = f"{original_size / compressed_size:.2f}:1"

        # Image quality metrics (ensure shapes match)
        if original_img_arr.shape != compressed_img_arr.shape:
             # Try to handle grayscale vs RGB comparison
             if original_img_arr.ndim == 2 and compressed_img_arr.ndim == 3 and compressed_img_arr.shape[2] == 3:
                 # Convert compressed RGB to grayscale for comparison
                 print("Warning: Comparing original grayscale to reconstructed RGB. Converting RGB to grayscale for metrics.")
                 compressed_gray = np.dot(compressed_img_arr[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
                 original_for_metric = original_img_arr
                 compressed_for_metric = compressed_gray
             elif compressed_img_arr.ndim == 2 and original_img_arr.ndim == 3 and original_img_arr.shape[2] == 3:
                  print("Warning: Comparing original RGB to reconstructed grayscale. Converting RGB to grayscale for metrics.")
                  original_gray = np.dot(original_img_arr[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
                  original_for_metric = original_gray
                  compressed_for_metric = compressed_img_arr
             else:
                 print(f"Warning: Original ({original_img_arr.shape}) and compressed ({compressed_img_arr.shape}) "
                       "image shapes differ significantly. Skipping PSNR/SSIM.")
                 metrics_dict["mse"] = "N/A"
                 metrics_dict["psnr_db"] = "N/A"
                 metrics_dict["ssim"] = "N/A"
                 original_for_metric = None # Flag to skip calculation
        else:
             # Shapes match or only differ by channel count if one is grayscale
             if original_img_arr.ndim == 3 and compressed_img_arr.ndim == 2: # Orig RGB, Comp Gray
                 print("Warning: Comparing original RGB to reconstructed grayscale. Converting RGB to grayscale for metrics.")
                 original_for_metric = np.dot(original_img_arr[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
                 compressed_for_metric = compressed_img_arr
             elif original_img_arr.ndim == 2 and compressed_img_arr.ndim == 3: # Orig Gray, Comp RGB
                 print("Warning: Comparing original grayscale to reconstructed RGB. Converting RGB to grayscale for metrics.")
                 original_for_metric = original_img_arr
                 compressed_for_metric = np.dot(compressed_img_arr[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
             else: # Both RGB or both Grayscale
                 original_for_metric = original_img_arr
                 compressed_for_metric = compressed_img_arr


        if original_for_metric is not None:
            try:
                 multichannel = original_for_metric.ndim == 3
                 channel_axis = 2 if multichannel else None
                 metrics_dict["mse"] = f"{metrics.mean_squared_error(original_for_metric, compressed_for_metric):.2f}"
                 metrics_dict["psnr_db"] = f"{metrics.peak_signal_noise_ratio(original_for_metric, compressed_for_metric, data_range=255):.2f}"
                 metrics_dict["ssim"] = f"{metrics.structural_similarity(original_for_metric, compressed_for_metric, multichannel=multichannel, channel_axis=channel_axis, data_range=255):.4f}"
            except ValueError as e:
                 print(f"Error calculating PSNR/SSIM, likely shape mismatch after conversion: {e}")
                 metrics_dict["mse"] = "Error"
                 metrics_dict["psnr_db"] = "Error"
                 metrics_dict["ssim"] = "Error"


        # Add entropy results if available
        total_estimated_bits = 0
        if entropy_results_y:
             metrics_dict["Y_zero_ratio"] = f"{entropy_results_y.get('zero_ratio', 0):.3f}"
             metrics_dict["Y_entropy_bits"] = f"{entropy_results_y.get('entropy_bits_per_coeff', 0):.3f}"
             total_estimated_bits += entropy_results_y.get('estimated_bits', 0)
        if entropy_results_cb:
             metrics_dict["Cb_zero_ratio"] = f"{entropy_results_cb.get('zero_ratio', 0):.3f}"
             metrics_dict["Cb_entropy_bits"] = f"{entropy_results_cb.get('entropy_bits_per_coeff', 0):.3f}"
             total_estimated_bits += entropy_results_cb.get('estimated_bits', 0)
        if entropy_results_cr:
             metrics_dict["Cr_zero_ratio"] = f"{entropy_results_cr.get('zero_ratio', 0):.3f}"
             metrics_dict["Cr_entropy_bits"] = f"{entropy_results_cr.get('entropy_bits_per_coeff', 0):.3f}"
             total_estimated_bits += entropy_results_cr.get('estimated_bits', 0)

        # Estimated compression ratio based on entropy
        original_bits_estimate = original_img_arr.size * 8 # 8 bits per value (uint8)
        if total_estimated_bits > 0:
             metrics_dict["compression_ratio_entropy"] = f"{original_bits_estimate / total_estimated_bits:.2f}:1"
        else:
             metrics_dict["compression_ratio_entropy"] = "N/A"

    except FileNotFoundError:
        print("Error: Could not find image files for metric calculation.")
        return {"Error": "File not found"}
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return {"Error": str(e)}

    return metrics_dict


# --- Visualization Functions ---
def visualize_comparison(original_img: npt.NDArray[np.uint8],
                         compressed_img: npt.NDArray[np.uint8],
                         metrics_dict: Optional[Dict[str, Union[float, str]]] = None,
                         title_prefix: str = "") -> plt.Figure:
    """Enhanced visualization comparing original and compressed images with metrics."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Determine if original is grayscale
    is_original_gray = original_img.ndim == 2 or (original_img.ndim == 3 and original_img.shape[2] == 1)
    cmap_original = 'gray' if is_original_gray else None
    img_original_plot = original_img.squeeze() if is_original_gray else original_img

    axes[0].imshow(img_original_plot, cmap=cmap_original)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Determine if compressed is grayscale
    is_compressed_gray = compressed_img.ndim == 2 or (compressed_img.ndim == 3 and compressed_img.shape[2] == 1)
    cmap_compressed = 'gray' if is_compressed_gray else None
    img_compressed_plot = compressed_img.squeeze() if is_compressed_gray else compressed_img

    axes[1].imshow(img_compressed_plot, cmap=cmap_compressed)

    # Add metrics to title if available
    title = title_prefix + 'Compressed Image'
    if metrics_dict:
        title += '\n'
        ratio = metrics_dict.get("compression_ratio_file", "N/A")
        psnr = metrics_dict.get("psnr_db", "N/A")
        ssim = metrics_dict.get("ssim", "N/A")
        title += f"File Ratio: {ratio}, PSNR: {psnr} dB, SSIM: {ssim}"

    axes[1].set_title(title)
    axes[1].axis('off')

    fig.tight_layout()
    return fig


def visualize_dct_heatmap(dct_blocks: npt.NDArray[np.float32], channel_name: str = 'Y', block_index: int = 0) -> plt.Figure:
    """Visualizes the DCT coefficients of a specific block as a heatmap."""
    if block_index >= dct_blocks.shape[0]:
        print(f"Warning: Block index {block_index} out of range. Showing block 0.")
        block_index = 0

    dct_coeffs = dct_blocks[block_index]

    fig, ax = plt.subplots(figsize=(7, 6))
    # Use LogNorm for better visualization of coefficient magnitudes
    # Add a small epsilon to handle zero coefficients in log scale
    im = ax.imshow(np.abs(dct_coeffs) + 1e-9, cmap='viridis', norm=LogNorm())

    # Add text annotations for coefficient values (optional, can be slow for many blocks)
    # for i in range(BLOCK_SIZE):
    #     for j in range(BLOCK_SIZE):
    #         ax.text(j, i, f"{dct_coeffs[i, j]:.1f}", ha='center', va='center', color='white', fontsize=8)

    ax.set_title(f'DCT Coefficients Heatmap (Log Scale)\nChannel: {channel_name}, Block: {block_index}')
    ax.set_xticks(np.arange(BLOCK_SIZE))
    ax.set_yticks(np.arange(BLOCK_SIZE))
    ax.set_xticklabels(np.arange(BLOCK_SIZE))
    ax.set_yticklabels(np.arange(BLOCK_SIZE))
    plt.colorbar(im, ax=ax, label='Absolute DCT Coefficient Magnitude (Log Scale)')
    fig.tight_layout()
    return fig

def visualize_quantization_matrices(y_quant: npt.NDArray[np.float32], cbcr_quant: npt.NDArray[np.float32], quality: int) -> plt.Figure:
    """Visualizes the quantization matrices used."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    im_y = axes[0].imshow(y_quant, cmap='plasma', vmin=1, vmax=255)
    axes[0].set_title(f'Y Quantization Matrix (Q={quality})')
    axes[0].set_xticks(np.arange(BLOCK_SIZE))
    axes[0].set_yticks(np.arange(BLOCK_SIZE))
    fig.colorbar(im_y, ax=axes[0], label='Quantization Step Size')

    im_cbcr = axes[1].imshow(cbcr_quant, cmap='plasma', vmin=1, vmax=255)
    axes[1].set_title(f'Cb/Cr Quantization Matrix (Q={quality})')
    axes[1].set_xticks(np.arange(BLOCK_SIZE))
    axes[1].set_yticks(np.arange(BLOCK_SIZE))
    fig.colorbar(im_cbcr, ax=axes[1], label='Quantization Step Size')

    fig.tight_layout()
    return fig

def visualize_frequency_domain(dct_blocks: npt.NDArray[np.float32], channel_name: str = 'Y') -> plt.Figure:
    """Visualizes the magnitude of frequency components across all blocks."""
    # Average the absolute DCT coefficients across all blocks
    avg_dct_magnitude = np.mean(np.abs(dct_blocks), axis=0)

    fig, ax = plt.subplots(figsize=(7, 6))
    # Use LogNorm to see both low and high frequency components clearly
    im = ax.imshow(avg_dct_magnitude + 1e-9, cmap='magma', norm=LogNorm())

    ax.set_title(f'Average DCT Coefficient Magnitude (Log Scale)\nChannel: {channel_name}')
    ax.set_xlabel('Horizontal Frequency')
    ax.set_ylabel('Vertical Frequency')
    ax.set_xticks(np.arange(BLOCK_SIZE))
    ax.set_yticks(np.arange(BLOCK_SIZE))
    ax.set_xticklabels(['0 (DC)', '1', '2', '3', '4', '5', '6', '7'])
    ax.set_yticklabels(['0 (DC)', '1', '2', '3', '4', '5', '6', '7'])
    plt.colorbar(im, ax=ax, label='Avg. Absolute DCT Magnitude (Log Scale)')
    fig.tight_layout()
    return fig

def create_comparison_slider(original_img: npt.NDArray[np.uint8], compressed_img: npt.NDArray[np.uint8]) -> plt.Figure:
    """Creates a figure with a slider to compare original and compressed images."""
    fig = plt.figure(figsize=(8, 6))
    gs = gridspec.GridSpec(2, 1, height_ratios=[10, 1]) # Allocate space for image and slider
    ax_img = fig.add_subplot(gs[0])
    ax_slider = fig.add_subplot(gs[1])

    # Initial display (show original)
    img_display = ax_img.imshow(original_img)
    ax_img.set_title('Image Comparison (Use Slider)')
    ax_img.axis('off')

    # Slider setup
    slider = Slider(ax_slider, 'Original <-> Compressed', 0.0, 1.0, valinit=0.0, valstep=0.01)

    # Update function for the slider
    def update(val):
        alpha = slider.val # 0 = original, 1 = compressed
        # Simple alpha blend (might not be perfect for visual comparison)
        # blended_img = ((1 - alpha) * original_img.astype(float) + alpha * compressed_img.astype(float)).astype(np.uint8)
        # img_display.set_data(blended_img)

        # Alternative: Switch image based on threshold
        if alpha < 0.5:
            img_display.set_data(original_img)
            ax_img.set_title('Original Image')
        else:
            img_display.set_data(compressed_img)
            ax_img.set_title('Compressed Image')
        fig.canvas.draw_idle() # Redraw the figure

    slider.on_changed(update)

    # Keep a reference to the slider to prevent garbage collection if used interactively
    fig._slider_ref = slider

    fig.tight_layout()
    return fig

# --- Main Compression Pipeline ---
def compress_image(image_path: str, output_path: str, quality: int,
                   subsample_mode: str = "4:2:0", num_threads: int = 4,
                   visualize: bool = False, use_tqdm: bool = True,
                   is_grayscale: bool = False) -> Tuple[Optional[npt.NDArray[np.uint8]], Optional[Dict[str, Union[float, str]]]]:
    """Performs the full JPEG-like compression process."""
    print(f"\n--- Compressing {os.path.basename(image_path)} (Q={quality}, Subsample={subsample_mode}) ---")
    start_time = time.time()

    # 1. Load Image
    img_rgb_uint8, is_grayscale_load = load_image(image_path)
    if img_rgb_uint8 is None:
        return None, None
    # Override grayscale flag if explicitly passed
    is_grayscale = is_grayscale if is_grayscale is not None else is_grayscale_load

    # 2. Adjust Quantization Matrices
    y_quant, cbcr_quant = adjust_quantization_matrices(quality)
    if visualize:
        fig_quant = visualize_quantization_matrices(y_quant, cbcr_quant, quality)
        fig_quant.show() # Or save fig_quant


    # 3. Color Space Conversion (if not grayscale) & Level Shift
    if is_grayscale:
        print("Processing as grayscale.")
        img_ycbcr_float = img_rgb_uint8[:,:,0].astype(np.float32) # Use only one channel
        y_channel = level_shift(img_ycbcr_float) # Level shift the single channel
        cb_channel, cr_channel = None, None # No chroma channels
        original_dims_y = y_channel.shape
        original_dims_cb, original_dims_cr = None, None
    else:
        print("Converting RGB to YCbCr...")
        img_ycbcr_float = rgb_to_ycbcr(img_rgb_uint8.astype(np.float32))
        original_dims_y = img_ycbcr_float.shape[:2]

        # 4. Chroma Subsampling (if applicable)
        if subsample_mode != "4:4:4":
            print(f"Applying Chroma Subsampling ({subsample_mode})...")
            cbcr_channels = img_ycbcr_float[:, :, 1:]
            cbcr_subsampled = apply_chroma_subsampling(cbcr_channels, subsample_mode)
            print(f"Subsampled Cb/Cr shape: {cbcr_subsampled.shape}")
            y_channel_shifted = level_shift(img_ycbcr_float[:, :, 0]) # Level shift Y
            cb_channel_shifted = level_shift(cbcr_subsampled[:, :, 0]) # Level shift Cb_sub
            cr_channel_shifted = level_shift(cbcr_subsampled[:, :, 1]) # Level shift Cr_sub
            original_dims_cb = cb_channel_shifted.shape
            original_dims_cr = cr_channel_shifted.shape
        else:
            print("No Chroma Subsampling (4:4:4).")
            y_channel_shifted = level_shift(img_ycbcr_float[:, :, 0])
            cb_channel_shifted = level_shift(img_ycbcr_float[:, :, 1])
            cr_channel_shifted = level_shift(img_ycbcr_float[:, :, 2])
            original_dims_cb = cb_channel_shifted.shape
            original_dims_cr = cr_channel_shifted.shape


    # 5. Blocking
    print("Splitting channels into blocks...")
    y_blocks, nbh_y, nbw_y, _ = efficient_block_splitter(y_channel_shifted)
    cb_blocks, nbh_cb, nbw_cb, _ = (efficient_block_splitter(cb_channel_shifted)
                                     if cb_channel_shifted is not None else (None, 0, 0, None))
    cr_blocks, nbh_cr, nbw_cr, _ = (efficient_block_splitter(cr_channel_shifted)
                                     if cr_channel_shifted is not None else (None, 0, 0, None))

    # 6. DCT & Quantization (Parallelized per channel)
    print("Applying DCT and Quantization...")
    y_quant_blocks = process_channel_blocks(y_blocks, y_quant, operation='compress',
                                            num_threads=num_threads, use_tqdm=use_tqdm, desc='Y Channel')
    cb_quant_blocks = (process_channel_blocks(cb_blocks, cbcr_quant, operation='compress',
                                               num_threads=num_threads, use_tqdm=use_tqdm, desc='Cb Channel')
                       if cb_blocks is not None else None)
    cr_quant_blocks = (process_channel_blocks(cr_blocks, cbcr_quant, operation='compress',
                                               num_threads=num_threads, use_tqdm=use_tqdm, desc='Cr Channel')
                       if cr_blocks is not None else None)


    if visualize:
        # Visualize DCT heatmap and frequency domain for Y channel (most important)
        fig_dct_h = visualize_dct_heatmap(y_quant_blocks, channel_name='Y Quantized', block_index=0)
        fig_dct_h.show()
        fig_freq = visualize_frequency_domain(y_quant_blocks, channel_name='Y Quantized')
        fig_freq.show()

    # 7. Zigzag Coding & Entropy Estimation
    print("Applying Zigzag Coding and Estimating Entropy...")
    y_zigzag = apply_zigzag_coding(y_quant_blocks)
    cb_zigzag = apply_zigzag_coding(cb_quant_blocks) if cb_quant_blocks is not None else None
    cr_zigzag = apply_zigzag_coding(cr_quant_blocks) if cr_quant_blocks is not None else None

    entropy_y = estimate_entropy(y_zigzag)
    entropy_cb = estimate_entropy(cb_zigzag) if cb_zigzag is not None else None
    entropy_cr = estimate_entropy(cr_zigzag) if cr_zigzag is not None else None
    print(f"  Y Channel - Zero Ratio: {entropy_y['zero_ratio']:.3f}, Entropy: {entropy_y['entropy_bits_per_coeff']:.3f} bits/coeff")
    if entropy_cb: print(f" Cb Channel - Zero Ratio: {entropy_cb['zero_ratio']:.3f}, Entropy: {entropy_cb['entropy_bits_per_coeff']:.3f} bits/coeff")
    if entropy_cr: print(f" Cr Channel - Zero Ratio: {entropy_cr['zero_ratio']:.3f}, Entropy: {entropy_cr['entropy_bits_per_coeff']:.3f} bits/coeff")

    # --- Decompression Steps ---
    print("\n--- Starting Decompression ---")
    # 8. Inverse Zigzag
    y_dezig = inverse_zigzag(y_zigzag, y_quant_blocks.shape[0])
    cb_dezig = inverse_zigzag(cb_zigzag, cb_quant_blocks.shape[0]) if cb_zigzag is not None else None
    cr_dezig = inverse_zigzag(cr_zigzag, cr_quant_blocks.shape[0]) if cr_zigzag is not None else None

    # 9. Dequantization & Inverse DCT
    print("Applying Dequantization and Inverse DCT...")
    y_idct_blocks = process_channel_blocks(y_dezig, y_quant, operation='decompress',
                                          num_threads=num_threads, use_tqdm=use_tqdm, desc='Y Channel')
    cb_idct_blocks = (process_channel_blocks(cb_dezig, cbcr_quant, operation='decompress',
                                             num_threads=num_threads, use_tqdm=use_tqdm, desc='Cb Channel')
                      if cb_dezig is not None else None)
    cr_idct_blocks = (process_channel_blocks(cr_dezig, cbcr_quant, operation='decompress',
                                             num_threads=num_threads, use_tqdm=use_tqdm, desc='Cr Channel')
                      if cr_dezig is not None else None)

    # 10. Block Merging
    print("Merging blocks...")
    y_merged = efficient_block_merger(y_idct_blocks, nbh_y, nbw_y, original_dims_y)
    cb_merged = (efficient_block_merger(cb_idct_blocks, nbh_cb, nbw_cb, original_dims_cb)
                 if cb_idct_blocks is not None else None)
    cr_merged = (efficient_block_merger(cr_idct_blocks, nbh_cr, nbw_cr, original_dims_cr)
                 if cr_idct_blocks is not None else None)

    # 11. Upscale Chroma Channels (if subsampled)
    if not is_grayscale and subsample_mode != "4:4:4":
        print("Upscaling chroma channels...")
        cbcr_merged = np.stack((cb_merged, cr_merged), axis=-1)
        cbcr_upscaled = upscale_chroma(cbcr_merged, original_dims_y)
        cb_final = cbcr_upscaled[:, :, 0]
        cr_final = cbcr_upscaled[:, :, 1]
    elif not is_grayscale: # 4:4:4 case
        cb_final = cb_merged
        cr_final = cr_merged
    else: # Grayscale case
         cb_final, cr_final = None, None


    # 12. Inverse Level Shift & Combine Channels
    y_final = inverse_level_shift(y_merged)
    if not is_grayscale:
         cb_final_ishift = inverse_level_shift(cb_final)
         cr_final_ishift = inverse_level_shift(cr_final)
         # Stack channels: Y, Cb, Cr
         img_ycbcr_reconstructed = np.stack((y_final, cb_final_ishift, cr_final_ishift), axis=-1)

         # 13. Convert back to RGB
         print("Converting YCbCr back to RGB...")
         img_rgb_reconstructed_uint8 = ycbcr_to_rgb(img_ycbcr_reconstructed)
    else:
         # For grayscale, just clip and convert type
         img_rgb_reconstructed_uint8 = np.clip(y_final, 0, 255).astype(np.uint8)


    # 14. Save Compressed Image
    if not save_image(img_rgb_reconstructed_uint8, output_path, quality=quality):
        print("Failed to save the compressed image.")
        return None, None

    end_time = time.time()
    print(f"Compression and Decompression finished in {end_time - start_time:.2f} seconds.")

    # 15. Calculate Metrics
    print("Calculating metrics...")
    metrics_dict = calculate_metrics(image_path, output_path,
                                     img_rgb_uint8, img_rgb_reconstructed_uint8, # Compare original RGB with final RGB
                                     entropy_y, entropy_cb, entropy_cr)

    print("\n--- Compression Report ---")
    for key, value in metrics_dict.items():
        print(f"- {key.replace('_', ' ').title()}: {value}")
    print("--------------------------\n")


    if visualize:
        # Final comparison visualization
        fig_comp = visualize_comparison(img_rgb_uint8, img_rgb_reconstructed_uint8, metrics_dict, title_prefix=f"Q={quality}, {subsample_mode} | ")
        fig_comp.show()
        # Comparison slider
        # fig_slider = create_comparison_slider(img_rgb_uint8, img_rgb_reconstructed_uint8)
        # fig_slider.show() # Slider often works best interactively

    return img_rgb_reconstructed_uint8, metrics_dict


# --- GUI Application ---
class CompressionGUI:
    def __init__(self, master):
        self.master = master
        master.title("Enhanced JPEG Compressor")
        master.geometry("800x600")

        self.input_path = tk.StringVar()
        self.output_dir = tk.StringVar(value=os.getcwd()) # Default to current dir
        self.quality = tk.IntVar(value=75)
        self.subsample_mode = tk.StringVar(value="4:2:0")
        self.num_threads = tk.IntVar(value=os.cpu_count() or 4) # Default to CPU count
        self.visualize = tk.BooleanVar(value=False)
        self.batch_mode = tk.BooleanVar(value=False)

        # Input Frame
        input_frame = ttk.LabelFrame(master, text="Input/Output")
        input_frame.pack(padx=10, pady=5, fill="x")

        ttk.Label(input_frame, text="Input Image/Folder:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        ttk.Entry(input_frame, textvariable=self.input_path, width=50).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(input_frame, text="Browse...", command=self.browse_input).grid(row=0, column=2, padx=5, pady=5)

        ttk.Label(input_frame, text="Output Directory:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        ttk.Entry(input_frame, textvariable=self.output_dir, width=50).grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(input_frame, text="Browse...", command=self.browse_output).grid(row=1, column=2, padx=5, pady=5)

        ttk.Checkbutton(input_frame, text="Batch Mode (Process Folder)", variable=self.batch_mode, command=self.toggle_batch_mode).grid(row=0, column=3, padx=10)


        # Settings Frame
        settings_frame = ttk.LabelFrame(master, text="Compression Settings")
        settings_frame.pack(padx=10, pady=5, fill="x")

        ttk.Label(settings_frame, text="Quality (1-100):").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        ttk.Scale(settings_frame, from_=1, to=100, orient=tk.HORIZONTAL, variable=self.quality, length=200, command=lambda s: self.quality_label.config(text=f"{int(float(s))}")).grid(row=0, column=1, padx=5, pady=5)
        self.quality_label = ttk.Label(settings_frame, text=f"{self.quality.get()}")
        self.quality_label.grid(row=0, column=2, padx=5, pady=5)

        ttk.Label(settings_frame, text="Chroma Subsampling:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        ttk.Combobox(settings_frame, textvariable=self.subsample_mode, values=["4:4:4", "4:2:2", "4:2:0"]).grid(row=1, column=1, columnspan=2, padx=5, pady=5, sticky="w")

        ttk.Label(settings_frame, text="Threads:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        ttk.Spinbox(settings_frame, from_=1, to=os.cpu_count() or 16, textvariable=self.num_threads, width=5).grid(row=2, column=1, padx=5, pady=5, sticky="w")

        ttk.Checkbutton(settings_frame, text="Show Visualizations", variable=self.visualize).grid(row=3, column=0, columnspan=2, padx=5, pady=5, sticky="w")


        # Action Frame
        action_frame = ttk.Frame(master)
        action_frame.pack(padx=10, pady=10)

        self.run_button = ttk.Button(action_frame, text="Compress", command=self.run_compression_gui)
        self.run_button.pack(side=tk.LEFT, padx=5)

        self.status_label = ttk.Label(master, text="Status: Idle")
        self.status_label.pack(pady=5)

        # Progress Bar
        self.progress = ttk.Progressbar(master, orient=tk.HORIZONTAL, length=300, mode='determinate')
        self.progress.pack(pady=5)

        # Image Preview Frame (Optional) - Placeholder
        preview_frame = ttk.LabelFrame(master, text="Preview")
        preview_frame.pack(padx=10, pady=5, fill="both", expand=True)
        self.canvas_preview = tk.Canvas(preview_frame, bg='lightgrey') # Placeholder Canvas
        self.canvas_preview.pack(fill="both", expand=True)
        # You would need more logic here to load and display images on the canvas

    def browse_input(self):
        if self.batch_mode.get():
            path = filedialog.askdirectory(title="Select Input Folder")
        else:
            path = filedialog.askopenfilename(title="Select Input Image", filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("All Files", "*.*")])
        if path:
            self.input_path.set(path)

    def browse_output(self):
        path = filedialog.askdirectory(title="Select Output Directory")
        if path:
            self.output_dir.set(path)

    def toggle_batch_mode(self):
        # Update button text or label if needed
        pass

    def update_status(self, message):
        self.status_label.config(text=f"Status: {message}")
        self.master.update_idletasks() # Force GUI update

    def run_compression_gui(self):
        input_val = self.input_path.get()
        output_dir_val = self.output_dir.get()
        quality_val = self.quality.get()
        subsample_val = self.subsample_mode.get()
        threads_val = self.num_threads.get()
        visualize_val = self.visualize.get()
        is_batch = self.batch_mode.get()

        if not input_val or not output_dir_val:
            messagebox.showerror("Error", "Please select input and output paths.")
            return

        if not os.path.exists(output_dir_val):
             os.makedirs(output_dir_val)

        self.run_button.config(state=tk.DISABLED) # Disable button during run
        self.update_status("Starting...")
        self.progress['value'] = 0

        # Run compression in a separate thread to keep GUI responsive
        thread = threading.Thread(target=self._run_compression_thread, args=(
            input_val, output_dir_val, quality_val, subsample_val, threads_val, visualize_val, is_batch
        ))
        thread.start()


    def _run_compression_thread(self, input_val, output_dir_val, quality_val, subsample_val, threads_val, visualize_val, is_batch):
         try:
             if is_batch:
                 if not os.path.isdir(input_val):
                      messagebox.showerror("Error", "Batch mode selected, but input is not a directory.")
                      return
                 image_files = [os.path.join(input_val, f) for f in os.listdir(input_val)
                                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
                 if not image_files:
                      messagebox.showwarning("Warning", "No supported image files found in the input directory.")
                      return

                 total_files = len(image_files)
                 self.progress['maximum'] = total_files
                 for i, img_path in enumerate(image_files):
                     self.update_status(f"Processing {os.path.basename(img_path)} ({i+1}/{total_files})...")
                     output_filename = f"{os.path.splitext(os.path.basename(img_path))[0]}_q{quality_val}.jpg" # Save as JPEG
                     output_filepath = os.path.join(output_dir_val, output_filename)
                     _, _ = compress_image(img_path, output_filepath, quality_val, subsample_val, threads_val, visualize_val, use_tqdm=False) # Disable tqdm for GUI
                     self.progress['value'] = i + 1
                     self.master.update_idletasks()

             else:
                 if not os.path.isfile(input_val):
                      messagebox.showerror("Error", "Input path is not a file.")
                      return
                 self.progress['maximum'] = 1
                 self.update_status(f"Processing {os.path.basename(input_val)}...")
                 output_filename = f"{os.path.splitext(os.path.basename(input_val))[0]}_q{quality_val}.jpg"
                 output_filepath = os.path.join(output_dir_val, output_filename)
                 _, _ = compress_image(input_val, output_filepath, quality_val, subsample_val, threads_val, visualize_val, use_tqdm=False)
                 self.progress['value'] = 1

             self.update_status("Finished.")
             messagebox.showinfo("Success", "Compression process completed.")

         except Exception as e:
             self.update_status(f"Error: {e}")
             messagebox.showerror("Error", f"An error occurred: {e}")
             import traceback
             traceback.print_exc() # Print full error for debugging
         finally:
             self.run_button.config(state=tk.NORMAL) # Re-enable button
             self.progress['value'] = 0


# --- Command Line Interface ---
def main():
    parser = argparse.ArgumentParser(description="Enhanced JPEG-like Image Compression Tool")
    parser.add_argument("input_path", help="Path to the input image or folder (for batch mode).")
    parser.add_argument("-o", "--output", default="output", help="Output directory for compressed files (default: ./output).")
    parser.add_argument("-q", "--quality", type=int, default=75, help="Compression quality factor (1-100, default: 75).")
    parser.add_argument("-s", "--subsample", type=str, default="4:2:0", choices=["4:4:4", "4:2:2", "4:2:0"],
                        help="Chroma subsampling mode (default: 4:2:0).")
    parser.add_argument("-t", "--threads", type=int, default=os.cpu_count() or 4,
                        help="Number of threads for parallel processing (default: CPU count).")
    parser.add_argument("-v", "--visualize", action="store_true", help="Show visualization plots during compression.")
    parser.add_argument("-b", "--batch", action="store_true", help="Enable batch mode: process all images in the input_path folder.")
    parser.add_argument("--gui", action="store_true", help="Launch the graphical user interface.")

    args = parser.parse_args()

    if args.gui:
        print("Launching GUI...")
        root = tk.Tk()
        gui = CompressionGUI(root)
        root.mainloop()
    else:
        # Command Line Execution
        if not os.path.exists(args.input_path):
            print(f"Error: Input path not found: {args.input_path}")
            return

        if not os.path.exists(args.output):
            os.makedirs(args.output)
            print(f"Created output directory: {args.output}")

        if args.batch:
            if not os.path.isdir(args.input_path):
                print(f"Error: Batch mode requires input_path to be a directory.")
                return
            print(f"--- Starting Batch Compression ---")
            print(f"Input Folder: {args.input_path}")
            print(f"Output Folder: {args.output}")
            print(f"Quality: {args.quality}, Subsampling: {args.subsample}, Threads: {args.threads}")

            image_files = [f for f in os.listdir(args.input_path)
                           if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

            if not image_files:
                print("No compatible image files found in the directory.")
                return

            total_files = len(image_files)
            print(f"Found {total_files} images to process.")

            for i, filename in enumerate(image_files):
                 input_filepath = os.path.join(args.input_path, filename)
                 # Construct output filename (save as JPEG)
                 output_filename = f"{os.path.splitext(filename)[0]}_q{args.quality}.jpg"
                 output_filepath = os.path.join(args.output, output_filename)
                 print(f"\nProcessing [{i+1}/{total_files}]: {filename}")
                 compress_image(input_filepath, output_filepath, args.quality,
                                args.subsample, args.threads, args.visualize, use_tqdm=True) # Enable tqdm for CLI batch

            print("\n--- Batch Compression Finished ---")

        else:
            if not os.path.isfile(args.input_path):
                 print(f"Error: Input path must be a file (or use --batch for folders).")
                 return
            print(f"--- Starting Single Image Compression ---")
            # Construct output filename (save as JPEG)
            base_name = os.path.basename(args.input_path)
            output_filename = f"{os.path.splitext(base_name)[0]}_q{args.quality}.jpg"
            output_filepath = os.path.join(args.output, output_filename)
            compress_image(args.input_path, output_filepath, args.quality,
                           args.subsample, args.threads, args.visualize, use_tqdm=True)
            print("\n--- Single Image Compression Finished ---")


if __name__ == "__main__":
    # Set backend for matplotlib if needed, especially for GUI/threading
    # import matplotlib
    # matplotlib.use('TkAgg') # Or another suitable backend
    main()