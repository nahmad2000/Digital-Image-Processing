# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
import time
import argparse
import glob
from typing import List, Any, Optional, Tuple, Dict
import math # For checking inf/nan

# --- Attempt to import DCT from SciPy ---
try:
    from scipy.fftpack import dct
except ImportError:
    print("Error: SciPy library not found.")
    print("Please install it using: pip install scipy")
    exit(1)

# --- Attempt to import Matplotlib ---
try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Warning: Matplotlib library not found.")
    print("Plotting will be disabled. Install it using: pip install matplotlib")
    plt = None # Disable plotting

# --- Attempt to import Huffman module ---
try:
    from huffman import huffman_encoding # Only encoding needed for size comparison
except ImportError:
    print("Error: Could not import huffman module. Make sure huffman.py is accessible.")
    exit(1)

# --- Helper Functions (Loading, Saving, File Size - Keep as before) ---

def load_image(image_path: str) -> Optional[np.ndarray]:
    """Loads an image using OpenCV."""
    try:
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Error: Could not load image from '{image_path}'. Skipping.")
            return None
        if img.ndim == 3 and img.shape[2] == 4:
             print(f"Info: Dropping alpha channel from '{os.path.basename(image_path)}'.")
             img = img[:, :, :3]
        return img
    except Exception as e:
        print(f"An unexpected error occurred while loading '{image_path}': {e}. Skipping.")
        return None

def get_file_size(file_path: str) -> Optional[int]:
    """Gets file size in bytes, returns None if file not found or error."""
    try:
        # Check if file exists before getting size
        if os.path.isfile(file_path):
            return os.path.getsize(file_path)
        else:
            # Expected if encoding failed, don't warn here.
            return None
    except Exception as e:
        print(f"Error getting size for file '{file_path}': {e}")
        return None


# --- DCT and Quantization Function (Keep as before) ---

def apply_dct_quantize(image_data: np.ndarray, q_factor: int = 10) -> Optional[np.ndarray]:
    """Simplified Demonstration: Apply 2D DCT and basic quantization."""
    if q_factor <= 0:
        print("Error: Quantization factor must be positive.")
        return None
    try:
        if image_data.ndim == 3:
            coeffs = np.zeros_like(image_data, dtype=np.float32)
            for i in range(image_data.shape[2]):
                channel_coeffs = dct(dct(image_data[:,:,i].astype(np.float32), axis=0, norm='ortho'), axis=1, norm='ortho')
                coeffs[:,:,i] = np.round(channel_coeffs / q_factor)
            return coeffs.astype(np.int32)
        elif image_data.ndim == 2:
             channel_coeffs = dct(dct(image_data.astype(np.float32), axis=0, norm='ortho'), axis=1, norm='ortho')
             coeffs = np.round(channel_coeffs / q_factor)
             return coeffs.astype(np.int32)
        else:
            print(f"Error: Unsupported image dimensions for DCT: {image_data.ndim}")
            return None
    except Exception as e:
        print(f"Error during DCT/Quantization process: {e}")
        return None


# --- Core Processing Function for COMPARISON (Keep as before) ---

def process_image_comparison(
    input_path: str,
    raw_huff_path: str, # Path for Huffman on raw pixels output
    dct_huff_path: str, # Path for Huffman on DCT coeffs output
    q_factor: int
) -> Optional[Dict[str, Any]]:
    """
    Processes one image using two methods:
    1. Huffman directly on raw pixels.
    2. Huffman on DCT+Quantized coefficients.
    Returns comparison results.
    """
    start_time = time.time()
    print(f"Comparing '{os.path.basename(input_path)}' (Q={q_factor})...")

    results = {
        "filename": os.path.basename(input_path),
        "status": "Failed",
        "q_factor": q_factor,
        "original_size": None,
        "raw_huffman_size": None,   # Size of Huffman on raw pixels
        "dct_huffman_size": None,   # Size of Huffman on DCT coeffs
        "raw_ratio": None,          # Orig / Raw Huffman
        "dct_ratio": None,          # Orig / DCT Huffman
        "time": None,
    }

    # 1. Load Image (Lossless like PNG, BMP recommended)
    img = load_image(input_path)
    if img is None: return results
    results["original_size"] = get_file_size(input_path)
    if results["original_size"] is None:
         print(f"Warning: Could not get original file size for {input_path}.")

    # --- Method 1: Huffman on Raw Pixels ---
    print(f"  - Applying Huffman directly to pixels...")
    raw_status = "Not Run"
    try:
        if img.ndim in [2, 3]:
            pixel_list = [int(p) for p in img.ravel()]
        else:
            raise ValueError("Unsupported image dimensions")

        if pixel_list:
            os.makedirs(os.path.dirname(raw_huff_path), exist_ok=True)
            _ , _ = huffman_encoding(pixel_list, raw_huff_path)
            results["raw_huffman_size"] = get_file_size(raw_huff_path)
            raw_status = "Raw Encoded"
            if results["raw_huffman_size"] is None:
                 print(f"    Warning: Huffman encoding (raw) seemed OK but failed to get size of {raw_huff_path}")
                 raw_status = "Raw Enc OK, Size Fail"
        else:
             print(f"    Warning: Raw pixel list is empty for {input_path}.")
             raw_status = "Raw Empty Pixels"

    except Exception as e:
        print(f"    Error during Huffman encoding of raw pixels: {e}")
        raw_status = "Raw Huffman Failed"


    # --- Method 2: Huffman on DCT + Quantized Coefficients ---
    print(f"  - Applying DCT (Q={q_factor}) + Huffman...")
    dct_status = "Not Run"
    # 2a. Apply DCT and Quantization
    quantized_coeffs = apply_dct_quantize(img, q_factor=q_factor)
    if quantized_coeffs is None:
        dct_status = "DCT/Quant Failed"
    else:
        # 2b. Flatten Coefficients
        coefficient_list = [int(c) for c in quantized_coeffs.ravel()]

        if not coefficient_list:
             print(f"    Warning: Coefficient list is empty after DCT/Quantization for {input_path}.")
             dct_status = "DCT Empty Coeffs"
        else:
            # 2c. Apply Huffman Encoding to Coefficients
            try:
                os.makedirs(os.path.dirname(dct_huff_path), exist_ok=True)
                _ , _ = huffman_encoding(coefficient_list, dct_huff_path)
                results["dct_huffman_size"] = get_file_size(dct_huff_path)
                dct_status = "DCT Encoded"
                if results["dct_huffman_size"] is None:
                     print(f"    Warning: Huffman encoding (DCT) seemed OK but failed to get size of {dct_huff_path}")
                     dct_status = "DCT Enc OK, Size Fail"

            except Exception as e:
                print(f"    Error during Huffman encoding of coefficients: {e}")
                dct_status = "DCT Huffman Failed"

    # Determine overall status based on individual steps
    if "Failed" in raw_status or "Failed" in dct_status:
         results["status"] = f"Partial Fail ({raw_status} / {dct_status})"
    elif "OK" in raw_status or "OK" in dct_status: # If size failed but encode worked
         results["status"] = f"Size Error ({raw_status} / {dct_status})"
    elif raw_status == "Raw Encoded" and dct_status == "DCT Encoded":
         results["status"] = "Comparison Done"
    else: # Handle cases like empty pixels/coeffs
         results["status"] = f"Issue ({raw_status} / {dct_status})"


    # 3. Calculate Ratios
    if results["original_size"] is not None and results["original_size"] > 0:
        if results["raw_huffman_size"] is not None and results["raw_huffman_size"] > 0:
            results["raw_ratio"] = results["original_size"] / results["raw_huffman_size"]
        if results["dct_huffman_size"] is not None and results["dct_huffman_size"] > 0:
            results["dct_ratio"] = results["original_size"] / results["dct_huffman_size"]

    end_time = time.time()
    results["time"] = end_time - start_time
    print(f"  Finished comparison for '{os.path.basename(input_path)}' in {results['time']:.2f} sec.")
    return results

# --- Function to Create Comparison Plot ---

def create_comparison_plot(results_list: List[Dict[str, Any]], output_dir: str, q_factor: int):
    """Creates and saves a bar chart comparing file sizes."""
    if not plt:
        print("Matplotlib not available. Skipping plot generation.")
        return
    if not results_list:
        print("No results to plot.")
        return

    filenames = [r['filename'] for r in results_list]
    # Convert sizes to KB, handle None values by setting them to 0 for plotting
    original_kb = np.array([r['original_size']/1024 if r['original_size'] is not None else 0 for r in results_list])
    raw_huff_kb = np.array([r['raw_huffman_size']/1024 if r['raw_huffman_size'] is not None else 0 for r in results_list])
    dct_huff_kb = np.array([r['dct_huffman_size']/1024 if r['dct_huffman_size'] is not None else 0 for r in results_list])

    # Ensure no NaN or Inf values which can cause plotting errors
    original_kb = np.nan_to_num(original_kb, nan=0.0, posinf=0.0, neginf=0.0)
    raw_huff_kb = np.nan_to_num(raw_huff_kb, nan=0.0, posinf=0.0, neginf=0.0)
    dct_huff_kb = np.nan_to_num(dct_huff_kb, nan=0.0, posinf=0.0, neginf=0.0)


    x = np.arange(len(filenames))  # the label locations
    width = 0.25  # the width of the bars

    try:
        fig, ax = plt.subplots(figsize=(max(10, len(filenames) * 1.5), 6)) # Dynamic width
        rects1 = ax.bar(x - width, original_kb, width, label='Original Size', color='skyblue')
        rects2 = ax.bar(x , raw_huff_kb, width, label='Huffman(Raw Pixels)', color='salmon')
        rects3 = ax.bar(x + width, dct_huff_kb, width, label=f'Huffman(DCT Q={q_factor})', color='lightgreen')

        # Add some text for labels, title and axes ticks
        ax.set_ylabel('Size (KB)')
        ax.set_title(f'File Size Comparison (Q Factor = {q_factor})')
        ax.set_xticks(x)
        ax.set_xticklabels(filenames, rotation=45, ha="right") # Rotate labels if many files
        ax.legend()

        # Optional: Add labels on top of bars
        # ax.bar_label(rects1, padding=3, fmt='%.0f')
        # ax.bar_label(rects2, padding=3, fmt='%.0f')
        # ax.bar_label(rects3, padding=3, fmt='%.0f')

        ax.yaxis.grid(True, linestyle='--', alpha=0.7) # Add horizontal grid lines
        fig.tight_layout() # Adjust layout

        # Save the plot
        plot_filename = f"compression_comparison_q{q_factor}.png"
        plot_path = os.path.join(output_dir, plot_filename)
        plt.savefig(plot_path)
        print(f"\nComparison plot saved to: '{plot_path}'")
        plt.close(fig) # Close the figure to free memory

    except Exception as e:
        print(f"\nError generating plot: {e}")
        print("Plotting failed. Check Matplotlib installation and data.")

# --- Main Execution Logic ---

def main():
    """Finds lossless images, processes using both methods, displays comparison table and plot."""
    parser = argparse.ArgumentParser(
        description="Compares Huffman Coding on Raw Pixels vs. DCT Coefficients.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input_dir", default="images",
                        help="Directory containing input images (PNG, BMP recommended).")
    parser.add_argument("--output_dir", default="comparison_output",
                        help="Directory to save compressed binary files (.bin) and plot.")
    parser.add_argument("--q_factor", type=int, default=10,
                        help="Quantization factor for DCT method (integer > 0).")
    parser.add_argument("--img_format", default="png",
                        help="Image format to search for (e.g., png, bmp, tif).")

    args = parser.parse_args()

    if args.q_factor <= 0:
        print("Error: --q_factor must be greater than 0.")
        exit(1)

    # --- Find Images ---
    search_pattern = os.path.join(args.input_dir, f'*.{args.img_format}')
    image_paths = sorted(glob.glob(search_pattern))

    if not image_paths:
        print(f"Error: No '*.{args.img_format}' images found in directory '{args.input_dir}'.")
        print("Note: This demo requires lossless formats like PNG or BMP.")
        exit(1)

    print(f"Found {len(image_paths)} images ('*.{args.img_format}') in '{args.input_dir}'.")
    print(f"Comparing Huffman(Raw) vs. Huffman(DCT(Q={args.q_factor}))")
    print(f"Output files (.bin, .png) will be saved to: '{args.output_dir}'")
    print("-" * 80)

    # --- Create output directory ---
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Process Images ---
    all_results = []
    total_start_time = time.time()

    for input_path in image_paths:
        base_filename = os.path.splitext(os.path.basename(input_path))[0]
        raw_huff_filename = f"{base_filename}_raw_huffman.bin"
        dct_huff_filename = f"{base_filename}_q{args.q_factor}_dct_huffman.bin"

        raw_huff_path = os.path.join(args.output_dir, raw_huff_filename)
        dct_huff_path = os.path.join(args.output_dir, dct_huff_filename)

        result = process_image_comparison(input_path, raw_huff_path, dct_huff_path, args.q_factor)
        if result:
            all_results.append(result)

    total_end_time = time.time()
    print("-" * 80)
    print(f"Finished processing all images in {total_end_time - total_start_time:.2f} seconds.")
    print("-" * 80)

    # --- Display Results Table ---
    print("\n--- Huffman Compression Comparison Summary ---")
    # Refined header for better alignment
    hdr = f"{'Filename':<20} | {'Orig (KB)':>10} | {'Raw Huff (KB)':>14} | {'Ratio':>9} | {'DCT Huff (KB)':>14} | {'Ratio':>9} | {'Time (s)':>8}"
    print(hdr)
    print("-" * len(hdr)) # Dynamic separator length

    valid_results_for_plot = [] # Collect only results with valid sizes for plotting

    for r in all_results:
        # Format KB values, handle None
        orig_kb_str = f"{r['original_size']/1024:.1f}" if r['original_size'] is not None else "N/A"
        raw_kb_str = f"{r['raw_huffman_size']/1024:.1f}" if r['raw_huffman_size'] is not None else "N/A"
        dct_kb_str = f"{r['dct_huffman_size']/1024:.1f}" if r['dct_huffman_size'] is not None else "N/A"
        # Format ratios, handle None and potential math errors
        raw_ratio_str = f"{r['raw_ratio']:.2f}" if r['raw_ratio'] is not None and math.isfinite(r['raw_ratio']) else "N/A"
        dct_ratio_str = f"{r['dct_ratio']:.2f}" if r['dct_ratio'] is not None and math.isfinite(r['dct_ratio']) else "N/A"
        time_str = f"{r['time']:.2f}" if r['time'] is not None else "N/A"

        print(f"{r['filename']:<20} | {orig_kb_str:>10} | {raw_kb_str:>14} | {raw_ratio_str:>9} | {dct_kb_str:>14} | {dct_ratio_str:>9} | {time_str:>8}")

        # Add to plot list only if all relevant sizes are available
        if r['original_size'] is not None and r['raw_huffman_size'] is not None and r['dct_huffman_size'] is not None:
             valid_results_for_plot.append(r)


    print("-" * len(hdr))
    # Add more summary stats if desired (e.g., average ratios)

    # --- Generate Plot ---
    if valid_results_for_plot:
         create_comparison_plot(valid_results_for_plot, args.output_dir, args.q_factor)
    else:
         print("\nSkipping plot generation as no valid results with all sizes were found.")


if __name__ == "__main__":
    main()