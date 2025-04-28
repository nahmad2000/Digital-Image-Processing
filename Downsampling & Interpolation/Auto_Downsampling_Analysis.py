#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automatic Image Downsampling and Interpolation Analysis

This script automatically processes an image with multiple downsampling and 
interpolation methods, generating comprehensive comparisons and quality metrics.
All results are saved without requiring user input.

Usage:
    python Auto_Downsampling_Analysis.py <path_to_image> [--factor FACTOR]

Example:
    python Auto_Downsampling_Analysis.py image.jpg --factor 8
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse
import os
import sys
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import time

# Constants
DOWNSAMPLE_METHODS = {
    1: "Simple (Pixel Skipping)",
    2: "Anti-aliased (Gaussian)",
    3: "Area-based"
}

UPSAMPLE_METHODS = {
    1: "Nearest Neighbor",
    2: "Bilinear",
    3: "Bicubic", 
    4: "Lanczos"
}

OPENCV_INTERPOLATION = {
    1: cv2.INTER_NEAREST,
    2: cv2.INTER_LINEAR,
    3: cv2.INTER_CUBIC,
    4: cv2.INTER_LANCZOS4
}

def downsample_image(image, factor, method):
    """Downsample an image using the specified method."""
    if method == 1:  # Simple
        return image[::factor, ::factor]
    elif method == 2:  # Anti-aliased
        sigma = 0.3 * factor
        blurred = cv2.GaussianBlur(image, (0, 0), sigma)
        return blurred[::factor, ::factor]
    elif method == 3:  # Area-based
        h, w = image.shape[:2]
        return cv2.resize(image, (w // factor, h // factor), interpolation=cv2.INTER_AREA)

def calculate_metrics(original, reconstructed):
    """Calculate PSNR and SSIM between original and reconstructed images."""
    # Convert to grayscale for metrics if needed
    if len(original.shape) == 3 and original.shape[2] == 3:
        original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        reconstructed_gray = cv2.cvtColor(reconstructed, cv2.COLOR_BGR2GRAY)
    else:
        original_gray = original
        reconstructed_gray = reconstructed
    
    return psnr(original_gray, reconstructed_gray), ssim(original_gray, reconstructed_gray)

def save_individual_image(image, filename, results_dir):
    """Save an individual image with proper name."""
    filepath = os.path.join(results_dir, filename)
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Save color image as is
        cv2.imwrite(filepath, image)
    else:
        # For grayscale, ensure proper saving
        cv2.imwrite(filepath, image)

def create_results_dir(base_dir="results"):
    """Create timestamped results directory."""
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    results_dir = os.path.join(base_dir, f"analysis_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    return results_dir

def process_all_combinations(original_image, factor, results_dir):
    """Process all combinations of methods and save results."""
    # Create subdirectories
    images_dir = os.path.join(results_dir, "images")
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    # Get original dimensions
    h, w = original_image.shape[:2]
    original_dims = (w, h)
    
    # Save original image
    save_individual_image(original_image, f"original.jpg", images_dir)
    
    # Results storage
    results = []
    
    # Process all combinations
    print(f"Processing all combinations with downsampling factor {factor}...")
    
    # Create downsampled versions first (to avoid redundant calculations)
    downsampled_images = {}
    for down_id, down_name in DOWNSAMPLE_METHODS.items():
        print(f"  Downsampling with {down_name}...")
        downsampled = downsample_image(original_image, factor, down_id)
        downsampled_images[down_id] = downsampled
        
        # Save downsampled image
        down_filename = f"downsampled_x{factor}_{down_id}_{down_name.replace(' ', '_')}.jpg"
        save_individual_image(downsampled, down_filename, images_dir)
    
    # Now process all upsampling methods for each downsampled image
    best_psnr = -float('inf')
    best_ssim = -float('inf')
    best_psnr_combo = None
    best_ssim_combo = None
    
    for down_id, downsampled in downsampled_images.items():
        down_name = DOWNSAMPLE_METHODS[down_id]
        
        for up_id, up_name in UPSAMPLE_METHODS.items():
            print(f"  Upsampling {down_name} with {up_name}...")
            
            # Upsample
            interpolation = OPENCV_INTERPOLATION[up_id]
            upsampled = cv2.resize(downsampled, original_dims, interpolation=interpolation)
            
            # Calculate metrics
            psnr_value, ssim_value = calculate_metrics(original_image, upsampled)
            
            # Save result image
            result_filename = f"result_x{factor}_down{down_id}_up{up_id}_{down_name.replace(' ', '_')}_{up_name.replace(' ', '_')}.jpg"
            save_individual_image(upsampled, result_filename, images_dir)
            
            # Save comparison figure
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Convert to RGB for display if color
            if len(original_image.shape) == 3 and original_image.shape[2] == 3:
                disp_original = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
                disp_downsampled = cv2.cvtColor(downsampled, cv2.COLOR_BGR2RGB)
                disp_upsampled = cv2.cvtColor(upsampled, cv2.COLOR_BGR2RGB)
            else:
                disp_original = original_image
                disp_downsampled = downsampled
                disp_upsampled = upsampled
                
            # Plot images
            axes[0].imshow(disp_original, cmap='gray' if len(original_image.shape) < 3 else None)
            axes[0].set_title(f"Original ({w}x{h})")
            axes[0].axis('off')
            
            axes[1].imshow(disp_downsampled, cmap='gray' if len(downsampled.shape) < 3 else None)
            axes[1].set_title(f"Downsampled x{factor}\n{down_name}")
            axes[1].axis('off')
            
            axes[2].imshow(disp_upsampled, cmap='gray' if len(upsampled.shape) < 3 else None)
            axes[2].set_title(f"Upsampled\n{up_name}\nPSNR: {psnr_value:.2f}dB\nSSIM: {ssim_value:.4f}")
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f"comparison_down{down_id}_up{up_id}.png"), dpi=150)
            plt.close()
            
            # Store results
            results.append({
                'down_id': down_id,
                'down_name': down_name,
                'up_id': up_id,
                'up_name': up_name,
                'psnr': psnr_value,
                'ssim': ssim_value
            })
            
            # Check if best
            if psnr_value > best_psnr:
                best_psnr = psnr_value
                best_psnr_combo = (down_id, up_id)
            if ssim_value > best_ssim:
                best_ssim = ssim_value
                best_ssim_combo = (down_id, up_id)
    
    return results, best_psnr_combo, best_ssim_combo

def create_summary_plots(results, factor, results_dir):
    """Create summary plots for visualization."""
    plots_dir = os.path.join(results_dir, "plots")
    
    # Sort results by PSNR and SSIM
    results_by_psnr = sorted(results, key=lambda x: x['psnr'], reverse=True)
    results_by_ssim = sorted(results, key=lambda x: x['ssim'], reverse=True)
    
    # === Bar charts of all methods ===
    fig, axes = plt.subplots(2, 1, figsize=(12, 12))
    
    # Prepare data for bar charts
    labels = [f"{r['down_name']}\n+\n{r['up_name']}" for r in results_by_psnr]
    psnr_values = [r['psnr'] for r in results_by_psnr]
    ssim_values = [r['ssim'] for r in results_by_ssim]
    
    # Plot PSNR
    bars = axes[0].bar(range(len(psnr_values)), psnr_values, color='skyblue')
    axes[0].set_xticks(range(len(labels)))
    axes[0].set_xticklabels(labels, rotation=90)
    axes[0].set_title(f'PSNR Values for All Method Combinations (Factor {factor})')
    axes[0].set_ylabel('PSNR (dB)')
    for i, v in enumerate(psnr_values):
        axes[0].text(i, v + 0.5, f"{v:.2f}", ha='center')
    
    # Plot SSIM
    labels_ssim = [f"{r['down_name']}\n+\n{r['up_name']}" for r in results_by_ssim]
    bars = axes[1].bar(range(len(ssim_values)), ssim_values, color='lightgreen')
    axes[1].set_xticks(range(len(labels_ssim)))
    axes[1].set_xticklabels(labels_ssim, rotation=90)
    axes[1].set_title(f'SSIM Values for All Method Combinations (Factor {factor})')
    axes[1].set_ylabel('SSIM')
    for i, v in enumerate(ssim_values):
        axes[1].text(i, v + 0.01, f"{v:.4f}", ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"metrics_comparison_factor{factor}.png"), dpi=150)
    plt.close()
    
    # === Heatmap of PSNR and SSIM ===
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Create matrices for heatmaps
    psnr_matrix = np.zeros((len(DOWNSAMPLE_METHODS), len(UPSAMPLE_METHODS)))
    ssim_matrix = np.zeros((len(DOWNSAMPLE_METHODS), len(UPSAMPLE_METHODS)))
    
    for result in results:
        down_idx = result['down_id'] - 1  # 0-indexed
        up_idx = result['up_id'] - 1      # 0-indexed
        psnr_matrix[down_idx, up_idx] = result['psnr']
        ssim_matrix[down_idx, up_idx] = result['ssim']
    
    # PSNR heatmap
    im1 = axes[0].imshow(psnr_matrix, cmap='hot')
    axes[0].set_title(f'PSNR Heatmap (Factor {factor})')
    axes[0].set_xticks(range(len(UPSAMPLE_METHODS)))
    axes[0].set_yticks(range(len(DOWNSAMPLE_METHODS)))
    axes[0].set_xticklabels([f"{i+1}. {name}" for i, name in enumerate(UPSAMPLE_METHODS.values())])
    axes[0].set_yticklabels([f"{i+1}. {name}" for i, name in enumerate(DOWNSAMPLE_METHODS.values())])
    plt.colorbar(im1, ax=axes[0], label='PSNR (dB)')
    
    # SSIM heatmap
    im2 = axes[1].imshow(ssim_matrix, cmap='viridis')
    axes[1].set_title(f'SSIM Heatmap (Factor {factor})')
    axes[1].set_xticks(range(len(UPSAMPLE_METHODS)))
    axes[1].set_yticks(range(len(DOWNSAMPLE_METHODS)))
    axes[1].set_xticklabels([f"{i+1}. {name}" for i, name in enumerate(UPSAMPLE_METHODS.values())])
    axes[1].set_yticklabels([f"{i+1}. {name}" for i, name in enumerate(DOWNSAMPLE_METHODS.values())])
    plt.colorbar(im2, ax=axes[1], label='SSIM')
    
    # Add text annotations to heatmap
    for i in range(len(DOWNSAMPLE_METHODS)):
        for j in range(len(UPSAMPLE_METHODS)):
            axes[0].text(j, i, f"{psnr_matrix[i, j]:.1f}", ha="center", va="center", color="white")
            axes[1].text(j, i, f"{ssim_matrix[i, j]:.3f}", ha="center", va="center", color="black")
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"heatmap_metrics_factor{factor}.png"), dpi=150)
    plt.close()
    
    # Create summary CSV
    with open(os.path.join(results_dir, "results_summary.csv"), 'w') as f:
        f.write("Downsampling,Upsampling,PSNR,SSIM\n")
        for result in results:
            f.write(f"{result['down_name']},{result['up_name']},{result['psnr']:.4f},{result['ssim']:.4f}\n")

def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(
        description='Automatic image downsampling and interpolation analysis.',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('image_path', help='Path to the input image file')
    parser.add_argument('--factor', type=int, default=4, help='Downsampling factor (default: 4)')
    args = parser.parse_args()
    
    # Load image
    print(f"Loading image: {args.image_path}")
    original_image = cv2.imread(args.image_path, cv2.IMREAD_UNCHANGED)
    
    if original_image is None:
        print(f"Error: Could not load image {args.image_path}")
        sys.exit(1)
    
    # Handle alpha channel if present
    if original_image.ndim == 3 and original_image.shape[2] == 4:
        original_image = original_image[:,:,:3]
        print("Note: Alpha channel removed from input image")
    
    # Determine image type
    image_type = "Color" if original_image.ndim == 3 else "Grayscale"
    h, w = original_image.shape[:2]
    print(f"Loaded {image_type} image, {w}x{h} pixels")
    
    # Create results directory
    results_dir = create_results_dir()
    print(f"Results will be saved to: {results_dir}")
    
    # Process all combinations
    results, best_psnr, best_ssim = process_all_combinations(original_image, args.factor, results_dir)
    
    # Create summary plots
    create_summary_plots(results, args.factor, results_dir)
    
    # Print conclusion
    best_psnr_down = DOWNSAMPLE_METHODS[best_psnr[0]]
    best_psnr_up = UPSAMPLE_METHODS[best_psnr[1]]
    best_ssim_down = DOWNSAMPLE_METHODS[best_ssim[0]]
    best_ssim_up = UPSAMPLE_METHODS[best_ssim[1]]
    
    print("\n=== Results Summary ===")
    print(f"Downsampling factor: {args.factor}")
    print(f"Best PSNR combination: {best_psnr_down} + {best_psnr_up}")
    print(f"Best SSIM combination: {best_ssim_down} + {best_ssim_up}")
    print(f"All results saved to: {results_dir}")
    print("========================")

if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"\nProcessing completed in {time.time() - start_time:.2f} seconds")