# -*- coding: utf-8 -*-
"""
Enhanced Image Downsampling and Interpolation using OpenCV and Matplotlib.

This improved script provides multiple downsampling techniques and interpolation
methods for both grayscale and color images, with anti-aliasing filters and
better quality options.

Usage:
    python Enhanced_Downsampling_Interpolation.py <path_to_your_image>

Example:
    python Enhanced_Downsampling_Interpolation.py images/my_color_photo.jpg
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse
import sys
import os
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def get_user_input():
    """
    Prompts the user to enter a valid downsampling factor and method.

    Returns:
        tuple: (downsampling_factor, downsample_method, interpolation_method)
    """
    print("\n=== Enhanced Image Downsampling & Interpolation (Color/Grayscale) ===")
    print("This program demonstrates advanced downsampling and interpolation on your image.")
    print("\n🔹 Downsampling Options:")
    print("   1. Simple (Pixel skipping - prone to aliasing)")
    print("   2. Anti-aliased (Gaussian pre-filtering)")
    print("   3. Area-based (OpenCV's optimized area interpolation)")
    print("\n🔹 Interpolation Options:")
    print("   1. Nearest Neighbor (Fast but blocky)")
    print("   2. Bilinear (Smoother than NN)")
    print("   3. Bicubic (Better quality but slower)")
    print("   4. Lanczos (High quality, preserves edges)")
    print("\nYou will now choose options for processing.")
    print("============================================")

    # Define valid factors
    valid_factors = [2, 4, 8, 16, 32, 64, 128, 256, 512]

    # Get downsampling factor
    while True:
        try:
            factor_str = ", ".join(map(str, valid_factors))
            downsampling_factor = int(input(f"\nChoose a downsampling factor [{factor_str}]: "))
            if downsampling_factor in valid_factors:
                break
            else:
                print(f"⚠️ Invalid choice! Please enter one of the following numbers: {factor_str}")
        except ValueError:
            print("⚠️ Invalid input! Please enter a valid number.")
    
    # Get downsampling method
    while True:
        try:
            downsample_method = int(input("\nChoose a downsampling method [1-3]: "))
            if 1 <= downsample_method <= 3:
                break
            else:
                print("⚠️ Invalid choice! Please enter 1, 2, or 3.")
        except ValueError:
            print("⚠️ Invalid input! Please enter a valid number.")
    
    # Get interpolation method
    while True:
        try:
            interpolation_method = int(input("\nChoose an interpolation method [1-4]: "))
            if 1 <= interpolation_method <= 4:
                break
            else:
                print("⚠️ Invalid choice! Please enter 1, 2, 3, or 4.")
        except ValueError:
            print("⚠️ Invalid input! Please enter a valid number.")
    
    return downsampling_factor, downsample_method, interpolation_method

def downsample_image(image, factor, method=1):
    """
    Downsamples an image using the specified method.
    
    Args:
        image (np.array): Input image
        factor (int): Downsampling factor
        method (int): 1=Simple, 2=Anti-aliased, 3=Area
        
    Returns:
        np.array: Downsampled image
    """
    if method == 1:
        # Simple downsampling (pixel skipping)
        return image[::factor, ::factor]
    
    elif method == 2:
        # Anti-aliased downsampling with Gaussian blur
        # Calculate sigma based on downsampling factor
        sigma = 0.3 * factor
        
        # Apply Gaussian blur before downsampling
        if len(image.shape) == 3:  # Color image
            blurred = cv2.GaussianBlur(image, (0, 0), sigma)
        else:  # Grayscale
            blurred = cv2.GaussianBlur(image, (0, 0), sigma)
            
        # Then apply simple downsampling
        return blurred[::factor, ::factor]
    
    elif method == 3:
        # Area-based downsampling (OpenCV's optimized implementation)
        if len(image.shape) == 3:  # Color image
            h, w, _ = image.shape
        else:  # Grayscale
            h, w = image.shape
            
        new_size = (w // factor, h // factor)
        return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    
    else:
        raise ValueError("Invalid downsampling method")

def get_interpolation_method(method_id):
    """
    Returns the OpenCV interpolation constant based on the method ID.
    
    Args:
        method_id (int): 1=NN, 2=Bilinear, 3=Bicubic, 4=Lanczos
        
    Returns:
        int: OpenCV interpolation constant
    """
    interpolation_methods = {
        1: cv2.INTER_NEAREST,
        2: cv2.INTER_LINEAR,
        3: cv2.INTER_CUBIC,
        4: cv2.INTER_LANCZOS4
    }
    
    return interpolation_methods.get(method_id, cv2.INTER_LINEAR)

def get_method_name(method_id, is_downsampling=True):
    """
    Returns the human-readable name of the method.
    
    Args:
        method_id (int): Method ID
        is_downsampling (bool): True if downsampling method, False if interpolation
        
    Returns:
        str: Method name
    """
    if is_downsampling:
        method_names = {
            1: "Simple (Pixel Skipping)",
            2: "Anti-aliased (Gaussian)",
            3: "Area-based"
        }
    else:
        method_names = {
            1: "Nearest Neighbor",
            2: "Bilinear",
            3: "Bicubic",
            4: "Lanczos"
        }
    
    return method_names.get(method_id, "Unknown")

def calculate_metrics(original, reconstructed):
    """
    Calculates image quality metrics between original and reconstructed image.
    
    Args:
        original (np.array): Original image
        reconstructed (np.array): Reconstructed image
        
    Returns:
        tuple: (PSNR value, SSIM value)
    """
    # Convert BGR to grayscale for metric calculation if needed
    if len(original.shape) == 3 and original.shape[2] == 3:
        original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        reconstructed_gray = cv2.cvtColor(reconstructed, cv2.COLOR_BGR2GRAY)
    else:
        original_gray = original
        reconstructed_gray = reconstructed
    
    # Calculate PSNR
    psnr_value = psnr(original_gray, reconstructed_gray)
    
    # Calculate SSIM
    ssim_value = ssim(original_gray, reconstructed_gray)
    
    return psnr_value, ssim_value

def display_images(original, downsampled, upsampled, metrics=None):
    """
    Displays the original, downsampled, and upsampled images.
    
    Args:
        original (np.array): Original image
        downsampled (np.array): Downsampled image
        upsampled (np.array): Upsampled image
        metrics (tuple, optional): (PSNR, SSIM) values
    """
    plt.figure(figsize=(15, 5))
    
    # Display original
    plt.subplot(1, 3, 1)
    if len(original.shape) == 3 and original.shape[2] == 3:
        plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(original, cmap='gray')
    plt.title(f"Original ({original.shape[1]}x{original.shape[0]})")
    plt.axis("off")
    
    # Display downsampled
    plt.subplot(1, 3, 2)
    if len(downsampled.shape) == 3 and downsampled.shape[2] == 3:
        plt.imshow(cv2.cvtColor(downsampled, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(downsampled, cmap='gray')
    plt.title(f"Downsampled ({downsampled.shape[1]}x{downsampled.shape[0]})")
    plt.axis("off")
    
    # Display upsampled with metrics
    plt.subplot(1, 3, 3)
    if len(upsampled.shape) == 3 and upsampled.shape[2] == 3:
        plt.imshow(cv2.cvtColor(upsampled, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(upsampled, cmap='gray')
    
    title = f"Upsampled ({upsampled.shape[1]}x{upsampled.shape[0]})"
    if metrics:
        title += f"\nPSNR: {metrics[0]:.2f}dB, SSIM: {metrics[1]:.4f}"
    plt.title(title)
    plt.axis("off")
    
    plt.tight_layout(pad=1.5)
    plt.show()

def save_images(original, downsampled, upsampled, factor, down_method, up_method, metrics=None, results_dir="results"):
    """
    Saves the processed images into a specified directory.
    
    Args:
        original (np.array): Original image
        downsampled (np.array): Downsampled image
        upsampled (np.array): Upsampled image
        factor (int): Downsampling factor
        down_method (int): Downsampling method ID
        up_method (int): Interpolation method ID
        metrics (tuple, optional): (PSNR, SSIM) values
        results_dir (str): Output directory
    """
    # Create the results directory if it doesn't exist
    try:
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
            print(f"📁 Created directory: '{results_dir}'")
    except OSError as e:
        print(f"❌ Error creating directory '{results_dir}': {e}")
        print("   Saving images in the current directory instead.")
        results_dir = "."
    
    # Generate base filenames
    down_method_name = get_method_name(down_method, True).replace(" ", "_")
    up_method_name = get_method_name(up_method, False).replace(" ", "_")
    
    # Define filenames
    original_filename = "Original.jpg"
    downsampled_filename = f"Downsampled_x{factor}_{down_method_name}.jpg"
    upsampled_filename = f"Upsampled_x{factor}_{down_method_name}_{up_method_name}.jpg"
    
    # Create comparison image with metrics
    if metrics:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Display original
        if len(original.shape) == 3 and original.shape[2] == 3:
            axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        else:
            axes[0].imshow(original, cmap='gray')
        axes[0].set_title(f"Original ({original.shape[1]}x{original.shape[0]})")
        axes[0].axis("off")
        
        # Display downsampled
        if len(downsampled.shape) == 3 and downsampled.shape[2] == 3:
            axes[1].imshow(cv2.cvtColor(downsampled, cv2.COLOR_BGR2RGB))
        else:
            axes[1].imshow(downsampled, cmap='gray')
        axes[1].set_title(f"Downsampled\n{down_method_name} (x{factor})")
        axes[1].axis("off")
        
        # Display upsampled with metrics
        if len(upsampled.shape) == 3 and upsampled.shape[2] == 3:
            axes[2].imshow(cv2.cvtColor(upsampled, cv2.COLOR_BGR2RGB))
        else:
            axes[2].imshow(upsampled, cmap='gray')
        
        title = f"Upsampled\n{up_method_name}"
        title += f"\nPSNR: {metrics[0]:.2f}dB, SSIM: {metrics[1]:.4f}"
        axes[2].set_title(title)
        axes[2].axis("off")
        
        plt.tight_layout(pad=1.5)
        comparison_filename = f"Comparison_x{factor}_{down_method_name}_{up_method_name}.png"
        comparison_path = os.path.join(results_dir, comparison_filename)
        plt.savefig(comparison_path, dpi=150)
        plt.close()
    
    # Construct full paths
    original_path = os.path.join(results_dir, original_filename)
    downsampled_path = os.path.join(results_dir, downsampled_filename)
    upsampled_path = os.path.join(results_dir, upsampled_filename)
    
    # Save the images
    try:
        cv2.imwrite(original_path, original)
        cv2.imwrite(downsampled_path, downsampled)
        cv2.imwrite(upsampled_path, upsampled)
        
        print(f"\n✅ Images saved successfully in '{results_dir}' folder!")
        print(f"   - '{original_filename}'")
        print(f"   - '{downsampled_filename}'")
        print(f"   - '{upsampled_filename}'")
        if metrics:
            print(f"   - '{comparison_filename}'")
    except Exception as e:
        print(f"\n❌ Error saving images: {e}")

def generate_comparison_grid(original, factor, results_dir="results/comparison"):
    """
    Generates a grid comparing all downsampling and interpolation methods.
    
    Args:
        original (np.array): Original image
        factor (int): Downsampling factor
        results_dir (str): Output directory
    """
    # Create the comparison directory if it doesn't exist
    try:
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
            print(f"📁 Created directory: '{results_dir}'")
    except OSError as e:
        print(f"❌ Error creating directory '{results_dir}': {e}")
        return
    
    # Set up the figure
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle(f"Comparison of All Methods (Downsampling Factor: {factor})", fontsize=16)
    
    # Original size
    if len(original.shape) == 3:  # Color image
        h, w, _ = original.shape
    else:  # Grayscale
        h, w = original.shape
    original_dimensions_cv = (w, h)
    
    # Track best method
    best_psnr = -float('inf')
    best_ssim = -float('inf')
    best_psnr_methods = None
    best_ssim_methods = None
    
    # Create a table for results
    results_table = {
        'Downsampling': [],
        'Interpolation': [],
        'PSNR': [],
        'SSIM': []
    }
    
    # Iterate through all method combinations
    for down_method in range(1, 4):  # 3 downsampling methods
        down_method_name = get_method_name(down_method, True)
        
        # Downsample the image
        downsampled = downsample_image(original, factor, down_method)
        
        for up_method in range(1, 5):  # 4 interpolation methods
            up_method_name = get_method_name(up_method, False)
            
            # Upsample the image
            interpolation = get_interpolation_method(up_method)
            upsampled = cv2.resize(downsampled, original_dimensions_cv, interpolation=interpolation)
            
            # Calculate metrics
            psnr_value, ssim_value = calculate_metrics(original, upsampled)
            
            # Update best methods
            if psnr_value > best_psnr:
                best_psnr = psnr_value
                best_psnr_methods = (down_method, up_method)
            
            if ssim_value > best_ssim:
                best_ssim = ssim_value
                best_ssim_methods = (down_method, up_method)
            
            # Add to results table
            results_table['Downsampling'].append(down_method_name)
            results_table['Interpolation'].append(up_method_name)
            results_table['PSNR'].append(f"{psnr_value:.2f}")
            results_table['SSIM'].append(f"{ssim_value:.4f}")
            
            # Plot the result
            row = down_method - 1
            col = up_method - 1
            
            if len(upsampled.shape) == 3 and upsampled.shape[2] == 3:
                axes[row, col].imshow(cv2.cvtColor(upsampled, cv2.COLOR_BGR2RGB))
            else:
                axes[row, col].imshow(upsampled, cmap='gray')
            
            title = f"Down: {down_method_name}\nUp: {up_method_name}"
            title += f"\nPSNR: {psnr_value:.2f}dB\nSSIM: {ssim_value:.4f}"
            axes[row, col].set_title(title)
            axes[row, col].axis("off")
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust to fit the suptitle
    
    # Save the comparison grid
    comparison_path = os.path.join(results_dir, f"Method_Comparison_Grid_x{factor}.png")
    plt.savefig(comparison_path, dpi=200)
    plt.close()
    
    # Create a results table figure
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Create the table
    cell_text = []
    for i in range(len(results_table['Downsampling'])):
        cell_text.append([
            results_table['Downsampling'][i],
            results_table['Interpolation'][i],
            results_table['PSNR'][i],
            results_table['SSIM'][i]
        ])
    
    # Highlight best results
    cell_colors = []
    for i in range(len(results_table['Downsampling'])):
        row_colors = ['white', 'white', 'white', 'white']
        down_method = results_table['Downsampling'][i]
        up_method = results_table['Interpolation'][i]
        
        # Check if this is the best PSNR method
        if (get_method_name(best_psnr_methods[0], True) == down_method and 
            get_method_name(best_psnr_methods[1], False) == up_method):
            row_colors[2] = 'lightgreen'
        
        # Check if this is the best SSIM method
        if (get_method_name(best_ssim_methods[0], True) == down_method and 
            get_method_name(best_ssim_methods[1], False) == up_method):
            row_colors[3] = 'lightgreen'
        
        cell_colors.append(row_colors)
    
    table = ax.table(
        cellText=cell_text,
        colLabels=['Downsampling', 'Interpolation', 'PSNR (dB)', 'SSIM'],
        cellColours=cell_colors,
        loc='center'
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.2)
    
    plt.title(f"Quality Metrics for Factor {factor} (Highlighted = Best)", fontsize=14)
    
    # Save the results table
    results_path = os.path.join(results_dir, f"Quality_Metrics_x{factor}.png")
    plt.savefig(results_path, dpi=150)
    plt.close()
    
    print(f"\n✅ Comparison grid and metrics saved in '{results_dir}'")
    print(f"📊 Best PSNR ({best_psnr:.2f}dB): {get_method_name(best_psnr_methods[0], True)} + {get_method_name(best_psnr_methods[1], False)}")
    print(f"📊 Best SSIM ({best_ssim:.4f}): {get_method_name(best_ssim_methods[0], True)} + {get_method_name(best_ssim_methods[1], False)}")
    
    return best_psnr_methods, best_ssim_methods

# === Main Program Execution ===
if __name__ == "__main__":
    # --- Setup Argument Parser ---
    parser = argparse.ArgumentParser(
        description='Enhanced image downsampling and interpolation using OpenCV (Color/Grayscale).',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        'image_path',
        type=str,
        help='Path to the input image file.'
    )
    parser.add_argument(
        '--compare-all',
        action='store_true',
        help='Generate a comparison of all method combinations.'
    )
    args = parser.parse_args()
    
    # --- Load Image ---
    print(f"⏳ Loading image: {args.image_path}")
    original_image = cv2.imread(args.image_path, cv2.IMREAD_UNCHANGED)
    
    # Check if image is loaded successfully
    if original_image is None:
        print(f"❌ Error: Image not found or unable to read: {args.image_path}")
        sys.exit(1)
    
    # Determine if image is color or grayscale
    if original_image.ndim == 3 and original_image.shape[2] >= 3:
        # Keep only first 3 channels if 4 (e.g., RGBA)
        if original_image.shape[2] == 4:
            original_image = original_image[:,:,:3]
            print("   (Note: Alpha channel discarded from input image)")
        image_type = "Color (BGR)"
        original_height, original_width, _ = original_image.shape
    elif original_image.ndim == 2:
        image_type = "Grayscale"
        original_height, original_width = original_image.shape
    else:
        print(f"❌ Error: Unsupported image format/dimensions: {original_image.shape}")
        sys.exit(1)
    
    print(f"✅ Image loaded successfully ({image_type}). Dimensions: {original_width}x{original_height}")
    
    # Store original dimensions for upsampling target size (W, H for OpenCV)
    original_dimensions_cv = (original_width, original_height)
    
    if args.compare_all:
        # Generate comparison of all methods
        print("\n⏳ Generating comparison of all method combinations...")
        factor = get_user_input()[0]  # Only need the factor for this mode
        best_methods = generate_comparison_grid(original_image, factor)
        
        # Use the best methods to generate a high-quality result
        best_psnr_methods, best_ssim_methods = best_methods
        
        # Ask user which metric to prioritize
        print("\nBased on the comparison, would you like to use:")
        print(f"1. Best PSNR method ({get_method_name(best_psnr_methods[0], True)} + {get_method_name(best_psnr_methods[1], False)})")
        print(f"2. Best SSIM method ({get_method_name(best_ssim_methods[0], True)} + {get_method_name(best_ssim_methods[1], False)})")
        
        while True:
            try:
                choice = int(input("\nEnter your choice [1-2]: "))
                if 1 <= choice <= 2:
                    break
                else:
                    print("⚠️ Invalid choice! Please enter 1 or 2.")
            except ValueError:
                print("⚠️ Invalid input! Please enter 1 or 2.")
        
        if choice == 1:
            down_method, up_method = best_psnr_methods
        else:
            down_method, up_method = best_ssim_methods
    else:
        # Interactive mode
        factor, down_method, up_method = get_user_input()
    
    # --- Process the image ---
    print(f"\n⏳ Downsampling using {get_method_name(down_method, True)}...")
    downsampled_image = downsample_image(original_image, factor, down_method)
    print(f"✅ Downsampling complete. New dimensions: {downsampled_image.shape[1]}x{downsampled_image.shape[0]}")
    
    print(f"\n⏳ Upsampling using {get_method_name(up_method, False)}...")
    interpolation = get_interpolation_method(up_method)
    upsampled_image = cv2.resize(downsampled_image, original_dimensions_cv, interpolation=interpolation)
    print("✅ Upsampling complete.")
    
    # --- Calculate quality metrics ---
    print("\n⏳ Calculating quality metrics...")
    psnr_value, ssim_value = calculate_metrics(original_image, upsampled_image)
    print(f"✅ PSNR: {psnr_value:.2f}dB, SSIM: {ssim_value:.4f}")
    
    # --- Display Results ---
    print("\n⏳ Displaying images...")
    display_images(original_image, downsampled_image, upsampled_image, (psnr_value, ssim_value))
    
    # --- Save Results ---
    save_images(
        original_image, 
        downsampled_image, 
        upsampled_image, 
        factor, 
        down_method, 
        up_method, 
        (psnr_value, ssim_value)
    )
    
    # --- Final Notes ---
    print("\n💡 Notes:")
    print("   - Higher PSNR and SSIM values indicate better image quality")
    print("   - Area-based downsampling often yields better results for photographs")
    print("   - Bicubic and Lanczos typically produce sharper upsampled images")
    print("   - Check the saved images in the 'results' folder for true resolution")
    print("\n✅ Processing finished.")