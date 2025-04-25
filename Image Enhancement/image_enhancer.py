import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse
import os
from typing import Tuple, List, Optional


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments for the image enhancement script.
    
    Returns:
        argparse.Namespace: The parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description='Image Enhancement Tool - Apply power law transformation and histogram equalization')
    
    parser.add_argument('--image', '-i', type=str, required=False,
                        help='Path to the input image file')
    parser.add_argument('--gamma', '-g', type=float, 
                        help='Gamma value for power law transformation (recommended: 0 < γ < 1 for brightening)')
    parser.add_argument('--constant', '-c', type=float, default=1.0,
                        help='Constant value for power law transformation (default: 1.0)')
    parser.add_argument('--output', '-o', type=str, default='enhanced_images',
                        help='Output directory for saving enhanced images (default: enhanced_images)')
    parser.add_argument('--no-display', action='store_true',
                        help='Do not display images and plots (useful for batch processing)')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save the enhanced images')
    
    return parser.parse_args()


def get_user_inputs() -> Tuple[float, float]:
    """
    Prompts user for gamma (γ) and c values for Power Law Transformation, ensuring valid input.
    
    Returns:
        Tuple[float, float]: A tuple containing the gamma and c values.
    """
    print("\n=== Image Enhancement Parameters ===")
    print("Your image appears dark, making details hard to see. We will apply both enhancements:")
    print("\n1️⃣ Power Law Transformation (Gamma Correction)")
    print("   - Formula: s = c * r^γ")
    print("   - If γ < 1 → The image becomes brighter.")
    print("   - If γ > 1 → The image becomes darker.")
    print("\n2️⃣ Histogram Equalization")
    print("   - Automatically adjusts contrast by redistributing pixel intensities.")
    print("\n=================================")

    while True:
        try:
            gamma = float(input("\nEnter γ value (Recommended: 0 < γ < 1 for brightening): "))
            if gamma > 0:
                while True:
                    try:
                        c = float(input("Enter c value (Recommended: c > 0, typically 1): "))
                        if c > 0:
                            return gamma, c
                        else:
                            print("⚠️ Invalid input! c must be greater than 0.")
                    except ValueError:
                        print("⚠️ Invalid input! Please enter a numeric value for c.")
            else:
                print("⚠️ Invalid input! γ must be greater than 0.")
        except ValueError:
            print("⚠️ Invalid input! Please enter a numeric value for γ.")


def compute_pdf(image: np.ndarray) -> np.ndarray:
    """
    Computes and returns the Probability Density Function (PDF) of an image.
    
    Args:
        image (np.ndarray): Input grayscale image.
    
    Returns:
        np.ndarray: The PDF of the image.
    """
    # More efficient histogram calculation using numpy
    hist, _ = np.histogram(image.flatten(), bins=256, range=[0, 256])
    pdf = hist / np.sum(hist)
    return pdf


def power_law_transformation(image: np.ndarray, gamma: float, c: float) -> np.ndarray:
    """
    Applies Power Law (Gamma) Transformation to an image.
    
    Args:
        image (np.ndarray): Input grayscale image.
        gamma (float): Gamma value for the power law transformation.
        c (float): Constant multiplier for the power law transformation.
    
    Returns:
        np.ndarray: The power law transformed image.
    """
    # Normalize image to [0, 1]
    normalized_image = image.astype(np.float32) / 255.0
    
    # Apply power law transformation
    power_image = c * np.power(normalized_image, gamma)
    
    # Scale back to [0, 255] and convert to uint8
    power_image = np.clip(power_image * 255, 0, 255).astype(np.uint8)
    
    return power_image


def histogram_equalization(image: np.ndarray) -> np.ndarray:
    """
    Applies Histogram Equalization to an image.
    
    Args:
        image (np.ndarray): Input grayscale image.
    
    Returns:
        np.ndarray: The histogram equalized image.
    """
    # Compute PDF and CDF
    pdf = compute_pdf(image)
    cdf = np.cumsum(pdf)
    
    # Scale CDF to [0, 255]
    cdf_normalized = np.round(cdf * 255).astype(np.uint8)
    
    # Apply mapping to create enhanced image
    return cdf_normalized[image]


def plot_pdfs(images: List[np.ndarray], titles: List[str], show_display: bool = True, output_dir: Optional[str] = None) -> None:
    """
    Plots the PDFs of one or multiple images.
    
    Args:
        images (List[np.ndarray]): List of images to plot PDFs for.
        titles (List[str]): List of titles for each PDF plot.
        show_display (bool): Whether to display the plots.
        output_dir (Optional[str]): Directory to save the plot. If None, the plot is not saved.
    """
    plt.figure(figsize=(15, 5))

    for i, (image, title) in enumerate(zip(images, titles), 1):
        pdf = compute_pdf(image)
        plt.subplot(1, len(images), i)
        plt.bar(range(256), pdf, color='black')
        plt.title(title)
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Probability")
        plt.xlim([0, 255])  # Keep x-axis range consistent

    plt.tight_layout()
    
    # Save the plot if output_dir is provided
    if output_dir is not None:
        pdf_plot_path = os.path.join(output_dir, "pdf_plots.png")
        plt.savefig(pdf_plot_path, dpi=300, bbox_inches='tight')
        print(f"PDF plots saved as '{pdf_plot_path}'")
    
    # Show the plot if show_display is True
    if show_display:
        plt.show()
    else:
        plt.close()


def display_images(original: np.ndarray, power: np.ndarray, hist: np.ndarray, power_hist: np.ndarray, 
                  show_display: bool = True, output_dir: Optional[str] = None) -> None:
    """
    Displays the original and enhanced images.
    
    Args:
        original (np.ndarray): Original image.
        power (np.ndarray): Power law transformed image.
        hist (np.ndarray): Histogram equalized image.
        power_hist (np.ndarray): Power law + histogram equalized image.
        show_display (bool): Whether to display the images.
        output_dir (Optional[str]): Directory to save the comparison image. If None, the image is not saved.
    """
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.imshow(original, cmap='gray')
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.imshow(power, cmap='gray')
    plt.title("Power Law Transformation")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.imshow(hist, cmap='gray')
    plt.title("Histogram Equalization")
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.imshow(power_hist, cmap='gray')
    plt.title("Power Law >> Histogram")
    plt.axis("off")

    plt.tight_layout()
    
    # Save the comparison image if output_dir is provided
    if output_dir is not None:
        comparison_path = os.path.join(output_dir, "comparison.png")
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        print(f"Comparison image saved as '{comparison_path}'")
    
    # Show the images if show_display is True
    if show_display:
        plt.show()
    else:
        plt.close()


def save_images(power: np.ndarray, hist: np.ndarray, power_hist: np.ndarray, output_dir: str) -> None:
    """
    Saves the transformed images.
    
    Args:
        power (np.ndarray): Power law transformed image.
        hist (np.ndarray): Histogram equalized image.
        power_hist (np.ndarray): Power law + histogram equalized image.
        output_dir (str): Directory to save the images.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    power_path = os.path.join(output_dir, "power_image.jpg")
    hist_path = os.path.join(output_dir, "histogram_image.jpg")
    power_hist_path = os.path.join(output_dir, "power_histogram_image.jpg")
    
    cv2.imwrite(power_path, power)
    cv2.imwrite(hist_path, hist)
    cv2.imwrite(power_hist_path, power_hist)
    
    print("\n✅ Images saved successfully!")
    print(f"Power Image saved as '{power_path}'")
    print(f"Histogram Image saved as '{hist_path}'")
    print(f"Power Histogram Image saved as '{power_hist_path}'")


def select_image() -> str:
    """
    Prompts the user to select an image file.
    
    Returns:
        str: The path to the selected image file.
    """
    print("\n=== Image Selection ===")
    print("Please enter the path to your image file:")
    
    while True:
        image_path = input("Image path: ").strip()
        
        # Remove quotes if the user included them
        if (image_path.startswith('"') and image_path.endswith('"')) or \
           (image_path.startswith("'") and image_path.endswith("'")):
            image_path = image_path[1:-1]
        
        if os.path.isfile(image_path):
            return image_path
        else:
            print(f"⚠️ File not found: '{image_path}'. Please enter a valid file path.")


def main() -> None:
    """Main function to run the image enhancement process."""
    # Parse command line arguments
    args = parse_arguments()
    
    # If no image path is provided via command line, prompt the user to select an image
    image_path = args.image if args.image else select_image()
    
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"❌ Error: Failed to load image '{image_path}'. Please check if the file exists and is a valid image.")
        return
    
    # Display original image if show_display is True
    if not args.no_display:
        plt.figure(figsize=(6, 6))
        plt.imshow(image, cmap='gray')
        plt.title("Original Image")
        plt.axis("off")
        plt.show()
    
    # If gamma is not provided via command line, ask for user input
    if args.gamma is None:
        gamma, c = get_user_inputs()
    else:
        gamma = args.gamma
        c = args.constant
        print(f"\nUsing provided parameters: γ = {gamma}, c = {c}")
    
    # Apply transformations
    print("\n⏳ Applying transformations...")
    power_image = power_law_transformation(image, gamma, c)
    histogram_image = histogram_equalization(image)
    power_histogram_image = histogram_equalization(power_image)
    print("✅ Transformations applied successfully!")
    
    # Save images if no_save is False
    if not args.no_save:
        save_images(power_image, histogram_image, power_histogram_image, args.output)
    
    # Display images and plots if no_display is False
    if not args.no_display:
        display_images(image, power_image, histogram_image, power_histogram_image, True, args.output if not args.no_save else None)
        
        # Plot PDFs
        images = [image, histogram_image, power_histogram_image]
        titles = ["PDF of Original Image", "PDF of Histogram Equalized Image", "PDF of Power Law >> Histogram"]
        plot_pdfs(images, titles, True, args.output if not args.no_save else None)
    
    print("\n🎉 Image enhancement completed successfully!")


if __name__ == "__main__":
    main()