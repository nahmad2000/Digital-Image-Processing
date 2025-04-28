import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from datetime import datetime


class LightingCorrection:
    """
    A class that provides methods for correcting uneven lighting in images.
    Implements both spatial domain (Gaussian blur) and frequency domain (homomorphic filtering) approaches.
    """
    
    def __init__(self, verbose=True):
        """Initialize the lighting correction object."""
        self.verbose = verbose
        
    def load_image(self, image_path):
        """Load an image from the specified path."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at: {image_path}")
            
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to read image at: {image_path}")
            
        # Convert to grayscale if it's a color image
        if len(img.shape) == 3:
            if self.verbose:
                print("Converting color image to grayscale")
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img
            
        return img_gray
    
    def spatial_correction(self, image, kernel_size=None, sigma=None):
        """
        Correct uneven lighting using a spatial domain approach with Gaussian blur.
        
        Args:
            image: Input grayscale image
            kernel_size: Size of Gaussian kernel (odd numbers only)
            sigma: Standard deviation for Gaussian kernel
            
        Returns:
            Tuple of (corrected_image, shading_estimate)
        """
        # Auto-determine kernel size if not specified (about 1/3 of image dimensions)
        if kernel_size is None:
            k_size = max(51, min(511, int(min(image.shape) / 3)))
            # Make sure it's odd
            kernel_size = (k_size if k_size % 2 == 1 else k_size + 1,
                          k_size if k_size % 2 == 1 else k_size + 1)
        
        # Auto-determine sigma if not specified
        if sigma is None:
            sigma = max(20, min(image.shape) / 12)
            
        if self.verbose:
            print(f"Spatial correction using kernel size {kernel_size} and sigma {sigma:.2f}")
            
        # Step 1: Estimate the shading using a Gaussian blur
        shading_estimate = cv2.GaussianBlur(image, kernel_size, sigma)
        
        # Step 2: Normalize the image by dividing by shading estimate
        # Add small constant to avoid division by zero
        normalized_image = cv2.divide(
            image.astype(np.float32), 
            shading_estimate.astype(np.float32) + 1e-5
        )
        
        # Step 3: Rescale intensities to 0-255
        normalized_image = cv2.normalize(
            normalized_image, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)
        
        return normalized_image, shading_estimate
    
    def frequency_correction(self, image, d0=10, gamma_l=0.3, gamma_h=1.2, 
                            apply_clahe=True, clahe_clip=2.0, clahe_tile=(8, 8)):
        """
        Correct uneven lighting using frequency domain approach (homomorphic filtering).
        
        Args:
            image: Input grayscale image
            d0: Gaussian filter cutoff frequency
            gamma_l: Low frequency gain (controls lighting removal)
            gamma_h: High frequency gain (controls texture enhancement)
            apply_clahe: Whether to apply CLAHE enhancement after filtering
            clahe_clip: CLAHE clip limit
            clahe_tile: CLAHE tile grid size
            
        Returns:
            Tuple of (corrected_image, illumination_component)
        """
        if self.verbose:
            print(f"Frequency correction using D0={d0}, γL={gamma_l}, γH={gamma_h}")
            
        # Convert image to float
        img_float = image.astype(np.float32)
        
        # Apply DFT
        log_img = np.log1p(img_float)  # Log transform (log(1+x) to avoid log(0))
        dft = np.fft.fft2(log_img)
        dft_shift = np.fft.fftshift(dft)
        
        # Create Gaussian low-pass filter
        rows, cols = image.shape
        crow, ccol = rows // 2, cols // 2
        u = np.arange(cols)
        v = np.arange(rows)
        U, V = np.meshgrid(u, v)
        D_sq = (U - ccol)**2 + (V - crow)**2
        glpf = np.exp(-D_sq / (2 * d0**2))
        
        # Modified Homomorphic filter
        H = (gamma_h - gamma_l) * (1 - glpf) + gamma_l
        
        # Apply filter
        reflect_fft = dft_shift * H
        illum_fft = dft_shift * glpf
        
        # Inverse DFT
        illum = np.real(np.fft.ifft2(np.fft.ifftshift(illum_fft)))
        reflect = np.real(np.fft.ifft2(np.fft.ifftshift(reflect_fft)))
        
        # Convert back from log domain
        illum_norm = cv2.normalize(np.expm1(illum), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        reflect_norm = cv2.normalize(np.expm1(reflect), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Apply CLAHE if requested
        if apply_clahe:
            clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_tile)
            reflect_norm = clahe.apply(reflect_norm)
            
        return reflect_norm, illum_norm
    
    def apply_correction(self, image_path, output_dir='results', method='frequency', 
                        params=None, show_plot=True, save_results=True):
        """
        Apply lighting correction to an image.
        
        Args:
            image_path: Path to input image
            output_dir: Directory to save results
            method: 'spatial', 'frequency', or 'both'
            params: Dictionary of parameters for the chosen method
            show_plot: Whether to display results
            save_results: Whether to save results to disk
            
        Returns:
            Dictionary of result images
        """
        # Create output directory if it doesn't exist
        if save_results and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Load image
        image = self.load_image(image_path)
        
        # Get base filename without extension
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Set default parameters if not provided
        if params is None:
            params = {}
            
        results = {'original': image}
        
        # Apply spatial correction if requested
        if method in ['spatial', 'both']:
            kernel_size = params.get('kernel_size', None)
            sigma = params.get('sigma', None)
            
            corrected, shading = self.spatial_correction(image, kernel_size, sigma)
            results['spatial_corrected'] = corrected
            results['spatial_shading'] = shading
            
            if save_results:
                cv2.imwrite(os.path.join(output_dir, f"{base_name}_spatial_corrected.png"), corrected)
                cv2.imwrite(os.path.join(output_dir, f"{base_name}_spatial_shading.png"), shading)
        
        # Apply frequency correction if requested
        if method in ['frequency', 'both']:
            d0 = params.get('d0', 10)
            gamma_l = params.get('gamma_l', 0.3)
            gamma_h = params.get('gamma_h', 1.2)
            apply_clahe = params.get('apply_clahe', True)
            clahe_clip = params.get('clahe_clip', 2.0)
            clahe_tile = params.get('clahe_tile', (8, 8))
            
            corrected, illumination = self.frequency_correction(
                image, d0, gamma_l, gamma_h, apply_clahe, clahe_clip, clahe_tile
            )
            results['frequency_corrected'] = corrected
            results['frequency_illumination'] = illumination
            
            if save_results:
                cv2.imwrite(os.path.join(output_dir, f"{base_name}_frequency_corrected.png"), corrected)
                cv2.imwrite(os.path.join(output_dir, f"{base_name}_frequency_illumination.png"), illumination)
        
        # Create comparison visualization
        if show_plot:
            self.visualize_results(results, method, params)
            
            if save_results:
                plt.savefig(os.path.join(output_dir, f"{base_name}_comparison.png"), dpi=300, bbox_inches='tight')
                
        return results
    
    def visualize_results(self, results, method, params):
        """
        Visualize the correction results.
        
        Args:
            results: Dictionary of result images
            method: Method used for correction
            params: Parameters used for correction
        """
        # Determine the layout based on the method
        if method == 'both':
            fig, axs = plt.subplots(2, 3, figsize=(15, 10))
            plt.subplots_adjust(wspace=0.05, hspace=0.2)
            
            # First row: original + spatial method
            axs[0, 0].imshow(results['original'], cmap='gray')
            axs[0, 0].set_title('Original Image')
            
            axs[0, 1].imshow(results['spatial_shading'], cmap='gray')
            axs[0, 1].set_title('Spatial - Shading Estimate')
            
            axs[0, 2].imshow(results['spatial_corrected'], cmap='gray')
            axs[0, 2].set_title('Spatial - Corrected')
            
            # Second row: empty + frequency method
            axs[1, 0].axis('off')
            
            axs[1, 1].imshow(results['frequency_illumination'], cmap='gray')
            axs[1, 1].set_title('Frequency - Illumination')
            
            axs[1, 2].imshow(results['frequency_corrected'], cmap='gray')
            axs[1, 2].set_title('Frequency - Corrected')
            
        else:
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            
            axs[0].imshow(results['original'], cmap='gray')
            axs[0].set_title('Original Image')
            
            if method == 'spatial':
                axs[1].imshow(results['spatial_shading'], cmap='gray')
                axs[1].set_title('Shading Estimate')
                
                axs[2].imshow(results['spatial_corrected'], cmap='gray')
                axs[2].set_title('Corrected Image')
                
            else:  # frequency
                axs[1].imshow(results['frequency_illumination'], cmap='gray')
                axs[1].set_title('Illumination Component')
                
                axs[2].imshow(results['frequency_corrected'], cmap='gray')
                axs[2].set_title('Corrected Image')
        
        # Add parameters to the title
        if method == 'spatial':
            kernel_size = params.get('kernel_size', 'auto')
            sigma = params.get('sigma', 'auto')
            title = f'Spatial Domain Correction - Kernel: {kernel_size}, Sigma: {sigma}'
            
        elif method == 'frequency':
            d0 = params.get('d0', 10)
            gamma_l = params.get('gamma_l', 0.3)
            gamma_h = params.get('gamma_h', 1.2)
            title = f'Frequency Domain Correction - D0: {d0}, γL: {gamma_l}, γH: {gamma_h}'
            
        else:
            title = 'Comparison of Spatial and Frequency Domain Methods'
            
        plt.suptitle(title, fontsize=16, y=0.98)
        
        for ax in axs.flat:
            ax.axis('off')
            
        plt.tight_layout()
        plt.show()


def main():
    # Set up command-line arguments
    parser = argparse.ArgumentParser(description='Lighting Correction for Images')
    
    parser.add_argument('image_path', help='Path to the input image')
    
    parser.add_argument('--method', type=str, default='frequency',
                        choices=['spatial', 'frequency', 'both'],
                        help='Correction method to use (default: frequency)')
    
    parser.add_argument('--output', '-o', type=str, default='results',
                        help='Directory to save results (default: results)')
    
    parser.add_argument('--no-plot', action='store_true',
                        help='Disable visualization plot')
    
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save results to disk')
    
    # Spatial method parameters
    parser.add_argument('--kernel-size', type=int,
                        help='Kernel size for Gaussian blur (spatial method)')
    
    parser.add_argument('--sigma', type=float,
                        help='Sigma value for Gaussian blur (spatial method)')
    
    # Frequency method parameters
    parser.add_argument('--d0', type=float, default=10,
                        help='Cutoff frequency for Gaussian filter (frequency method)')
    
    parser.add_argument('--gamma-l', type=float, default=0.3,
                        help='Low frequency gain (frequency method)')
    
    parser.add_argument('--gamma-h', type=float, default=1.2,
                        help='High frequency gain (frequency method)')
    
    parser.add_argument('--no-clahe', action='store_true',
                        help='Disable CLAHE enhancement (frequency method)')
    
    parser.add_argument('--clahe-clip', type=float, default=2.0,
                        help='CLAHE clip limit (frequency method)')
    
    parser.add_argument('--clahe-tile', type=int, default=8,
                        help='CLAHE tile size (frequency method)')
    
    args = parser.parse_args()
    
    # Create parameters dictionary
    params = {}
    
    # Spatial method parameters
    if args.kernel_size:
        # Convert to tuple of odd numbers
        k = args.kernel_size
        k = k if k % 2 == 1 else k + 1
        params['kernel_size'] = (k, k)
        
    if args.sigma:
        params['sigma'] = args.sigma
        
    # Frequency method parameters
    params['d0'] = args.d0
    params['gamma_l'] = args.gamma_l
    params['gamma_h'] = args.gamma_h
    params['apply_clahe'] = not args.no_clahe
    params['clahe_clip'] = args.clahe_clip
    params['clahe_tile'] = (args.clahe_tile, args.clahe_tile)
    
    # Create the lighting correction object
    corrector = LightingCorrection()
    
    # Process the image
    start_time = datetime.now()
    results = corrector.apply_correction(
        args.image_path,
        output_dir=args.output,
        method=args.method,
        params=params,
        show_plot=not args.no_plot,
        save_results=not args.no_save
    )
    end_time = datetime.now()
    
    print(f"Processing completed in {(end_time - start_time).total_seconds():.2f} seconds")
    
    
if __name__ == "__main__":
    main()