# -*- coding: utf-8 -*-

"""
Optimized Image Compression Analyzer

This script analyzes and compares different image compression formats (PNG, JPEG, WebP)
using multiple quality metrics and provides visualization of results.
It incorporates I/O optimizations and parallel processing.
"""

import os
import io  # Added for in-memory buffer
import argparse
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union
import concurrent.futures # Added for parallel processing
from functools import partial # Added for passing arguments in parallel processing

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
# Ignore specific warnings from PIL about potentially large images
Image.MAX_IMAGE_PIXELS = None # Disable decompression bomb check if needed, use with caution
from skimage import metrics
from tqdm import tqdm

# --- Dataclass for Results ---
@dataclass
class CompressionResult:
    """Class to store compression results for an image"""
    format: str
    compression_ratio: float
    mse: float
    psnr: float
    ssim: float
    file_size: int

# --- Worker Function for Parallel Processing ---
# Defined outside the class to potentially improve pickling reliability
def _process_single_image(image_path: Path,
                           output_dir: Path,
                           quality: int,
                           formats: List[str],
                           log_level: int) -> Tuple[str, Dict[str, CompressionResult]]:
    """
    Processes a single image: compresses to specified formats, calculates metrics.
    Designed to be run in parallel.

    Args:
        image_path: Path to the original image.
        output_dir: Directory to save compressed images.
        quality: Quality level for lossy compression.
        formats: List of formats to test.
        log_level: Logging level to configure logger in the subprocess (currently unused).

    Returns:
        A tuple containing (image_name, dictionary_of_results_for_this_image).
        Returns an empty dictionary if processing fails.
    """
    image_name = image_path.stem
    results_for_image: Dict[str, CompressionResult] = {}
    # Optional: Configure logging within the process if needed.
    # logger = logging.getLogger(f"worker_{os.getpid()}")
    # logger.setLevel(log_level)
    # Simple print for errors in worker is used here instead.
    # print(f"Processing {image_name} in process {os.getpid()}") # Uncomment for debug

    try:
        original_image = Image.open(image_path)
        original_size = image_path.stat().st_size

        # Convert original image to numpy array ONCE
        # Convert to a standard format like RGB for consistent metric calculation.
        # Handle potential conversion issues (e.g., palette modes) safely.
        try:
            # Use 'RGB' for metrics consistency. If alpha is important, use 'RGBA'
            # but ensure metrics handle the 4th channel correctly.
            original_array = np.array(original_image.convert('RGB'))
            original_image_mode = 'RGB' # Track mode used for array comparison
        except Exception as e:
            # Log error in worker process (simple print)
            print(f"[Worker PID: {os.getpid()}] Error: Could not convert original image {image_name} to array: {e}")
            original_image.close() # Ensure file handle is closed
            return image_name, {} # Fail this image if conversion fails

        for fmt in formats:
            compressed_image_obj: Optional[Image.Image] = None
            compressed_size: int = 0
            compressed_path: Optional[Path] = None

            # --- Compression Step (using buffer) ---
            try:
                buffer = io.BytesIO()
                save_options = {}
                output_filename = f"{image_name}_compressed.{fmt}"
                compressed_path = output_dir / output_filename

                # Determine format-specific save options and image object to save
                fmt_upper = fmt.upper()
                save_image = original_image # Start with original

                if fmt_upper in ['JPG', 'JPEG']:
                    # JPEG doesn't support alpha, convert if necessary
                    if save_image.mode == 'RGBA' or save_image.mode == 'P': # Handle RGBA and Palette
                        save_image = save_image.convert('RGB')
                    save_options = {'format': 'JPEG', 'quality': quality}
                elif fmt_upper == 'WEBP':
                    # WebP supports RGBA, use original unless it needs conversion
                    save_options = {'format': 'WEBP', 'quality': quality}
                elif fmt_upper == 'PNG':
                    # PNG supports various modes including RGBA, L, P
                    save_options = {'format': 'PNG', 'optimize': True}
                else:
                    # This case should not be reached due to format validation in main()
                    print(f"[Worker PID: {os.getpid()}] Warning: Unsupported format {fmt} for {image_name}, skipping.")
                    continue

                # Save to buffer
                save_image.save(buffer, **save_options)
                compressed_size = buffer.tell()

                # Save to disk from buffer (essential part of the original script's goal)
                buffer.seek(0)
                with open(compressed_path, 'wb') as f:
                    f.write(buffer.read())

                # Rewind buffer and load image back from buffer for metrics
                buffer.seek(0)
                compressed_image_obj = Image.open(buffer)
                # Keep buffer open until compressed_image_obj is processed

            except Exception as e:
                print(f"[Worker PID: {os.getpid()}] Error: Failed to compress {image_name} to {fmt}: {e}")
                continue # Skip this format if compression fails

            # --- Metrics Calculation Step ---
            if compressed_image_obj and compressed_size > 0:
                try:
                    # Ensure compressed image matches original mode for comparison ('RGB' used above)
                    if compressed_image_obj.mode != original_image_mode:
                        compressed_image_converted = compressed_image_obj.convert(original_image_mode)
                    else:
                        compressed_image_converted = compressed_image_obj

                    compressed_array = np.array(compressed_image_converted)

                    # Handle potentially different image shapes after compression/conversion
                    if original_array.shape != compressed_array.shape:
                        print(f"[Worker PID: {os.getpid()}] Warning: Shapes mismatch for {image_name} ({fmt}): "
                              f"{original_array.shape} vs {compressed_array.shape}. Resizing compressed.")
                        # Resize compressed image to match original dimensions using PIL
                        # PIL resize expects (width, height) which corresponds to (shape[1], shape[0])
                        compressed_image_resized = compressed_image_converted.resize(
                            (original_array.shape[1], original_array.shape[0]),
                            resample=Image.Resampling.LANCZOS # Use a high-quality resampler
                        )
                        compressed_array = np.array(compressed_image_resized)

                        # Double-check shape after resize
                        if original_array.shape != compressed_array.shape:
                             print(f"[Worker PID: {os.getpid()}] Error: Shape mismatch persisted after resize for {image_name} ({fmt}). Skipping metrics.")
                             continue # Skip metrics if still mismatched

                    # --- Calculate Metrics ---
                    # Assuming 8-bit images (0-255 range) after conversion to RGB.
                    # Adjust data_range if using higher bit depth images.
                    data_range = 255.0

                    compression_ratio = original_size / compressed_size if compressed_size > 0 else 0

                    # MSE
                    mse = metrics.mean_squared_error(original_array, compressed_array)

                    # PSNR - Handle potential division by zero if MSE is 0 (identical images)
                    if mse == 0:
                        psnr = float('inf') # Or a very large number
                    else:
                        psnr = metrics.peak_signal_noise_ratio(original_array, compressed_array, data_range=data_range)

                    # SSIM - Requires channel axis for multi-channel images
                    # Determine if multichannel based on array dimensions
                    is_multichannel = compressed_array.ndim == 3 and compressed_array.shape[2] > 1
                    channel_axis_param = 2 if is_multichannel else None # Axis 2 for (H, W, C)

                    # Use channel_axis (newer skimage) or multichannel (older)
                    try:
                        ssim = metrics.structural_similarity(
                            original_array, compressed_array,
                            channel_axis=channel_axis_param,
                            data_range=data_range
                        )
                    except TypeError:
                         # Fallback for older skimage versions if channel_axis is not recognized
                         try:
                            ssim = metrics.structural_similarity(
                                original_array, compressed_array,
                                multichannel=is_multichannel,
                                data_range=data_range
                            )
                         except Exception as ssim_err:
                            print(f"[Worker PID: {os.getpid()}] Error calculating SSIM for {image_name} ({fmt}): {ssim_err}")
                            ssim = 0.0 # Assign default on error


                    results_for_image[fmt] = CompressionResult(
                        format=fmt,
                        compression_ratio=compression_ratio,
                        mse=mse,
                        psnr=psnr,
                        ssim=ssim,
                        file_size=compressed_size
                    )

                except Exception as e:
                    print(f"[Worker PID: {os.getpid()}] Error: Failed to calculate metrics for {image_name} ({fmt}): {e}")
                    # Optionally add a placeholder error result instead of skipping
                    # results_for_image[fmt] = CompressionResult(format=fmt, ..., file_size=compressed_size) # with error values (0 or NaN)

                finally:
                    # Ensure PIL image object from buffer is closed
                    if compressed_image_obj:
                         compressed_image_obj.close()
                    if 'compressed_image_converted' in locals() and compressed_image_converted:
                         compressed_image_converted.close()
                    if 'compressed_image_resized' in locals() and compressed_image_resized:
                         compressed_image_resized.close()

        # Close the original image file handle after processing all formats
        original_image.close()

    except FileNotFoundError:
        print(f"[Worker PID: {os.getpid()}] Error: Image file not found: {image_path}")
        return image_name, {}
    except Exception as e:
        print(f"[Worker PID: {os.getpid()}] Error: Major failure processing image {image_path}: {e}")
        # Ensure image is closed if opened before failure
        if 'original_image' in locals() and original_image:
            try:
                original_image.close()
            except Exception:
                pass # Ignore errors during cleanup close
        return image_name, {} # Return empty dict on major failure for this image

    return image_name, results_for_image


# --- Main Analyzer Class ---
class ImageCompressionAnalyzer:
    """Class to analyze and compare image compression formats"""

    def __init__(self,
                 input_dir: str = 'images',
                 output_dir: str = 'compressed_images',
                 quality: int = 85,
                 formats: Optional[List[str]] = None,
                 max_workers: Optional[int] = None):
        """
        Initialize the analyzer

        Args:
            input_dir: Directory containing original images.
            output_dir: Directory to save compressed images.
            quality: Quality level for lossy compression formats (1-100).
            formats: List of formats to test (e.g., ['png', 'jpg', 'webp']).
            max_workers: Max number of processes for parallel execution (default: os.cpu_count()).
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.quality = quality
        self.formats = formats or ['png', 'jpg', 'webp']
        self.max_workers = max_workers
        self.results: Dict[str, Dict[str, CompressionResult]] = {}
        self.summary: Optional[pd.DataFrame] = None

        # Create output directory if it doesn't exist, including parents
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Configure logging for the main process
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - [%(process)d] - %(message)s', # Added process ID
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.logger = logging.getLogger(__name__)
        # Store log level for potential use in workers (though currently using print in workers)
        self.log_level = logging.INFO

    def get_image_files(self) -> List[Path]:
        """Get all valid image files from the input directory."""
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}

        if not self.input_dir.is_dir():
             self.logger.error(f"Input directory not found or is not a directory: {self.input_dir}")
             return []

        try:
            image_files = [
                f for f in self.input_dir.iterdir()
                if f.is_file() and f.suffix.lower() in valid_extensions
            ]
        except OSError as e:
             self.logger.error(f"Error reading input directory {self.input_dir}: {e}")
             return []


        if not image_files:
            self.logger.warning(f"No valid image files found in {self.input_dir}")

        return image_files

    def analyze_images_parallel(self):
         """Analyze all images in parallel using ProcessPoolExecutor."""
         image_files = self.get_image_files()
         if not image_files:
             self.logger.error("No image files to process. Exiting analysis.")
             return # Exit if no files found

         num_images = len(image_files)
         # Determine actual number of workers
         num_workers = self.max_workers if self.max_workers is not None else os.cpu_count()
         self.logger.info(f"Found {num_images} images to analyze using up to {num_workers} worker processes.")

         # Clear previous results
         self.results = {}

         # Prepare arguments for the worker function using partial
         worker_func = partial(_process_single_image,
                               output_dir=self.output_dir,
                               quality=self.quality,
                               formats=self.formats,
                               log_level=self.log_level)

         processed_count = 0
         successful_images = 0
         # Use ProcessPoolExecutor for CPU-bound tasks like image processing
         with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
             # Submit tasks and get futures
             futures = {executor.submit(worker_func, img_path): img_path for img_path in image_files}

             # Process results as they complete using as_completed for better progress tracking
             pbar = tqdm(concurrent.futures.as_completed(futures), total=num_images, desc="Analyzing images (parallel)")
             for future in pbar:
                 image_path = futures[future]
                 image_name = image_path.stem
                 processed_count += 1
                 try:
                     # Get the result from the future (this will re-raise exceptions from the worker)
                     img_name_result, result_dict = future.result()
                     if result_dict: # Check if the worker returned any valid results
                         self.results[img_name_result] = result_dict
                         successful_images += 1
                     else:
                         # Log failure if worker returned empty dict but didn't raise exception
                         self.logger.warning(f"Processing failed for image: {img_name_result} (worker returned empty result).")
                 except Exception as exc:
                     # Log exceptions raised by the worker process
                     self.logger.error(f"Image '{image_name}' generated an exception during parallel processing: {exc}")

                 # Update progress bar description (optional)
                 pbar.set_postfix_str(f"Success: {successful_images}/{processed_count}", refresh=True)


         self.logger.info(f"Parallel analysis complete. Processed {processed_count} files. "
                          f"{successful_images} images analyzed successfully.")


    def create_summary(self) -> Optional[pd.DataFrame]:
        """
        Create a summary Pandas DataFrame from the collected results.

        Returns:
            DataFrame with results or None if no results are available.
        """
        if not self.results:
            self.logger.warning("No results available to create summary.")
            self.summary = None
            return None

        data = []
        for image_name, formats_results in self.results.items():
            for fmt, result in formats_results.items():
                # Ensure the result is the correct type before adding
                if isinstance(result, CompressionResult):
                    data.append({
                        'Image': image_name,
                        'Format': fmt,
                        'CompressionRatio': result.compression_ratio,
                        'MSE': result.mse,
                        'PSNR': result.psnr,
                        'SSIM': result.ssim,
                        'FileSize': result.file_size
                    })
                else:
                    self.logger.warning(f"Skipping invalid result type for image '{image_name}', format '{fmt}'")

        if not data:
             self.logger.warning("No valid result data found to create summary DataFrame.")
             self.summary = None
             return None

        try:
            self.summary = pd.DataFrame(data)
            # Add FileSizeKB column for convenience
            if 'FileSize' in self.summary.columns:
                self.summary['FileSizeKB'] = self.summary['FileSize'] / 1024.0
        except Exception as e:
            self.logger.error(f"Error creating summary DataFrame: {e}")
            self.summary = None

        return self.summary

    def print_results(self):
        """Print detailed results per image and average results to the console."""
        if not self.results:
            self.logger.info("No results to print.")
            return

        # Print results for each image
        for image_name, formats_results in self.results.items():
            print(f"\n==== Image: {image_name} ====")
            if not formats_results:
                 print("  No valid results for this image.")
                 continue
            for fmt, result in sorted(formats_results.items()): # Sort formats for consistent output
                 if isinstance(result, CompressionResult):
                    print(f"\n-- Format: {fmt.upper()} --")
                    print(f"  Compression Ratio: {result.compression_ratio:.2f}")
                    print(f"  MSE:               {result.mse:.2f}")
                    # Handle infinite PSNR
                    psnr_str = f"{result.psnr:.2f} dB" if np.isfinite(result.psnr) else "Infinite (lossless)"
                    print(f"  PSNR:              {psnr_str}")
                    print(f"  SSIM:              {result.ssim:.4f}")
                    print(f"  File Size:         {result.file_size / 1024:.1f} KB")
                 else:
                    print(f"\n-- Format: {fmt.upper()} --")
                    print(f"  Result:            Invalid or Error Recorded")

        # --- Print Averages ---
        # Ensure summary exists and is not empty
        if self.summary is None:
            self.logger.info("Summary not generated, cannot print averages.")
            return
        if self.summary.empty:
            self.logger.info("Summary is empty, cannot print averages.")
            return

        print("\n\n======== Average Results Across All Images ========")
        try:
            # Select numeric columns suitable for averaging, exclude infinite PSNR for mean calculation
            numeric_cols = ['CompressionRatio', 'MSE', 'PSNR', 'SSIM', 'FileSizeKB']
            summary_finite = self.summary.replace([np.inf, -np.inf], np.nan) # Replace inf with NaN for mean calc

            valid_cols = [col for col in numeric_cols if col in summary_finite.columns]
            if not valid_cols:
                self.logger.warning("No numeric columns found in summary for averaging.")
                return

            # Calculate mean, ignoring NaN values
            avg_df = summary_finite.groupby('Format')[valid_cols].mean(numeric_only=True)

            for fmt_index, row in avg_df.iterrows():
                fmt = str(fmt_index) # Ensure format is string
                print(f"\n-- Format: {fmt.upper()} --")
                if 'CompressionRatio' in row and pd.notna(row['CompressionRatio']): print(f"  Avg Compression Ratio: {row['CompressionRatio']:.2f}")
                if 'MSE' in row and pd.notna(row['MSE']):              print(f"  Avg MSE:               {row['MSE']:.2f}")
                if 'PSNR' in row and pd.notna(row['PSNR']):             print(f"  Avg PSNR:              {row['PSNR']:.2f} dB")
                if 'SSIM' in row and pd.notna(row['SSIM']):             print(f"  Avg SSIM:              {row['SSIM']:.4f}")
                if 'FileSizeKB' in row and pd.notna(row['FileSizeKB']): print(f"  Avg File Size:         {row['FileSizeKB']:.1f} KB")
        except Exception as e:
            self.logger.error(f"Error calculating or printing average results: {e}")
            import traceback
            traceback.print_exc()


    def create_visualizations(self, output_dir: Union[str, Path] = 'figures'):
        """
        Create and save visualizations summarizing the compression results.

        Args:
            output_dir: Directory where the plot images will be saved.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Ensure summary exists and is not empty
        if self.summary is None:
            self.logger.warning("Summary DataFrame not created. Attempting to create it now.")
            self.create_summary() # Attempt creation
            if self.summary is None: # Check again
                self.logger.error("Cannot create visualizations without a summary DataFrame.")
                return
        if self.summary.empty:
            self.logger.warning("Summary DataFrame is empty. Cannot create visualizations.")
            return

        sns.set_theme(style="whitegrid", palette="viridis") # Use Seaborn's theme context manager
        plot_data = self.summary.copy()
        # Replace inf PSNR with a large value for plotting or handle differently if needed
        plot_data['PSNR'] = plot_data['PSNR'].replace([np.inf, -np.inf], 100) # Cap PSNR at 100 for plots

        # --- Plot 1: Bar plots for key metrics ---
        metrics_to_plot = ['CompressionRatio', 'PSNR', 'MSE', 'SSIM', 'FileSizeKB']
        num_metrics = len(metrics_to_plot)
        cols = 2
        rows = (num_metrics + cols - 1) // cols
        try:
            fig1, axes1 = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), squeeze=False) # squeeze=False ensures axes is 2D array
            axes1 = axes1.flatten() # Flatten axes array for easy iteration

            plot_idx = 0
            for metric in metrics_to_plot:
                if metric in plot_data.columns:
                    sns.barplot(x='Format', y=metric, data=plot_data, ax=axes1[plot_idx], errorbar=('ci', 95)) # Added error bars
                    title = metric.replace('FileSizeKB', 'File Size (KB)').replace('CompressionRatio', 'Compression Ratio')
                    ylabel = title
                    if metric == 'PSNR': ylabel += ' (dB)'
                    axes1[plot_idx].set_title(f'Average {title} by Format')
                    axes1[plot_idx].set_ylabel(ylabel)
                    axes1[plot_idx].tick_params(axis='x', rotation=30)
                    plot_idx += 1

            # Hide any unused subplots
            for j in range(plot_idx, len(axes1)):
                axes1[j].set_visible(False)

            plt.tight_layout(pad=2.0) # Add some padding
            plot_path = output_dir / 'metrics_comparison_barplots.png'
            fig1.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close(fig1) # Close the figure to free memory
            self.logger.info(f"Saved metrics bar plots to {plot_path}")
        except Exception as e:
             self.logger.error(f"Failed to create or save metrics barplot: {e}")


        # --- Plot 2: Box plot for File Size ---
        if 'FileSizeKB' in plot_data.columns:
            try:
                plt.figure(figsize=(10, 6))
                sns.boxplot(x='Format', y='FileSizeKB', data=plot_data)
                plt.title('File Size Distribution by Format')
                plt.ylabel('File Size (KB)')
                plt.tight_layout()
                plot_path = output_dir / 'filesize_boxplot.png'
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                self.logger.info(f"Saved file size box plot to {plot_path}")
            except Exception as e:
                 self.logger.error(f"Failed to create or save filesize boxplot: {e}")

        # --- Plot 3: Scatterplot PSNR vs Compression Ratio ---
        if 'PSNR' in plot_data.columns and 'CompressionRatio' in plot_data.columns:
            try:
                plt.figure(figsize=(10, 8))
                sns.scatterplot(
                    x='CompressionRatio', y='PSNR',
                    hue='Format', style='Format',
                    s=80, data=plot_data, alpha=0.8, edgecolor='k' # Added alpha, edge
                )
                plt.title('Image Quality (PSNR) vs. Compression Ratio')
                plt.xlabel('Compression Ratio (Higher = Smaller File Size)')
                plt.ylabel('PSNR (dB) (Higher = Better Quality)')
                plt.legend(title='Format', bbox_to_anchor=(1.05, 1), loc='upper left') # Move legend outside
                plt.grid(True, linestyle='--', alpha=0.6)
                plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout for external legend
                plot_path = output_dir / 'psnr_vs_compression_scatter.png'
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                self.logger.info(f"Saved PSNR vs Compression scatter plot to {plot_path}")
            except Exception as e:
                 self.logger.error(f"Failed to create or save scatter plot: {e}")

        # Visualization generation complete
        self.logger.info(f"Visualizations saved to directory: {output_dir}")


    def export_results(self, output_path: Union[str, Path] = 'compression_results.csv'):
        """
        Export the results summary DataFrame to a CSV file.

        Args:
            output_path: The file path where the CSV will be saved.
        """
        # Ensure summary exists and is not empty
        if self.summary is None:
            self.logger.warning("Summary DataFrame not created. Attempting to create it now.")
            self.create_summary()
            if self.summary is None:
                self.logger.error("Cannot export results without a summary DataFrame.")
                return
        if self.summary.empty:
             self.logger.warning("Summary DataFrame is empty. No data to export.")
             return

        try:
            self.summary.to_csv(output_path, index=False, float_format='%.4f') # Format floats
            self.logger.info(f"Exported results summary to {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to export results to CSV '{output_path}': {e}")


# --- Argument Parsing and Main Execution ---
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze and compare image compression formats (Optimized & Parallel)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--input', '-i', type=str, default='images',
        help='Directory containing original images.'
    )
    parser.add_argument(
        '--output', '-o', type=str, default='compressed_images',
        help='Directory to save compressed images.'
    )
    parser.add_argument(
        '--quality', '-q', type=int, default=85, choices=range(1, 101),
        metavar='[1-100]',
        help='Quality level for lossy compression formats (JPEG, WebP).'
    )
    parser.add_argument(
        '--formats', '-f', nargs='+', default=['png', 'jpg', 'webp'],
        choices=['png', 'jpg', 'jpeg', 'webp'],
        help='Formats to test (space-separated list).'
    )
    parser.add_argument(
        '--workers', '-w', type=int, default=None,
        help='Number of parallel worker processes (default: number of CPU cores).'
    )
    parser.add_argument(
        '--visualize', '-v', action='store_true',
        help='Create visualizations of the results.'
    )
    parser.add_argument(
        '--csv', '-c', type=str, default='compression_results.csv',
        help='Path to export the results summary CSV file.'
    )
    parser.add_argument(
        '--figures', type=str, default='figures',
        help='Directory to save visualization figures.'
    )

    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()

    # Normalize format names (e.g., jpeg -> jpg) and remove duplicates
    formats_processed = set('jpg' if f.lower() == 'jpeg' else f.lower() for f in args.formats)
    formats_sorted = sorted(list(formats_processed))

    analyzer = ImageCompressionAnalyzer(
        input_dir=args.input,
        output_dir=args.output,
        quality=args.quality,
        formats=formats_sorted,
        max_workers=args.workers
    )

    analyzer.logger.info("Starting image analysis...")
    analyzer.analyze_images_parallel() # Use the parallel analysis method

    analyzer.logger.info("Creating results summary...")
    analyzer.create_summary() # Create summary DataFrame after analysis

    analyzer.logger.info("Printing results...")
    analyzer.print_results() # Print results (includes averages if summary created)

    if args.csv:
        analyzer.logger.info(f"Exporting summary to '{args.csv}'...")
        analyzer.export_results(args.csv) # Export summary to CSV

    if args.visualize:
        analyzer.logger.info(f"Creating visualizations in '{args.figures}'...")
        analyzer.create_visualizations(args.figures) # Generate plots

    analyzer.logger.info("Image Compression Analyzer finished.")


if __name__ == '__main__':
    # The `if __name__ == '__main__':` block is crucial for multiprocessing
    # to work correctly, especially on Windows. It prevents child processes
    # from re-executing the script's main part upon import.
    main()