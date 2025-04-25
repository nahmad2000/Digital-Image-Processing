# -*- coding: utf-8 -*-

"""
Image Compression Analyzer

This script analyzes and compares different image compression formats (PNG, JPEG, WebP)
using multiple quality metrics and provides visualization of results.
"""

import os
import argparse
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from skimage import metrics
from tqdm import tqdm


@dataclass
class CompressionResult:
    """Class to store compression results for an image"""
    format: str
    compression_ratio: float
    mse: float
    psnr: float
    ssim: float
    file_size: int
    

class ImageCompressionAnalyzer:
    """Class to analyze and compare image compression formats"""
    
    def __init__(self, 
                 input_dir: str = 'images', 
                 output_dir: str = 'compressed_images',
                 quality: int = 85,
                 formats: List[str] = None):
        """
        Initialize the analyzer
        
        Args:
            input_dir: Directory containing original images
            output_dir: Directory to save compressed images
            quality: Quality level for lossy compression formats (1-100)
            formats: List of formats to test (default: ['png', 'jpg', 'webp'])
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.quality = quality
        self.formats = formats or ['png', 'jpg', 'webp']
        self.results = {}  # Dictionary to store results by image name
        self.summary = None  # DataFrame to store summary results
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def get_image_files(self) -> List[Path]:
        """Get all image files from the input directory"""
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
        image_files = [
            f for f in self.input_dir.iterdir() 
            if f.is_file() and f.suffix.lower() in valid_extensions
        ]
        
        if not image_files:
            self.logger.warning(f"No valid image files found in {self.input_dir}")
        
        return image_files
    
    def compress_image(self, 
                       image_path: Path, 
                       output_format: str) -> Tuple[Path, Image.Image]:
        """
        Compress an image using the specified format
        
        Args:
            image_path: Path to the original image
            output_format: Format to compress to ('png', 'jpg', or 'webp')
            
        Returns:
            Tuple of (output path, compressed image)
        """
        # Get output path
        output_filename = f"{image_path.stem}_compressed.{output_format}"
        output_path = self.output_dir / output_filename
        
        # Open original image
        try:
            image = Image.open(image_path)
            
            # Save with specified format and quality
            if output_format in ['jpg', 'jpeg']:
                image.save(output_path, format='JPEG', quality=self.quality)
            elif output_format == 'webp':
                image.save(output_path, format='WEBP', quality=self.quality)
            elif output_format == 'png':
                image.save(output_path, format='PNG', optimize=True)
            else:
                self.logger.warning(f"Unsupported format: {output_format}, using PNG")
                output_path = output_path.with_suffix('.png')
                image.save(output_path, format='PNG', optimize=True)
                
            # Open the saved image to get the compressed version
            compressed_image = Image.open(output_path)
            return output_path, compressed_image
            
        except Exception as e:
            self.logger.error(f"Error compressing {image_path}: {str(e)}")
            return None, None
    
    def calculate_metrics(self, 
                         original_image: Image.Image, 
                         compressed_image: Image.Image,
                         original_size: int,
                         compressed_size: int) -> CompressionResult:
        """
        Calculate image quality metrics between original and compressed images
        
        Args:
            original_image: Original image
            compressed_image: Compressed image
            original_size: Original file size in bytes
            compressed_size: Compressed file size in bytes
            
        Returns:
            CompressionResult with calculated metrics
        """
        # Convert images to numpy arrays
        try:
            # Ensure images have the same mode
            if original_image.mode != compressed_image.mode:
                compressed_image = compressed_image.convert(original_image.mode)
                
            original_array = np.array(original_image)
            compressed_array = np.array(compressed_image)
            
            # Handle differently sized images (might happen with some formats)
            if original_array.shape != compressed_array.shape:
                self.logger.warning(
                    f"Image shapes don't match: {original_array.shape} vs {compressed_array.shape}. "
                    f"Resizing compressed image."
                )
                compressed_image = compressed_image.resize(original_image.size)
                compressed_array = np.array(compressed_image)
            
            # Calculate compression ratio
            compression_ratio = original_size / compressed_size if compressed_size > 0 else 0
            
            # Calculate MSE
            mse = metrics.mean_squared_error(original_array, compressed_array)
            
            # Calculate PSNR
            psnr = metrics.peak_signal_noise_ratio(
                original_array, compressed_array, data_range=None
            )
            
            # Calculate SSIM
            ssim = metrics.structural_similarity(
                original_array, compressed_array, 
                channel_axis=2 if len(original_array.shape) > 2 else None,
                data_range=original_array.max() - original_array.min()
            )
            
            return CompressionResult(
                format=compressed_image.format.lower() if compressed_image.format else "unknown",
                compression_ratio=compression_ratio,
                mse=mse,
                psnr=psnr,
                ssim=ssim,
                file_size=compressed_size
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {str(e)}")
            return CompressionResult(
                format="error",
                compression_ratio=0,
                mse=0,
                psnr=0,
                ssim=0,
                file_size=0
            )
    
    def analyze_images(self):
        """Analyze all images in the input directory"""
        image_files = self.get_image_files()
        self.logger.info(f"Found {len(image_files)} images to analyze")
        
        # Process each image
        for image_path in tqdm(image_files, desc="Analyzing images"):
            image_name = image_path.stem
            self.results[image_name] = {}
            
            try:
                # Open original image and get its size
                original_image = Image.open(image_path)
                original_size = image_path.stat().st_size
                
                # Process each format
                for fmt in self.formats:
                    output_path, compressed_image = self.compress_image(image_path, fmt)
                    
                    if output_path and compressed_image:
                        compressed_size = output_path.stat().st_size
                        
                        # Calculate metrics
                        result = self.calculate_metrics(
                            original_image, compressed_image, original_size, compressed_size
                        )
                        
                        self.results[image_name][fmt] = result
            
            except Exception as e:
                self.logger.error(f"Error processing {image_name}: {str(e)}")
    
    def create_summary(self) -> pd.DataFrame:
        """
        Create a summary DataFrame of all results
        
        Returns:
            DataFrame with results for all images and formats
        """
        data = []
        
        for image_name, formats in self.results.items():
            for fmt, result in formats.items():
                data.append({
                    'Image': image_name,
                    'Format': fmt,
                    'CompressionRatio': result.compression_ratio,
                    'MSE': result.mse,
                    'PSNR': result.psnr,
                    'SSIM': result.ssim,
                    'FileSize': result.file_size
                })
        
        self.summary = pd.DataFrame(data)
        return self.summary
    
    def print_results(self):
        """Print results for each image and format to console"""
        for image_name, formats in self.results.items():
            print(f"\n==== Image: {image_name} ====")
            
            for fmt, result in formats.items():
                print(f"\n-- {fmt.upper()} --")
                print(f"Compression Ratio: {result.compression_ratio:.2f}")
                print(f"MSE: {result.mse:.2f}")
                print(f"PSNR: {result.psnr:.2f} dB")
                print(f"SSIM: {result.ssim:.4f}")
                print(f"File Size: {result.file_size / 1024:.1f} KB")
        
        # Print averages
        print("\n==== Average Results ====")
        avg_df = self.summary.groupby('Format').mean()
        for idx, row in avg_df.iterrows():
            print(f"\n-- {idx.upper()} --")
            print(f"Avg Compression Ratio: {row['CompressionRatio']:.2f}")
            print(f"Avg MSE: {row['MSE']:.2f}")
            print(f"Avg PSNR: {row['PSNR']:.2f} dB")
            print(f"Avg SSIM: {row['SSIM']:.4f}")
            print(f"Avg File Size: {row['FileSize'] / 1024:.1f} KB")
    
    def create_visualizations(self, output_dir: Union[str, Path] = 'figures'):
        """
        Create visualizations of results
        
        Args:
            output_dir: Directory to save figures to
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        if not self.summary is not None:
            self.create_summary()
        
        sns.set_style("whitegrid")
        
        # Set up figure size and style
        plt.figure(figsize=(12, 9))
        
        # 1. Compression Ratio Comparison
        plt.subplot(2, 2, 1)
        sns.barplot(x='Format', y='CompressionRatio', data=self.summary)
        plt.title('Compression Ratio by Format')
        plt.ylabel('Compression Ratio')
        plt.grid(True)
        
        # 2. PSNR Comparison
        plt.subplot(2, 2, 2)
        sns.barplot(x='Format', y='PSNR', data=self.summary)
        plt.title('PSNR by Format')
        plt.ylabel('PSNR (dB)')
        plt.grid(True)
        
        # 3. MSE Comparison
        plt.subplot(2, 2, 3)
        sns.barplot(x='Format', y='MSE', data=self.summary)
        plt.title('MSE by Format')
        plt.ylabel('Mean Squared Error')
        plt.grid(True)
        
        # 4. SSIM Comparison
        plt.subplot(2, 2, 4)
        sns.barplot(x='Format', y='SSIM', data=self.summary)
        plt.title('SSIM by Format')
        plt.ylabel('Structural Similarity Index')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'format_comparison.png', dpi=300)
        
        # Additional plots
        
        # 5. File size comparison
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='Format', y='FileSize', data=self.summary)
        plt.title('File Size by Format')
        plt.ylabel('File Size (bytes)')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_dir / 'filesize_comparison.png', dpi=300)
        
        # 6. Metrics by image
        metrics_long = pd.melt(
            self.summary, 
            id_vars=['Image', 'Format'], 
            value_vars=['CompressionRatio', 'PSNR', 'MSE', 'SSIM'],
            var_name='Metric', value_name='Value'
        )
        
        plt.figure(figsize=(15, 10))
        g = sns.FacetGrid(metrics_long, col='Metric', hue='Format', col_wrap=2, height=4)
        g.map(sns.barplot, 'Image', 'Value', alpha=0.7)
        g.add_legend()
        g.set_titles('{col_name}')
        plt.tight_layout()
        plt.savefig(output_dir / 'metrics_by_image.png', dpi=300)
        
        # 7. Scatterplot of PSNR vs Compression Ratio
        plt.figure(figsize=(10, 8))
        sns.scatterplot(
            x='CompressionRatio', y='PSNR', 
            hue='Format', style='Format',
            s=100, data=self.summary
        )
        plt.title('PSNR vs Compression Ratio')
        plt.xlabel('Compression Ratio')
        plt.ylabel('PSNR (dB)')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_dir / 'psnr_vs_compression.png', dpi=300)
        
        self.logger.info(f"Saved visualizations to {output_dir}")
        
    def export_results(self, output_path: Union[str, Path] = 'compression_results.csv'):
        """
        Export results to CSV
        
        Args:
            output_path: Path to save CSV file to
        """
        if self.summary is None:
            self.create_summary()
            
        self.summary.to_csv(output_path, index=False)
        self.logger.info(f"Exported results to {output_path}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Analyze and compare image compression formats",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--input', '-i', type=str, default='images',
        help='Directory containing original images'
    )
    parser.add_argument(
        '--output', '-o', type=str, default='compressed_images',
        help='Directory to save compressed images'
    )
    parser.add_argument(
        '--quality', '-q', type=int, default=85, choices=range(1, 101),
        help='Quality level for lossy compression formats (1-100)'
    )
    parser.add_argument(
        '--formats', '-f', nargs='+', default=['png', 'jpg', 'webp'],
        help='Formats to test'
    )
    parser.add_argument(
        '--visualize', '-v', action='store_true',
        help='Create visualizations of results'
    )
    parser.add_argument(
        '--csv', '-c', type=str, default='compression_results.csv',
        help='Path to export CSV results'
    )
    parser.add_argument(
        '--figures', type=str, default='figures',
        help='Directory to save figures to'
    )
    
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()
    
    analyzer = ImageCompressionAnalyzer(
        input_dir=args.input,
        output_dir=args.output,
        quality=args.quality,
        formats=args.formats
    )
    
    analyzer.analyze_images()
    analyzer.create_summary()
    analyzer.print_results()
    
    if args.csv:
        analyzer.export_results(args.csv)
    
    if args.visualize:
        analyzer.create_visualizations(args.figures)

if __name__ == '__main__':
    main()