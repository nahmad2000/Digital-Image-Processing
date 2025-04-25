# Image Compression Analyzer

A powerful tool for analyzing and comparing different image compression formats (PNG, JPEG, WebP) using multiple quality metrics.

## Features

- **Multiple Format Support**: Compare PNG, JPEG, and WebP compression formats
- **Comprehensive Metrics**: Analyze compression ratio, MSE, PSNR, and SSIM
- **Rich Visualizations**: Generate insightful charts and graphs
- **Batch Processing**: Process multiple images at once
- **Flexible Configuration**: Customize quality settings and formats
- **Results Export**: Save analysis results to CSV for further analysis

## Installation

### Prerequisites

- Python 3.7 or higher
- pip (Python package manager)

### Steps

1. Clone this repository or download the script:

```bash
git clone https://github.com/yourusername/image-compression-analyzer.git
cd image-compression-analyzer
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

Or install dependencies manually:

```bash
pip install numpy pandas matplotlib seaborn pillow scikit-image tqdm
```

## Usage

### Basic Usage

Place your images in a folder called `images` in the same directory as the script, then run:

```bash
python image_compression_analyzer.py
```

This will:
1. Process all images in the `images` folder
2. Save compressed versions to `compressed_images` folder
3. Print analysis results to the console

### Advanced Options

```bash
python image_compression_analyzer.py --input my_images --output compressed --quality 90 --formats png jpg webp --visualize
```

### Full Command-Line Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--input` | `-i` | Directory containing original images | `images` |
| `--output` | `-o` | Directory to save compressed images | `compressed_images` |
| `--quality` | `-q` | Quality level for lossy compression (1-100) | `85` |
| `--formats` | `-f` | Formats to test | `png jpg webp` |
| `--visualize` | `-v` | Create visualizations of results | `False` |
| `--csv` | `-c` | Path to export CSV results | `compression_results.csv` |
| `--figures` | | Directory to save figures to | `figures` |

## Output Examples

### Console Output

```
==== Image: sample_image ====

-- PNG --
Compression Ratio: 1.20
MSE: 0.00
PSNR: Infinity dB
SSIM: 1.0000
File Size: 1024.5 KB

-- JPG --
Compression Ratio: 10.50
MSE: 5.23
PSNR: 40.92 dB
SSIM: 0.9820
File Size: 117.2 KB

-- WEBP --
Compression Ratio: 15.80
MSE: 6.12
PSNR: 39.22 dB
SSIM: 0.9780
File Size: 77.8 KB
```

### Visualizations

The `--visualize` option generates several charts:

1. **Format Comparison**: Bar charts comparing compression ratio, PSNR, MSE, and SSIM across formats
2. **File Size Comparison**: Box plot of file sizes by format
3. **Metrics by Image**: Detailed breakdown of metrics for each image
4. **PSNR vs Compression Ratio**: Scatter plot showing the quality-size tradeoff

## Interpreting Results

- **Compression Ratio**: Higher is better - indicates how much smaller the compressed file is
- **MSE (Mean Squared Error)**: Lower is better - measures average squared difference between original and compressed images
- **PSNR (Peak Signal-to-Noise Ratio)**: Higher is better - measures quality of compression (typically 30-50 dB is good)
- **SSIM (Structural Similarity Index)**: Higher is better (max 1.0) - measures perceived quality, accounting for human visual system

## Tips for Best Results

- For highest quality preservation, use PNG (lossless)
- For web images with transparency, WebP often provides the best compression ratio while maintaining quality
- For photos, JPEG at 85% quality often provides a good balance between size and quality
- For large batches, consider running in smaller groups to avoid memory issues

## Troubleshooting

### Common Issues

1. **Missing dependencies**: Make sure you've installed all required packages
2. **Permission errors**: Ensure you have write access to output directories
3. **Memory errors**: Try processing fewer images at once or use smaller image files

### Error Logs

The script creates detailed logs that can help diagnose issues. Look for error messages in the console output.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.