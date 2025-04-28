# Image Downsampling and Interpolation Analysis

This tool provides comprehensive analysis of image downsampling and interpolation methods, showing how different combinations affect image quality.

## Quick Start

Run the analysis with:

```
python Auto_Downsampling_Analysis.py sample.jpg --factor 8
```

This will process your image using 12 different method combinations and generate detailed comparisons.

## Results Summary

Analysis results for `sample.jpg` with downsampling factor 8:

| Method Combination | PSNR (dB) | SSIM |
|-------------------|-----------|------|
| **Area-based + Lanczos** | **27.88** | **0.766** |
| Area-based + Bicubic | 27.77 | 0.765 |
| Area-based + Bilinear | 26.97 | 0.754 |
| Anti-aliased + Lanczos | 26.16 | 0.746 |
| Anti-aliased + Bicubic | 26.13 | 0.746 |
| Anti-aliased + Bilinear | 25.77 | 0.739 |
| Area-based + Nearest Neighbor | 25.75 | 0.699 |
| Simple + Bilinear | 25.19 | 0.731 |
| Simple + Bicubic | 24.84 | 0.718 |
| Simple + Lanczos | 24.67 | 0.708 |
| Anti-aliased + Nearest Neighbor | 24.41 | 0.683 |
| Simple + Nearest Neighbor | 23.07 | 0.638 |

> **Best overall:** Area-based downsampling + Lanczos interpolation

## Visual Comparison

### Downsampling Methods
![Downsampling Methods](results/analysis_20250429-012933/plots/downsampling_comparison.png)

### Interpolation Methods
The following table shows sample results for different method combinations:

| Downsampling | Nearest Neighbor | Bilinear | Bicubic | Lanczos |
|-------------|-----------------|----------|---------|---------|
| **Simple** | ![Simple+NN](results/analysis_20250429-012933/plots/comparison_down1_up1.png) | ![Simple+Bilinear](results/analysis_20250429-012933/plots/comparison_down1_up2.png) | ![Simple+Bicubic](results/analysis_20250429-012933/plots/comparison_down1_up3.png) | ![Simple+Lanczos](results/analysis_20250429-012933/plots/comparison_down1_up4.png) |
| **Anti-aliased** | ![Anti-aliased+NN](results/analysis_20250429-012933/plots/comparison_down2_up1.png) | ![Anti-aliased+Bilinear](results/analysis_20250429-012933/plots/comparison_down2_up2.png) | ![Anti-aliased+Bicubic](results/analysis_20250429-012933/plots/comparison_down2_up3.png) | ![Anti-aliased+Lanczos](results/analysis_20250429-012933/plots/comparison_down2_up4.png) |
| **Area-based** | ![Area+NN](results/analysis_20250429-012933/plots/comparison_down3_up1.png) | ![Area+Bilinear](results/analysis_20250429-012933/plots/comparison_down3_up2.png) | ![Area+Bicubic](results/analysis_20250429-012933/plots/comparison_down3_up3.png) | ![Area+Lanczos](results/analysis_20250429-012933/plots/comparison_down3_up4.png) |

### Performance Heatmaps
![Metrics Heatmap](results/analysis_20250429-012933/plots/heatmap_metrics_factor8.png)

## Understanding the Results

### Image Quality Metrics

- **PSNR (Peak Signal-to-Noise Ratio)**: Higher is better. Measures pixel-level accuracy.
- **SSIM (Structural Similarity Index)**: Higher is better. Measures perceived quality.

### Method Descriptions

#### Downsampling Methods:
- **Simple**: Basic pixel skipping - fast but prone to aliasing
- **Anti-aliased**: Gaussian blur before downsampling - reduces artifacts
- **Area-based**: Averages pixel regions - usually best for photographs

#### Upsampling (Interpolation) Methods:
- **Nearest Neighbor**: Fast but blocky
- **Bilinear**: Smoother than NN, good balance of speed/quality
- **Bicubic**: Better edge preservation than bilinear
- **Lanczos**: High quality with sharp edges, but slower

## Output Files

The script creates a timestamped folder with:

- **images/** - Individual result images
- **plots/** - Comparison visualizations
- **results_summary.csv** - Complete metrics table

## Requirements

- Python 3.6+
- OpenCV
- NumPy
- Matplotlib
- scikit-image