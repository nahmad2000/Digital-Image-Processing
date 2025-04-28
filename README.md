# 🖼️ Digital Image Processing Projects

Welcome to the **Digital Image Processing** collection!  
This repository showcases a variety of small but powerful projects exploring classic and advanced image processing techniques.

Each folder contains a focused project with sample results, easy-to-run scripts, and clean code to help you understand and experiment with key concepts.

---

## 📂 Project Overview

| Project | Description | Example |
|:--------|:------------|:--------|
| [Downsampling & Interpolation](#downsampling--interpolation) | Analyze different downsampling and interpolation combinations for image resizing and quality evaluation. | ![Example](Downsampling%20&%20Interpolation/results/analysis_20250429-012933/plots/heatmap_metrics_factor8.png) |
| [Geometric Transformations](#geometric-transformations) | Apply rotations, scaling, translations, and shears to images using OpenCV. | ![Example](Geometric%20Transformations/results/image1_rotation_180.0.jpg) |
| [Image Compression](#image-compression) | Compare compression formats (JPEG, PNG, WebP) based on size vs. quality trade-offs. | ![Example](Image%20Compression/figures/metrics_comparison_barplots.png) |
| [Image Denoising](#image-denoising) | Add synthetic noise and evaluate denoising filters like Median, Gaussian, and Non-Local Means. | ![Example](Image%20Denoising/figures/denoising_results_comparison.png) |
| [Image Enhancement](#image-enhancement) | Improve brightness and contrast using Power Law Transformation and Histogram Equalization. | ![Example](Image%20Enhancement/enhanced_images/comparison.png) |
| [Shading Correction](#shading-correction) | Correct uneven lighting using spatial and frequency domain filtering. | ![Example](Shading%20Correction/results/sample_frequency_corrected.png) |

---

## 📋 Detailed Folder Descriptions

### Downsampling & Interpolation
> **Location:** `Downsampling & Interpolation/`

- Explore how different downsampling (Simple, Anti-aliased, Area-based) and upsampling (Nearest, Bilinear, Bicubic, Lanczos) methods affect image quality.
- Analyze results using PSNR and SSIM metrics.
- Auto-generate visual heatmaps and side-by-side comparisons.

🔗 [See Folder](./Downsampling%20&%20Interpolation)

---

### Geometric Transformations
> **Location:** `Geometric Transformations/`

- Apply multiple geometric transformations in a single script.
- Transformations include Rotation, Scaling, Translation, Vertical Shear, and Horizontal Shear.
- Batch process multiple images and automatically save outputs.

🔗 [See Folder](./Geometric%20Transformations)

---

### Image Compression
> **Location:** `Image Compression/`

- Compress images into JPEG, PNG, and WebP formats.
- Measure file size, compression ratio, PSNR, SSIM, and visualize the differences.
- Supports parallel processing for faster performance.

🔗 [See Folder](./Image%20Compression)

---

### Image Denoising
> **Location:** `Image Denoising/`

- Add synthetic noise (Gaussian, Salt & Pepper) to clean images.
- Apply various denoising filters and evaluate their effectiveness.
- Generate comparison plots and save final cleaned images.

🔗 [See Folder](./Image%20Denoising)

---

### Image Enhancement
> **Location:** `Image Enhancement/`

- Brighten dark images and improve contrast using:
  - Gamma Correction (Power Law Transformation)
  - Histogram Equalization
- Interactive and command-line modes available for easy use.

🔗 [See Folder](./Image%20Enhancement)

---

### Shading Correction
> **Location:** `Shading Correction/`

- Correct uneven lighting using two methods:
  - Spatial Domain (Gaussian blur based)
  - Frequency Domain (Homomorphic filtering)
- Visualize corrected images and extracted lighting components.

🔗 [See Folder](./Shading%20Correction)

---

## 🛠️ Installation and Setup

Most folders have individual setup instructions, but in general:

1. Install the basic requirements:

```bash
pip install numpy opencv-python matplotlib scikit-image seaborn tqdm
