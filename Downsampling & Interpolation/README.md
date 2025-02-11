# Downsampling & Interpolation

This script demonstrates **downsampling an image** and restoring it using:
- **Nearest Neighbor Interpolation**
- **Bilinear Interpolation**

## 🔹 How It Works
1. The user selects a **downsampling factor**.
2. The image is **downsampled** by keeping 1 pixel every `N` pixels.
3. Two interpolation methods restore the original size:
   - **Nearest Neighbor:** Copies the nearest pixel.
   - **Bilinear Interpolation:** Uses a weighted average of neighboring pixels.

## 🚀 How to Run