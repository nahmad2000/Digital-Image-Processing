# Shading Correction

This project demonstrates **shading correction** by removing non-uniform illumination from an image using **lowpass filtering**.

## 🔹 How It Works

1. The script applies **Gaussian blur** to estimate the shading pattern in an image.
2. The original image is **divided by the shading estimate** to remove lighting effects.
3. The corrected image is **rescaled** to maintain contrast.

## 🚀 How to Run
```bash
python Shading_Correction.py
```
The program will process the image and generate a **corrected version**.

## 📌 Requirements
- Python 3.x
- OpenCV (`cv2`)
- NumPy
- Matplotlib

Install dependencies using:
```bash
pip install numpy opencv-python matplotlib
```

## 📸 Example Outputs

| **Before (Original Image)** | **After (Shading Corrected Image)** |
|-----------------------------|--------------------------------------|
| ![Original](wall_image.png) | ![Corrected](wall_corrected.png) |

## 📝 Notes
- **This method works best for images with gradual lighting variations.**
- **Gaussian blur size and sigma should be adjusted for different images.**
- **The corrected image is saved as `wall_corrected.png`.**

