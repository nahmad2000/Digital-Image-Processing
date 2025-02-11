# Geometric Transformations

This project demonstrates **geometric transformations** applied to an image, including:

- **Rotation**: Rotates the image by a specified angle.
- **Scaling**: Resizes the image with independent scaling factors for x and y.
- **Translation**: Moves the image by a given distance in x and y directions.
- **Shearing (Vertical & Horizontal)**: Skews the image along the x or y axis.

## 🔹 How It Works

1. The user selects a transformation type.
2. The program prompts for the necessary parameters (e.g., rotation angle, scaling factors).
3. The transformation is applied manually (without built-in OpenCV functions).
4. The original and transformed images are displayed and saved.

## 🚀 How to Run
```bash
python geometric_transformations.py
```
The program will guide you through selecting a transformation and entering values.

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
| Transformation | Example |
|---------------|---------|
| **Rotation (30°)** | ![Rotated](images/rotated_example.jpg) |
| **Scaling (0.5x, 1.5y)** | ![Scaled](images/scaled_example.jpg) |
| **Translation (Tx=50, Ty=30)** | ![Translated](images/translated_example.jpg) |

## 📝 Notes
⚠ **Matplotlib normalizes all images to the same display size**, so check the saved images for correct visualization.

✅ The transformed images are saved in the folder as:
- `Transformed_Image.jpg`
