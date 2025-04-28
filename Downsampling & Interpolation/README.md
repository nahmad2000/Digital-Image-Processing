# Image Downsampling and Interpolation

## Overview

This Python script demonstrates the effects of image downsampling (reducing resolution) and subsequent upsampling using different interpolation techniques (Nearest Neighbor and Bilinear). It uses the highly optimized functions from the OpenCV library (`cv2`) for efficient processing and Matplotlib for displaying the results.

The script allows you to visually compare the original image, the downsampled version, and the results of restoring the original dimensions using the two common interpolation methods.

## Features

* Loads grayscale images from a specified file path.
* Prompts the user to select a downsampling factor from a predefined list (e.g., 2, 4, 8...).
* Performs downsampling by simple pixel skipping (`image[::factor, ::factor]`).
* Upsamples the downsampled image back to the original dimensions using:
    * **Nearest Neighbor Interpolation** (`cv2.INTER_NEAREST`)
    * **Bilinear Interpolation** (`cv2.INTER_LINEAR`)
* Displays a comparison plot showing the Original, Downsampled, Nearest Neighbor Upsampled, and Bilinear Upsampled images side-by-side using Matplotlib.
* Saves the downsampled image and the two upsampled images as separate JPG files, including the factor in the filename (e.g., `Downsampled_Image_x8.jpg`).
* Uses command-line arguments for easy integration and specifying the input image.

## Requirements

* Python 3.x
* The script file itself (e.g., `Downsampling_Interpolation_Optimized.py`)
* Required Python libraries:
    * NumPy
    * OpenCV for Python (`opencv-python`)
    * Matplotlib

## Setup

1.  **Ensure you have the script:** Make sure the Python script file (e.g., `Downsampling_Interpolation_Optimized.py`) is located in your working directory.

2.  **Install Dependencies:** Open your terminal or command prompt in the script's directory and install the required Python libraries:
    ```bash
    pip install numpy opencv-python matplotlib
    ```
    *(Using a Python virtual environment is recommended for managing project dependencies.)*

## Usage

Run the script from your terminal, providing the path to the image you want to process as a command-line argument.

```bash
python Downsampling_Interpolation_Optimized.py <path_to_your_image>
````

**Example:**

Bash

```
# If your image is in the same folder as the script:
python Downsampling_Interpolation_Optimized.py my_photo.jpg

# If your image is in a subfolder called 'images':
python Downsampling_Interpolation_Optimized.py images/my_photo.jpg
```

The script will then:

1. Load the specified image.
2. Prompt you to enter a downsampling factor (e.g., 2, 4, 8...).
3. Perform the downsampling and interpolation.
4. Display the comparison plot.
5. Save the resulting images in the same directory where the script is run.

## Input

- A path to an image file (e.g., JPG, PNG, BMP, TIFF) provided as a command-line argument. The script will load it as grayscale.

## Output

1. **Display Window:** A Matplotlib window showing four subplots:
    
    - Original Image (with dimensions)
    - Downsampled Image (with dimensions - will appear smaller)
    - Nearest Neighbor Upsampled Image (restored dimensions)
    - Bilinear Interpolation Upsampled Image (restored dimensions)
2. **Saved Files:** Three image files saved in the script's directory:
    
    - `Downsampled_Image_x<Factor>.jpg`
    - `Nearest_Neighbor_Image_x<Factor>.jpg`
    - `Bilinear_Image_x<Factor>.jpg` _(Where `<Factor>` is the downsampling factor you chose)_

## Notes

- The Matplotlib display window automatically scales all subplots to fit. This means the "Downsampled" image might appear larger in the plot than its actual pixel dimensions relative to the others. To see the true resolution differences, examine the saved image files.
- The downsampling method used is simple pixel skipping. For potentially higher-quality downsampling (anti-aliasing), especially with large factors, methods like `cv2.resize` with `interpolation=cv2.INTER_AREA` could be considered as an alternative.
