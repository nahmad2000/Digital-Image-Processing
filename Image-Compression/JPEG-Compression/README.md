# Enhanced JPEG-Style Image Compressor

## What is this?

This tool lets you compress images (like photos) using techniques similar to the standard JPEG format. You can experiment with different compression levels and settings to see how they affect the final image quality and file size. It's designed to be educational and allows you to visualize what happens during compression.

You can use it either through a simple command-line interface or a more visual graphical interface (GUI).

## Key Features

* **Quality Control:** Easily adjust the compression level using a simple quality factor (1-100). Lower numbers mean smaller files but potentially lower quality.
* **Color Handling:** Works with both color (RGB) and grayscale images.
* **Chroma Subsampling:** Optionally reduces color information more than brightness information (like standard JPEG does with modes like 4:2:0 or 4:2:2) for better compression.
* **Performance:** Uses multiple processor threads to speed up compression on larger images.
* **Visualizations (Optional):** See visual comparisons of the original and compressed images, understand how the compression algorithm "sees" the image frequencies, and examine the settings (quantization matrices) used.
* **Batch Processing:** Compress multiple images in a folder at once.
* **Detailed Reports:** Get information about the compression ratio, image quality (PSNR, SSIM), and estimated data size after different steps.
* **Easy to Use:** Choose between a command-line interface for scripting or a graphical interface for interactive use.

## Files in this Project

```

./

├── jpeg_compressor.py # The main Python script containing all the code

├── requirements.txt # List of software libraries needed

├── README.md # This file!

└── output/ # Default folder where compressed images are saved (created automatically)

└── example_q75.jpg # Example output file name structure

```

## Setup Instructions

You'll need Python 3 installed on your computer.

1.  **Download:** Get all the project files (`jpeg_compressor.py`, `requirements.txt`, `README.md`).
2.  **Open Terminal or Command Prompt:** Navigate to the directory where you saved the files.
3.  **(Optional but Recommended) Create a Virtual Environment:** This keeps the required libraries separate from your system's Python.
    * Run: `python -m venv venv`
    * Activate it:
        * Windows: `venv\Scripts\activate`
        * Mac/Linux: `source venv/bin/activate`
4.  **Install Required Libraries:**
    * Run: `pip install -r requirements.txt`

Now you're ready to use the tool!

## How to Use

You have two main ways to use the compressor:

### 1. Graphical User Interface (GUI) - Easiest!

This is the recommended way for most users.

* **Launch:** Open your terminal or command prompt (activate the virtual environment if you created one) and run:
    ```bash
    python jpeg_compressor.py --gui
    ```
* **Using the GUI:**
    * **Input:** Click "Browse..." to select an image file OR a folder containing images (if you check "Batch Mode").
    * **Output:** Click "Browse..." to choose where the compressed images will be saved.
    * **Quality:** Drag the slider to set the desired quality (1=lowest, 100=highest).
    * **Chroma Subsampling:** Select how much color information to reduce (4:2:0 is common for good compression, 4:4:4 keeps all color info).
    * **Threads:** Usually leave this at the default (your computer's core count).
    * **Visualizations:** Check the box if you want plots comparing images, showing frequency info, etc., to pop up during compression.
    * **Compress:** Click the "Compress" button to start. The status bar and progress bar will show updates.

### 2. Command Line Interface (CLI)

This is useful for automating compression or using it in scripts.

* **Basic Syntax:**
    ```bash
    python jpeg_compressor.py <input_path> [options]
    ```
* **`input_path`:** The *only required* argument. This is the path to the image file you want to compress.
* **Common Options:**
    * `-o <directory>` or `--output <directory>`: Specify where to save the output file(s). (Default: creates an `output` folder).
    * `-q <number>` or `--quality <number>`: Set the quality factor (1-100). (Default: 75).
    * `-s <mode>` or `--subsample <mode>`: Set chroma subsampling (`4:4:4`, `4:2:2`, `4:2:0`). (Default: `4:2:0`).
    * `-t <number>` or `--threads <number>`: Set the number of processor threads to use.
    * `-v` or `--visualize`: Show the visualization plots.
    * `-b` or `--batch`: Process all images in the folder specified by `input_path`.

* **Examples:**
    * Compress a single image with default settings:
        ```bash
        python jpeg_compressor.py my_photo.png
        ```
        *(Output will be `output/my_photo_q75.jpg`)*

    * Compress an image with quality 90 and save it to a specific folder:
        ```bash
        python jpeg_compressor.py my_photo.png -q 90 -o ./compressed_images/
        ```
        *(Output will be `compressed_images/my_photo_q90.jpg`)*

    * Compress all images in a folder named `input_pics` with quality 50:
        ```bash
        python jpeg_compressor.py ./input_pics -q 50 --batch
        ```
        *(Outputs will be in `output/`, named like `imagename_q50.jpg`)*

## Understanding the Output

* **Compressed Images:** The tool saves the compressed image(s) in the output folder, usually as `.jpg` files.
* **Console Report:** When using the command line, a report is printed showing:
    * `Original Size Kb` / `Compressed Size Kb`: File sizes before and after.
    * `Compression Ratio File`: How much smaller the file is (e.g., 10:1 means 10 times smaller).
    * `Psnr Db`: A measure of image quality (higher is better, often 30+ dB is good).
    * `Ssim`: Another quality measure (closer to 1.00 is better).
    * `Entropy` / `Zero Ratio`: Technical details about how much information was removed during compression.
* **Visualizations (if enabled):** Pop-up windows showing:
    * Side-by-side comparison of the original and compressed image.
    * Heatmaps showing the image data in the "frequency domain" (how the algorithm represents sharp vs. smooth areas).
    * The quantization matrices (grids of numbers) used for the chosen quality level.

Enjoy experimenting with image compression!
```