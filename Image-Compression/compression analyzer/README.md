# ✨ Image Compression Analyzer ✨

[![Python Version](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)

A handy Python tool to compare different image compression formats (like JPEG, PNG, WebP) side-by-side! 🖼️➡️📊

This script takes a folder of your images, compresses them using different settings, and tells you exactly how much space you saved and how much quality was potentially lost. It calculates common metrics like PSNR and SSIM and even draws charts to help you visualize the results!

It's designed to be fast, using parallel processing to analyze multiple images at once on multi-core computers. 🚀

## 🌟 Features

* **Compare Formats:** Analyze and compare PNG, JPEG, and WebP compression.
* **Quality Metrics:** Calculates Compression Ratio, Mean Squared Error (MSE), Peak Signal-to-Noise Ratio (PSNR), and Structural Similarity Index (SSIM).
* **Fast Analysis:** Uses multiple CPU cores to process images in parallel, speeding things up significantly for large batches.
* **Save Compressed Files:** Saves the newly compressed images in a separate output folder.
* **Easy Reporting:**
    * Prints results directly to your console.
    * Exports a detailed summary to a CSV file (great for spreadsheets!).
* **Visualizations:** Generates plots (bar charts, box plots, scatter plots) comparing formats if you ask it to (`--visualize`). 📊
* **Configurable:** You can easily set the quality level for JPEG/WebP compression.
* **User-Friendly:** Simple command-line interface.

---
## ⚙️ Requirements

* **Python:** Version 3.7 or newer.
* **Libraries:** You'll need a few common Python libraries for data handling, image processing, and plotting:
    * `numpy`
    * `pandas`
    * `matplotlib`
    * `seaborn`
    * `Pillow` (or the faster `Pillow-SIMD`)
    * `scikit-image`
    * `tqdm` (for the nice progress bars!)

## 🛠️ Installation

1.  **Clone or Download:** Get the script (`compression_analyzer.py`) onto your computer.
2.  **Open Terminal/Command Prompt:** Navigate to the directory where you saved the script.
3.  **(Recommended) Create a Virtual Environment:** This keeps dependencies organized.
    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```
4.  **Install Libraries:** Run the following command:
    ```bash
    pip install numpy pandas matplotlib seaborn Pillow scikit-image tqdm
    ```
    *Optional:* For potentially faster performance, you can try installing `Pillow-SIMD` instead of `Pillow`:
    ```bash
    pip uninstall Pillow # If already installed
    pip install Pillow-SIMD scikit-image tqdm numpy pandas matplotlib seaborn
    ```

---
## 🚀 How to Use

Run the script from your terminal using `python compression_analyzer.py` followed by options.

```bash
python compression_analyzer.py [OPTIONS]
````


### 📋 Command-Line Arguments

|Option|Alias|Description|Default|
|---|---|---|---|
|`--input`|`-i`|**(Required)** Path to the directory containing your original images.|`images`|
|`--output`|`-o`|Path to the directory where compressed images will be saved.|`compressed_images`|
|`--quality`|`-q`|Quality level (1–100) for lossy formats. Higher = better quality, larger file.|`85`|
|`--formats`|`-f`|Space-separated list of formats to test (`png`, `jpg`, `jpeg`, `webp`).|`png jpg webp`|
|`--workers`|`-w`|Number of CPU cores to use for parallel processing. Blank = use all available cores.|`None`|
|`--visualize`||Add this flag to generate and save comparison plots.|Not set|
|`--csv`||Path to save the output summary CSV file.|`compression_results.csv`|
|`--figures`||Directory to save visualization plots (only if `--visualize` is set).|`figures`|
|`--help`|`-h`|Show the help message and exit.|–|

## ✨ Example

Let's say you have your original photos in a folder named `holiday_pics` and you want to:

- Compare JPEG and WebP formats.
- Use a quality setting of 90.
- Save the compressed files to a folder called `compressed_output`.
- Save the results summary to `holiday_report.csv`.
- Generate the comparison plots.
- Use 4 CPU cores for the analysis.

You would run this command:

Bash

```
python compression_analyzer.py -i holiday_pics -o compressed_output -q 90 -f jpg webp --csv holiday_report.csv --visualize -w 4
```

The script will then process the images, print results, save the compressed files, create the CSV report, and generate the plots in the `figures` directory!

---
### Input

- Create a directory (e.g., `my_images/`) and place your original images inside it.
- Supported input formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`, `.webp`.

### Output

The script will create/use the following (by default):

1. **Console Output:** Shows progress, results for each image, and average statistics.
2. `compressed_images/`: A new folder containing the compressed versions of your images.
3. `compression_results.csv`: A CSV file with detailed metrics for every image and format analyzed. Perfect for opening in Excel or Google Sheets!
4. `figures/`: (Only if you use the `--visualize` flag) A folder containing `.png` plots comparing the different formats.

## 🤔 How it Works (Briefly)

This script is optimized for speed:

1. **Parallel Processing:** It uses Python's `concurrent.futures` to distribute the work of compressing and analyzing images across multiple CPU cores simultaneously.
2. **Efficient I/O:** When calculating quality metrics, it compresses images into memory buffers first, avoiding unnecessary reads and writes to the hard drive, which speeds up the process.

---
## 📜 License

This project is open-source. You can modify and distribute it as needed.

---

Happy Analyzing! Let me know if you have suggestions or find any bugs.