# Image Compression Project

## Overview
This project focuses on image compression using wavelet decomposition and thresholding. The code is implemented in Python and utilizes the PyWavelets library.


## Project Structure
- `EE413Project.ipynb`: Jupyter Notebook containing the main code.
- `wallpaper.jpg`: Sample input image.


## Dependencies
- PyWavelets
- NumPy
- Pillow (PIL)


## Usage
1. Install the required dependencies:
```
pip install pywavelets numpy Pillow
```
2. Open and run the Jupyter Notebook EE413Project.ipynb using a platform like Google Colab.
3. Configure parameters: Modify the parameters in the notebook as needed, such as the input and output paths, wavelet types, decomposition levels, and thresholds.
4. Execute the notebook to perform image compression.


## Parameters
1. input_path: Path to the input image.
2. output_path: Path to save the compressed output image.
3. wave_dec_1: Parameters for the first wavelet decomposition.
4. wave_dec_2: Parameters for the second wavelet decomposition.
5. thresholds_percentage: List of thresholds as a percentage of the maximum wavelet coefficient value.


## Results
The notebook provides information on the size reduction achieved after thresholding and quantization for both 3-level and 5-level decompositions.
