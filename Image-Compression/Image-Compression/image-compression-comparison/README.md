# Image Compression and Quality Metrics


This code utilizes the Pillow library to compress images using its built-in compression algorithms while calculating quality metrics for PNG, JPEG, and WebP formats. The quality metrics include the compression ratio, mean squared error (MSE), peak signal-to-noise ratio (PSNR), and structural similarity index (SSIM).


## Compression Algorithms

The code employs the following compression algorithms:

### PNG Compression

Pillow utilizes the Deflate algorithm for PNG compression. Deflate combines the LZ77 algorithm and Huffman coding, resulting in an efficient lossless compression algorithm. It achieves high compression ratios with fast compression and decompression speeds. The implementation of Deflate in Pillow relies on the zlib library.


### JPEG Compression

For JPEG compression, Pillow uses the Discrete Cosine Transform (DCT) algorithm. This algorithm converts image data from the spatial domain to the frequency domain. Pillow quantizes the DCT coefficients and applies entropy encoding using the Huffman coding algorithm to achieve lossy compression. The compression level is determined by the degree of quantization applied to the DCT coefficients. Additionally, the quality of JPEG compression can be adjusted by changing the quantization level.

### WebP Compression

Pillow utilizes the WebP library, based on the VP8 video codec, for WebP compression. WebP format employs a combination of lossy and lossless compression techniques, including variable length coding (VLC) and predictive coding, to achieve high compression rates without significant loss of image quality. Pillow allows you to adjust the quality of the image and the compression level when saving an image in WebP format.



## Installation

Before running the code, ensure that you have the following packages installed:

* NumPy
* Pillow
* scikit-image
* OpenCV (cv2)

You can install these packages by running the following command:

```
pip install numpy Pillow scikit-image opencv-python

```


## Usage and Output


To utilize the image compression and quality assessment code, follow these steps:

1. Prepare the Images: Place the images you want to compress in the "images" folder located in the same directory as the Python code. Alternatively, you can specify a different folder path by modifying the "image_folder_path" variable in the code.

2. Run the Code: Execute the code using a Python interpreter.

3. Folder Creation: After running the program, the following folders will be created:

	* Noisy Image Folders: Four new folders will be created, each containing noisy images based on the specific noise type. The folder names will correspond to the respective noise types.

	* Compressed Image Folders: Five new folders will be created, with "_compressed" appended to their names. These folders will store the compressed images for the compression that has been on the crossponding folder.

4. TXT File Generation: Furthermore, five TXT files will be generated, each representing the quality metrics for a specific folder.

Each TXT file will include the following metrics for each image:

* Compression Ratio (JPEG, PNG, WEBP)
* Mean Squared Error (MSE) (JPEG, PNG, WEBP)
* Peak Signal-to-Noise Ratio (PSNR) (JPEG, PNG, WEBP)
* Structural Similarity Index (SSIM) (JPEG, PNG, WEBP)

Additionally, each TXT file will provide the average metrics for each compression algorithm:

* Average Compression Ratio (JPEG, PNG, WEBP)
* Average MSE (JPEG, PNG, WEBP)
* Average PSNR (JPEG, PNG, WEBP)
* Average SSIM (JPEG, PNG, WEBP)

By following these steps, you can conveniently assess the quality of image compression using various algorithms and evaluate the performance under different types of noise.


# Code Modifications:

The code has been modified to enhance its functionality, readability, and performance. The following improvements have been implemented:

## structural similarity index (SSIM)

The code now includes the structural similarity index (SSIM) as a quality metric. SSIM offers a more comprehensive evaluation of image similarity by considering both pixel-level differences and structural information. It captures perceptual quality by assessing the similarity of structural patterns and textures between images. SSIM is particularly valuable for detecting distortions related to texture, edges, and global structure, providing a more accurate representation of image quality perceived by humans.


## Noise Addition

To provide a better assessment of each compression algorithm's performance, the code introduces four types of noise: Gaussian, Salt and Pepper, Poisson, and Speckle. These noises are added to all images, resulting in new sets of noisy images. The code organizes the noisy images into specific folders corresponding to each noise type. This enables a comprehensive evaluation of compression algorithms under various noise conditions, allowing for a better understanding of their performance characteristics.

## Output Representation

The code now generates TXT files to present the output in a more organized and concise manner. Instead of displaying the output solely in the console, the code stores the output information in TXT files. Each TXT file provides clear and structured data, facilitating easier analysis and interpretation of the quality metrics for each image and compression algorithm.

## Code Refactoring: 

The code has undergone significant refactoring to improve its clarity, maintainability, and efficiency. The key refinements include:

* Restructuring the code into modular functions, enhancing code organization and reusability.
* Adopting meaningful variable and function names, making the code more self-explanatory.
* Incorporating proper documentation and comments throughout the code to explain its purpose, inputs, and outputs.
* Optimizing code logic and reducing redundancy to improve performance.

These code modifications ensure that the codebase is easier to understand, maintain, and extend. The overall result is a more powerful, user-friendly, and reliable image compression and quality assessment tool.

Please feel free to reach out if you have any further questions or need additional assistance!