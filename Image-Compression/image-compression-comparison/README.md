# Image Compression and Quality Metrics

This code utilizes the pillow library to compress images using its built-in compression algorithms, while also calculating quality metrics for both PNG and JPEG formats. These quality metrics include the compression ratio, mean squared error (MSE), and peak signal-to-noise ratio (PSNR).

For PNG compression, Pillow uses the Deflate algorithm, which is a lossless compression algorithm that combines the LZ77 algorithm and Huffman coding. This algorithm compresses image data efficiently while maintaining high compression ratios and fast compression and decompression speeds. To implement Deflate, Pillow uses the zlib library.

For JPEG compression, Pillow uses the Discrete Cosine Transform (DCT) algorithm to convert image data from the spatial domain to the frequency domain. Pillow then quantizes the coefficients obtained from the DCT and applies entropy encoding using the Huffman coding algorithm to achieve lossy compression. The level of compression is determined by the degree of quantization used on the DCT coefficients. Pillow also allows you to adjust the quality of JPEG compression by changing the level of quantization.

When saving images as WebP, Pillow utilizes the WebP library, which is based on the VP8 video codec. WebP format uses a combination of lossy and lossless compression techniques, such as variable length coding (VLC) and predictive coding, to achieve high compression rates without significant loss of image quality. With Pillow, you can adjust the quality of the image and the level of compression used when saving an image in WebP format.


## Installation

Before running the code, ensure that you have the following packages installed:

* NumPy
* Pillow
* scikit-image

You can install these packages by running the following command:

```
pip install numpy Pillow scikit-image
```


## Usage

1. Place the images you want to compress in the "images" folder in the same path of the python code.
2. Run the code using a Python interpreter.
3. The compressed images will be saved in the "compressed_images" folder and the quality metrics will be displayed in the console.


## Output

The output includes the following information:

1. it will include for each image

* PNG compression ratio, MSE, and PSNR 
* JPEG compression ratio, MSE, and PSNR
* WebP compression ratio, MSE, and PSNR


2. it will include the average 

* PNG compression ratio, MSE, and PSNR 
* JPEG compression ratio, MSE, and PSNR
* WebP compression ratio, MSE, and PSNR