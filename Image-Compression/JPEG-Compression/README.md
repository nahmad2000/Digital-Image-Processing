# JPEG-DCT Image Compression


This code implements the JPEG compression process steps for color images, with a few modifications. First, an RGB to YCbCr color space conversion is performed to specify the color. Next, the original image is divided into blocks of 8 x 8, and the pixel values within each block are shifted from [0-255] to [-128 to 127], so they can be processed by the DCT.

The DCT is applied to each block from left to right and top to bottom, compressing the block through quantization. However, this code does not perform entropy encoding and decoding.

Overall, this implementation of the JPEG compression process steps without the encoding and decoding can efficiently compress digital color images while maintaining their visual quality.



## Requirements

* numpy
* PIL
* skimage
* cv2
* scipy
* os
* time

## Installation

Install the required libraries by running:

```
pip install numpy
pip install Pillow
pip install scikit-image
pip install opencv-python
pip install scipy
```

### Usage

* Download or clone this repository.
* Place the image you want to compress in the same directory as the JPEG.py file.
* Change the image_path variable in the compress.py file to the name of your image file.
* Run JPEG.py.
* The compressed image will be saved in the same directory.

