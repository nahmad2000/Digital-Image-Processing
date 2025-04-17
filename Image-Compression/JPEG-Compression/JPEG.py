# -*- coding: utf-8 -*-

#---------------------
# Importing Libraries
#---------------------
import numpy as np
from PIL import Image
from scipy.fftpack import dct, idct
from skimage import metrics
import cv2
import os
import time

#---------------------
# Stating the time and Loading the image
#---------------------

start_time = time.time()

image_path = "r30109005t.tif"
image = Image.open(image_path)

# Convert the image to a numpy array
image_array = np.array(image)



#---------------------
# Defining Functions
#---------------------

def RGB2YCbCr(rgb_array):

    # Conversion matrix from RGB to YCbCr
    global conversion_matrix
    conversion_matrix = np.array([[0.299, 0.587, 0.114],
                                  [-0.169, -0.334, 0.500],
                                  [0.500, -0.419, -0.081]])

    # Shifts for Cb, and Cr components (the Y shift is zero)
    Cb_shift = 128
    Cr_shift = 128

    # Initialize YCbCr image array with zeros
    ycbcr_array = np.zeros_like(rgb_array, dtype=np.float32)

    # Iterate over each pixel in the image
    for i in range(rgb_array.shape[0]):
        for j in range(rgb_array.shape[1]):
            # Convert RGB to YCbCr for each pixel
            rgb_pixel = rgb_array[i, j, :]
            ycbcr_pixel = np.dot(conversion_matrix, rgb_pixel)
            ycbcr_pixel[1] += Cb_shift
            ycbcr_pixel[2] += Cr_shift
            ycbcr_array[i, j, :] = ycbcr_pixel

    return ycbcr_array

def shifting_Y(ycbcr_array):

    # Iterate over each pixel in the image
    for i in range(ycbcr_array.shape[0]):
        for j in range(ycbcr_array.shape[1]):
          ycbcr_array[i,j,0] -= 128
    return ycbcr_array

def divide_blocks(ycbcr_array):
  
    # Determine the dimensions of the image
    height, width, _ = ycbcr_array.shape

    # Calculate the number of blocks in the height and width directions
    num_blocks_h = height // 8
    num_blocks_w = width // 8

    # Initialize the list of blocks
    blocks = []

    # Loop over each block and extract it from the YCbCr image
    for i in range(num_blocks_h):
        for j in range(num_blocks_w):
            # Extract the block from the YCbCr image
            ycbcr_block = ycbcr_array[i*8:(i+1)*8, j*8:(j+1)*8, :]
            
            # Append the block to the list
            blocks.append(ycbcr_block)

    # Convert the list of blocks to a numpy array
    blocks = np.array(blocks)

    return blocks

def DCT(blocks):

    for i in range(blocks.shape[0]):
     
        blocks[i,:,:,0] = dct(dct(blocks[i,:,:,0].T, norm = 'ortho').T, norm = 'ortho') # finding DCT coff of Block_Y
        blocks[i,:,:,1] = dct(dct(blocks[i,:,:,1].T, norm = 'ortho').T, norm = 'ortho') # finding DCT coff of Block_Cb
        blocks[i,:,:,2] = dct(dct(blocks[i,:,:,2].T, norm = 'ortho').T, norm = 'ortho') # finding DCT coff of Block_Cr
        
    return blocks

def Quantization(dct_blocks): 

    # The shape of DCT blocks is (number of blocks, 8, 8, 3)

    # Define the standard quantization matrices for Y, Cb, and Cr components
    global y_quant_mtx
    global cbcr_quant_mtx
    
    y_quant_mtx = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                            [12, 12, 14, 19, 26, 58, 60, 55],
                            [14, 13, 16, 24, 40, 57, 69, 56],
                            [14, 17, 22, 29, 51, 87, 80, 62],
                            [18, 22, 37, 56, 68, 109, 103, 77],
                            [24, 35, 55, 64, 81, 104, 113, 92],
                            [49, 64, 78, 87, 103, 121, 120, 101],
                            [72, 92, 95, 98, 112, 100, 103, 99]])

    cbcr_quant_mtx = np.array([[17, 18, 24, 47, 99, 99, 99, 99],
                               [18, 21, 26, 66, 99, 99, 99, 99],
                               [24, 26, 56, 99, 99, 99, 99, 99],
                               [47, 66, 99, 99, 99, 99, 99, 99],
                               [99, 99, 99, 99, 99, 99, 99, 99],
                               [99, 99, 99, 99, 99, 99, 99, 99],
                               [99, 99, 99, 99, 99, 99, 99, 99],
                               [99, 99, 99, 99, 99, 99, 99, 99]])

    # Iterate over all blocks
    for i in range(dct_blocks.shape[0]):
        # Apply quantization to the Y component
        dct_blocks[i, :, :, 0] = np.round(dct_blocks[i, :, :, 0] / y_quant_mtx)

        # Apply quantization to the Cb and Cr components
        dct_blocks[i, :, :, 1] = np.round(dct_blocks[i, :, :, 1] / cbcr_quant_mtx) 
        dct_blocks[i, :, :, 2] = np.round(dct_blocks[i, :, :, 2] / cbcr_quant_mtx) 

    return dct_blocks

def deQuantization(dct_blocks): 

    # The shape of DCT blocks is (number of blocks, 8, 8, 3)
    # y_quant_mtx and cbcr_quant_mtx are global variables defined in Quantization function
    # because quantization matrices need to be the same as the ones used during quantization

    # Iterate over all blocks
    for i in range(dct_blocks.shape[0]):
        # Apply dequantization to the Y component
        dct_blocks[i, :, :, 0] = np.multiply(dct_blocks[i, :, :, 0], y_quant_mtx)

        # Apply dequantization to the Cb and Cr components
        dct_blocks[i, :, :, 1] = np.multiply(dct_blocks[i, :, :, 1], cbcr_quant_mtx) 
        dct_blocks[i, :, :, 2] = np.multiply(dct_blocks[i, :, :, 2], cbcr_quant_mtx) 

    return dct_blocks

def inverse_DCT(blocks):


    for i in range(blocks.shape[0]):
     
        blocks[i,:,:,0] = idct(idct(blocks[i,:,:,0].T, norm = 'ortho').T, norm = 'ortho') # finding inverse DCT coff of Block_Y
        blocks[i,:,:,1] = idct(idct(blocks[i,:,:,1].T, norm = 'ortho').T, norm = 'ortho') # finding inverse DCT coff of Block_Cb
        blocks[i,:,:,2] = idct(idct(blocks[i,:,:,2].T, norm = 'ortho').T, norm = 'ortho') # finding inverse DCT coff of Block_Cr
        
    return blocks

def add_blocks(blocks, height, width):
  
    # Calculate the number of blocks in the height and width directions
    num_blocks_h = height // 8
    num_blocks_w = width // 8

    # Initialize the empty array to hold the reconstructed image
    add_blocks = np.zeros((height, width, 3))

    # Loop over each block and place it back into the reconstructed image
    block_index = 0
    for i in range(num_blocks_h):
        for j in range(num_blocks_w):
            # Get the block from the list of blocks
            block = blocks[block_index]
            block_index += 1

            # Determine the location to place the block in the reconstructed image
            y_start = i*8
            y_end = (i+1)*8
            x_start = j*8
            x_end = (j+1)*8

            # Place the block into the reconstructed image
            add_blocks[y_start:y_end, x_start:x_end, :] = block

    return add_blocks

def shifting_Y_inverse(ycbcr_array):

    # Iterate over each pixel in the image
    for i in range(ycbcr_array.shape[0]):
        for j in range(ycbcr_array.shape[1]):
          ycbcr_array[i,j,0] += 128
    return ycbcr_array

def YCbCr2RGB(ycbcr_array):

    # Inverse conversion matrix from YCbCr to RGB
    conversion_matrix_inv = np.linalg.inv(conversion_matrix)

    # Shifts for Cb, and Cr components (The Y shift is 0)
    Cb_shift = 128
    Cr_shift = 128

    # Initialize RGB image array with zeros
    rgb_array = np.zeros_like(ycbcr_array, dtype=np.float32)

    # Iterate over each pixel in the image
    for i in range(ycbcr_array.shape[0]):
        for j in range(ycbcr_array.shape[1]):
            # Convert YCbCr to RGB for each pixel
            ycbcr_pixel = ycbcr_array[i, j, :]
            ycbcr_pixel[1] -= Cb_shift
            ycbcr_pixel[2] -= Cr_shift
            rgb_pixel = np.dot(conversion_matrix_inv, ycbcr_pixel)
            rgb_array[i, j, :] = rgb_pixel


    # Convert to uint8 data type
    rgb_array = rgb_array.astype(np.uint8)

    return rgb_array



#---------------------
# Compression
#---------------------

# converting from RGB to YCbCr
ycbcr_array = RGB2YCbCr(image_array)

# shifting the Y by -128
ycbcr_array = shifting_Y(ycbcr_array)

# dividing the image into 8x8 blocks ==> blocks.shape = (number of blocks, 8, 8, 3)
blocks = divide_blocks(ycbcr_array)

# finding the DCT of each block
dct_blocks = DCT(blocks)

# quantizing the DCT blocks
quantized_blocks = Quantization(dct_blocks)

# Encoding



#---------------------
# Decompression
#---------------------

# Decoding

# deQuantization
dequantized_blocks = deQuantization(quantized_blocks)

# inverse_DCT_Version2
inverse_dct_blocks = inverse_DCT(dequantized_blocks)

# add_blocks
added_blocks = add_blocks(inverse_dct_blocks, ycbcr_array.shape[0], ycbcr_array.shape[1])

# shifting_Y_inverse
shifted_ycbcr_array = shifting_Y_inverse(added_blocks)

# YCbCr2RGB
rgb_array = YCbCr2RGB(shifted_ycbcr_array)



#---------------------
# Saving the compressed image and reading the images
#---------------------

Image.fromarray(rgb_array, 'RGB').save('compressed_image.jpeg')
compressed_img = cv2.imread("compressed_image.jpeg", 1)
refference_img = cv2.imread("r30109005t.tif", 1)


#---------------------
# image quality metrics
#---------------------


# get the size of the images
refference_size = os.path.getsize("r30109005t.tif")
compressed_size = os.path.getsize('compressed_image.jpeg')

# Calculate the compression ratio
compression_ratio = refference_size / compressed_size
print('Compression Ratio:', compression_ratio)

# Calculate mse
mse = metrics.mean_squared_error(refference_img, compressed_img)
print("mse:", mse)

# Calculate PSNR
psnr = metrics.peak_signal_noise_ratio(refference_img, compressed_img, data_range=None)
print("PSNR:", psnr)

#---------------------
# end
#---------------------

end_time = time.time()
total_time = end_time - start_time

print(f"Total time taken: {total_time} seconds")