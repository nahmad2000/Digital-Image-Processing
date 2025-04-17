# -*- coding: utf-8 -*-

import numpy as np
from PIL import Image
from skimage import metrics
import os

images_names_list = []
images_names_list_without_ext = []
images_names_list_png= []
images_names_list_jpg= []
images_names_list_webp= []
compression_data_list = [] # will contain dict similar to "compression_data_dict"



def ShowOutputs(compression_data_list, average_compression_data_list, images_names_list_without_ext):
    
    for i in range(len(compression_data_list)):
        
        print("image quality metrics for the", images_names_list_without_ext[i],"image:\n")
        print("-----------------")
        print("PNG compression:\n")
        print('Compression Ratio:', compression_data_list[i]["png"][0])
        print('mse', compression_data_list[i]["png"][1])
        print('PSNR:', compression_data_list[i]["png"][2])
        print("-----------------")
        print("JPEG compression:\n")
        print('Compression Ratio:', compression_data_list[i]["jpg"][0])
        print('mse', compression_data_list[i]["jpg"][1])
        print('PSNR:', compression_data_list[i]["jpg"][2])
        print("-----------------")
        print("WebP compression:\n")
        print('Compression Ratio:', compression_data_list[i]["webp"][0])
        print('mse', compression_data_list[i]["webp"][1])
        print('PSNR:', compression_data_list[i]["webp"][2])
        print("====================================================")
        print("====================================================")


    print("==============Average Quality Metrics===============")
    print("====================================================")
    print("====================================================")
    print("PNG compression:\n")
    print("cr =", average_compression_data_list[0]["png"])
    print("mse =", average_compression_data_list[1]["png"])
    print("psnr =", average_compression_data_list[2]["png"])
    print("-----------------")
    print("JPEG compression:\n")
    print("cr =", average_compression_data_list[0]["jpg"])
    print("mse =", average_compression_data_list[1]["jpg"])
    print("psnr =", average_compression_data_list[2]["jpg"])
    print("-----------------")
    print("WebP compression:\n")
    print("cr =", average_compression_data_list[0]["webp"])
    print("mse =", average_compression_data_list[1]["webp"])
    print("psnr =", average_compression_data_list[2]["webp"])
    print("-----------------")
    

def AverageOutputs(compression_data_list):
    
    average_cr = {"png": 0, "jpg": 0, "webp": 0}
    average_mse = {"png": 0, "jpg": 0, "webp": 0}
    average_psnr = {"png": 0, "jpg": 0, "webp": 0}
    
    for i in range(len(compression_data_list)):
        
        average_cr["png"] += compression_data_list[i]["png"][0]
        average_cr["jpg"] += compression_data_list[i]["jpg"][0]
        average_cr["webp"] += compression_data_list[i]["webp"][0]
        
        average_mse["png"] += compression_data_list[i]["png"][1]
        average_mse["jpg"] += compression_data_list[i]["jpg"][1]
        average_mse["webp"] += compression_data_list[i]["webp"][1]
        
        average_psnr["png"] += compression_data_list[i]["png"][2]
        average_psnr["jpg"] += compression_data_list[i]["jpg"][2]
        average_psnr["webp"] += compression_data_list[i]["webp"][2]
    
    average_cr = {k: v /len(compression_data_list) for k, v in average_cr.items()}
    average_mse = {k: v /len(compression_data_list) for k, v in average_mse.items()}
    average_psnr = {k: v /len(compression_data_list) for k, v in average_psnr.items()}
    
    average_compression_data_list = [average_cr, average_mse, average_psnr]
    
    return average_compression_data_list



if not os.path.exists('compressed_images'):
    os.mkdir('compressed_images')

# Get all image file names from the "images" folder
image_folder_path = 'images/'
image_files = [f for f in os.listdir(image_folder_path) if os.path.isfile(os.path.join(image_folder_path, f))]

number_of_images = len(image_files)

for i in range(number_of_images):
    image_path = image_folder_path + image_files[i]
    images_names_list.append(image_path)

for i in range(number_of_images):
    image = Image.open(images_names_list[i])
    images_names_list_without_ext.append(image_files[i].split('.')[0])
    images_names_list_png.append('compressed_images/' + images_names_list_without_ext[i] + ' _compressed' + '.png')
    images_names_list_jpg.append('compressed_images/' + images_names_list_without_ext[i] + ' _compressed' + '.jpg')
    images_names_list_webp.append('compressed_images/' + images_names_list_without_ext[i] + ' _compressed' + '.webp')
    image.save(images_names_list_png[i])
    image.save(images_names_list_jpg[i])
    image.save(images_names_list_webp[i])

#---------------------
# image quality metrics
#---------------------
for i in range(number_of_images):
    
    compression_data_dict = {"png": [0,0,0], "jpg": [0,0,0], "webp": [0,0,0]}
    
    # get the size of the images
    refference_size = os.path.getsize(images_names_list[i])
    png_compressed_size = os.path.getsize(images_names_list_png[i])
    jpg_compressed_size = os.path.getsize(images_names_list_jpg[i])
    webp_compressed_size = os.path.getsize(images_names_list_webp[i])
    
    # Calculate the compression ratio
    png_cr = refference_size / png_compressed_size
    jpg_cr = refference_size / jpg_compressed_size
    webp_cr = refference_size / webp_compressed_size
    
    # Open the images
    refference_img = Image.open(images_names_list[i])
    png_img = Image.open(images_names_list_png[i])
    jpg_img = Image.open(images_names_list_jpg[i])
    webp_img = Image.open(images_names_list_webp[i])
    
    # Converting the images into numpy arrays
    png_img_array = np.asarray(png_img)
    jpg_img_array = np.asarray(jpg_img)
    webp_img_array = np.asarray(webp_img)
    refference_img_array = np.asarray(refference_img)
    
    # Calculate mse
    png_mse = metrics.mean_squared_error(refference_img_array, png_img_array)
    jpg_mse = metrics.mean_squared_error(refference_img_array, jpg_img_array)
    webp_mse = metrics.mean_squared_error(refference_img_array, webp_img_array)
    
    # Calculate PSNR
    png_psnr = metrics.peak_signal_noise_ratio(refference_img_array, png_img_array, data_range=None)
    jpg_psnr = metrics.peak_signal_noise_ratio(refference_img_array, jpg_img_array, data_range=None)
    webp_psnr = metrics.peak_signal_noise_ratio(refference_img_array, webp_img_array, data_range=None)
    
    compression_data_dict["png"][:3] = [png_cr, png_mse, png_psnr]
    compression_data_dict["jpg"][:3] = [jpg_cr, jpg_mse, jpg_psnr]
    compression_data_dict["webp"][:3] = [webp_cr, webp_mse, webp_psnr]
    
    compression_data_list.append(compression_data_dict)
    

# Calculate Average
average_compression_data_list = AverageOutputs(compression_data_list)

# Showing Output
ShowOutputs(compression_data_list, average_compression_data_list, images_names_list_without_ext)
