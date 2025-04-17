# -*- coding: utf-8 -*-

import cv2
import numpy as np
from PIL import Image
from skimage import metrics
import os

image_folder_path = "images"

# Create folders to store the modified images
noise_folders = ["Gaussian_Noise_Images", "Salt_Pepper_Noise_Images", "Poisson_Noise_Images", "Speckle_Noise_Images"]

for folder in noise_folders:
    os.makedirs(folder, exist_ok=True)

# Function to add Gaussian noise to an image
def add_gaussian_noise(image):
    mean = 0
    stddev = 50  # Adjust the standard deviation to control the amount of noise
    noise = np.random.normal(mean, stddev, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, noise)
    return noisy_image

# Function to add salt and pepper noise to an image
def add_salt_and_pepper_noise(image):
    amount = 0.05  # Adjust the amount of noise (percentage of pixels to be affected)
    noisy_image = np.copy(image)
    height, width, channels = image.shape
    num_salt = int(amount * height * width * 0.5)
    salt_coords = [np.random.randint(0, height, size=num_salt), np.random.randint(0, width, size=num_salt)]
    num_pepper = int(amount * height * width * 0.5)
    pepper_coords = [np.random.randint(0, height, size=num_pepper), np.random.randint(0, width, size=num_pepper)]
    noisy_image[salt_coords[0], salt_coords[1]] = 255  # Add salt noise
    noisy_image[pepper_coords[0], pepper_coords[1]] = 0  # Add pepper noise
    return noisy_image


# Function to add Poisson noise to an image
def add_poisson_noise(image):
    noisy_image = np.random.poisson(image.astype(np.float64)).astype(np.uint8)
    return noisy_image

# Function to add speckle noise to an image
def add_speckle_noise(image):
    stddev = 50  # Adjust the standard deviation to control the amount of noise
    noise = np.random.normal(0, stddev, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, noise)
    return noisy_image

# Get all image file names from the "images" folder
image_files = [f for f in os.listdir(image_folder_path + "/") if os.path.isfile(os.path.join(image_folder_path + "/", f))]

# Process each image
for image_file in image_files:
    # Load the original image
    image_path = os.path.join(image_folder_path, image_file)
    image = cv2.imread(image_path)

    # Add Gaussian noise
    gaussian_noise_image = add_gaussian_noise(image)
    gaussian_output_path = os.path.join(noise_folders[0], os.path.splitext(image_file)[0] + "_gaussian.TIFF")
    cv2.imwrite(gaussian_output_path, gaussian_noise_image)

    # Add salt and pepper noise
    salt_pepper_noise_image = add_salt_and_pepper_noise(image)
    salt_pepper_output_path = os.path.join(noise_folders[1], os.path.splitext(image_file)[0] + "_salt_pepper.TIFF")
    cv2.imwrite(salt_pepper_output_path, salt_pepper_noise_image)

    # Add Poisson noise
    poisson_noise_image = add_poisson_noise(image)
    poisson_output_path = os.path.join(noise_folders[2], os.path.splitext(image_file)[0] + "_poisson.TIFF")
    cv2.imwrite(poisson_output_path, poisson_noise_image)

    # Add speckle noise
    speckle_noise_image = add_speckle_noise(image)
    speckle_output_path = os.path.join(noise_folders[3], os.path.splitext(image_file)[0] + "_speckle.TIFF")
    cv2.imwrite(speckle_output_path, speckle_noise_image)



def processing(folder_name, new_folder_name):

    parameters = {
        "folder_name": folder_name,
        "new_folder_name": new_folder_name,
        "images_names": [],
        "images_full_path" : [],
        "jpg_full_path": [],
        "png_full_path": [],
        "webp_full_path": [],
    }

    # Create the folder to save the compressed images
    if not os.path.exists(parameters["new_folder_name"]):
        os.makedirs(parameters["new_folder_name"])

    # Get all image file names from the "images" folder
    parameters["images_names"] = [f for f in os.listdir(parameters["folder_name"] + "/") if os.path.isfile(os.path.join(parameters["folder_name"] + "/", f))]
    jpg_images_names = [img.split(".", 1)[0] + ".jpg" for img in parameters["images_names"]]
    png_images_names = [img.split(".", 1)[0] + ".png" for img in parameters["images_names"]]
    webp_images_names = [img.split(".", 1)[0] + ".webp" for img in parameters["images_names"]]

    # Get the full path
    parameters["images_full_path"] = [os.path.join(parameters["folder_name"], img) for img in parameters["images_names"]]
    parameters["jpg_full_path"] = [os.path.join(parameters["new_folder_name"], img) for img in jpg_images_names]
    parameters["png_full_path"] = [os.path.join(parameters["new_folder_name"], img) for img in png_images_names]
    parameters["webp_full_path"] = [os.path.join(parameters["new_folder_name"], img) for img in webp_images_names]

    # Perform Compression and saving images
    def compression(parameters):
        for img_path, jpg_path, png_path, webp_path in zip(parameters["images_full_path"], parameters["jpg_full_path"], parameters["png_full_path"], parameters["webp_full_path"]):
            image = Image.open(img_path)
            image.save(jpg_path) # JPEG Compression
            image.save(png_path) # PNG Compression
            image.save(webp_path) # WEBP Compression

    compression(parameters)

  

    def calculate_metrics(parameters):

        CR = {"jpg" : [], "png" : [], "webp" : []}
        MSE = {"jpg" : [], "png" : [], "webp" : []}
        PSNR = {"jpg" : [], "png" : [], "webp" : []}
        SSIM = {"jpg" : [], "png" : [], "webp" : []}

        # Open a file to save the results
        results_file = open(parameters["folder_name"]+ "_results.txt", "w")

        for img_name, img_path, jpg_path, png_path, webp_path in zip(parameters["images_names"], parameters["images_full_path"], parameters["jpg_full_path"], parameters["png_full_path"], parameters["webp_full_path"]):
            
            # Calculate the sizes
            img_size = os.path.getsize(img_path)
            jpg_size = os.path.getsize(jpg_path)
            png_size = os.path.getsize(png_path)
            webp_size = os.path.getsize(webp_path)

            # Open the images
            img = Image.open(img_path)
            jpg = Image.open(jpg_path)
            png = Image.open(png_path)
            webp = Image.open(webp_path)

            # Convert images to numpy arrays
            img_arr = np.asarray(img)
            jpg_arr = np.asarray(jpg)
            png_arr = np.asarray(png)
            webp_arr = np.asarray(webp)

            # Write the results to the file
            results_file.write("===================================================\n===================================================\n")
            results_file.write("QUALITY METRICS FOR THE " + img_name + " IMAGE\n")

            # Store CR
            CR["jpg"].append(img_size/jpg_size)
            CR["png"].append(img_size/png_size)
            CR["webp"].append(img_size/webp_size)

            # Write CR
            results_file.write("************CR************\n")
            results_file.write("JPEG ==> " + str(CR["jpg"][-1]) + "\n")
            results_file.write("PNG ==> " + str(CR["png"][-1]) + "\n")
            results_file.write("WEBP ==> " + str(CR["webp"][-1]) + "\n")

            # Store MSE
            MSE["jpg"].append(metrics.mean_squared_error(img_arr, jpg_arr))
            MSE["png"].append(metrics.mean_squared_error(img_arr, png_arr))
            MSE["webp"].append(metrics.mean_squared_error(img_arr, webp_arr))

            # Write MSE
            results_file.write("************MSE************\n")
            results_file.write("JPEG ==> " + str(MSE["jpg"][-1]) + "\n")
            results_file.write("PNG ==> " + str(MSE["png"][-1]) + "\n")
            results_file.write("WEBP ==> " + str( MSE["webp"][-1]) + "\n")

            # Store PSNR
            PSNR["jpg"].append(metrics.peak_signal_noise_ratio(img_arr, jpg_arr, data_range=None))
            PSNR["png"].append(metrics.peak_signal_noise_ratio(img_arr, png_arr, data_range=None))
            PSNR["webp"].append(metrics.peak_signal_noise_ratio(img_arr, webp_arr, data_range=None))

            # Write PSNR
            results_file.write("************PSNR************\n")
            results_file.write("JPEG ==> " + str(PSNR["jpg"][-1]) + "\n")
            results_file.write("PNG ==> " + str(PSNR["png"][-1]) + "\n")
            results_file.write("WEBP ==> " + str(PSNR["webp"][-1]) + "\n")

            # Store SSIM
            SSIM["jpg"].append(metrics.structural_similarity(img_arr, jpg_arr, win_size=3, channel_axis=None))
            SSIM["png"].append(metrics.structural_similarity(img_arr, png_arr, win_size=3, channel_axis=None))
            SSIM["webp"].append(metrics.structural_similarity(img_arr, webp_arr, win_size=3, channel_axis=None))

            # Write SSIM
            results_file.write("************SSIM************\n")
            results_file.write("JPEG ==> " + str(SSIM["jpg"][-1]) + "\n")
            results_file.write("PNG ==> " + str(SSIM["png"][-1]) + "\n")
            results_file.write("WEBP ==> " + str(SSIM["webp"][-1]) + "\n")


        # Calculate the average
        number_of_images = len(parameters["images_names"])
        results_file.write("===================================================\n===================================================\n")
        results_file.write("AVERAGE QUALITY METRICS FOR ALL THE IMAGE\n")
        results_file.write("************CR************\n")
        results_file.write("JPEG ==>" + str(sum(CR["jpg"])/number_of_images) + "\n")
        results_file.write("PNG ==>" + str(sum(CR["png"])/number_of_images) + "\n")
        results_file.write("WEBP ==>" + str(sum(CR["webp"])/number_of_images) + "\n")
        results_file.write("************MSE************\n")
        results_file.write("JPEG ==>" + str(sum(MSE["jpg"])/number_of_images) + "\n")
        results_file.write("PNG ==>" + str(sum(MSE["png"])/number_of_images) + "\n")
        results_file.write("WEBP ==>" + str(sum(MSE["webp"])/number_of_images) + "\n")
        results_file.write("************PSNR************\n")
        results_file.write("JPEG ==>" + str(sum(PSNR["jpg"])/number_of_images) + "\n")
        results_file.write("PNG ==>" + str(sum(PSNR["png"])/number_of_images) + "\n")
        results_file.write("WEBP ==>" + str(sum(PSNR["webp"])/number_of_images) + "\n")
        results_file.write("************SSIM************\n")
        results_file.write("JPEG ==>" + str(sum(SSIM["jpg"])/number_of_images) + "\n")
        results_file.write("PNG ==>" + str(sum(SSIM["png"])/number_of_images) + "\n")
        results_file.write("WEBP ==>" + str(sum(SSIM["webp"])/number_of_images) + "\n")

        # Close the results file
        results_file.close()

    calculate_metrics(parameters)


noise_folders.insert(0, image_folder_path)
output_folders = [f + "_compressed" for f in noise_folders]

for input, output in zip(noise_folders, output_folders):
    processing(input, output)
