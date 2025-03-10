# -*- coding: utf-8 -*-
"""Untitled0.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1zHCN9EH2YKICt3wiuJSzzSTo-9FALmZG
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

def get_user_input():
    """
    Prompts the user to select a transformation type and enter the required parameters.

    Returns:
        tuple: (transformation_choice, parameters)
    """

    # Explanation before asking for input
    print("\n=== Geometric Transformations ===")
    print("This program applies various geometric transformations on an image, including:")
    print("🔹 Rotation (1) - Rotate the image by a given angle.")
    print("🔹 Scaling (2) - Resize the image with scaling factors.")
    print("🔹 Translation (3) - Shift the image in x and y directions.")
    print("🔹 Shear (Vertical) (4) - Skew the image vertically.")
    print("🔹 Shear (Horizontal) (5) - Skew the image horizontally.")
    print("===================================")

    while True:
        try:
            choice = int(input("\nEnter the transformation number (1-5): "))
            if choice not in [1, 2, 3, 4, 5]:
                print("⚠️ Invalid choice! Please enter a number between 1 and 5.")
                continue

            # Collect additional parameters based on transformation type
            if choice == 1:  # Rotation
                angle = float(input("Enter rotation angle (in degrees): "))
                return choice, (angle,)

            elif choice == 2:  # Scaling
                sx = float(input("Enter scaling factor for x (Sx): "))
                sy = float(input("Enter scaling factor for y (Sy): "))
                return choice, (sx, sy)

            elif choice == 3:  # Translation
                tx = int(input("Enter translation in x direction (Tx): "))
                ty = int(input("Enter translation in y direction (Ty): "))
                return choice, (tx, ty)

            elif choice == 4:  # Vertical Shear
                sv = float(input("Enter vertical shearing factor (Sv): "))
                return choice, (sv,)

            elif choice == 5:  # Horizontal Shear
                sh = float(input("Enter horizontal shearing factor (Sh): "))
                return choice, (sh,)

        except ValueError:
            print("⚠️ Invalid input! Please enter numeric values.")


def apply_rotation(image, angle):
    """Rotates an image manually by the given angle (in degrees)."""

    # Convert angle to radians
    theta = np.radians(angle)

    # Get image dimensions
    h, w = image.shape

    # Compute the center of the image
    cx, cy = w // 2, h // 2

    # Define the rotation matrix
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])

    # Create an empty output image
    rotated_image = np.ones((h, w), dtype=image.dtype) * 255
    # Loop over each pixel in the output image
    for y in range(h):
        for x in range(w):
            # Translate pixel to origin (center)
            x_shifted = x - cx
            y_shifted = y - cy

            # Apply rotation transformation
            new_coords = rotation_matrix @ np.array([x_shifted, y_shifted])
            new_x, new_y = new_coords[0] + cx, new_coords[1] + cy

            # Ensure new coordinates are within bounds
            if 0 <= int(new_x) < w and 0 <= int(new_y) < h:
                rotated_image[int(new_y), int(new_x)] = image[y, x]

    return rotated_image


def apply_scaling(image, sx, sy):
    """Scales an image manually by the given factors (sx, sy), supporting negative scaling (flipping)."""

    # Get original dimensions
    h, w = image.shape

    # Compute new dimensions (absolute values to avoid negative sizes)
    new_h, new_w = int(h * abs(sy)), int(w * abs(sx))

    # Create an empty output image with new dimensions
    scaled_image = np.ones((new_h, new_w), dtype=image.dtype) * 255

    # Loop through the original image
    for y in range(h):
        for x in range(w):
            # Compute new coordinates
            new_x = int(x * abs(sx))
            new_y = int(y * abs(sy))

            # Ensure new coordinates are within bounds
            if 0 <= new_x < new_w and 0 <= new_y < new_h:
                scaled_image[new_y, new_x] = image[y, x]

    # **Manually flip the image if scaling factors were negative**

    # Flip horizontally if sx < 0 (reverse columns)
    if sx < 0:
        flipped_image = np.zeros_like(scaled_image)
        for i in range(new_h):
            for j in range(new_w):
                flipped_image[i, j] = scaled_image[i, new_w - 1 - j]  # Swap left and right
        scaled_image = flipped_image

    # Flip vertically if sy < 0 (reverse rows)
    if sy < 0:
        flipped_image = np.zeros_like(scaled_image)
        for i in range(new_h):
            for j in range(new_w):
                flipped_image[i, j] = scaled_image[new_h - 1 - i, j]  # Swap top and bottom
        scaled_image = flipped_image

    return scaled_image


def apply_translation(image, tx, ty):
    """Translates an image manually by shifting pixels by (Tx, Ty)."""

    # Get image dimensions
    h, w = image.shape

    # Create an empty output image (same size)
    translated_image = np.ones((h, w), dtype=image.dtype) * 255

    # Loop through each pixel in the original image
    for y in range(h):
        for x in range(w):
            # Compute new coordinates
            new_x = x + tx
            new_y = y + ty

            # Ensure the new coordinates are within bounds
            if 0 <= new_x < w and 0 <= new_y < h:
                translated_image[new_y, new_x] = image[y, x]

    return translated_image


def apply_vertical_shear(image, sv):
    """Applies vertical shearing."""

    # Get original dimensions
    h, w = image.shape

    # Compute new width to accommodate shearing (to prevent cutting)
    new_h = int(h + abs(sv) * w)

    # Create an empty output image with expanded width
    sheared_image = np.ones((new_h, w), dtype=image.dtype) * 255  # White background

    # Define shearing transformation matrix (2×2)
    shear_matrix = np.array([
        [1, 0],  # x changes based on y
        [sv, 1]    # y remains the same
    ])

    # Loop through each pixel
    for y in range(h):
        for x in range(w):
            # Apply matrix multiplication
            new_coords = shear_matrix @ np.array([x, y])
            new_x, new_y = int(new_coords[0]), int(new_coords[1])

            # Adjust new_x to avoid negative indices (shift image right)
            new_y += int(abs(sv) * w) // 2

            # Ensure new coordinates are within bounds
            if 0 <= new_x < w and 0 <= new_y < new_h:
                sheared_image[new_y, new_x] = image[y, x]

    return sheared_image


def apply_horizontal_shear(image, sh):
    """Applies horizontal shearing."""

    # Get original dimensions
    h, w = image.shape

    # Compute new width to accommodate shearing (to prevent cutting)
    new_w = int(w + abs(sh) * h)

    # Create an empty output image with expanded width
    sheared_image = np.ones((h, new_w), dtype=image.dtype) * 255  # White background

    # Define shearing transformation matrix (2×2)
    shear_matrix = np.array([
        [1, sh],  # x changes based on y
        [0, 1]    # y remains the same
    ])

    # Loop through each pixel
    for y in range(h):
        for x in range(w):
            # Apply matrix multiplication
            new_coords = shear_matrix @ np.array([x, y])
            new_x, new_y = int(new_coords[0]), int(new_coords[1])

            # Adjust new_x to avoid negative indices (shift image right)
            new_x += int(abs(sh) * h) // 2

            # Ensure new coordinates are within bounds
            if 0 <= new_x < new_w and 0 <= new_y < h:
                sheared_image[new_y, new_x] = image[y, x]

    return sheared_image


def apply_transformation(image, choice, params):
    """Applies the selected transformation to the image."""
    if choice == 1:
        return apply_rotation(image, params[0])
    elif choice == 2:
        return apply_scaling(image, params[0], params[1])
    elif choice == 3:
        return apply_translation(image, params[0], params[1])
    elif choice == 4:
        return apply_vertical_shear(image, params[0])
    elif choice == 5:
        return apply_horizontal_shear(image, params[0])

# === Main Program ===
image_path = r"C:\Users\ahmad\Python codes\Digital Image Processing\images\name.jpg"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    print("❌ Error: Image not found. Please check the file path.")
    exit()

plt.imshow(image, cmap='gray')
plt.title("Original Image")
plt.axis("off")
plt.show()

user_choice, parameters = get_user_input()
print(f"\nYou selected transformation {user_choice} with parameters {parameters}")

transformed_image = apply_transformation(image, user_choice, parameters)


plt.figure(figsize=(10, 5))

# Show original image
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title("Original Image")
plt.axis("off")

# Show transformed image
plt.subplot(1, 2, 2)
plt.imshow(transformed_image, cmap='gray')
plt.title("Transformed Image")
plt.axis("off")

plt.show()

# Save the transformed image
cv2.imwrite("Transformed_Image.jpg", transformed_image)
print("✅ Transformed image saved as 'Transformed_Image.jpg'")

# Notify the user to check saved images for correct size comparison
print("\n⚠️ Note: Matplotlib scales all images to the same display size, which may make them look similar.")
print("✅ To see the true difference, please check the saved images in your folder!")