import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# === CONFIGURATION ===

# 🔹 Path to the input grayscale image
image_path = r"C:\Users\ahmad\Python codes\Digital Image Processing\HW3\wall_original.png"

# === TUNABLE PARAMETERS ===

# 🎚 D0 — Gaussian cutoff frequency
# Controls how much low-frequency content is removed
# Lower D0 → more aggressive lighting removal (but may lose detail)
# Higher D0 → softer lighting correction (may keep more lighting)
D0 = 10  # Recommended range: 5–30

# 🎚 gamma_L — Low-frequency gain
# Controls brightness preservation from lighting
# Lower gamma_L (e.g., 0.1) removes more lighting
# Higher gamma_L (e.g., 0.4) keeps more base brightness
gamma_L = 0.3  # Range: 0.1 – 0.5

# 🎚 gamma_H — High-frequency gain
# Controls how much contrast/detail to boost in the texture
# Higher gamma_H → more texture pop (but risk of noise)
gamma_H = 1.2  # Range: 1.0 – 2.0

# 🎚 CLAHE Contrast Enhancement (applied after method 2)
# Clip limit controls contrast enhancement strength
# Tile size affects local patch granularity
clahe_clip_limit = 2.0  # Range: 1.0 – 4.0
clahe_tile_size = (8, 8)  # Try (8,8), (4,4), (16,16)


# === UTILITY FUNCTIONS ===
def apply_dft(img_float):
    log_img = np.log1p(img_float)
    dft = np.fft.fft2(log_img)
    return log_img, np.fft.fftshift(dft)

def gaussian_lowpass(shape, D0):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    u = np.arange(cols)
    v = np.arange(rows)
    U, V = np.meshgrid(u, v)
    D_sq = (U - ccol)**2 + (V - crow)**2
    return np.exp(-D_sq / (2 * D0**2))

def postprocess(reflectance, illumination, name_prefix):
    # Normalize for saving
    illum_norm = cv2.normalize(np.expm1(illumination), None, 0, 255, cv2.NORM_MINMAX)
    illum_out = illum_norm.astype(np.uint8)

    reflect_norm = cv2.normalize(np.expm1(reflectance), None, 0, 255, cv2.NORM_MINMAX)
    reflect_out = reflect_norm.astype(np.uint8)

    # Save results
    cv2.imwrite(f"illumination_{name_prefix}.png", illum_out)
    cv2.imwrite(f"corrected_{name_prefix}.png", reflect_out)
    return illum_out, reflect_out


# === METHOD 1: Standard Homomorphic Filtering (No Gain, No CLAHE) ===
def method_1_basic(img_float, D0):
    log_img, fft_shift = apply_dft(img_float)
    glpf = gaussian_lowpass(img_float.shape, D0)
    ghpf = 1 - glpf

    illum_fft = fft_shift * glpf
    reflect_fft = fft_shift * ghpf

    illum = np.real(np.fft.ifft2(np.fft.ifftshift(illum_fft)))
    reflect = np.real(np.fft.ifft2(np.fft.ifftshift(reflect_fft)))
    return postprocess(reflect, illum, "code1")


# === METHOD 2: Improved Homomorphic Filtering (Gain control + CLAHE) ===
def method_2_advanced(img_float, D0, gamma_L, gamma_H, clahe_clip, clahe_tile):
    log_img, fft_shift = apply_dft(img_float)
    glpf = gaussian_lowpass(img_float.shape, D0)

    # Modified Homomorphic filter
    H = (gamma_H - gamma_L) * (1 - glpf) + gamma_L
    reflect_fft = fft_shift * H
    illum_fft = fft_shift * glpf

    illum = np.real(np.fft.ifft2(np.fft.ifftshift(illum_fft)))
    reflect = np.real(np.fft.ifft2(np.fft.ifftshift(reflect_fft)))
    illum_norm = cv2.normalize(np.expm1(illum), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Normalize + CLAHE
    reflect_norm = cv2.normalize(np.expm1(reflect), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_tile)
    reflect_clahe = clahe.apply(reflect_norm)

    cv2.imwrite("illumination_code2.png", illum_norm)
    cv2.imwrite("corrected_code2.png", reflect_clahe)
    return illum_norm, reflect_clahe


# === MAIN EXECUTION ===
def main():
    if not os.path.exists(image_path):
        print(f"Image not found at: {image_path}")
        return

    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        print("Failed to read image.")
        return

    img_float = img_gray.astype(np.float32)
    print("Running Method 1 (basic)...")
    illum1, corrected1 = method_1_basic(img_float, D0)

    print("Running Method 2 (advanced)...")
    illum2, corrected2 = method_2_advanced(img_float, D0, gamma_L, gamma_H, clahe_clip_limit, clahe_tile_size)

    # --- Visualization ---
    plt.figure(figsize=(15, 5))
    titles = [
        "Original Image",
        "Illumination (Method 1)", "Corrected (Method 1)",
        "Illumination (Method 2)", "Corrected (Method 2)"
    ]
    images = [img_gray, illum1, corrected1, illum2, corrected2]

    for i in range(5):
        plt.subplot(1, 5, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i], fontsize=10)
        plt.axis('off')

    plt.suptitle(
        f'Lighting Removal Results — D0={D0}, γL={gamma_L}, γH={gamma_H}',
        fontsize=12, y=0.95
    )
    plt.tight_layout(rect=[0, 0, 1, 1.3]) 
    plt.savefig("comparison_plot.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()