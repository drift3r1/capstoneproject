import numpy as np
from skimage import io, img_as_float
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import cv2

def nlm_cpu(image, patch_size=3, h=10.0):
    padded_image = np.pad(image, patch_size, mode='reflect')
    denoised_image = np.zeros_like(image)
    height, width = image.shape
    
    for i in range(height):
        for j in range(width):
            patch = padded_image[i:i+2*patch_size+1, j:j+2*patch_size+1]
            weights = np.exp(-((patch - padded_image[i+patch_size, j+patch_size]) ** 2) / (h ** 2))
            weights /= weights.sum()
            denoised_image[i, j] = np.sum(weights * patch)
    
    return denoised_image

# Load the preprocessed image
input_image = img_as_float(io.imread('data/preprocessed/image.tiff'))

# Apply NLM on CPU
denoised_image_cpu = nlm_cpu(input_image)

# Save the denoised image
io.imsave('data/preprocessed/denoised_image_cpu.tiff', denoised_image_cpu)
