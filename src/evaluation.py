import numpy as np
from skimage import io, img_as_float
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import time
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
start_time = time.time()
denoised_image_cpu = nlm_cpu(input_image)
cpu_time = time.time() - start_time

# Save the denoised image
io.imsave('data/preprocessed/denoised_image_cpu.tiff', denoised_image_cpu)

# Load GPU denoised image
denoised_image_gpu = img_as_float(io.imread('data/preprocessed/denoised_image_gpu.tiff'))

# Calculate PSNR and SSIM
psnr_cpu = psnr(input_image, denoised_image_cpu)
ssim_cpu = ssim(input_image, denoised_image_cpu)

psnr_gpu = psnr(input_image, denoised_image_gpu)
ssim_gpu = ssim(input_image, denoised_image_gpu)

# Display results
print(f"CPU PSNR: {psnr_cpu}, CPU SSIM: {ssim_cpu}, CPU Time: {cpu_time}s")
print(f"GPU PSNR: {psnr_gpu}, GPU SSIM: {ssim_gpu}")

