import numpy as np
from skimage import io, img_as_float
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import time

# Load the preprocessed image
input_image = img_as_float(io.imread('data/preprocessed/image.tiff'))

# Load CPU denoised image
denoised_image_cpu = img_as_float(io.imread('data/preprocessed/denoised_image_cpu.tiff'))

# Load GPU denoised image
denoised_image_gpu = img_as_float(io.imread('data/preprocessed/denoised_image_gpu.tiff'))

# Calculate PSNR and SSIM
psnr_cpu = psnr(input_image, denoised_image_cpu)
ssim_cpu = ssim(input_image, denoised_image_cpu)

psnr_gpu = psnr(input_image, denoised_image_gpu)
ssim_gpu = ssim(input_image, denoised_image_gpu)

# Display results
print(f"CPU PSNR: {psnr_cpu}, CPU SSIM: {ssim_cpu}")
print(f"GPU PSNR: {psnr_gpu}, GPU SSIM: {ssim_gpu}")
