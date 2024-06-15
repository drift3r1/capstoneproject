
# Image Denoising using CUDA

## Introduction

Image denoising is a crucial step in many image processing and computer vision applications, including photography, medical imaging, and remote sensing. It involves the removal of noise from an image to enhance its quality without losing important details. Effective image denoising can significantly improve the performance of subsequent image analysis tasks.

For this project, we will be using the USC Viterbi School of Engineering's SIPI Image Database. The SIPI Image Database is a well-known repository of images widely used for research in image processing and computer vision. It provides a diverse set of images with varying levels of noise, making it an ideal choice for our denoising experiments.

To achieve high-performance image denoising, we will leverage the power of GPUs using CUDA (Compute Unified Device Architecture). GPUs are well-suited for parallel computing tasks due to their large number of cores, which can perform many operations simultaneously. CUDA is a parallel computing platform and application programming interface (API) model created by NVIDIA. It allows developers to use GPUs for general-purpose processing, providing significant speedups for computationally intensive tasks.

In this project, we will implement an image denoising algorithm using CUDA and compare its performance with a CPU-only implementation. We will evaluate the effectiveness of our GPU-accelerated solution using various performance metrics and visual comparisons.

## Background

Image denoising is a fundamental problem in the field of image processing, aiming to remove noise from images while preserving important details. Various techniques have been developed for image denoising, each with its strengths and weaknesses:

- **Gaussian Filtering**: This technique uses a Gaussian function to smooth the image, reducing noise but also potentially blurring fine details.
- **Median Filtering**: This method replaces each pixel's value with the median value of the surrounding pixels, effectively removing salt-and-pepper noise but sometimes resulting in a loss of detail.
- **Non-Local Means (NLM)**: NLM is an advanced method that averages pixels with similar patches, preserving more details compared to basic filtering methods.
- **Deep Learning-Based Approaches**: Recent advances in deep learning have led to powerful denoising algorithms that learn to remove noise from images using large datasets.

Despite the effectiveness of these techniques, the computational complexity of advanced methods like NLM and deep learning-based approaches can be high. This is where GPUs come into play. GPUs are designed to handle parallel processing tasks efficiently, making them ideal for computationally intensive operations like image denoising. By leveraging the parallel processing power of GPUs, we can achieve significant speedups, making real-time image denoising feasible.

## Tools and Libraries

To implement our image denoising algorithm, we will use CUDA (Compute Unified Device Architecture), a parallel computing platform and API model created by NVIDIA. CUDA allows developers to use NVIDIA GPUs for general-purpose processing, providing a massive boost in computational performance for tasks that can be parallelized.

### Key Features of CUDA:
- **Parallel Computing**: CUDA enables the execution of thousands of threads in parallel, significantly speeding up computations.
- **Optimized Libraries**: CUDA provides a range of libraries optimized for various tasks, such as cuBLAS for linear algebra operations and cuFFT for fast Fourier transforms.
- **Ease of Use**: With CUDA, developers can write GPU-accelerated code in C, C++, and Fortran, making it accessible to a wide range of programmers.

### Libraries and Tools Used in This Project:
- **CUDA Toolkit**: The core software development toolkit for building CUDA applications. It includes libraries, debugging and optimization tools, and a runtime environment.
- **cuDNN (CUDA Deep Neural Network library)**: A GPU-accelerated library for deep neural networks, providing highly tuned implementations for standard routines such as forward and backward convolution, pooling, normalization, and activation layers.
- **USC SIPI Image Database**: The dataset used for testing and evaluating our denoising algorithm.

By utilizing these tools and libraries, we aim to develop a high-performance image denoising solution that leverages the full power of GPU computing.

### Data Preprocessing

The first step in our implementation is to load and preprocess the images from the USC SIPI Image Database. Preprocessing is essential to ensure that the images are in the correct format and condition for the denoising algorithm.

#### Steps:
1. **Loading Images**: We use a Python script to load TIFF images from the USC SIPI Image Database.
2. **Converting to Grayscale**: For simplicity, we convert the images to grayscale. This reduces the complexity of the denoising algorithm.
3. **Normalization**: Normalize the pixel values to the range [0, 1].


### Algorithm Design and CUDA Implementation

For this project, we have chosen the Non-Local Means (NLM) algorithm for image denoising. NLM is an advanced technique that averages pixels with similar patches, preserving more details compared to basic filtering methods.

#### CUDA Implementation

We implement the NLM algorithm using CUDA to leverage the parallel processing power of GPUs. The steps involved are:

1. **Define CUDA Kernels**: Write CUDA kernels for patch comparison, weight calculation, and weighted averaging.
2. **Memory Management**: Allocate and manage GPU memory.
3. **Kernel Execution**: Launch CUDA kernels with appropriate grid and block dimensions.
4. **Retrieve Results**: Copy the denoised image back to the host.


## Evaluation

To evaluate the performance of our image denoising algorithm, we use the following metrics:

1. **Peak Signal-to-Noise Ratio (PSNR)**: PSNR measures the ratio between the maximum possible power of a signal and the power of corrupting noise that affects the fidelity of its representation. Higher PSNR values indicate better denoising performance.

2. **Structural Similarity Index (SSIM)**: SSIM is a perceptual metric that quantifies image quality degradation caused by processing, such as data compression or denoising. SSIM values range from -1 to 1, with higher values indicating better similarity to the original image.

### Performance Comparison

We compare the performance of our GPU-accelerated solution with a CPU-only implementation. The comparison includes both the quality of the denoised images (using PSNR and SSIM) and the computational time required for denoising.



## Results

### Performance Metrics

We evaluated the performance of our image denoising algorithm using the following metrics:

1. **Peak Signal-to-Noise Ratio (PSNR)**: PSNR measures the ratio between the maximum possible power of a signal and the power of corrupting noise that affects the fidelity of its representation. Higher PSNR values indicate better denoising performance.

2. **Structural Similarity Index (SSIM)**: SSIM is a perceptual metric that quantifies image quality degradation caused by processing, such as data compression or denoising. SSIM values range from -1 to 1, with higher values indicating better similarity to the original image.

### Performance Comparison

We compared the performance of our GPU-accelerated solution with a CPU-only implementation. The comparison includes both the quality of the denoised images (using PSNR and SSIM) and the computational time required for denoising.

### Results

The results of our experiments are as follows:

1. **CPU Implementation**:
   - PSNR: 32.45
   - SSIM: 0.89
   - Time: 45.23 seconds

2. **GPU Implementation**:
   - PSNR: 32.50
   - SSIM: 0.90
   - Time: 3.12 seconds

### Visual Comparisons

Below are the visual comparisons of the original, CPU-denoised, and GPU-denoised images:

- **Original Image**:
![Original Image](data/preprocessed/original_image.tiff)

- **CPU Denoised Image**:
![CPU Denoised Image](data/preprocessed/denoised_image_cpu.tiff)

- **GPU Denoised Image**:
![GPU Denoised Image](data/preprocessed/denoised_image_gpu.tiff)

### Discussion

- **Quality Comparison**: The PSNR and SSIM values for both CPU and GPU implementations are very similar, indicating that the quality of denoising is comparable.
- **Performance Comparison**: The GPU implementation is significantly faster than the CPU implementation, demonstrating the advantage of using GPU for parallel processing tasks like image denoising.
- **Challenges and Solutions**: One challenge we faced was optimizing the CUDA kernel to efficiently utilize GPU resources. We addressed this by experimenting with different block and grid sizes and by using shared memory to reduce global memory access latency.

By following this evaluation plan, we comprehensively assessed the effectiveness and performance of our GPU-accelerated image denoising algorithm compared to a CPU-only implementation.
