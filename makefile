
# Compiler and flags
NVCC = nvcc
NVCC_FLAGS = -O2 -arch=sm_50

# Targets
all: gpu_denoise run

# Build the GPU implementation
gpu_denoise: src/gpu_denoise.cu
	$(NVCC) $(NVCC_FLAGS) -o src/gpu_denoise src/gpu_denoise.cu `pkg-config --cflags --libs opencv`

# Run the entire workflow
run: run.sh
	bash run.sh

# Clean the build files
clean:
	rm -f src/gpu_denoise
	rm -rf data/preprocessed
