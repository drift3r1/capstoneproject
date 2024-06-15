#!/bin/bash

# Create the preprocessed directory if it doesn't exist
mkdir -p data/preprocessed

# Step 1: Preprocess the images
echo "Preprocessing images..."
python3 src/preprocess_images.py

# Step 2: Run CPU implementation
echo "Running CPU implementation..."
python3 src/cpu_denoise.py

# Step 3: Run GPU implementation
echo "Running GPU implementation..."
./src/gpu_denoise

# Step 4: Generate results
echo "Generating results..."
python3 src/generate_results.py

echo "All steps completed!"
