#!/bin/bash

# Exit immediately if a command fails
set -e

# Activate virtual environment
source ~/swift-venv/bin/activate

# Go to the directory containing the script and files
cd /home/marti-ayuso/uni/bachelors_thesis/bachelors-thesis/results/swift_mod/inner_gradient_2

# List of chunk files
FILES=(
    "chunk_0000_0049.hdf5"
    "chunk_0050_0099.hdf5"
    "chunk_0100_0149.hdf5"
    "chunk_0150_0199.hdf5"
    "chunk_0200_0249.hdf5"
    "chunk_0250_0299.hdf5"
    "chunk_0300_0349.hdf5"
    "chunk_0350_0399.hdf5"
)

# Run the plotting command sequentially
for FILE in "${FILES[@]}"; do
    echo "Processing $FILE ..."

    python3 plotSphericalProfile2.py \
        -y density \
        --xmin 0.01 \
        --xmax 10 \
        --rmax 10 \
        --nr 300 \
        --log xy \
        "$FILE"

    echo "Finished $FILE"
    echo "-----------------------------------"
done

echo "All plots generated successfully."
