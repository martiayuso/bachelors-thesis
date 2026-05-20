#!/bin/bash

# Exit immediately if a command fails
set -e

# Activate virtual environment
source ~/swift-venv/bin/activate

# Go to the directory containing the script and files
cd /home/marti-ayuso/uni/bachelors_thesis/bachelors-thesis/results/swift_mod/inner_gradient

# List of chunk files
FILES=(
    "chunk_0000_0009.hdf5"
    "chunk_0010_0019.hdf5"
    "chunk_0020_0029.hdf5"
    "chunk_0030_0039.hdf5"
    "chunk_0040_0049.hdf5"
    "chunk_0050_0059.hdf5"
    "chunk_0060_0069.hdf5"
    "chunk_0070_0079.hdf5"
    "chunk_0080_0089.hdf5"
    "chunk_0090_0099.hdf5"
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
