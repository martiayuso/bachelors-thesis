#!/bin/bash

# Exit immediately if a command fails
set -e

# Activate virtual environment
source ~/swift-venv/bin/activate

# Go to the directory containing the script and files
cd /home/marti-ayuso/uni/bachelors_thesis/bachelors-thesis/results/swift_mod/test_7

# List of chunk files
FILES=(
    "chunk_0000_0004.hdf5"
    "chunk_0010_0014.hdf5"
    "chunk_0020_0024.hdf5"
    "chunk_0030_0034.hdf5"
    "chunk_0040_0044.hdf5"
    "chunk_0050_0054.hdf5"
    "chunk_0060_0064.hdf5"
    "chunk_0070_0074.hdf5"
    "chunk_0080_0084.hdf5"
    "chunk_0090_0094.hdf5"
    "chunk_0100_0104.hdf5"
)

# Run the plotting command sequentially
for FILE in "${FILES[@]}"; do
    echo "Processing $FILE ..."

    python3 plotSphericalProfile2.py \
        -y density \
        --xmin 0.01 \
        --xmax 10 \
        --rmax 10 \
        --nr 500 \
        --log xy \
        "$FILE"

    echo "Finished $FILE"
    echo "-----------------------------------"
done

echo "All plots generated successfully."
