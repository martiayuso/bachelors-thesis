#!/bin/bash

# Exit immediately if a command fails
set -e

# Activate virtual environment
source ~/swift-venv/bin/activate

# Go to the directory containing the script and files
cd /home/marti-ayuso/uni/bachelors_thesis/bachelors-thesis/results/swift_mod/NEW_TESTS/gradient_50

# List of chunk files
FILES=(
    "chunk_0100_0104.hdf5"
    "chunk_0200_0204.hdf5"
    "chunk_0300_0304.hdf5"
    "chunk_0400_0404.hdf5"
    "chunk_0500_0504.hdf5"
    "chunk_0600_0604.hdf5"
    "chunk_0700_0704.hdf5"
    "chunk_0800_0804.hdf5"
    "chunk_0900_0904.hdf5"
    "chunk_0995_0999.hdf5"
)

# Run the plotting command sequentially
for FILE in "${FILES[@]}"; do
    echo "Processing $FILE ..."

    python3 plotSphericalProfile2.py \
        -y density \
        --xmin 0.01 \
        --xmax 10 \
        --rmax 10 \
        --nr 200 \
        --log xy \
        "$FILE"

    echo "Finished $FILE"
    echo "-----------------------------------"
done

echo "All plots generated successfully."
