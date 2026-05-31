#!/bin/bash

# Exit immediately if a command fails
set -e

# Activate virtual environment
source ~/swift-venv/bin/activate

# Go to the directory containing the script and files
cd /home/marti-ayuso/uni/bachelors_thesis/bachelors-thesis/results/core_test/big_chunks

# List of chunk files
FILES=(
	"chunk_0000_0049.hdf5"
        "chunk_0100_0149.hdf5"
        "chunk_0200_0249.hdf5"
        "chunk_0300_0349.hdf5"
        "chunk_0400_0449.hdf5"
        "chunk_0500_0549.hdf5"
        "chunk_0600_0649.hdf5"
        "chunk_0700_0749.hdf5"
        "chunk_0800_0849.hdf5"
        "chunk_0900_0949.hdf5"
        "chunk_0950_0999.hdf5"
)

# Run the plotting command sequentially
for FILE in "${FILES[@]}"; do
    echo "Processing $FILE ..."

    python3 plotSphericalProfile2.py \
        -y density \
        --xmin 0.01 \
        --xmax 10 \
        --rmax 10 \
        --nr 100 \
        --log xy \
        "$FILE"

    echo "Finished $FILE"
    echo "-----------------------------------"
done

echo "All plots generated successfully."
