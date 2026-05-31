1#!/bin/bash

# Exit immediately if a command fails
set -e

# Activate virtual environment
source ~/swift-venv/bin/activate

# Go to the directory containing the script and files
cd /home/marti-ayuso/uni/bachelors_thesis/bachelors-thesis/results/swift_mod/NEW_TESTS/third_test

# List of chunk files
FILES=(
    "snap_01.hdf5"
    "snap_02.hdf5"
    "snap_03.hdf5"
    "snap_04.hdf5"
    "snap_05.hdf5"
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
