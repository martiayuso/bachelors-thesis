#!/bin/bash

# Exit immediately if a command fails
set -e

# Activate virtual environment
source ~/swift-venv/bin/activate

# Go to the directory containing the script and files
cd /home/marti-ayuso/uni/bachelors_thesis/bachelors-thesis/results/swift_mod/test_6

# List of chunk files
FILES=(
    "snapshot_0000.hdf5"
    "snapshot_0010.hdf5"
    "snapshot_0020.hdf5"
    "snapshot_0030.hdf5"
    "snapshot_0040.hdf5"
    "snapshot_0050.hdf5"
    "snapshot_0060.hdf5"
    "snapshot_0070.hdf5"
    "snapshot_0080.hdf5"
    "snapshot_0090.hdf5"
    "snapshot_0100.hdf5"
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
