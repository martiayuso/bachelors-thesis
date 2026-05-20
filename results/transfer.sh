#!/bin/bash

# Exit immediately if a command fails
set -e

# Activate virtual environment
source ~/swift-venv/bin/activate

# Destination directory
DEST="/home/marti-ayuso/uni/bachelors_thesis/bachelors-thesis/results/swift_mod/inner_gradient_2"

# Create destination directory if it doesn't exist
mkdir -p "$DEST"

# List of files to transfer
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

# Remote base path
REMOTE_BASE="ayuso@jed.hpc.epfl.ch:/scratch/ayuso/sims/swift_mod/inner_gradient/snap/chunks"

# Sequentially transfer files
for FILE in "${FILES[@]}"; do
    echo "Transferring $FILE ..."
    rsync -azP "${REMOTE_BASE}/${FILE}" "$DEST"
    echo "Finished $FILE"
    echo "-----------------------------------"
done

echo "All transfers completed successfully."
