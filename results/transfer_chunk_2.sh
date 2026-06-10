#!/bin/bash

# Exit immediately if a command fails
set -e

# Activate virtual environment
source ~/swift-venv/bin/activate

# Destination directory
DEST="/home/marti-ayuso/uni/bachelors_thesis/bachelors-thesis/results/swift_mod/NEW_TESTS"

# Create destination directory if it doesn't exist
mkdir -p "$DEST"

# List of files to transfer
FILES=(
    "chunk_0100_0101.hdf5"
    "chunk_0200_0201.hdf5"
    "chunk_0300_0301.hdf5"
    "chunk_0400_0401.hdf5"
    "chunk_0500_0501.hdf5"
    "chunk_0600_0601.hdf5"
    "chunk_0700_0701.hdf5"
    "chunk_0800_0801.hdf5"
    "chunk_0900_0901.hdf5"
    "chunk_0998_0999.hdf5"
)

# Remote base path
REMOTE_BASE="ayuso@jed.hpc.epfl.ch:/scratch/ayuso/sims/swift_mod/truetest15/snap/chunks"

# Sequentially transfer files
for FILE in "${FILES[@]}"; do
    echo "Transferring $FILE ..."
    rsync -azP "${REMOTE_BASE}/${FILE}" "$DEST"
    echo "Finished $FILE"
    echo "-----------------------------------"
done

echo "All transfers completed successfully."
