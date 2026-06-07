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

# Remote base path
REMOTE_BASE="ayuso@jed.hpc.epfl.ch:/scratch/ayuso/sims/swift_mod/truetest10/snap/chunks"

# Sequentially transfer files
for FILE in "${FILES[@]}"; do
    echo "Transferring $FILE ..."
    rsync -azP "${REMOTE_BASE}/${FILE}" "$DEST"
    echo "Finished $FILE"
    echo "-----------------------------------"
done

echo "All transfers completed successfully."
