#!/bin/bash

# Exit immediately if a command fails
set -e

# Activate virtual environment
source ~/swift-venv/bin/activate

# Destination directory
DEST="/home/marti-ayuso/uni/bachelors_thesis/bachelors-thesis/results/kin_heating/orig_detail/snap"

# Create destination directory if it doesn't exist
mkdir -p "$DEST"

# List of files to transfer
FILES=(
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

# Remote base path
REMOTE_BASE="ayuso@jed.hpc.epfl.ch:/scratch/ayuso/sims/orig/snap"

# Sequentially transfer files
for FILE in "${FILES[@]}"; do
    echo "Transferring $FILE ..."
    rsync -azP "${REMOTE_BASE}/${FILE}" "$DEST"
    echo "Finished $FILE"
    echo "-----------------------------------"
done

echo "All transfers completed successfully."
