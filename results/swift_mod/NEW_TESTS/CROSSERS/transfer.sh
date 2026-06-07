#!/bin/bash

# Exit immediately if a command fails
set -e

# Activate virtual environment
source ~/swift-venv/bin/activate

# Destination directory
DEST="/home/marti-ayuso/uni/bachelors_thesis/bachelors-thesis/results/swift_mod/NEW_TESTS/CROSSERS"

# Create destination directory if it doesn't exist
mkdir -p "$DEST"

# List of files to transfer
FILES=($(seq -f "snapshot_%04g.hdf5" 0 100))

# Remote base path
REMOTE_BASE="ayuso@jed.hpc.epfl.ch:/scratch/ayuso/sims/swift_mod/truetest15/snap"

# Sequentially transfer files
for FILE in "${FILES[@]}"; do
    echo "Transferring $FILE ..."
    rsync -azP "${REMOTE_BASE}/${FILE}" "$DEST"
    echo "Finished $FILE"
    echo "-----------------------------------"
done

echo "All transfers completed successfully."
