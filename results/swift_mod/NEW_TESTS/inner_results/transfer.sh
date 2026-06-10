#!/bin/bash

# Exit immediately if a command fails
set -e

# Activate virtual environment
source ~/swift-venv/bin/activate

# ------------------------------------------------------------------
# Local destination base directory
# ------------------------------------------------------------------
DEST_BASE="/home/marti-ayuso/uni/bachelors_thesis/bachelors-thesis/results/swift_mod/NEW_TESTS/LARGE_PLOT"

# Create base destination directory
mkdir -p "$DEST_BASE"

# ------------------------------------------------------------------
# Simulation folders
# ------------------------------------------------------------------
SIMS=(
    "test4"
    "test5"
    "test6"
    "test7"
    "test8"
    "test9"
    "test10"
    "test11"
)

# ------------------------------------------------------------------
# Files to transfer
# ------------------------------------------------------------------
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

# ------------------------------------------------------------------
# Loop over simulations
# ------------------------------------------------------------------
for SIM in "${SIMS[@]}"; do

    echo "=========================================="
    echo "Processing simulation: $SIM"
    echo "=========================================="

    # Create local destination subfolder
    DEST="${DEST_BASE}/${SIM}"
    mkdir -p "$DEST"

    # Remote base path for this simulation
    REMOTE_BASE="ayuso@jed.hpc.epfl.ch:/scratch/ayuso/sims/swift_mod/${SIM}/snap/chunks"

    # Transfer files
    for FILE in "${FILES[@]}"; do

        echo "Transferring ${SIM}/${FILE} ..."

        rsync -azP \
            "${REMOTE_BASE}/${FILE}" \
            "$DEST"

        echo "Finished ${SIM}/${FILE}"
        echo "-----------------------------------"

    done

done

echo "All transfers completed successfully."
