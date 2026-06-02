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
    "gravtest"
    "gravtest2"
    "gravtest4"
    "gravtest6"
    "gravtest7"
    "gravtest8"
    "gravtest9"
    "gravtest10"
    "gravtest11"
)

# ------------------------------------------------------------------
# Files to transfer
# ------------------------------------------------------------------
FILES=(
    "chunk_0010_0014.hdf5"
    "chunk_0020_0024.hdf5"
    "chunk_0030_0034.hdf5"
    "chunk_0040_0044.hdf5"
    "chunk_0050_0054.hdf5"
    "chunk_0060_0064.hdf5"
    "chunk_0070_0074.hdf5"
    "chunk_0080_0084.hdf5"
    "chunk_0090_0094.hdf5"
    "chunk_0095_0099.hdf5"
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
