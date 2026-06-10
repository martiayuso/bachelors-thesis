#!/bin/bash

# Exit immediately if a command fails
set -e

# Activate virtual environment
source ~/swift-venv/bin/activate

# ------------------------------------------------------------------
# Local destination base directory
# ------------------------------------------------------------------
DEST_BASE="/home/marti-ayuso/uni/bachelors_thesis/bachelors-thesis/results/swift_mod/NEW_TESTS/POTENTIAL"

# Create base destination directory
mkdir -p "$DEST_BASE"

# ------------------------------------------------------------------
# Simulation folders
# ------------------------------------------------------------------
SIMS=(
    "truetest"
    "truetest2"
    "truetest3"
    "truetest6"
    "truetest10"
    "truetest13"
    "truetest14"
)

# ------------------------------------------------------------------
# Files to transfer
# ------------------------------------------------------------------
FILES=(
            "snapshot_0001.hdf5"
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
    REMOTE_BASE="ayuso@jed.hpc.epfl.ch:/scratch/ayuso/sims/swift_mod/${SIM}/snap"

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
