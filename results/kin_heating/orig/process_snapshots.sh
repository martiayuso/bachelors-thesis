#!/bin/bash

set -e  # stop on error

# ----------------------------
# Parameters
# ----------------------------
INPUT_DIR="snap"
CENTERED_DIR="snap/centered"

# ----------------------------
# Create directories
# ----------------------------
echo "Creating directories..."
mkdir -p "$CENTERED_DIR"

# ----------------------------
# Step 1: Center snapshots
# ----------------------------
echo "Centering snapshots..."
snapshots_exec ${INPUT_DIR}/snapshot_*.hdf5 \
	--exec "nb.cmcenter()" \
    -o "$CENTERED_DIR"

echo "Centering complete."

# ----------------------------
# Step 2: Sort files safely
# ----------------------------
echo "Preparing file list..."
mapfile -t FILES < <(ls ${CENTERED_DIR}/*.hdf5 | sort)

TOTAL=${#FILES[@]}
echo "Total centered snapshots: $TOTAL"
