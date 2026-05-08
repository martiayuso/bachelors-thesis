#!/bin/bash

set -e  # stop on error

# ----------------------------
# Parameters
# ----------------------------
INPUT_DIR="snap"
CENTERED_DIR="snap/centered"
CHUNKS_DIR="snap/chunks"
CHUNK_SIZE=50

# ----------------------------
# Create directories
# ----------------------------
echo "Creating directories..."
#mkdir -p "$CENTERED_DIR"
mkdir -p "$CHUNKS_DIR"

# ----------------------------
# Step 1: Center snapshots
# ----------------------------
#echo "Centering snapshots..."
#snapshots_exec ${INPUT_DIR}/snapshot_*.hdf5 \
#    --exec "nb.potcenter()" \
#    -o "$CENTERED_DIR"

#echo "Centering complete."

# ----------------------------
# Step 2: Sort files safely
# ----------------------------
echo "Preparing file list..."
mapfile -t FILES < <(ls ${CENTERED_DIR}/snapshot_*.hdf5 | sort)

TOTAL=${#FILES[@]}
echo "Total centered snapshots: $TOTAL"

# ----------------------------
# Step 3: Concatenate in chunks
# ----------------------------
echo "Concatenating in chunks of $CHUNK_SIZE..."

for ((i=0; i<TOTAL; i+=CHUNK_SIZE)); do

    # Build chunk file list
    CHUNK_FILES=("${FILES[@]:i:CHUNK_SIZE}")

    START_INDEX=$(printf "%04d" "$i")
    END_INDEX=$(printf "%04d" "$((i + ${#CHUNK_FILES[@]} - 1))")

    OUTPUT_FILE="${CHUNKS_DIR}/chunk_${START_INDEX}_${END_INDEX}.hdf5"

    echo "Creating $OUTPUT_FILE"

    snapshots_concatenate "${CHUNK_FILES[@]}" -o "$OUTPUT_FILE"

done

echo "All chunks created in $CHUNKS_DIR"
