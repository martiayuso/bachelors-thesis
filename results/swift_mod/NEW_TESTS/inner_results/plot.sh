#!/bin/bash

# Exit immediately if a command fails
set -e

# Activate virtual environment
source ~/swift-venv/bin/activate

# Base directory containing many simulation folders
BASE_DIR="/home/marti-ayuso/uni/bachelors_thesis/bachelors-thesis/results/swift_mod/NEW_TESTS/inner_results"

# Loop through all subfolders
for DIR in "$BASE_DIR"/ttest*/; do

    # Skip if not a directory
    [ -d "$DIR" ] || continue

    echo "==================================="
    echo "Entering folder: $DIR"
    echo "==================================="

    cd "$DIR"

    # Loop through all snapshot files automatically
    for FILE in chunk_*.hdf5; do

        # Skip if no files match
        [ -e "$FILE" ] || continue

        echo "Processing $FILE ..."

        python3 ../plotSphericalProfile2.py \
            -y density \
            --xmin 0.01 \
            --xmax 10 \
            --rmax 10 \
            --nr 200 \
            --log xy \
            "$FILE"

        echo "Finished $FILE"
        echo "-----------------------------------"

    done
done

echo "All plots generated successfully."
