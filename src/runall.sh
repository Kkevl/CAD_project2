#!/bin/bash

# Input and output directories
INPUT_DIR="../testcase"
OUTPUT_DIR="../output"

# Loop through all .cts files
for file in "$INPUT_DIR"/*.cts; do
    # Extract the base filename (e.g., case0.cts → case0)
    base=$(basename "$file" .cts)

    # Extract the number from the case filename (e.g., case0 → 0)
    num=$(echo "$base" | grep -o '[0-9]\+')

    # Construct output filename (e.g., output0.cbi)
    output="output${num}.cbi"

    # Run the command
    ./cbi "$file" "$OUTPUT_DIR/$output"
    python ../cbi_checker.py "$file" "$OUTPUT_DIR/$output"
    echo ""
done
