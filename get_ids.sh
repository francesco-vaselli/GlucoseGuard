#!/bin/bash

# Define the directory where the JSON files are located
SOURCE_DIR="/home/fvaselli/Documents/TSA/data/data_oh"

# Define the file where you want to log the unique patient IDs
LOG_FILE="unique_patient_ids.log"

# Clear the log file
> "$LOG_FILE"

# Use a associative array to store unique IDs
declare -A unique_ids

# Iterate over each .json file in the source directory
for file in "$SOURCE_DIR"/*.json; do
    # Extract the ID from the filename using a regex
    if [[ $(basename "$file") =~ ([0-9]+) ]]; then
        id="${BASH_REMATCH[1]}"
        unique_ids["$id"]=1
    fi
done

# Write unique IDs to the log file
for id in "${!unique_ids[@]}"; do
    echo "'$id'," >> "$LOG_FILE"
done

echo "Script execution completed."
