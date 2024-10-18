#!/bin/bash

# Directories for healthy and epileptic patients
HEALTHY_DIR="/home/jagan/Development/eeg_final/backend/Healthy Individuals"
EPILEPTIC_DIR="/home/jagan/Development/eeg_final/backend/Epileptic Patients"

# Function to clean CSV files
clean_csv_files() {
    local directory=$1
    echo "Cleaning CSV files in $directory..."
    
    for file in "$directory"/*.csv; do
        if [ -f "$file" ]; then
            # Use pandas to clean the file
            python3 - <<END
import pandas as pd

# Load the CSV file
df = pd.read_csv('$file', header=None)

# Remove rows with NaN or empty values
df.dropna(inplace=True)
df = df[df.astype(str).apply(lambda x: x.str.isnumeric()).all(axis=1)]  # Keep only numeric rows

# Save the cleaned data back to the CSV file
df.to_csv('$file', index=False, header=False)
END
            echo "Cleaned $file"
        fi
    done
}

# Clean healthy individuals' data
clean_csv_files "$HEALTHY_DIR"

# Clean epileptic patients' data
clean_csv_files "$EPILEPTIC_DIR"

echo "Cleaning complete!"
