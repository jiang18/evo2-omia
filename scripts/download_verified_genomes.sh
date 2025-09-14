#!/bin/bash

# Input CSV file with verified Name,NCBI_Accession,Species
VERIFIED_CSV="verified_genomes.csv"

# Directory where genome packages will be downloaded
OUTPUT_DIR="reference_genomes_verified"

# --- Configuration ---
# Delay in seconds between download attempts
DOWNLOAD_DELAY=10 # Adjust as needed (e.g., 5, 10, 15)

# --- Check Dependencies ---
if ! command -v datasets &> /dev/null; then
    echo "Error: 'datasets' command not found. Please install ncbi-datasets-cli."
    exit 1
fi
if ! command -v tail &> /dev/null || ! command -v cut &> /dev/null || ! command -v grep &> /dev/null || ! command -v sort &> /dev/null; then
    echo "Error: One or more required core utilities (tail, cut, grep, sort) not found."
    exit 1
fi
if ! command -v sleep &> /dev/null; then
    echo "Error: 'sleep' command not found."
    exit 1
fi


# --- Setup ---
mkdir -p "$OUTPUT_DIR"

# --- Check if CSV file exists ---
if [ ! -f "$VERIFIED_CSV" ]; then
  echo "Error: Verified CSV file '$VERIFIED_CSV' not found."
  exit 1
fi

echo "Extracting unique accessions from $VERIFIED_CSV..."

# Process the CSV to get a clean, unique list of accessions:
# 1. tail -n +2: Skip header line
# 2. cut -d',' -f2: Get the second column (NCBI_Accession)
# 3. grep -Eo '(GCF|GCA)_[0-9.]+': Extract valid accession patterns cleanly
# 4. sort -u: Get only unique accessions
unique_accessions=$(tail -n +2 "$VERIFIED_CSV" | cut -d',' -f2 | grep -Eo '(GCF|GCA)_[0-9.]+' | sort -u)

if [ -z "$unique_accessions" ]; then
    echo "Error: No valid GCF/GCA accessions found in '$VERIFIED_CSV' after processing."
    exit 1
fi

echo "Found unique accessions to download:"
echo "$unique_accessions"
echo "-----------------------------------------"
echo "Starting downloads (with a ${DOWNLOAD_DELAY} second delay between attempts)..."

# --- Loop Through Unique Accessions ---
echo "$unique_accessions" | while IFS= read -r accession; do
  echo "-----------------------------------------"
  echo "Processing Accession: $accession"
  echo "-----------------------------------------"

  output_package="${OUTPUT_DIR}/${accession}.zip"

  # Check if already downloaded
  if [ -f "$output_package" ]; then
      echo "Package $output_package already exists. Skipping."
  else
      echo "Downloading genome for $accession..."
      # Run the datasets download command
      # --include genome : Downloads the genome sequence (FASTA)
      # --filename : Specifies the output zip file name
      datasets download genome accession "$accession" --include genome --filename "$output_package" # --no-progressbar
      status=$? # Capture the exit status of the datasets command

      # Check download status
      if [ $status -eq 0 ]; then
          echo "Successfully downloaded $accession to $output_package"
      else
          # Provide more specific feedback based on the error code if possible in future versions
          echo "!!! Error downloading $accession (Exit Status: $status). Check datasets output. !!!"
          # Consider removing partially downloaded file if it exists and status is non-zero
          # if [ -f "$output_package" ]; then
          #     echo "Removing potentially incomplete file: $output_package"
          #     rm "$output_package"
          # fi
      fi
  fi # End check if file exists

  # --- ADD DELAY HERE ---
  # Pause regardless of whether the file existed or the download succeeded/failed,
  # before starting the next iteration.
  echo "Pausing for ${DOWNLOAD_DELAY} seconds before next item..."
  sleep ${DOWNLOAD_DELAY}
  # --------------------

done # End while loop

echo "-----------------------------------------"
echo "Batch download process completed."
echo "Downloaded genomes are in '$OUTPUT_DIR'."
echo "If any downloads failed, you can re-run this script to attempt them again."
echo "-----------------------------------------"

