#!/usr/bin/env python3

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import time

# --- Check for and import Evo2 ---
try:
    from evo2.models import Evo2
except ImportError:
    print("Error: The 'evo2' library is required but not found.", file=sys.stderr)
    sys.exit(1)

# --- Configuration ---
INPUT_CSV = 'sampled_vcf_snv_sequences_by_type.csv' # CSV from the previous step
OUTPUT_CSV = 'sampled_vcf_snv_sequences_by_type_scored.csv' # Output file
MODEL_NAME = 'evo2_7b' # Specify the Evo 2 model to use

REF_SEQ_COL = 'Ref_Sequence_Window'
ALT_SEQ_COL = 'Alt_Sequence_Window'
SCORE_COL_NAME = 'Evo2_Delta_Score'

# --- Main Script Logic ---
if __name__ == "__main__":
    print(f"Starting Evo 2 delta score calculation for {INPUT_CSV}...")
    start_time = time.time()

    # --- Input Validation ---
    input_csv_path = Path(INPUT_CSV)
    if not input_csv_path.is_file():
        sys.exit(f"Error: Input CSV file not found: {INPUT_CSV}")

    # --- Load Input Data ---
    print(f"Loading data from {input_csv_path}...")
    try:
        # Load the sampled variants data
        variants_df = pd.read_csv(input_csv_path)
        print(f"Loaded {len(variants_df)} sampled variants.")
    except Exception as e:
        sys.exit(f"Error reading input CSV file {INPUT_CSV}: {e}")

    # --- Filter for Rows with Valid Sequences ---
    # Check if required sequence columns exist
    if REF_SEQ_COL not in variants_df.columns or ALT_SEQ_COL not in variants_df.columns:
        sys.exit(f"Error: Required columns '{REF_SEQ_COL}' or '{ALT_SEQ_COL}' not found in {INPUT_CSV}")

    # Keep only rows where both sequences are present and not 'NA' placeholder
    # Convert to string first to handle potential non-string types robustly
    variants_df[REF_SEQ_COL] = variants_df[REF_SEQ_COL].astype(str)
    variants_df[ALT_SEQ_COL] = variants_df[ALT_SEQ_COL].astype(str)
    original_count = len(variants_df)

    # Filter out rows where either sequence column is NA (using pandas method)
    # or where the string value is literally 'NA' (case-insensitive)
    valid_seq_df = variants_df.dropna(subset=[REF_SEQ_COL, ALT_SEQ_COL]).copy()
    valid_seq_df = valid_seq_df[valid_seq_df[REF_SEQ_COL].str.upper() != 'NA']
    valid_seq_df = valid_seq_df[valid_seq_df[ALT_SEQ_COL].str.upper() != 'NA']
    # Add check for empty strings as well
    valid_seq_df = valid_seq_df[valid_seq_df[REF_SEQ_COL].str.strip() != '']
    valid_seq_df = valid_seq_df[valid_seq_df[ALT_SEQ_COL].str.strip() != '']


    num_valid = len(valid_seq_df)
    num_skipped = original_count - num_valid
    if num_skipped > 0:
        print(f"Filtered out {num_skipped} rows due to missing/invalid reference or alternate sequences.")
    if num_valid == 0:
        sys.exit("Error: No rows with valid reference and alternate sequences found.")
    print(f"Processing {num_valid} variants with valid sequences.")

    # --- Prepare Sequences for Evo 2 ---
    print("Preparing sequence lists...")
    ref_seqs = []             # Stores unique reference sequences
    ref_seq_to_index = {}     # Maps reference sequence string to its index in ref_seqs
    ref_seq_indexes = []      # Stores the index (from ref_seqs) for each row in valid_seq_df
    alt_seqs = []             # Stores alternate sequences for each row in valid_seq_df

    for index, row in valid_seq_df.iterrows():
        ref_sequence = row[REF_SEQ_COL]
        alt_sequence = row[ALT_SEQ_COL]

        if ref_sequence not in ref_seq_to_index:
            ref_seq_to_index[ref_sequence] = len(ref_seqs)
            ref_seqs.append(ref_sequence)

        ref_seq_indexes.append(ref_seq_to_index[ref_sequence])
        alt_seqs.append(alt_sequence)

    ref_seq_indexes = np.array(ref_seq_indexes)
    print(f"Prepared {len(ref_seqs)} unique reference sequences and {len(alt_seqs)} alternate sequences.")

    # --- Load Evo 2 Model ---
    print(f"Loading Evo 2 model: {MODEL_NAME}...")
    try:
        model = Evo2(MODEL_NAME)
        print("Model loaded successfully.")
    except Exception as e:
        sys.exit(f"Error loading Evo 2 model '{MODEL_NAME}': {e}\n"
                 f"Ensure the 'evo2' library is installed and the model weights are accessible.")

    # --- Score Sequences ---
    print(f"Scoring likelihoods of {len(ref_seqs)} unique reference sequences...")
    try:
        ref_scores = model.score_sequences(ref_seqs)
        print("Reference scoring complete.")
    except Exception as e:
        sys.exit(f"Error scoring reference sequences: {e}")

    print(f"Scoring likelihoods of {len(alt_seqs)} variant sequences...")
    try:
        alt_scores = model.score_sequences(alt_seqs)
        print("Alternate scoring complete.")
    except Exception as e:
        sys.exit(f"Error scoring alternate sequences: {e}")

    # --- Calculate Delta Scores ---
    print("Calculating delta scores...")
    ref_scores_np = np.array(ref_scores)
    alt_scores_np = np.array(alt_scores)
    delta_scores = alt_scores_np - ref_scores_np[ref_seq_indexes]
    print("Delta score calculation complete.")

    # --- Add Delta Scores Back to DataFrame ---
    # Add scores to the filtered dataframe first
    valid_seq_df[SCORE_COL_NAME] = delta_scores

    # Merge scores back into the original DataFrame structure
    # Add the new score column to the original df, initialized with NA
    variants_df[SCORE_COL_NAME] = pd.NA
    # Update the scores based on the index
    variants_df.loc[valid_seq_df.index, SCORE_COL_NAME] = valid_seq_df[SCORE_COL_NAME]


    # --- Write Output CSV ---
    output_csv_path = Path(OUTPUT_CSV)
    print(f"Writing output CSV file with scores: {output_csv_path}...")
    try:
        # Get original columns and add the new score column
        output_columns = list(pd.read_csv(input_csv_path, nrows=0).columns)
        if SCORE_COL_NAME not in output_columns:
             output_columns.append(SCORE_COL_NAME)

        # Ensure columns exist before selecting
        final_output_columns = [col for col in output_columns if col in variants_df.columns]
        variants_df_output = variants_df[final_output_columns]

        variants_df_output.to_csv(output_csv_path, index=False, na_rep='NA')
        print("Output written successfully.")
    except Exception as e:
        print(f"Error writing output CSV: {e}", file=sys.stderr)

    # --- Final Summary ---
    end_time = time.time()
    print("\n--- Processing Summary ---")
    print(f"Input variants: {original_count}")
    print(f"Variants with valid sequences processed: {num_valid}")
    print(f"Unique reference sequences scored: {len(ref_seqs)}")
    print(f"Alternate sequences scored: {len(alt_seqs)}")
    print(f"Output file: {output_csv_path}")
    print(f"Total time: {end_time - start_time:.2f} seconds")
    print("Done.")
