import pandas as pd
import re
import sys # Import sys to access command-line arguments and exit

# --- Check for Command-Line Argument ---
if len(sys.argv) < 2:
    # sys.argv[0] is the script name itself
    print(f"Usage: python {sys.argv[0]} <input_csv_file>", file=sys.stderr)
    sys.exit(1) # Exit the script indicating an error

# The first argument after the script name is the input file path
csv_filepath = sys.argv[1]

# --- Configuration ---

# 1. Columns to Extract (Added EVA IDs)
columns_to_keep = [
    'OMIA Variant ID',
    'OMIA Phene-Species ID(s)',
    'Species Name',
    'Gene',
    'Type of Variant',
    'Reference Sequence',
    'Chr.',
    'g. or m.',
    'EVA ID', # Added
    'Inferred EVA ID' # Added
]

# 2. Species Names to Keep
allowed_species = [
    "dog",
    "taurine cattle",
    "domestic cat",
    "pig",
    "sheep",
    "horse",
    "chicken",
    "rabbit",
    "goat"
]

# --- Regular Expression for SNV parsing ---
# (Same as before)
snv_pattern = re.compile(r'(?:g\.|:g\.)(\d+)([ACGTN])>([ACGTN])')

# --- Function to Parse g. notation ---
# (Same as before)
def parse_g_notation_for_snv(notation_str):
    if not isinstance(notation_str, str):
        return None, None, None
    match = snv_pattern.search(notation_str)
    if match:
        position, ref, alt = match.groups()
        return position, ref, alt
    else:
        return None, None, None

# --- Main Processing ---
try:
    print(f"Reading CSV data from '{csv_filepath}'...")
    # Read the CSV file specified by the command-line argument
    df = pd.read_csv(csv_filepath)
    print(f"Successfully read {len(df)} rows.")

    # === New Step: Filter by Species Name ===
    print(f"Filtering rows for species: {', '.join(allowed_species)}...")
    original_row_count = len(df)
    df_filtered_species = df[df['Species Name'].isin(allowed_species)].copy() # Use .copy()
    rows_kept = len(df_filtered_species)
    rows_removed = original_row_count - rows_kept
    print(f"Kept {rows_kept} rows based on species criteria (removed {rows_removed}).")

    if df_filtered_species.empty:
        print("No rows matched the specified species criteria. Exiting.")
        sys.exit(0) # Exit cleanly, no error but no data to process

    # 1. Select required columns from the species-filtered DataFrame
    # Check if all required columns exist before selecting
    missing_cols = [col for col in columns_to_keep if col not in df_filtered_species.columns]
    if missing_cols:
        print(f"Error: The following required columns are missing from '{csv_filepath}': {', '.join(missing_cols)}", file=sys.stderr)
        sys.exit(1) # Exit the script

    df_selected = df_filtered_species[columns_to_keep].copy() # Use .copy() to avoid SettingWithCopyWarning
    print(f"Selected {len(df_selected.columns)} columns.")

    # 2. Parse 'g. or m.' column for SNV details
    print("Parsing 'g. or m.' column for SNV position, ref, and alt alleles...")
    # Ensure the column exists before parsing
    if 'g. or m.' not in df_selected.columns:
         print(f"Error: Column 'g. or m.' not found after selection.", file=sys.stderr)
         sys.exit(1)

    # Apply parsing - handle potential errors if column type is unexpected
    try:
        parse_results = df_selected['g. or m.'].astype(str).apply(parse_g_notation_for_snv)
    except Exception as e:
        print(f"Error applying parsing function to 'g. or m.' column in '{csv_filepath}': {e}", file=sys.stderr)
        sys.exit(1)

    # Add parsed results as new columns
    df_selected['Position'] = [res[0] for res in parse_results]
    df_selected['Ref Allele'] = [res[1] for res in parse_results]
    df_selected['Alt Allele'] = [res[2] for res in parse_results]

    # 3. Filter rows where parsing was successful (i.e., it's an SNV)
    # We check if 'Position' is not None (or NaN after conversion)
    snv_filtered_count_before = len(df_selected)
    df_filtered_snvs = df_selected[df_selected['Position'].notna()].copy()
    snv_kept = len(df_filtered_snvs)
    snv_removed = snv_filtered_count_before - snv_kept
    print(f"Filtered down to {snv_kept} rows matching SNV pattern in 'g. or m.' (removed {snv_removed}).")

    # Optional: Convert Position column to numeric if needed (handle potential errors)
    if not df_filtered_snvs.empty:
       df_filtered_snvs['Position'] = pd.to_numeric(df_filtered_snvs['Position'], errors='coerce')
       # Check if any values failed conversion (became NaN)
       if df_filtered_snvs['Position'].isnull().any():
           print("Warning: Some 'Position' values could not be converted to numeric.")

    # Optional: Drop the original 'g. or m.' column
    df_final = df_filtered_snvs.drop(columns=['g. or m.'])

    # Reorder columns to put parsed ones near the end (optional, for readability)
    if not df_final.empty:
        cols_order = [
            'OMIA Variant ID', 'OMIA Phene-Species ID(s)', 'Species Name', 'Gene',
            'Type of Variant', 'Reference Sequence', 'Chr.',
            'Position', 'Ref Allele', 'Alt Allele', # Parsed columns
            'EVA ID', 'Inferred EVA ID' # Added columns
        ]
        # Ensure all columns in cols_order exist in df_final before reordering
        cols_order = [col for col in cols_order if col in df_final.columns]
        df_final = df_final[cols_order]


    # --- Output ---
    if df_final.empty:
        print("\n--- No SNV Data Found Matching All Criteria (Species and SNV pattern) ---")
    else:
        print("\n--- Extracted SNV Data ---")
        print(df_final.to_string()) # .to_string() prevents truncation in output

        # Determine output filename based on input filename
        base_name = csv_filepath.rsplit('.', 1)[0] # Get name before last dot
        output_filename = f'{base_name}_extracted_snvs.csv'
        try:
            df_final.to_csv(output_filename, index=False)
            print(f"\nResults saved to {output_filename}")
        except Exception as e:
            print(f"\nError saving results to '{output_filename}': {e}", file=sys.stderr)


except FileNotFoundError:
    print(f"Error: The file '{csv_filepath}' was not found. Please check the path and filename.", file=sys.stderr)
    sys.exit(1)
except pd.errors.EmptyDataError:
    print(f"Error: The file '{csv_filepath}' is empty.", file=sys.stderr)
    sys.exit(1)
except pd.errors.ParserError:
    print(f"Error: Could not parse '{csv_filepath}'. Check if it's a valid CSV file.", file=sys.stderr)
    sys.exit(1)
except KeyError as e:
    # This might be redundant due to the earlier check, but kept for robustness
    print(f"Error: Column '{e}' not found in the CSV '{csv_filepath}'. Check column names. Required for filtering or selection.", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred while processing '{csv_filepath}': {e}", file=sys.stderr)
    sys.exit(1)

