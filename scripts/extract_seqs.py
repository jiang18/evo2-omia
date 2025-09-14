#!/usr/bin/env python3

import pandas as pd
import os
import zipfile
import math
import sys
import shutil
from pathlib import Path
import re # Import regular expressions for parsing
import gzip # Needed for reading potentially gzipped fasta

# --- Check for and import pyfaidx ---
try:
    from pyfaidx import Fasta, Faidx, FastaIndexingError
except ImportError:
    print("Error: The 'pyfaidx' library is required but not found.", file=sys.stderr)
    print("Please install it: pip install pyfaidx", file=sys.stderr)
    sys.exit(1)

# --- Configuration ---
VARIANT_CSV_FILE = 'single_gene_all_extracted_snvs.csv'
GENOME_DIR = 'reference_genomes_verified' # Directory with GCF_*.zip files
OUTPUT_CSV_FILE = 'extracted_flanking_sequences_ref_alt_pyfaidx_hdr.csv' # Output CSV file name
TEMP_FASTA_DIR = 'temp_fasta_extraction' # Temporary dir for extracted fasta
WINDOW_SIZE = 8192 # Total window size (centered on variant)

# --- Genome Name to Accession Mapping ---
# (Based on the list you provided previously)
GENOME_NAME_TO_ACCESSION = {
    "GRCg6a": "GCF_000002315.6",
    "ARS-UCD1.2": "GCF_002263795.1",
    "ARS-UCD1.3": "GCF_002263795.2",
    "Sscrofa11.1": "GCF_000003025.6",
    "CanFam3.1": "GCF_000002285.3",
    "Dog10K_Boxer_Tasha": "GCF_000002285.5",
    "EquCab3.0": "GCF_002863925.1",
    "Felis_catus_9.0": "GCF_000181335.3",
    "Oar_rambouillet_v1.0": "GCF_002742125.1",
    "ROS_Cfam_1.0": "GCF_014441545.1",
    "UU_Cfam_GSD_1.0": "GCF_011100685.1", # Note: Was GCA in your list, ensure GCF zip exists or adjust check below
    "Oar_v3.1": "GCF_000298735.1",
    "F.catus_Fca126_mat1.0": "GCF_018350175.1",
    "UMD_3.1.1": "GCF_000003055.6",
    "ARS-UI_Ramb_v3.0": "GCF_016772045.2",
    "ARS1.2": "GCF_001704415.2",
    "Oar_v4.0": "GCF_000298735.2",
    "Felis_catus 9.0": "GCF_000181335.3", # Needs handling if spaces are exact in CSV
    "OryCun2.0": "GCF_000003625.3",
    "EquCab3": "GCF_002863925.1",
    "Oar_Rambouillet_v1.0": "GCF_002742125.1" # Corrected typo from GGCF_
}
# Handle specific quoted name if necessary (often pandas handles quotes automatically)
if 'Felis_catus 9.0' in GENOME_NAME_TO_ACCESSION:
     GENOME_NAME_TO_ACCESSION['Felis_catus 9.0'] = GENOME_NAME_TO_ACCESSION.pop('Felis_catus 9.0')

# --- Helper Functions ---
def build_chromosome_map_from_fasta_headers(fasta_path):
    """
    Builds a map from human-readable chr names (parsed from FASTA headers)
    to the sequence identifiers (first word after '>').
    Returns a dictionary like {'1': 'NC_006583.3', 'X': 'NC_006621.3', ...} or None on failure.
    Reads the FASTA file directly to parse headers.
    """
    print(f"Building chromosome map by parsing headers from {fasta_path.name}...")
    chr_map = {}
    if not fasta_path.is_file():
        print(f"Error: FASTA file not found for mapping: {fasta_path}", file=sys.stderr)
        return None

    try:
        # Determine if file is gzipped based on extension
        is_gzipped = str(fasta_path).endswith('.gz')
        open_func = gzip.open if is_gzipped else open
        read_mode = 'rt' # Read as text

        processed_ids = set() # Keep track of IDs found in headers

        with open_func(fasta_path, read_mode, encoding='latin-1', errors='ignore') as f_fasta:
            for line in f_fasta:
                if line.startswith('>'):
                    parts = line[1:].split(None, 1) # Split on first whitespace
                    if not parts: continue

                    seq_id = parts[0] # This is the ID pyfaidx will use
                    if seq_id in processed_ids: continue # Avoid redundant processing if ID appears twice (unlikely but possible)
                    processed_ids.add(seq_id)

                    header_description = parts[1] if len(parts) > 1 else ""

                    # --- Primary Parsing Strategy: Look for "chromosome XXX" ---
                    # Make regex case-insensitive and look for word chars or dots/dashes
                    # Handles "chromosome 1,", "chromosome X", "chromosome MT.", "chr. 1" etc.
                    # Improved regex to be more flexible
                    match = re.search(r'[Cc]hr(?:omosome)?\.?\s+([\w.-]+)', header_description)
                    human_chr = None
                    if match:
                        human_chr = match.group(1).rstrip(',.;') # Extract and clean

                    # --- Fallback Strategy: If seq_id itself looks like a chromosome name ---
                    # Useful if header is just ">chr1" or ">1" and no description match
                    if human_chr is None and re.match(r'^chr[\w.-]+$|^[0-9XYMTZW]+$', seq_id, re.I):
                         human_chr = seq_id # Treat the ID itself as the human name candidate

                    # --- Populate Map ---
                    if human_chr:
                        # Ensure self-mapping exists for the ID pyfaidx uses
                        if seq_id not in chr_map:
                             chr_map[seq_id] = seq_id

                        # Add mapping from human name to seq_id, checking conflicts
                        if human_chr in chr_map and chr_map[human_chr] != seq_id:
                            print(f"Warning: Map conflict! '{human_chr}' found for '{seq_id}' but already mapped to '{chr_map[human_chr]}'. Keeping first mapping.", file=sys.stderr)
                        elif human_chr not in chr_map:
                            chr_map[human_chr] = seq_id

                        # Add common variations ('1' <-> 'chr1', 'MT' <-> 'Mito')
                        if human_chr.isdigit():
                             chr_version = f"chr{human_chr}"
                             if chr_version not in chr_map: chr_map[chr_version] = seq_id
                        elif human_chr.lower().startswith('chr') and human_chr.replace('chr','',1) not in chr_map:
                             non_chr_version = human_chr.replace('chr','',1)
                             if non_chr_version not in chr_map: chr_map[non_chr_version] = seq_id

                        if human_chr.upper() == "MT" and "Mito" not in chr_map:
                            chr_map["Mito"] = seq_id
                        elif human_chr.upper() == "MITO" and "MT" not in chr_map:
                            chr_map["MT"] = seq_id
                    else:
                         # If no human_chr found, ensure self-mapping still exists
                         if seq_id not in chr_map:
                              chr_map[seq_id] = seq_id

        if not chr_map:
             print(f"Warning: Chromosome map is empty after parsing {fasta_path.name}. Check FASTA headers.", file=sys.stderr)
             return None # Indicate failure

        print(f"Built map with {len(chr_map)} entries by parsing headers.")
        # print(f"Debug Map Sample: {dict(list(chr_map.items())[:15])}") # Print sample
        return chr_map

    except FileNotFoundError:
         print(f"Error: FASTA file not found for mapping: {fasta_path}", file=sys.stderr)
         return None
    except Exception as e:
        print(f"Error parsing FASTA headers for {fasta_path.name}: {e}", file=sys.stderr)
        return None


# --- Main Script Logic ---
if __name__ == "__main__":
    print("Starting sequence extraction process (using pyfaidx, parsing headers)...")

    # --- Input Validation ---
    variant_csv_path = Path(VARIANT_CSV_FILE)
    genome_dir_path = Path(GENOME_DIR)
    if not variant_csv_path.is_file(): sys.exit(f"Error: Variant CSV file not found: {VARIANT_CSV_FILE}")
    if not genome_dir_path.is_dir(): sys.exit(f"Error: Genome directory not found: {GENOME_DIR}")

    # --- Create temporary directory ---
    temp_fasta_path = Path(TEMP_FASTA_DIR)
    temp_fasta_path.mkdir(exist_ok=True)
    print(f"Using temporary directory: {TEMP_FASTA_DIR}")

    # --- Load Variant Data ---
    print(f"Loading variants from {VARIANT_CSV_FILE}...")
    try:
        variants_df = pd.read_csv(variant_csv_path, dtype={'Chr': str}) # Read Chr as string
        variants_df['Position'] = pd.to_numeric(variants_df['Position'], errors='coerce').astype('Int64')
        variants_df.dropna(subset=['Position'], inplace=True) # Drop rows with invalid Position
        variants_df['Ref Allele'] = variants_df['Ref Allele'].astype(str)
        variants_df['Alt Allele'] = variants_df['Alt Allele'].astype(str)
        print(f"Loaded {len(variants_df)} variants.")
    except Exception as e:
        sys.exit(f"Error reading CSV file {VARIANT_CSV_FILE}: {e}")

    # --- Initialize new columns ---
    variants_df['Ref_Sequence_Window'] = pd.NA
    variants_df['Alt_Sequence_Window'] = pd.NA
    variants_df['Processing_Status'] = 'Pending'

    # --- Map Genome Name to Accession ---
    print("Mapping Reference Sequence name to NCBI Accession...")
    variants_df['NCBI_Accession'] = variants_df['Reference Sequence'].str.strip().map(GENOME_NAME_TO_ACCESSION)
    missing_map = variants_df[variants_df['NCBI_Accession'].isna()]
    if not missing_map.empty:
        print("\nWarning: Could not map the following 'Reference Sequence' names:", file=sys.stderr)
        print(missing_map['Reference Sequence'].unique(), file=sys.stderr)
        variants_df.loc[variants_df['NCBI_Accession'].isna(), 'Processing_Status'] = 'Failed: Cannot map Reference Sequence to Accession'

    # --- Dictionary to store chromosome maps and Fasta objects ---
    genome_data = {} # Stores {'accession_key': {'fasta': FastaObject, 'map': chr_map_dict}}

    # --- Group by Genome (only process rows with valid accession) ---
    valid_variants_df = variants_df.dropna(subset=['NCBI_Accession']).copy()
    if valid_variants_df.empty:
         print("No variants with valid Accessions found. Exiting processing loop.", file=sys.stderr)
    else:
        grouped_variants = valid_variants_df.groupby('NCBI_Accession')
        print(f"Processing variants grouped by {len(grouped_variants)} unique NCBI Accessions.")

        processed_genomes_prep_status = {} # Track if genome prep succeeded or failed

        # --- Process Each Genome Group ---
        for accession_group_key, group_df in grouped_variants:
            print(f"\n--- Processing Genome Group: {accession_group_key} ({len(group_df)} variants) ---")

            # If genome prep already failed, skip all variants for this group
            if accession_group_key in processed_genomes_prep_status and not processed_genomes_prep_status[accession_group_key]:
                 print(f"Skipping group {accession_group_key} as genome preparation previously failed.")
                 variants_df.loc[group_df.index, 'Processing_Status'] = 'Failed: Genome Preparation Error'
                 continue

            # Handle GCA/GCF check and determine actual accession/path used for files
            accession = accession_group_key # Start with the group key
            genome_zip_path_gcf = genome_dir_path / f"{accession}.zip"
            genome_zip_path_gca = genome_dir_path / f"{accession.replace('GCF','GCA')}.zip"
            genome_zip_path = None
            if genome_zip_path_gcf.is_file():
                genome_zip_path = genome_zip_path_gcf
            elif genome_zip_path_gca.is_file():
                 genome_zip_path = genome_zip_path_gca
                 accession = accession.replace('GCF','GCA') # Use the GCA name for file ops
                 print(f"Note: Using GCA archive {genome_zip_path} for accession group {accession_group_key}")
            else:
                 print(f"Error: Genome ZIP archive not found for {accession_group_key} (checked GCF/GCA).", file=sys.stderr)
                 variants_df.loc[group_df.index, 'Processing_Status'] = 'Failed: Genome ZIP not found'
                 processed_genomes_prep_status[accession_group_key] = False # Mark prep as failed
                 continue

            temp_genome_fasta = temp_fasta_path / f"{accession}.fna"
            current_fasta_obj = None
            current_chr_map = None

            # --- Prepare FASTA & Build Chromosome Map (if needed) ---
            if accession_group_key not in processed_genomes_prep_status:
                # 1. Ensure FASTA file exists (Extract if needed)
                if not temp_genome_fasta.is_file():
                    print(f"Preparing FASTA for {accession}...")
                    assembly_name = "UnknownAssembly"
                    for name, acc_map in GENOME_NAME_TO_ACCESSION.items():
                        # Compare accession, considering potential GCA/GCF swap
                        if acc_map == accession_group_key or acc_map.replace('GCF','GCA') == accession_group_key:
                            assembly_name = name.replace(" ", "_")
                            break
                    internal_fasta_path = f"ncbi_dataset/data/{accession}/{accession}_{assembly_name}_genomic.fna"
                    print(f"Attempting to extract: {internal_fasta_path} from {genome_zip_path}")
                    try:
                        with zipfile.ZipFile(genome_zip_path, 'r') as zip_ref:
                            if internal_fasta_path in zip_ref.namelist():
                                zip_ref.extract(internal_fasta_path, temp_fasta_path)
                                extracted_path = temp_fasta_path / internal_fasta_path
                                temp_genome_fasta.parent.mkdir(parents=True, exist_ok=True)
                                extracted_path.rename(temp_genome_fasta)
                                print(f"Extracted FASTA to: {temp_genome_fasta}")
                            else:
                                print(f"Error: FASTA file '{internal_fasta_path}' not found within {genome_zip_path}", file=sys.stderr)
                                variants_df.loc[group_df.index, 'Processing_Status'] = 'Failed: FASTA path not in ZIP'
                                processed_genomes_prep_status[accession_group_key] = False
                                continue # Skip group
                    except Exception as e:
                        print(f"Error extracting from {genome_zip_path}: {e}", file=sys.stderr)
                        variants_df.loc[group_df.index, 'Processing_Status'] = 'Failed: ZIP Extraction Error'
                        processed_genomes_prep_status[accession_group_key] = False
                        continue
                else:
                     print(f"Found existing extracted FASTA: {temp_genome_fasta}")

                # 2. Build Map by parsing headers
                # This reads the FASTA file directly
                current_chr_map = build_chromosome_map_from_fasta_headers(temp_genome_fasta)
                if current_chr_map is None:
                     print(f"Error: Failed to build chromosome map from headers for {accession}. Cannot process variants.", file=sys.stderr)
                     variants_df.loc[group_df.index, 'Processing_Status'] = 'Failed: Chromosome Map Build Error'
                     processed_genomes_prep_status[accession_group_key] = False
                     continue # Skip this group

                # 3. Initialize pyfaidx Fasta object (Creates/uses .fai)
                try:
                    print(f"Indexing/Loading FASTA using pyfaidx: {temp_genome_fasta}")
                    current_fasta_obj = Fasta(
                        str(temp_genome_fasta),
                        sequence_always_upper=True,
                        rebuild=False, # Assume map build didn't force rebuild, let Fasta check
                        read_ahead=None
                    )

                    # Sanity check: Ensure all values in the map are actual keys in Fasta object
                    missing_keys = [v for v in current_chr_map.values() if v not in current_fasta_obj]
                    if missing_keys:
                         print(f"Warning: The following sequence IDs from the map were not found as keys by pyfaidx for {accession}: {missing_keys[:5]}...", file=sys.stderr)
                         # This might indicate an issue with header parsing or pyfaidx key reading
                         # Depending on severity, might want to mark prep as failed

                    # Store data for reuse
                    genome_data[accession_group_key] = {'fasta': current_fasta_obj, 'map': current_chr_map}
                    processed_genomes_prep_status[accession_group_key] = True # Mark prep as successful

                except FastaIndexingError as e:
                     print(f"Error indexing FASTA {temp_genome_fasta} with pyfaidx: {e}", file=sys.stderr)
                     variants_df.loc[group_df.index, 'Processing_Status'] = 'Failed: pyfaidx Indexing Error'
                     processed_genomes_prep_status[accession_group_key] = False
                     if current_chr_map is not None: # Clear map if Fasta object failed
                         genome_data.pop(accession_group_key, None)
                     continue
                except Exception as e:
                     print(f"Error initializing pyfaidx Fasta object for {temp_genome_fasta}: {e}", file=sys.stderr)
                     variants_df.loc[group_df.index, 'Processing_Status'] = 'Failed: pyfaidx Init Error'
                     processed_genomes_prep_status[accession_group_key] = False
                     if current_chr_map is not None:
                         genome_data.pop(accession_group_key, None)
                     continue

            else:
                # Genome prep was already attempted, retrieve data if successful
                if processed_genomes_prep_status[accession_group_key]:
                    genome_info = genome_data.get(accession_group_key)
                    if genome_info:
                        current_fasta_obj = genome_info['fasta']
                        current_chr_map = genome_info['map']
                        if current_fasta_obj is None or current_chr_map is None:
                             print(f"Error: Lost Fasta object or map for already processed genome {accession_group_key}.", file=sys.stderr)
                             variants_df.loc[group_df.index, 'Processing_Status'] = 'Failed: Internal State Error'
                             continue
                    else:
                        print(f"Error: Genome data not found for successfully prepped genome {accession_group_key}.", file=sys.stderr)
                        variants_df.loc[group_df.index, 'Processing_Status'] = 'Failed: Internal State Error'
                        continue
                else:
                     # Prep failed previously, status already set, just continue loop
                     continue


            # --- Extract Sequences for Variants in this Group ---
            flank = WINDOW_SIZE // 2
            for index, row in group_df.iterrows():
                # Get row index in the *original* dataframe to update it
                original_index = index

                # Skip if status is already failed from prep stage
                if variants_df.loc[original_index, 'Processing_Status'].startswith('Failed'):
                    continue

                try:
                    csv_chr = str(row['Chr']).strip()
                    pos = int(row['Position']) # Position is 1-based from CSV
                    ref_allele = str(row['Ref Allele']).strip().upper()
                    alt_allele = str(row['Alt Allele']).strip().upper()
                    omia_id = row['OMIA Variant ID']

                    # Basic validation for typical SNVs
                    if len(ref_allele) != 1 or len(alt_allele) != 1:
                        variants_df.loc[original_index, 'Processing_Status'] = 'Failed: Not single base SNV'
                        continue

                    # --- Use the map to get pyfaidx chromosome ID ---
                    pyfaidx_chr_id = current_chr_map.get(csv_chr)

                    # Add specific fallbacks if primary lookup fails
                    if pyfaidx_chr_id is None:
                        if csv_chr.startswith('chr') and csv_chr.replace('chr','',1) in current_chr_map:
                             pyfaidx_chr_id = current_chr_map.get(csv_chr.replace('chr','',1))
                        elif not csv_chr.startswith('chr') and f"chr{csv_chr}" in current_chr_map:
                             pyfaidx_chr_id = current_chr_map.get(f"chr{csv_chr}")
                        elif csv_chr == "Mito" and "MT" in current_chr_map:
                             pyfaidx_chr_id = current_chr_map.get("MT")
                        elif csv_chr == "MT" and "Mito" in current_chr_map:
                             pyfaidx_chr_id = current_chr_map.get("Mito")
                        elif csv_chr in current_fasta_obj: # Try direct key match last
                             pyfaidx_chr_id = csv_chr
                             print(f"Note: Using direct CSV Chr '{csv_chr}' as key for OMIA ID {omia_id} (map lookup failed).", file=sys.stderr)

                    if pyfaidx_chr_id is None:
                        variants_df.loc[original_index, 'Processing_Status'] = f"Failed: Cannot map chromosome '{csv_chr}'"
                        continue # Cannot find appropriate key
                    # ----------------------------------------------------

                    # --- Extract sequence using pyfaidx ---
                    start = max(1, pos - flank + 1) # 1-based start of window
                    end = pos + flank             # 1-based end of window

                    try:
                        seq_record = current_fasta_obj[pyfaidx_chr_id]
                        chrom_len = len(seq_record)
                        fetch_start = max(1, start)
                        fetch_end = min(end, chrom_len)

                        if fetch_start > fetch_end:
                             variants_df.loc[original_index, 'Processing_Status'] = 'Failed: Position outside chr bounds'
                             continue

                        # Use get_seq for consistency with 1-based coords
                        sequence_record = current_fasta_obj.get_seq(
                             name=pyfaidx_chr_id,
                             start=fetch_start,
                             end=fetch_end
                        )
                        sequence = sequence_record.seq # Get sequence string (already uppercase)

                    # Catch expected errors from pyfaidx or standard Python errors
                    except (KeyError, ValueError, IndexError, FastaIndexingError) as fetch_error:
                        print(f"Failed to extract sequence for OMIA ID {omia_id} ({pyfaidx_chr_id}:{fetch_start}-{fetch_end}). Error: {fetch_error}", file=sys.stderr)
                        variants_df.loc[original_index, 'Processing_Status'] = f"Failed: pyfaidx extract error"
                        continue
                    # -----------------------------------------

                    if len(sequence) == 0:
                         variants_df.loc[original_index, 'Processing_Status'] = 'Failed: Extracted empty sequence'
                         continue

                    # --- Ref Allele Check ---
                    relative_pos_index = pos - fetch_start # 0-based index

                    if not (0 <= relative_pos_index < len(sequence)):
                        variants_df.loc[original_index, 'Processing_Status'] = 'Failed: Position outside extracted bounds'
                        continue

                    ref_base_in_sequence = sequence[relative_pos_index]
                    if ref_base_in_sequence != ref_allele:
                        variants_df.loc[original_index, 'Processing_Status'] = f"Failed: Ref Mismatch (Exp {ref_allele}, Got {ref_base_in_sequence})"
                        continue

                    # --- Generate Alt Allele Sequence ---
                    alt_sequence_list = list(sequence)
                    alt_sequence_list[relative_pos_index] = alt_allele
                    alt_sequence = "".join(alt_sequence_list)

                    # --- Store results in the main DataFrame ---
                    pad_start = fetch_start - start
                    pad_end = end - fetch_end

                    final_ref_sequence = ('N' * pad_start) + sequence + ('N' * pad_end)
                    final_alt_sequence = ('N' * pad_start) + alt_sequence + ('N' * pad_end)

                    # Ensure final length
                    final_ref_sequence = (final_ref_sequence + 'N' * WINDOW_SIZE)[:WINDOW_SIZE]
                    final_alt_sequence = (final_alt_sequence + 'N' * WINDOW_SIZE)[:WINDOW_SIZE]

                    variants_df.loc[original_index, 'Ref_Sequence_Window'] = final_ref_sequence
                    variants_df.loc[original_index, 'Alt_Sequence_Window'] = final_alt_sequence
                    variants_df.loc[original_index, 'Processing_Status'] = 'Success'

                except ValueError as e:
                    variants_df.loc[original_index, 'Processing_Status'] = 'Failed: Data format/ValueError'
                except KeyError as e:
                     variants_df.loc[original_index, 'Processing_Status'] = 'Failed: Missing Column/KeyError'
                except Exception as e:
                    print(f"Unexpected error processing variant row {original_index} (OMIA ID {row.get('OMIA Variant ID', 'N/A')}): {e}", file=sys.stderr)
                    variants_df.loc[original_index, 'Processing_Status'] = 'Failed: Unexpected Python Error'


    # --- Close Fasta file handles ---
    print("\nClosing open FASTA file handles...")
    for key, data in genome_data.items():
        if 'fasta' in data and data['fasta']:
            try: data['fasta'].close() # Close the pyfaidx Fasta object
            except Exception as e: print(f"Warning: Error closing Fasta object for {key}: {e}", file=sys.stderr)


    print("\n--- Processing Complete ---")

    # --- Write Output CSV ---
    print(f"Writing output CSV file: {OUTPUT_CSV_FILE}...")
    output_csv_path = Path(OUTPUT_CSV_FILE)
    try:
        # Define desired output columns order
        original_cols = list(pd.read_csv(variant_csv_path, nrows=0).columns)
        output_columns = original_cols + ['NCBI_Accession', 'Ref_Sequence_Window', 'Alt_Sequence_Window', 'Processing_Status']

        # Ensure columns exist before selecting, handle potential missing columns gracefully
        final_output_columns = [col for col in output_columns if col in variants_df.columns]

        variants_df_output = variants_df[final_output_columns]
        variants_df_output.to_csv(output_csv_path, index=False, na_rep='NA') # Use NA for missing values
        print(f"Output written successfully.")
    except Exception as e:
        print(f"Error writing output CSV: {e}", file=sys.stderr)


    # --- Final Summary ---
    success_count = (variants_df['Processing_Status'] == 'Success').sum()
    fail_count_total = (variants_df['Processing_Status'].str.startswith('Failed', na=False)).sum()
    pending_count = (variants_df['Processing_Status'] == 'Pending').sum() # Should be 0

    print("\n--- Final Summary ---")
    print(f"Successfully processed variants: {success_count}")
    print(f"Failed variants (total): {fail_count_total}")
    print("Failure Reasons Breakdown:")
    # Only print breakdown if there were failures
    failure_counts = variants_df.loc[variants_df['Processing_Status'].str.startswith('Failed', na=False), 'Processing_Status'].value_counts()
    if not failure_counts.empty:
        print(failure_counts)
    else:
        print("(No failures recorded)")
    print(f"Output CSV written to: {OUTPUT_CSV_FILE}")
    if pending_count > 0:
        print(f"Warning: {pending_count} variants still marked as 'Pending'. This might indicate an issue.")

    # --- Optional Cleanup ---
    # clean_temp = input(f"Remove temporary directory '{TEMP_FASTA_DIR}'? (y/N): ")
    # if clean_temp.lower() == 'y':
    #     print(f"Removing {TEMP_FASTA_DIR}...")
    #     shutil.rmtree(temp_fasta_path)
    # else:
    #      print(f"Temporary files kept in {TEMP_FASTA_DIR}")

    print("Done.")

