#!/usr/bin/env python3

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import time
import math # For math.ceil

# --- Check for and import Evo2 and PyTorch ---
try:
    from evo2.models import Evo2
    import torch # PyTorch is needed for your embedding example
    TORCH_AVAILABLE = True
except ImportError:
    print("Error: 'evo2' or 'torch' library is required but not found.", file=sys.stderr)
    print("Please install them: pip install evo2-cli torch torchvision torchaudio", file=sys.stderr)
    sys.exit(1)

# --- Configuration ---
INPUT_CSV = 'sampled_vcf_snv_sequences_ensembl_113.csv' # Or your latest CSV with sequences
OUTPUT_CSV_EMBEDDINGS = 'sampled_vcf_snv_sequences_ensembl_113_embeddings_simple_mean_pool.csv' # Output file
MODEL_NAME = 'evo2_7b'

# Specify the layer from which to extract embeddings
EMBEDDING_LAYER_NAME = 'blocks.20.mlp.l3' # As per your example

REF_SEQ_COL = 'Ref_Sequence_Window'
ALT_SEQ_COL = 'Alt_Sequence_Window'
EMBEDDING_PREFIX_REF = 'Ref_Emb'
EMBEDDING_PREFIX_ALT = 'Alt_Emb'
ROW_BATCH_SIZE = 16
DETERMINED_EMBEDDING_DIM = None

# --- Helper Function to Embed a BATCH of Sequences ---
def get_mean_pooled_embeddings_batch(sequence_strings_batch, model, tokenizer, device, layer_name):
    """
    Tokenizes a batch of sequence strings (one by one),
    pads them to form a batch tensor, gets embeddings via a single model call
    (WITHOUT explicit attention_mask passed to model),
    and returns SIMPLE mean-pooled embeddings across the sequence length.
    Returns a list of NumPy arrays (one per input sequence, or None for errors/invalid sequences).
    """
    global DETERMINED_EMBEDDING_DIM

    if not sequence_strings_batch:
        return []

    batch_embeddings_np = [None] * len(sequence_strings_batch)

    try:
        batch_token_ids_list = []
        actual_lengths = [] # Store actual token lengths before padding

        for idx, seq_str_single in enumerate(sequence_strings_batch):
            if not seq_str_single or not isinstance(seq_str_single, str) or seq_str_single.upper() == 'NA' or seq_str_single.strip() == '':
                batch_token_ids_list.append([])
                actual_lengths.append(0)
                continue
            try:
                token_ids = tokenizer.tokenize(seq_str_single)
                batch_token_ids_list.append(token_ids if token_ids else [])
                actual_lengths.append(len(token_ids) if token_ids else 0)
            except Exception as e_tok:
                print(f"Warning: Error tokenizing sequence '{seq_str_single[:30]}...': {e_tok}. Treating as empty.", file=sys.stderr)
                batch_token_ids_list.append([])
                actual_lengths.append(0)

        if not any(actual_lengths): # All sequences tokenized to empty or were invalid
            return batch_embeddings_np

        max_len = 0
        non_empty_token_lists_for_padding = [batch_token_ids_list[i] for i in range(len(batch_token_ids_list)) if batch_token_ids_list[i]]
        if non_empty_token_lists_for_padding:
             max_len = max(len(tokens) for tokens in non_empty_token_lists_for_padding)
        
        if max_len == 0: # All sequences effectively tokenized to empty
            return batch_embeddings_np

        padded_batch_token_ids = []
        pad_id = tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') else 0

        for i, tokens in enumerate(batch_token_ids_list):
            current_tokens = tokens if tokens else []
            padding_len = max_len - len(current_tokens)
            padded_tokens = current_tokens + [pad_id] * padding_len
            if len(padded_tokens) != max_len:
                padded_tokens = (current_tokens + [pad_id] * max_len)[:max_len]
            padded_batch_token_ids.append(padded_tokens)

        if not padded_batch_token_ids: return batch_embeddings_np

        input_ids = torch.tensor(padded_batch_token_ids, dtype=torch.long).to(device)

        with torch.no_grad():
            # Model call WITHOUT explicit attention_mask
            outputs, embeddings_dict = model(input_ids,
                                             return_embeddings=True,
                                             layer_names=[layer_name])
        layer_embeddings_tensor = embeddings_dict[layer_name] # Shape: [BATCH_SIZE, max_len, emb_dim]

        if DETERMINED_EMBEDDING_DIM is None and layer_embeddings_tensor is not None:
            DETERMINED_EMBEDDING_DIM = layer_embeddings_tensor.shape[-1]
            print(f"Determined embedding dimension: {DETERMINED_EMBEDDING_DIM}")

        # --- SIMPLE Mean Pooling across sequence length (dim=1) ---
        mean_pooled_batch_tensor = torch.mean(layer_embeddings_tensor, dim=1)
        # Resulting shape: [BATCH_SIZE, emb_dim]
        # --------------------------------------------------------
        
        mean_pooled_batch_tensor = mean_pooled_batch_tensor.to(torch.float32)

        for i in range(len(sequence_strings_batch)):
            if actual_lengths[i] == 0: # If original sequence tokenized to empty
                batch_embeddings_np[i] = None
            else:
                batch_embeddings_np[i] = mean_pooled_batch_tensor[i].cpu().numpy()
        
        return batch_embeddings_np
    except KeyError:
        print(f"FATAL Error: Layer name '{layer_name}' not found. Halting.", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e:
        print(f"FATAL RuntimeError during embedding batch: {e}. Halting.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error embedding batch: {e}", file=sys.stderr)
        return [None] * len(sequence_strings_batch)


# --- Main Script Logic ---
if __name__ == "__main__":
    print(f"Starting Evo 2 embedding extraction (batching rows, simple mean pool) for {INPUT_CSV}...")
    start_time = time.time()

    # --- Input Validation ---
    input_csv_path = Path(INPUT_CSV)
    if not input_csv_path.is_file(): sys.exit(f"Error: Input CSV file not found: {INPUT_CSV}")

    # --- Load Input Data ---
    print(f"Loading data from {input_csv_path}...")
    try:
        variants_df = pd.read_csv(input_csv_path)
        print(f"Loaded {len(variants_df)} variants.")
    except Exception as e: sys.exit(f"Error reading input CSV file {INPUT_CSV}: {e}")

    # --- Filter for Rows with Valid Sequences ---
    if REF_SEQ_COL not in variants_df.columns or ALT_SEQ_COL not in variants_df.columns:
        sys.exit(f"Error: Required columns '{REF_SEQ_COL}' or '{ALT_SEQ_COL}' not found")
    variants_df[REF_SEQ_COL] = variants_df[REF_SEQ_COL].astype(str)
    variants_df[ALT_SEQ_COL] = variants_df[ALT_SEQ_COL].astype(str)
    original_count = len(variants_df)
    variants_df['valid_for_embedding'] = (
        variants_df[REF_SEQ_COL].notna() & (variants_df[REF_SEQ_COL].str.upper() != 'NA') & (variants_df[REF_SEQ_COL].str.strip() != '') &
        variants_df[ALT_SEQ_COL].notna() & (variants_df[ALT_SEQ_COL].str.upper() != 'NA') & (variants_df[ALT_SEQ_COL].str.strip() != '')
    )
    valid_seq_df = variants_df[variants_df['valid_for_embedding']].copy()
    num_valid = len(valid_seq_df); num_skipped = original_count - num_valid
    if num_skipped > 0: print(f"Identified {num_skipped} rows with missing/invalid sequences that will be skipped.")
    if num_valid == 0: sys.exit("Error: No rows with valid sequences found for embedding.")
    print(f"Processing {num_valid} variants with valid sequences.")

    # --- Load Evo 2 Model ---
    print(f"Loading Evo 2 model: {MODEL_NAME}...")
    device_to_use = 'cuda:0'
    evo2_model = Evo2(MODEL_NAME)

    # --- Process DataFrame in Chunks for Embedding ---
    print(f"\n--- Extracting Embeddings in Chunks of {ROW_BATCH_SIZE} rows ---")
    all_ref_embeddings_list = []
    all_alt_embeddings_list = []
    num_chunks = math.ceil(num_valid / ROW_BATCH_SIZE)

    for i in range(num_chunks):
        start_idx_in_valid_df = i * ROW_BATCH_SIZE
        end_idx_in_valid_df = min((i + 1) * ROW_BATCH_SIZE, num_valid)
        chunk_df = valid_seq_df.iloc[start_idx_in_valid_df:end_idx_in_valid_df]
        print(f"Processing row chunk {i+1}/{num_chunks} (DataFrame rows {chunk_df.index.min()}-{chunk_df.index.max()}) "
              f"Number of sequences in chunk: {len(chunk_df)}")
        ref_sequences_in_chunk = chunk_df[REF_SEQ_COL].tolist()
        alt_sequences_in_chunk = chunk_df[ALT_SEQ_COL].tolist()
        ref_embeddings_batch = get_mean_pooled_embeddings_batch(ref_sequences_in_chunk, evo2_model, evo2_model.tokenizer, device_to_use, EMBEDDING_LAYER_NAME)
        all_ref_embeddings_list.extend(ref_embeddings_batch)
        alt_embeddings_batch = get_mean_pooled_embeddings_batch(alt_sequences_in_chunk, evo2_model, evo2_model.tokenizer, device_to_use, EMBEDDING_LAYER_NAME)
        all_alt_embeddings_list.extend(alt_embeddings_batch)
        if DETERMINED_EMBEDDING_DIM is None and (any(e is not None for e in ref_embeddings_batch) or any(e is not None for e in alt_embeddings_batch)):
            for emb_vec in ref_embeddings_batch + alt_embeddings_batch:
                if emb_vec is not None:
                    DETERMINED_EMBEDDING_DIM = len(emb_vec)
                    print(f"Determined embedding dimension from batch: {DETERMINED_EMBEDDING_DIM}")
                    break
    
    # --- Add Embeddings to the DataFrame (MORE EFFICIENT WAY) ---
    if DETERMINED_EMBEDDING_DIM is None:
        print("Warning: Embedding dimension could not be determined. No embedding columns will be added to the final CSV.", file=sys.stderr)
    else:
        print(f"Preparing embedding columns (Dimension: {DETERMINED_EMBEDDING_DIM}) for final DataFrame...")
        ref_emb_col_names = [f"{EMBEDDING_PREFIX_REF}_{k}" for k in range(DETERMINED_EMBEDDING_DIM)]
        alt_emb_col_names = [f"{EMBEDDING_PREFIX_ALT}_{k}" for k in range(DETERMINED_EMBEDDING_DIM)]

        # Create temporary NumPy arrays aligned with valid_seq_df
        ref_embeddings_temp_data = np.full((num_valid, DETERMINED_EMBEDDING_DIM), np.nan)
        alt_embeddings_temp_data = np.full((num_valid, DETERMINED_EMBEDDING_DIM), np.nan)

        for i in range(num_valid): # num_valid is len(valid_seq_df) and len(all_..._list)
            if all_ref_embeddings_list[i] is not None:
                ref_embeddings_temp_data[i, :] = all_ref_embeddings_list[i]
            if all_alt_embeddings_list[i] is not None:
                alt_embeddings_temp_data[i, :] = all_alt_embeddings_list[i]
        
        # Create pandas DataFrames from these numpy arrays, using the index of valid_seq_df
        ref_embeddings_to_add_df = pd.DataFrame(
            ref_embeddings_temp_data, 
            columns=ref_emb_col_names, 
            index=valid_seq_df.index # Align with the original index of valid rows
        )
        alt_embeddings_to_add_df = pd.DataFrame(
            alt_embeddings_temp_data, 
            columns=alt_emb_col_names, 
            index=valid_seq_df.index # Align with the original index of valid rows
        )

        # Merge these new embedding columns with the original variants_df
        # This will align based on the index. Rows not in valid_seq_df will get NaNs.
        variants_df = variants_df.merge(ref_embeddings_to_add_df, left_index=True, right_index=True, how='left')
        variants_df = variants_df.merge(alt_embeddings_to_add_df, left_index=True, right_index=True, how='left')
        
        print("Embeddings columns added via merge.")

    # --- Write Output CSV ---
    output_csv_path = Path(OUTPUT_CSV_EMBEDDINGS)
    print(f"Writing output CSV file with embeddings: {output_csv_path}...")
    try:
        variants_df.drop(columns=['valid_for_embedding'], inplace=True, errors='ignore')
        variants_df.to_csv(output_csv_path, index=False, na_rep='NA')
        print("Output written successfully.")
    except Exception as e: print(f"Error writing output CSV: {e}", file=sys.stderr)

    # --- Final Summary ---
    end_time = time.time()
    print("\n--- Embedding Extraction Summary ---")
    print(f"Total input rows: {original_count}")
    print(f"Rows with valid sequences considered for embedding: {num_valid}")
    if DETERMINED_EMBEDDING_DIM is not None:
        ref_emb_success_count = sum(1 for emb_list in all_ref_embeddings_list if emb_list is not None)
        alt_emb_success_count = sum(1 for emb_list in all_alt_embeddings_list if emb_list is not None)
        print(f"Ref sequences for which embeddings were generated: {ref_emb_success_count}")
        print(f"Alt sequences for which embeddings were generated: {alt_emb_success_count}")
        print(f"Embedding dimension: {DETERMINED_EMBEDDING_DIM}")
    else: print("Embedding dimension could not be determined.")
    print(f"Output file: {output_csv_path}")
    print(f"Total time: {end_time - start_time:.2f} seconds")
    print("Done.")

