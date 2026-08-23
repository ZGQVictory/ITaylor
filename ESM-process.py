"""
ESM-process.py (Version 3.0 - Multi-Precision Support)

Important notes for Version 3.0
To reduce memory usage, this version:
  1. Processes and stores embeddings only for positive samples.
  2. Stores embeddings in both float32 and float16 precision.
  3. Builds negative samples dynamically from CSV ``id_tcr`` values during training.

Features
1. ESM-C embedding generation:
   - Embedding length equals sequence length + 2 (start and end tokens included).
   - Each residue-level embedding has 1,152 dimensions.
   - Start- and end-token embeddings are retained.
   - Both float32 (full precision) and float16 (reduced memory) files are stored.

2. Length filtering:
   - Samples with seven-residue epitopes are removed automatically.
   - Both complete and filtered files (suffix ``_clear_peplen7``) are generated.

3. Multiple output precisions:
   - float32: used for positive samples at full precision.
   - float16: optional compact representation requiring approximately 50% less memory.

Output layout (positive samples):
esm-embedding/
  |-- training_positive_esm_embeddings_float32.pt                # Complete float32 embeddings
  |-- training_positive_esm_embeddings_float16.pt                # Complete float16 embeddings
  |-- training_positive_metadata.pt                              # Complete metadata
  |-- training_positive_clear_peplen7_esm_embeddings_float32.pt  # Filtered float32 embeddings
  |-- training_positive_clear_peplen7_esm_embeddings_float16.pt  # Filtered float16 embeddings
  |-- training_positive_metadata_clear_peplen7.pt                # Filtered metadata
  `-- training_positive_removed_peplen7.csv                      # Removed-sample records
"""

import os
import sys

# Set CUDA_VISIBLE_DEVICES before importing torch.
# Obtain the device index from the configured device string.
DEVICE = "cuda:0"  # Use cuda:0 by default.
if "cuda" in DEVICE:
    device_id = DEVICE.split(":")[-1]
    os.environ["CUDA_VISIBLE_DEVICES"] = device_id
    print(f"Set CUDA_VISIBLE_DEVICES={device_id}; using physical GPU {device_id} only")

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
from typing import Dict, List, Tuple, Any

# Fix the random seed for reproducibility.
RANDOM_SEED = 42


def sample_negative_data(
    negative_csv_path: str,
    positive_count: int,
    ratio: int,
    output_csv_path: str,
    random_seed: int = RANDOM_SEED
) -> pd.DataFrame:
    """
    Randomly sample a specified number of negative examples.

    Args:
        negative_csv_path: Path to the negative-sample CSV.
        positive_count: Number of positive samples.
        ratio: Negative-to-positive sampling ratio.
        output_csv_path: Path for the sampled output CSV.
        random_seed: Random seed.

    Returns:
        Sampled DataFrame.
    """
    print(f"\n{'='*60}")
    print(f"Sampling negative data...")
    print(f"Positive-to-negative ratio: 1:{ratio}")
    print(f"Random seed: {random_seed}")

    # Load negative samples.
    df_neg = pd.read_csv(negative_csv_path)
    print(f"Total negative samples: {len(df_neg)}")

    # Calculate the requested sample count.
    sample_size = positive_count * ratio
    print(f"Requested sample count: {sample_size}")

    if sample_size > len(df_neg):
        print(f"Warning: requested count ({sample_size}) exceeds the available negatives ({len(df_neg)})")
        print(f"Using all negative samples")
        sampled_df = df_neg
    else:
        # Sample with a fixed random seed.
        sampled_df = df_neg.sample(n=sample_size, random_state=random_seed)
        print(f"Sampling complete: {len(sampled_df)} samples")

    # Save the sampled CSV.
    sampled_df.to_csv(output_csv_path, index=False)
    print(f"Sampled data saved to: {output_csv_path}")
    print(f"{'='*60}\n")

    return sampled_df


def remove_peplen7_samples(
    all_embeddings: Dict[str, Dict[str, torch.Tensor]],
    metadata: Dict[str, List[Any]],
    output_dir: str,
    file_prefix: str
) -> Tuple[Dict[str, Dict[str, torch.Tensor]], Dict[str, List[Any]], pd.DataFrame]:
    """
    Remove samples with seven-residue epitopes.

    Args:
        all_embeddings: Dictionary containing all embeddings.
        metadata: Metadata dictionary.
        output_dir: Output directory.
        file_prefix: Output filename prefix, for example ``training_negative_ratio_1_5``.

    Returns:
        Filtered embeddings, filtered metadata, and a DataFrame of removed samples.
    """
    print(f"\n{'='*60}")
    print(f"Removing samples with seven-residue epitopes...")

    n_before = len(metadata['sample_ids'])
    print(f"Samples before filtering: {n_before}")

    # Find samples to remove.
    remove_indices = [i for i, L in enumerate(metadata['epitope_lengths']) if L == 7]
    remove_indices_set = set(remove_indices)

    if not remove_indices:
        print(f"No seven-residue epitopes found; skipping filtering")
        print(f"{'='*60}\n")
        return all_embeddings, metadata, pd.DataFrame()

    print(f"Found {len(remove_indices)} samples with seven-residue epitopes")

    # Collect metadata for removed samples.
    removed_records = []
    removed_ids = []

    for i in remove_indices:
        sid = metadata['sample_ids'][i]
        removed_ids.append(sid)
        removed_records.append({
            'id': sid,
            'epitope_length': metadata['epitope_lengths'][i],
            'hla_length': metadata['hla_lengths'][i],
            'tcra_length': metadata['tcra_lengths'][i],
            'tcrb_length': metadata['tcrb_lengths'][i],
            'hla_allele': metadata['hla_allele'][i],
            'label': metadata['labels'][i]
        })

    # Remove samples from the embedding dictionaries.
    new_embeddings = {
        'Epitope': {},
        'hla_alpha123_mature': {},
        'tcra_variable': {},
        'tcrb_variable': {}
    }

    for field in ['Epitope', 'hla_alpha123_mature', 'tcra_variable', 'tcrb_variable']:
        for sid in all_embeddings[field]:
            if sid not in removed_ids:
                new_embeddings[field][sid] = all_embeddings[field][sid]

    # Rebuild metadata without removed samples.
    new_metadata = {k: [] for k in metadata.keys()}
    for i in range(n_before):
        if i not in remove_indices_set:
            for k in metadata.keys():
                new_metadata[k].append(metadata[k][i])

    n_after = len(new_metadata['sample_ids'])
    print(f"Samples after filtering: {n_after} ({n_before - n_after} removed)")

    # Save records for removed samples.
    removed_df = pd.DataFrame(removed_records)
    removed_csv_path = os.path.join(output_dir, f"{file_prefix}_removed_peplen7.csv")
    removed_df.to_csv(removed_csv_path, index=False)
    print(f"Removed-sample records saved to: {removed_csv_path}")
    print(f"{'='*60}\n")

    return new_embeddings, new_metadata, removed_df


def process_sequence(client, sequence, device="cuda:0"):
    """
    Process one sequence with ESM-C.

    Args:
        client: ESM-C model.
        sequence: Amino-acid sequence.
        device: Compute device (cuda:0 after setting CUDA_VISIBLE_DEVICES).

    Returns:
        Embeddings with shape [seq_len + 2, 1152], including start and end tokens.
    """
    protein = ESMProtein(sequence=sequence)
    protein_tensor = client.encode(protein)
    logits_output = client.logits(
        protein_tensor, LogitsConfig(sequence=True, return_embeddings=True)
    )
    
    # Retrieve embeddings while retaining start and end tokens.
    # embeddings: [1, seq_len+2, 1152]
    full_embeddings = logits_output.embeddings[0]  # [seq_len+2, 1152]

    return full_embeddings


def process_and_save_embeddings(
    df: pd.DataFrame,
    client: ESMC,
    device: str,
    output_dir: str,
    file_prefix: str,
    remove_peplen7: bool = True
) -> None:
    """
    Generate and save embeddings for all sequences in a DataFrame.

    Args:
        df: Input DataFrame.
        client: ESM-C model.
        device: Compute device.
        output_dir: Output directory.
        file_prefix: Output filename prefix, for example ``training_positive``.
        remove_peplen7: Whether to remove samples with seven-residue epitopes.
    """
    print(f"\n{'='*60}")
    print(f"Generating ESM embeddings...")
    print(f"Number of samples: {len(df)}")
    print(f"Saving both float32 and float16 representations")
    print(f"{'='*60}\n")

    # Store all embeddings at float32 precision.
    all_embeddings_fp32 = {
        'Epitope': {},
        'hla_alpha123_mature': {},
        'tcra_variable': {},
        'tcrb_variable': {}
    }

    # Store sample metadata.
    metadata = {
        'sample_ids': [],
        'epitope_lengths': [],
        'hla_lengths': [],
        'tcra_lengths': [],
        'tcrb_lengths': [],
        'hla_allele': [],
        'labels': []
    }

    # Process samples one at a time.
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating ESM embeddings"):
        sample_id = row['id']

        with torch.no_grad():
            # Process all four sequences and retain start and end tokens.
            epitope_emb = process_sequence(client, row['Epitope'], device)
            hla_emb = process_sequence(client, row['hla_alpha123_mature'], device)
            tcra_emb = process_sequence(client, row['tcra_variable'], device)
            tcrb_emb = process_sequence(client, row['tcrb_variable'], device)

            # Move embeddings to CPU and store them at float32 precision.
            all_embeddings_fp32['Epitope'][sample_id] = epitope_emb.cpu().float()
            all_embeddings_fp32['hla_alpha123_mature'][sample_id] = hla_emb.cpu().float()
            all_embeddings_fp32['tcra_variable'][sample_id] = tcra_emb.cpu().float()
            all_embeddings_fp32['tcrb_variable'][sample_id] = tcrb_emb.cpu().float()

            # Store metadata.
            metadata['sample_ids'].append(sample_id)
            metadata['epitope_lengths'].append(len(row['Epitope']))
            metadata['hla_lengths'].append(len(row['hla_alpha123_mature']))
            metadata['tcra_lengths'].append(len(row['tcra_variable']))
            metadata['tcrb_lengths'].append(len(row['tcrb_variable']))
            metadata['hla_allele'].append(row['hla_allele'])
            metadata['labels'].append(row['label'])

    # Create float16 copies.
    print(f"\nConverting embeddings to float16...")
    all_embeddings_fp16 = {
        'Epitope': {},
        'hla_alpha123_mature': {},
        'tcra_variable': {},
        'tcrb_variable': {}
    }

    for field in ['Epitope', 'hla_alpha123_mature', 'tcra_variable', 'tcrb_variable']:
        for sample_id in all_embeddings_fp32[field]:
            all_embeddings_fp16[field][sample_id] = all_embeddings_fp32[field][sample_id].half()

    # Save complete embeddings (float32 and float16) and metadata.
    embeddings_fp32_path = os.path.join(output_dir, f"{file_prefix}_esm_embeddings_float32.pt")
    embeddings_fp16_path = os.path.join(output_dir, f"{file_prefix}_esm_embeddings_float16.pt")
    metadata_path = os.path.join(output_dir, f"{file_prefix}_metadata.pt")

    print(f"\nSaving complete float32 embeddings to: {embeddings_fp32_path}")
    torch.save(all_embeddings_fp32, embeddings_fp32_path)
    print(f"Saving complete float16 embeddings to: {embeddings_fp16_path}")
    torch.save(all_embeddings_fp16, embeddings_fp16_path)
    print(f"Saving complete metadata to: {metadata_path}")
    torch.save(metadata, metadata_path)

    # Remove seven-residue epitopes and save the filtered datasets.
    if remove_peplen7:
        clean_embeddings_fp32, clean_metadata, removed_df = remove_peplen7_samples(
            all_embeddings_fp32, metadata, output_dir, file_prefix
        )

        # Create a float16 copy of the filtered embeddings.
        print(f"\nCreating filtered float16 embeddings...")
        clean_embeddings_fp16 = {
            'Epitope': {},
            'hla_alpha123_mature': {},
            'tcra_variable': {},
            'tcrb_variable': {}
        }

        for field in ['Epitope', 'hla_alpha123_mature', 'tcra_variable', 'tcrb_variable']:
            for sample_id in clean_embeddings_fp32[field]:
                clean_embeddings_fp16[field][sample_id] = clean_embeddings_fp32[field][sample_id].half()

        # Save filtered embeddings at both precisions.
        clean_embeddings_fp32_path = os.path.join(output_dir, f"{file_prefix}_clear_peplen7_esm_embeddings_float32.pt")
        clean_embeddings_fp16_path = os.path.join(output_dir, f"{file_prefix}_clear_peplen7_esm_embeddings_float16.pt")
        clean_metadata_path = os.path.join(output_dir, f"{file_prefix}_metadata_clear_peplen7.pt")

        print(f"Saving filtered float32 embeddings to: {clean_embeddings_fp32_path}")
        torch.save(clean_embeddings_fp32, clean_embeddings_fp32_path)
        print(f"Saving filtered float16 embeddings to: {clean_embeddings_fp16_path}")
        torch.save(clean_embeddings_fp16, clean_embeddings_fp16_path)
        print(f"Saving filtered metadata to: {clean_metadata_path}")
        torch.save(clean_metadata, clean_metadata_path)

    # Print summary statistics.
    print(f"\n{'='*60}")
    print(f"=== Processing complete ===")
    print(f"Complete dataset samples: {len(metadata['sample_ids'])}")
    if remove_peplen7 and len(clean_metadata['sample_ids']) < len(metadata['sample_ids']):
        print(f"Filtered dataset samples: {len(clean_metadata['sample_ids'])} ({len(metadata['sample_ids']) - len(clean_metadata['sample_ids'])} seven-residue epitope samples removed)")
    print(f"\nNotes:")
    print(f"  - All embeddings include start and end tokens")
    print(f"  - Both float32 and float16 representations were saved")
    print(f"  - float16 requires approximately 50% less memory")
    print(f"\nExample embedding shape:")
    print(f"  Epitope: {all_embeddings_fp32['Epitope'][metadata['sample_ids'][0]].shape}")
    print(f"  Sequence length: {metadata['epitope_lengths'][0]}, embedding length: {all_embeddings_fp32['Epitope'][metadata['sample_ids'][0]].shape[0]} (sequence length + 2)")
    print(f"{'='*60}\n")


def main():
    """
    Generate positive-sample ESM embeddings at multiple precisions.

    Configuration:
    - Process positive samples only.
    - Save both float32 and float16 representations.
    - Build negative samples dynamically from CSV ``id_tcr`` values during training.
    """
    # ========== Configuration ==========
    # Data paths
    POSITIVE_CSV = "./data/Database_stage1/outputs_split/training_positive.csv"
    OUTPUT_DIR = "./data/Database_stage1/esm-embedding"

    # Model configuration
    DEVICE = "cuda:0"  # Use cuda:0 after setting CUDA_VISIBLE_DEVICES above.
    MODEL_NAME = "esmc_600m"

    # Processing options
    REMOVE_PEPLEN7 = True  # Remove samples with seven-residue epitopes.
    # ============================

    # Create the output directory.
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load the ESM-C model.
    print(f"\n{'='*60}")
    print(f"ESM-process.py Version 3.0 - Multi-Precision Support")
    print(f"{'='*60}")
    print(f"Loading ESM-C model: {MODEL_NAME} on {DEVICE}")
    print(f"GPUs visible to PyTorch: {torch.cuda.device_count()}")
    client = ESMC.from_pretrained(MODEL_NAME).to(DEVICE)
    client.eval()
    print(f"Model loaded")
    print(f"{'='*60}\n")

    # Process positive samples.
    print(f"Processing mode: POSITIVE DATA ONLY")
    print(f"Reading CSV: {POSITIVE_CSV}")
    df = pd.read_csv(POSITIVE_CSV)
    print(f"Number of samples: {len(df)}")

    process_and_save_embeddings(
        df=df,
        client=client,
        device=DEVICE,
        output_dir=OUTPUT_DIR,
        file_prefix="training_positive",
        remove_peplen7=REMOVE_PEPLEN7
    )

    print(f"\n{'#'*60}")
    print(f"### All processing completed ###")
    print(f"{'#'*60}")
    print(f"\nAll files were saved under {OUTPUT_DIR}/")
    print(f"\nNotes:")
    print(f"  1. Embeddings were saved at both float32 and float16 precision")
    print(f"  2. Negative-sample embeddings are not generated in advance")
    print(f"  3. Negative samples are built dynamically during training from")
    print(f"     the id_tcr column in training_negative_clear_peplen7.csv")
    print(f"  4. Ensure that negative_data_summary.py has generated the id_tcr column")


if __name__ == "__main__":
    main()
