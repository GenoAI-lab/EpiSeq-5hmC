"""
EpiSeq-5hmC: Prepare ternary methylation target labels.

Parses GEO processed matrices and Zenodo sample metadata to compute
per-CpG 5hmC and 5mC fractions for each biological sample (ZD).

Data sources:
    1. GEO (GSE290585): Two processed beta-value matrices containing
       mixed BS and bACE arrays identified by IDAT column headers.
       The two files are batches uploaded at different dates, not
       separate BS/bACE splits.
    2. Zenodo: MouseArrayMaster.xlsx — maps each biological sample (ZD)
       to its paired BS (bACE==0) and bACE (bACE==1) IDAT identifiers,
       along with tissue type and age information.

Calculation:
    5hmC = bACE beta value (direct readout)
    5mC  = BS beta value - bACE beta value (clipped to >= 0)

Output:
    ternary_targets.csv — columns: {ZD_ID}_5hmC, {ZD_ID}_5mC

Usage:
    python prepare_targets.py
    python prepare_targets.py --config config.yaml
    python prepare_targets.py --data_dir /path/to/data --output ternary_targets.csv
"""

import os
import argparse

import numpy as np
import pandas as pd

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


# ==============================================================================
# Config
# ==============================================================================
DEFAULT_CONFIG = {
    'paths': {
        'data_dir': '.',
        'target_csv': 'ternary_targets.csv',
    },
}

GEO_MATRIX_FILES = [
    "GSE290585_20250221_GEO_processed_matrix_GPL30650.csv.gz",
    "GSE290585_20250827_GEO_processed_matrix_GPL30650.csv.gz",
]
METADATA_FILE = "MouseArrayMaster.xlsx"
SAMPLE_TSV_FILE = "output_unique_matched.tsv"


def load_config(path=None):
    cfg = {k: dict(v) for k, v in DEFAULT_CONFIG.items()}
    if path and os.path.exists(path) and HAS_YAML:
        with open(path) as f:
            u = yaml.safe_load(f)
        for s in cfg:
            if s in u and isinstance(cfg[s], dict):
                cfg[s].update(u[s])
    return cfg


# ==============================================================================
# Args
# ==============================================================================
def get_args():
    p = argparse.ArgumentParser(description="EpiSeq-5hmC Target Label Preparation")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--data_dir", type=str, default=None,
                   help="Directory containing downloaded GEO and Zenodo files")
    p.add_argument("--output", type=str, default=None,
                   help="Output CSV path (default: data/ternary_targets.csv)")
    return p.parse_args()


# ==============================================================================
# Step 1: Parse sample metadata
# ==============================================================================
def parse_metadata(metadata_path):
    """
    Parse MouseArrayMaster to find valid BS/bACE paired samples per ZD.

    Each biological sample (ZD) has two arrays:
        bACE == 0 → BS array (measures 5mC + 5hmC combined)
        bACE == 1 → bACE array (measures 5hmC directly)

    Returns:
        dict: {ZD_ID: {'BS': idat_id, 'bACE': idat_id, 'info': 'tissue_age'}}
    """
    print(f"Parsing metadata: {metadata_path} ...")

    ext = os.path.splitext(metadata_path)[1].lower()
    if ext in ('.xlsx', '.xls'):
        meta = pd.read_excel(metadata_path)
    else:
        meta = pd.read_csv(metadata_path)

    meta.columns = [c.strip() for c in meta.columns]

    valid_pairs = {}
    grouped = meta.groupby('ZD')

    for zd_id, group in grouped:
        bs_rows = group[group['bACE'] == 0]
        bace_rows = group[group['bACE'] == 1]

        # Require exactly one BS and one bACE array per ZD
        if len(bs_rows) == 1 and len(bace_rows) == 1:
            tissue = bs_rows.iloc[0].get('Tissue', 'unknown')
            age = bs_rows.iloc[0].get('AgeInWeeks', 'NA')
            valid_pairs[zd_id] = {
                'BS': bs_rows.iloc[0]['IDAT'],
                'bACE': bace_rows.iloc[0]['IDAT'],
                'info': f"{tissue}_{age}w",
            }

    print(f"  Found {len(valid_pairs)} valid ZD pairs "
          f"(BS + bACE) out of {len(grouped)} groups")
    return valid_pairs


def filter_by_sample_tsv(paired_info, sample_tsv_path):
    """
    Filter paired_info to include only ZD_IDs present in the ENCODE
    sample metadata (output_unique_matched.tsv).

    This ensures the target file contains only the 7 tissues used
    in the EpiSeq-5hmC framework.
    """
    print(f"Filtering ZD_IDs by: {sample_tsv_path} ...")
    if not os.path.exists(sample_tsv_path):
        raise FileNotFoundError(f"{sample_tsv_path} not found")

    tsv_df = pd.read_csv(sample_tsv_path, sep='\t')
    valid_zd_ids = set(tsv_df['ZD_ID'].unique())

    filtered = {zd: info for zd, info in paired_info.items() if zd in valid_zd_ids}
    print(f"  {len(paired_info)} total → {len(filtered)} matched "
          f"({len(valid_zd_ids)} ZD_IDs in sample TSV)")
    return filtered


# ==============================================================================
# Step 2: Load and merge GEO matrices
# ==============================================================================
def load_and_merge_matrices(file_paths):
    """Load multiple GEO processed matrices and merge by probe ID (index)."""
    dfs = []
    for fpath in file_paths:
        if not os.path.exists(fpath):
            print(f"  WARNING: File not found, skipping: {fpath}")
            continue
        print(f"  Loading: {fpath} ...")
        df = pd.read_csv(fpath, index_col=0, compression='gzip', low_memory=False)
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError("No GEO matrix files found. Run download_data.sh first.")

    print("  Merging matrices...")
    merged = pd.concat(dfs, axis=1)
    merged = merged.loc[:, ~merged.columns.duplicated()]
    print(f"  Merged shape: {merged.shape}")
    return merged


# ==============================================================================
# Step 3: Compute ternary targets
# ==============================================================================
def compute_ternary_targets(merged_df, paired_info):
    """
    Compute per-CpG 5hmC and 5mC fractions for each ZD sample.

    Calculation (per the ternary atlas method):
        5hmC = bACE beta value (direct measurement)
        5mC  = BS beta value - bACE beta value (clipped to >= 0)
    """
    print("Computing ternary targets...")
    result_dfs = []
    processed = 0

    for zd_id, info in paired_info.items():
        bs_id = info['BS']
        bace_id = info['bACE']

        # Check if both IDAT columns exist in matrix
        if bs_id not in merged_df.columns or bace_id not in merged_df.columns:
            continue

        bs_beta = pd.to_numeric(merged_df[bs_id], errors='coerce')
        bace_beta = pd.to_numeric(merged_df[bace_id], errors='coerce')

        # 5hmC = bACE (direct readout)
        col_5hmc = bace_beta.copy()

        # 5mC = BS - bACE (subtraction, clipped to non-negative)
        col_5mc = (bs_beta - bace_beta).clip(lower=0.0)

        result_dfs.append(pd.DataFrame({
            f"{zd_id}_5hmC": col_5hmc,
            f"{zd_id}_5mC": col_5mc,
        }))
        processed += 1

    if not result_dfs:
        raise ValueError("No valid target data extracted. "
                         "Check that IDAT IDs in metadata match matrix columns.")

    print(f"  Processed {processed} ZD samples")
    return pd.concat(result_dfs, axis=1)


# ==============================================================================
# Main
# ==============================================================================
def main():
    args = get_args()
    cfg = load_config(args.config)

    data_dir = args.data_dir or cfg['paths']['data_dir']
    output_path = args.output or cfg['paths']['target_csv']

    print("=" * 70)
    print("EpiSeq-5hmC: Ternary Target Label Preparation")
    print("=" * 70)

    # Step 1: Parse metadata
    metadata_path = os.path.join(data_dir, METADATA_FILE)
    paired_info = parse_metadata(metadata_path)

    # Step 1b: Filter to ZD_IDs used in EpiSeq-5hmC
    sample_tsv_path = os.path.join(data_dir, SAMPLE_TSV_FILE)
    paired_info = filter_by_sample_tsv(paired_info, sample_tsv_path)

    # Step 2: Load GEO matrices
    matrix_paths = [os.path.join(data_dir, f) for f in GEO_MATRIX_FILES]
    merged_df = load_and_merge_matrices(matrix_paths)

    # Step 3: Compute targets
    target_df = compute_ternary_targets(merged_df, paired_info)

    # Save
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    target_df.to_csv(output_path)
    print(f"\n>>> Saved: {output_path}")
    print(f"    Shape: {target_df.shape}")
    print(f"    Columns: {list(target_df.columns[:6])} ...")
    print(">>> Done")


if __name__ == "__main__":
    main()
