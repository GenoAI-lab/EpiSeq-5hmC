import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import pyBigWig
import pysam


# ==============================================================================
# Default constants (overridden by config.yaml)
# ==============================================================================
DEFAULT_SEQ_LEN = 1000
DEFAULT_EPI_WINDOW = 10000
DEFAULT_EPI_BIN_SIZE = 500
DEFAULT_FEATURES = ['CTCF', 'DNase-seq', 'H3K4me1', 'H3K4me3', 'POLR2A']

ONEHOT_MAP = {
    'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0],
    'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1],
    'N': [0, 0, 0, 0],
}


# ==============================================================================
# Data loading functions
# ==============================================================================
def load_manifest(manifest_path):
    """Load MM285 probe manifest."""
    print(f"Loading Manifest: {manifest_path} ...")
    usecols = ['probeID', 'CpG_chrm', 'CpG_beg']
    try:
        df = pd.read_csv(manifest_path, sep='\t', compression='gzip', usecols=usecols)
    except Exception:
        df = pd.read_csv(manifest_path, sep='\t', compression='gzip')
    df = df.dropna(subset=['CpG_chrm', 'CpG_beg', 'probeID']).reset_index(drop=True)
    df['CpG_beg'] = df['CpG_beg'].astype(int)
    return df


def prepare_sample_map(tsv_path, features=None, exclude_tissue=None, target_tissue=None):
    """
    Parse ENCODE metadata TSV and return sample_map and sample_ids.

    Args:
        tsv_path:        path of ENCODE metadata TSV ()
        features:        list of chromatin marks (default: 5 marks)
        exclude_tissue:  tissue name to exclude (for LOTO training)
        target_tissue:   tissue name to include only (for calibration / evaluation)

    Returns:
        sample_map: {ZD_ID: {feature: bigwig_filename, ...}}
        sample_ids: list of valid ZD_IDs
    """
    features = features or DEFAULT_FEATURES
    if not os.path.exists(tsv_path):
        raise FileNotFoundError(f"{tsv_path} not found")

    df = pd.read_csv(tsv_path, sep='\t')

    # Tissue filtering
    if target_tissue:
        df = df[df['Tissue_Name'].str.contains(target_tissue, case=False, na=False)]
        if len(df) == 0:
            raise ValueError(f"No samples found for tissue: {target_tissue}")
        print(f"  Filtered to tissue '{target_tissue}': {len(df['ZD_ID'].unique())} samples")
    elif exclude_tissue:
        before = len(df['ZD_ID'].unique())
        df = df[~df['Tissue_Name'].str.contains(exclude_tissue, case=False, na=False)]
        after = len(df['ZD_ID'].unique())
        print(f"  [LOTO] Excluded '{exclude_tissue}': {before} -> {after} samples")

    # Quality filtering
    df = df[(df['ZD_Weeks'] == 8) & (df['Age_Diff'] <= 7.0)].copy()

    pivot = (
        df.pivot_table(
            index=['ZD_ID', 'Tissue_Name'], columns='Final_Type',
            values='BigWig_Filename', aggfunc='first',
        ).reset_index()
    )
    golden = pivot.dropna(subset=features)

    sample_map = {}
    for _, row in golden.iterrows():
        sample_map[row['ZD_ID']] = {f: row[f] for f in features}

    return sample_map, golden['ZD_ID'].tolist()


def load_split_chroms(split_file, idx):
    """Load chromosome list for a given split index."""
    if not os.path.exists(split_file):
        raise FileNotFoundError(f"Split file not found: {split_file}")
    with open(split_file, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    if idx >= len(lines):
        raise IndexError(f"Split index {idx} out of range (max {len(lines) - 1})")
    chroms = lines[idx].split(',')
    print(f">>> Loaded Split [{idx}]: {chroms}")
    return chroms


def build_train_val_indices(manifest_df, val_chroms, n_samples):
    """Build global train/val indices based on chromosome partition."""
    is_val = manifest_df['CpG_chrm'].isin(val_chroms)
    probe_val = np.where(is_val)[0]
    probe_train = np.where(~is_val)[0]
    n_probes = len(manifest_df)

    train_indices, val_indices = [], []
    for k in range(n_samples):
        offset = k * n_probes
        train_indices.extend((probe_train + offset).tolist())
        val_indices.extend((probe_val + offset).tolist())
    return train_indices, val_indices


# ==============================================================================
# Dataset
# ==============================================================================
class EpiSeq5hmCDataset(Dataset):
    """
    PyTorch Dataset for EpiSeq-5hmC.

    Returns:
        seq:   (4, seq_len)   one-hot encoded DNA
        epi:   (n_features, n_bins) binned chromatin signals
        label: (1,)           1.0=5hmC, 0.0=5mC, -1.0=ambiguous (masked)
    """

    def __init__(self, sample_ids, sample_map, manifest_df, target_df,
                 fasta_path, bigwig_dir,
                 th_hmc=0.20, th_mc=0.50, th_hmc_neg=0.05,
                 seq_len=DEFAULT_SEQ_LEN, epi_window=DEFAULT_EPI_WINDOW,
                 epi_bin_size=DEFAULT_EPI_BIN_SIZE, features=None):
        self.sample_ids = sample_ids
        self.sample_map = sample_map
        self.manifest = manifest_df
        self.target_df = target_df
        self.fasta_path = fasta_path
        self.bigwig_dir = bigwig_dir
        self.th_hmc = th_hmc
        self.th_mc = th_mc
        self.th_hmc_neg = th_hmc_neg
        self.seq_len = seq_len
        self.epi_window = epi_window
        self.epi_bin_size = epi_bin_size
        self.features = features or DEFAULT_FEATURES
        self.n_bins = (epi_window * 2) // epi_bin_size
        self.fasta = None  # lazy-opened per worker

    def _open_fasta(self):
        if self.fasta is None:
            self.fasta = pysam.FastaFile(self.fasta_path)

    def _get_seq_onehot(self, chrom, center):
        self._open_fasta()
        start = center - self.seq_len // 2
        end = center + self.seq_len // 2
        try:
            seq = self.fasta.fetch(chrom, start, end).upper()
        except Exception:
            seq = 'N' * self.seq_len
        return np.array([ONEHOT_MAP.get(b, [0, 0, 0, 0]) for b in seq], dtype=np.float32).T

    def _get_epi_signal(self, chrom, center, bw_files_dict):
        start = center - self.epi_window
        end = center + self.epi_window
        signals = []
        for feat in self.features:
            bw_path = os.path.join(self.bigwig_dir, bw_files_dict[feat])
            vals = np.zeros(self.n_bins, dtype=np.float32)
            if os.path.exists(bw_path):
                try:
                    bw = pyBigWig.open(bw_path)
                    raw = bw.stats(chrom, start, end, nBins=self.n_bins)
                    vals = np.array([v if v is not None else 0.0 for v in raw], dtype=np.float32)
                    bw.close()
                except Exception:
                    pass
            signals.append(vals)
        return np.stack(signals, axis=0)

    def _get_label(self, probe_id, zd_id):
        try:
            v_hmc = self.target_df.at[probe_id, f"{zd_id}_5hmC"]
            v_mc = self.target_df.at[probe_id, f"{zd_id}_5mC"]
            if v_hmc >= self.th_hmc:
                return 1.0
            elif v_mc >= self.th_mc and v_hmc < self.th_hmc_neg:
                return 0.0
        except Exception:
            pass
        return -1.0

    def __len__(self):
        return len(self.sample_ids) * len(self.manifest)

    def __getitem__(self, idx):
        n_probes = len(self.manifest)
        sample_idx = idx // n_probes
        probe_idx = idx % n_probes
        zd_id = self.sample_ids[sample_idx]
        row = self.manifest.iloc[probe_idx]
        chrom, center = str(row['CpG_chrm']), int(row['CpG_beg'])

        seq = torch.tensor(self._get_seq_onehot(chrom, center), dtype=torch.float32)
        epi = torch.tensor(
            self._get_epi_signal(chrom, center, self.sample_map[zd_id]),
            dtype=torch.float32,
        )
        label = torch.tensor([self._get_label(row['probeID'], zd_id)], dtype=torch.float32)
        return seq, epi, label
