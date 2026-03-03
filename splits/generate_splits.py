"""
EpiSeq-5hmC: Generate chromosome-based cross-validation splits.

Produces random chromosome combinations where each split contains
15-20% of total CpG probes, ensuring consistent validation set sizes.

Usage:
    python generate_splits.py
    python generate_splits.py --config ../config.yaml
    python generate_splits.py --num_splits 100 --seed 0 --output chrom_splits.txt
"""

import os
import argparse
import random

import numpy as np
import pandas as pd

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


# ==============================================================================
# Defaults
# ==============================================================================
DEFAULT_CONFIG = {
    'paths': {
        'manifest': 'data/MM285.mm10.manifest.gencode.vM25.tsv.gz',
    },
}

DEFAULT_NUM_SPLITS = 100
DEFAULT_TARGET_MIN = 0.15
DEFAULT_TARGET_MAX = 0.20
DEFAULT_SEED = 0
DEFAULT_OUTPUT = 'chrom_splits.txt'
MAX_ATTEMPTS = 10000


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
    p = argparse.ArgumentParser(description="EpiSeq-5hmC Chromosome Split Generation")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--manifest", type=str, default=None,
                   help="Override manifest path from config")
    p.add_argument("--num_splits", type=int, default=DEFAULT_NUM_SPLITS,
                   help=f"Number of unique splits to generate (default: {DEFAULT_NUM_SPLITS})")
    p.add_argument("--target_min", type=float, default=DEFAULT_TARGET_MIN,
                   help=f"Min validation ratio (default: {DEFAULT_TARGET_MIN})")
    p.add_argument("--target_max", type=float, default=DEFAULT_TARGET_MAX,
                   help=f"Max validation ratio (default: {DEFAULT_TARGET_MAX})")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    return p.parse_args()


# ==============================================================================
# Split generation
# ==============================================================================
def load_chrom_counts(manifest_path):
    """Load manifest and return per-chromosome probe counts (excluding random/Un/fix/chrM)."""
    print(f"Loading manifest: {manifest_path} ...")
    df = pd.read_csv(manifest_path, sep='\t', compression='gzip', usecols=['CpG_chrm'])
    mask = df['CpG_chrm'].astype(str).str.contains('Un|random|fix|chrM')
    df = df[~mask]
    counts = df['CpG_chrm'].value_counts().to_dict()
    print(f"  Chromosomes: {len(counts)} | Total probes: {sum(counts.values()):,}")
    return counts


def generate_splits(chrom_counts, num_splits, target_min, target_max, seed):
    """
    Generate unique chromosome combinations within target probe ratio.

    Args:
        chrom_counts: {chrom_name: probe_count}
        num_splits:   number of unique splits to find
        target_min:   minimum fraction of total probes
        target_max:   maximum fraction of total probes
        seed:         random seed

    Returns:
        list of comma-separated chromosome strings
    """
    random.seed(seed)
    np.random.seed(seed)

    total = sum(chrom_counts.values())
    min_count = total * target_min
    max_count = total * target_max
    chroms = list(chrom_counts.keys())

    print(f"  Target range: {min_count:,.0f} – {max_count:,.0f} "
          f"({target_min*100:.0f}% – {target_max*100:.0f}%)")

    found = set()
    attempts = 0

    while len(found) < num_splits and attempts < MAX_ATTEMPTS:
        attempts += 1
        random.shuffle(chroms)

        selection = []
        current_sum = 0

        for c in chroms:
            cnt = chrom_counts[c]
            if current_sum + cnt > max_count:
                continue
            selection.append(c)
            current_sum += cnt

            if min_count <= current_sum <= max_count:
                combo = ",".join(sorted(selection))
                if combo not in found:
                    found.add(combo)
                    ratio = current_sum / total * 100
                    print(f"  [{len(found):>3}/{num_splits}] {combo} ({ratio:.2f}%)")
                break

    if len(found) < num_splits:
        print(f"  WARNING: Only found {len(found)}/{num_splits} splits "
              f"after {MAX_ATTEMPTS} attempts")

    return sorted(found)


# ==============================================================================
# Main
# ==============================================================================
def main():
    args = get_args()
    cfg = load_config(args.config)

    manifest_path = args.manifest or cfg['paths']['manifest']

    print("=" * 70)
    print(f"Chromosome Split Generation | Splits: {args.num_splits} | Seed: {args.seed}")
    print("=" * 70)

    chrom_counts = load_chrom_counts(manifest_path)

    splits = generate_splits(
        chrom_counts,
        num_splits=args.num_splits,
        target_min=args.target_min,
        target_max=args.target_max,
        seed=args.seed,
    )

    with open(args.output, 'w') as f:
        for s in splits:
            f.write(s + '\n')

    print(f"\n>>> Saved {len(splits)} splits to '{args.output}'")


if __name__ == "__main__":
    main()
