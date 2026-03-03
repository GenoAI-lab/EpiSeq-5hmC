"""
EpiSeq-5hmC: Training script with chromosome-based cross-validation and early stopping.

Usage:
    python train.py --split_idx 0
    python train.py --split_idx 0 --config config.yaml --gpu_id 0

Requires:
    model.py   - EpiSeq5hmCModel
    dataset.py - EpiSeq5hmCDataset, load_manifest, prepare_sample_map,
                 load_split_chroms, build_train_val_indices
"""

import os
import argparse
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

from model import EpiSeq5hmCModel
from dataset import (
    EpiSeq5hmCDataset,
    load_manifest,
    prepare_sample_map,
    load_split_chroms,
    build_train_val_indices,
)


# ==============================================================================
# Reproducibility
# ==============================================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ==============================================================================
# Config loading
# ==============================================================================
DEFAULT_CONFIG = {
    'paths': {
        'fasta':      'data/mm10.fa',
        'bigwig_dir': 'data/ENCODE/',
        'manifest':   'data/MM285.mm10.manifest.gencode.vM25.tsv.gz',
        'sample_tsv': 'data/output_unique_matched.tsv',
        'target_csv': 'data/ternary_targets_ZD.csv',
        'split_file': 'splits/chrom_splits.txt',
        'output_dir': 'results/',
    },
    'model': {
        'seq_len':      1000,
        'epi_window':   10000,
        'epi_bin_size': 500,
        'features':     ['CTCF', 'DNase-seq', 'H3K4me1', 'H3K4me3', 'POLR2A'],
    },
    'labeling': {
        'th_5hmc':     0.20,
        'th_5mc':      0.50,
        'th_5hmc_neg': 0.05,
    },
    'training': {
        'batch_size':     1024,
        'learning_rate':  0.001,
        'max_epochs':     20,
        'patience':       5,
        'dropout':        0.5,
        'seed':           0,
    },
    'hardware': {
        'gpu_id':      0,
        'num_workers': 8,
    },
}


def load_config(config_path=None):
    """Load config from YAML file, falling back to defaults."""
    cfg = DEFAULT_CONFIG.copy()
    if config_path and os.path.exists(config_path):
        if not HAS_YAML:
            print("WARNING: PyYAML not installed. Using default config.")
            return cfg
        with open(config_path, 'r') as f:
            user_cfg = yaml.safe_load(f)
        # Deep merge: user overrides defaults
        for section in cfg:
            if section in user_cfg and isinstance(cfg[section], dict):
                cfg[section].update(user_cfg[section])
    return cfg


# ==============================================================================
# Argument parser
# ==============================================================================
def get_args():
    parser = argparse.ArgumentParser(description="EpiSeq-5hmC Training")
    parser.add_argument("--split_idx", type=int, required=True,
                        help="Chromosome split index (0-based)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config.yaml (optional)")
    parser.add_argument("--gpu_id", type=int, default=None,
                        help="Override GPU ID from config")
    parser.add_argument("--save_name", type=str, default=None,
                        help="Custom model save path")
    parser.add_argument("--log_step", type=int, default=100,
                        help="Log training metrics every N steps")
    return parser.parse_args()


# ==============================================================================
# Training loop
# ==============================================================================
def train_one_epoch(model, loader, optimizer, criterion, device, epoch, log_step):
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss, total_cnt = 0.0, 0
    running_loss, running_correct, running_total = 0.0, 0, 0
    global_step = getattr(train_one_epoch, '_step', 0)

    for seq, epi, label in loader:
        seq, epi, label = seq.to(device), epi.to(device), label.to(device)
        mask = (label != -1.0).float()
        if mask.sum() == 0:
            continue

        optimizer.zero_grad()
        output = torch.clamp(model(seq, epi), 1e-7, 1 - 1e-7)
        safe_label = torch.where(label == -1.0, torch.zeros_like(label), label)
        loss = (criterion(output, safe_label) * mask).sum() / mask.sum()
        loss.backward()
        optimizer.step()

        n = mask.sum().item()
        total_loss += loss.item() * n
        total_cnt += n

        pred = (output > 0.5).float()
        running_loss += loss.item() * n
        running_correct += ((pred == safe_label) * mask).sum().item()
        running_total += n
        global_step += 1

        if global_step % log_step == 0 and running_total > 0:
            avg_loss = running_loss / running_total
            avg_acc = running_correct / running_total * 100
            print(f"  [Ep {epoch} | Step {global_step}] "
                  f"Loss: {avg_loss:.4f} | Acc: {avg_acc:.2f}%")
            running_loss, running_correct, running_total = 0.0, 0, 0

    train_one_epoch._step = global_step
    return total_loss / (total_cnt + 1e-8)


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Validate and return (avg_loss, accuracy)."""
    model.eval()
    total_loss, total_cnt, correct = 0.0, 0, 0

    for seq, epi, label in loader:
        seq, epi, label = seq.to(device), epi.to(device), label.to(device)
        mask = (label != -1.0).float()
        if mask.sum() == 0:
            continue

        output = torch.clamp(model(seq, epi), 1e-7, 1 - 1e-7)
        safe_label = torch.where(label == -1.0, torch.zeros_like(label), label)
        total_loss += (criterion(output, safe_label) * mask).sum().item()
        total_cnt += mask.sum().item()
        correct += ((output > 0.5).float() == safe_label).mul(mask).sum().item()

    avg_loss = total_loss / (total_cnt + 1e-8)
    acc = correct / (total_cnt + 1e-8) * 100
    return avg_loss, acc


# ==============================================================================
# Main
# ==============================================================================
def main():
    args = get_args()
    cfg = load_config(args.config)

    paths = cfg['paths']
    mcfg = cfg['model']
    lcfg = cfg['labeling']
    tcfg = cfg['training']
    hcfg = cfg['hardware']

    seed = tcfg['seed']
    set_seed(seed)

    gpu_id = args.gpu_id if args.gpu_id is not None else hcfg['gpu_id']
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    # Output path
    os.makedirs(paths['output_dir'], exist_ok=True)
    if args.save_name:
        save_path = args.save_name
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(paths['output_dir'], f"model_cv{args.split_idx}_{ts}.pth")

    print("=" * 70)
    print(f"EpiSeq-5hmC Training | Split: {args.split_idx} | Device: {device}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    val_chroms = load_split_chroms(paths['split_file'], args.split_idx)
    sample_map, sample_ids = prepare_sample_map(
        paths['sample_tsv'], features=mcfg['features']
    )
    manifest_df = load_manifest(paths['manifest'])
    target_df = pd.read_csv(paths['target_csv'], index_col=0)

    print(f"  Samples : {len(sample_ids)}")
    print(f"  Probes  : {len(manifest_df):,}")
    print(f"  Val chroms: {val_chroms}")

    # ------------------------------------------------------------------
    # Build train / val splits
    # ------------------------------------------------------------------
    train_idx, val_idx = build_train_val_indices(
        manifest_df, val_chroms, len(sample_ids)
    )

    full_dataset = EpiSeq5hmCDataset(
        sample_ids=sample_ids,
        sample_map=sample_map,
        manifest_df=manifest_df,
        target_df=target_df,
        fasta_path=paths['fasta'],
        bigwig_dir=paths['bigwig_dir'],
        th_hmc=lcfg['th_5hmc'],
        th_mc=lcfg['th_5mc'],
        th_hmc_neg=lcfg['th_5hmc_neg'],
        seq_len=mcfg['seq_len'],
        epi_window=mcfg['epi_window'],
        epi_bin_size=mcfg['epi_bin_size'],
        features=mcfg['features'],
    )

    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)
    print(f"  Train set: {len(train_dataset):,}")
    print(f"  Val set  : {len(val_dataset):,}")

    g = torch.Generator()
    g.manual_seed(seed)
    loader_kwargs = dict(
        batch_size=tcfg['batch_size'],
        num_workers=hcfg['num_workers'],
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
    )
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    # ------------------------------------------------------------------
    # Model, optimizer, loss
    # ------------------------------------------------------------------
    n_bins = (mcfg['epi_window'] * 2) // mcfg['epi_bin_size']
    model = EpiSeq5hmCModel(
        seq_len=mcfg['seq_len'],
        n_features=len(mcfg['features']),
        n_bins=n_bins,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=tcfg['learning_rate'])
    criterion = nn.BCELoss(reduction='none')

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {total_params:,}")

    # ------------------------------------------------------------------
    # Training with early stopping
    # ------------------------------------------------------------------
    history = {'epoch': [], 'train_loss': [], 'val_loss': [], 'val_acc': []}
    best_val_loss = float('inf')
    patience_counter = 0
    train_one_epoch._step = 0  # reset global step counter

    for epoch in range(1, tcfg['max_epochs'] + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion,
            device, epoch, args.log_step,
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(f"Epoch [{epoch}/{tcfg['max_epochs']}] "
              f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | Acc: {val_acc:.2f}%")

        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"  >>> Best model saved: {save_path}")
        else:
            patience_counter += 1
            print(f"  [Early Stopping] {patience_counter}/{tcfg['patience']}")
            if patience_counter >= tcfg['patience']:
                print(f">>> Early stopping triggered at epoch {epoch}")
                break

    # ------------------------------------------------------------------
    # Save training history
    # ------------------------------------------------------------------
    hist_path = save_path.replace('.pth', '_history.csv')
    pd.DataFrame(history).to_csv(hist_path, index=False)
    print(f">>> History saved: {hist_path}")
    print(f">>> Finished split {args.split_idx}")


if __name__ == "__main__":
    main()
