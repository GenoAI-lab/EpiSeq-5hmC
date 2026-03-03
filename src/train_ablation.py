"""
EpiSeq-5hmC: Ablation study — train sequence-only, chromatin-only, or full model.

Usage:
    python train_ablation.py --mode seq  --split_idx 0
    python train_ablation.py --mode epi  --split_idx 0
    python train_ablation.py --mode full --split_idx 0
    python train_ablation.py --mode seq  --split_idx 0 --config config.yaml
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
    EpiSeq5hmCDataset, load_manifest, prepare_sample_map,
    load_split_chroms, build_train_val_indices,
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
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


# ==============================================================================
# Config
# ==============================================================================
DEFAULT_CONFIG = {
    'paths': {
        'fasta': 'data/mm10.fa', 'bigwig_dir': 'data/ENCODE/',
        'manifest': 'data/MM285.mm10.manifest.gencode.vM25.tsv.gz',
        'sample_tsv': 'data/output_unique_matched.tsv',
        'target_csv': 'data/ternary_targets_ZD.csv',
        'split_file': 'splits/chrom_splits.txt', 'output_dir': 'results/',
    },
    'model': {
        'seq_len': 1000, 'epi_window': 10000, 'epi_bin_size': 500,
        'features': ['CTCF', 'DNase-seq', 'H3K4me1', 'H3K4me3', 'POLR2A'],
    },
    'labeling': {'th_5hmc': 0.20, 'th_5mc': 0.50, 'th_5hmc_neg': 0.05},
    'training': {
        'batch_size': 1024, 'learning_rate': 0.001,
        'max_epochs': 20, 'patience': 3, 'dropout': 0.5, 'seed': 0,
    },
    'hardware': {'gpu_id': 0, 'num_workers': 4},
}

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
    p = argparse.ArgumentParser(description="EpiSeq-5hmC Ablation Training")
    p.add_argument("--mode", type=str, required=True, choices=['seq', 'epi', 'full'])
    p.add_argument("--split_idx", type=int, required=True)
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--gpu_id", type=int, default=None)
    p.add_argument("--save_name", type=str, default=None)
    p.add_argument("--log_step", type=int, default=100)
    return p.parse_args()


# ==============================================================================
# Train / Validate
# ==============================================================================
def train_one_epoch(model, loader, optimizer, criterion, device, epoch, log_step):
    model.train()
    total_loss, total_n = 0.0, 0
    run_loss, run_correct, run_n = 0.0, 0, 0
    step = getattr(train_one_epoch, '_step', 0)

    for seq, epi, label in loader:
        seq, epi, label = seq.to(device), epi.to(device), label.to(device)
        mask = (label != -1.0).float()
        if mask.sum() == 0:
            continue
        optimizer.zero_grad()
        out = torch.clamp(model(seq, epi), 1e-7, 1 - 1e-7)
        safe = torch.where(label == -1.0, torch.zeros_like(label), label)
        loss = (criterion(out, safe) * mask).sum() / mask.sum()
        loss.backward()
        optimizer.step()

        n = mask.sum().item()
        total_loss += loss.item() * n
        total_n += n
        run_loss += loss.item() * n
        run_correct += ((out > 0.5).float() == safe).mul(mask).sum().item()
        run_n += n
        step += 1

        if step % log_step == 0 and run_n > 0:
            print(f"  [Ep {epoch} | Step {step}] "
                  f"Loss: {run_loss/run_n:.4f} | Acc: {run_correct/run_n*100:.2f}%")
            run_loss, run_correct, run_n = 0.0, 0, 0

    train_one_epoch._step = step
    return total_loss / (total_n + 1e-8)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss, total_n, correct = 0.0, 0, 0
    for seq, epi, label in loader:
        seq, epi, label = seq.to(device), epi.to(device), label.to(device)
        mask = (label != -1.0).float()
        if mask.sum() == 0:
            continue
        out = torch.clamp(model(seq, epi), 1e-7, 1 - 1e-7)
        safe = torch.where(label == -1.0, torch.zeros_like(label), label)
        total_loss += (criterion(out, safe) * mask).sum().item()
        total_n += mask.sum().item()
        correct += ((out > 0.5).float() == safe).mul(mask).sum().item()
    return total_loss / (total_n + 1e-8), correct / (total_n + 1e-8) * 100


# ==============================================================================
# Main
# ==============================================================================
def main():
    args = get_args()
    cfg = load_config(args.config)
    P, M, L, T, H = cfg['paths'], cfg['model'], cfg['labeling'], cfg['training'], cfg['hardware']

    set_seed(T['seed'])
    gpu = args.gpu_id if args.gpu_id is not None else H['gpu_id']
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

    os.makedirs(P['output_dir'], exist_ok=True)
    save_path = args.save_name or os.path.join(
        P['output_dir'], f"model_{args.mode}_cv{args.split_idx}.pth"
    )

    print("=" * 70)
    print(f"Ablation Training | Mode: {args.mode.upper()} | Split: {args.split_idx}")
    print("=" * 70)

    # Data
    val_chroms = load_split_chroms(P['split_file'], args.split_idx)
    sample_map, sample_ids = prepare_sample_map(P['sample_tsv'], features=M['features'])
    manifest_df = load_manifest(P['manifest'])
    target_df = pd.read_csv(P['target_csv'], index_col=0)

    train_idx, val_idx = build_train_val_indices(manifest_df, val_chroms, len(sample_ids))

    ds = EpiSeq5hmCDataset(
        sample_ids, sample_map, manifest_df, target_df,
        P['fasta'], P['bigwig_dir'],
        th_hmc=L['th_5hmc'], th_mc=L['th_5mc'], th_hmc_neg=L['th_5hmc_neg'],
        seq_len=M['seq_len'], epi_window=M['epi_window'],
        epi_bin_size=M['epi_bin_size'], features=M['features'],
    )
    g = torch.Generator(); g.manual_seed(T['seed'])
    kw = dict(batch_size=T['batch_size'], num_workers=H['num_workers'],
              pin_memory=True, worker_init_fn=seed_worker, generator=g)
    train_loader = DataLoader(Subset(ds, train_idx), shuffle=True, **kw)
    val_loader = DataLoader(Subset(ds, val_idx), shuffle=False, **kw)

    print(f"  Train: {len(train_idx):,} | Val: {len(val_idx):,}")

    # Model
    n_bins = (M['epi_window'] * 2) // M['epi_bin_size']
    model = EpiSeq5hmCModel(
        mode=args.mode, seq_len=M['seq_len'],
        n_features=len(M['features']), n_bins=n_bins, dropout=T['dropout'],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=T['learning_rate'])
    criterion = nn.BCELoss(reduction='none')

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Mode: {args.mode} | Params: {params:,}")

    # Training loop
    history = {'epoch': [], 'train_loss': [], 'val_loss': [], 'val_acc': []}
    best_val_loss = float('inf')
    patience_cnt = 0
    train_one_epoch._step = 0

    for epoch in range(1, T['max_epochs'] + 1):
        t_loss = train_one_epoch(model, train_loader, optimizer, criterion,
                                 device, epoch, args.log_step)
        v_loss, v_acc = validate(model, val_loader, criterion, device)

        print(f"[{args.mode.upper()}] Epoch [{epoch}/{T['max_epochs']}] "
              f"Train: {t_loss:.4f} | Val: {v_loss:.4f} | Acc: {v_acc:.2f}%")

        history['epoch'].append(epoch)
        history['train_loss'].append(t_loss)
        history['val_loss'].append(v_loss)
        history['val_acc'].append(v_acc)

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            patience_cnt = 0
            torch.save(model.state_dict(), save_path)
            print(f"  >>> Best model saved: {save_path}")
        else:
            patience_cnt += 1
            print(f"  [Early Stopping] {patience_cnt}/{T['patience']}")
            if patience_cnt >= T['patience']:
                print(f">>> Early stopping at epoch {epoch}")
                break

    pd.DataFrame(history).to_csv(save_path.replace('.pth', '_history.csv'), index=False)
    print(f">>> Finished ablation [{args.mode}] split {args.split_idx}")


if __name__ == "__main__":
    main()
