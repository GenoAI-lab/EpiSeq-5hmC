"""
EpiSeq-5hmC: Tissue-specific threshold calibration via Youden's index.

Loads a LOTO-trained model, runs inference on the held-out tissue,
finds the optimal decision threshold, and reports accuracy gain.

Usage:
    python calibrate.py --tissue Liver  --model_path results/model_loto_no_Liver_cv5.pth
    python calibrate.py --tissue Spleen --model_path results/model_loto_no_Spleen_cv5.pth
    python calibrate.py --tissue Liver  --model_path results/model_loto_no_Liver_cv5.pth --config config.yaml
"""

import os
import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

from model import EpiSeq5hmCModel
from dataset import EpiSeq5hmCDataset, load_manifest, prepare_sample_map


# ==============================================================================
# Config
# ==============================================================================
DEFAULT_CONFIG = {
    'paths': {
        'fasta': 'data/mm10.fa', 'bigwig_dir': 'data/ENCODE/',
        'manifest': 'data/MM285.mm10.manifest.gencode.vM25.tsv.gz',
        'sample_tsv': 'data/output_unique_matched.tsv',
        'target_csv': 'data/ternary_targets_ZD.csv',
        'output_dir': 'results/',
    },
    'model': {
        'seq_len': 1000, 'epi_window': 10000, 'epi_bin_size': 500,
        'features': ['CTCF', 'DNase-seq', 'H3K4me1', 'H3K4me3', 'POLR2A'],
    },
    'labeling': {'th_5hmc': 0.20, 'th_5mc': 0.50, 'th_5hmc_neg': 0.05},
    'hardware': {'gpu_id': 0, 'num_workers': 8, 'batch_size': 1024},
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
    p = argparse.ArgumentParser(description="EpiSeq-5hmC Threshold Calibration")
    p.add_argument("--tissue", type=str, required=True,
                   help="Target tissue for calibration (e.g., 'Liver')")
    p.add_argument("--model_path", type=str, required=True,
                   help="Path to LOTO-trained model (.pth)")
    p.add_argument("--mode", type=str, default='full', choices=['seq', 'epi', 'full'])
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--gpu_id", type=int, default=None)
    p.add_argument("--no_plot", action='store_true', help="Skip saving plot")
    return p.parse_args()


# ==============================================================================
# Inference
# ==============================================================================
@torch.no_grad()
def run_inference(model, loader, device):
    """Run inference and return (y_true, y_scores) for valid labels only."""
    model.eval()
    all_probs, all_labels = [], []

    for i, (seq, epi, label) in enumerate(loader):
        seq, epi = seq.to(device), epi.to(device)
        out = model(seq, epi).cpu().numpy().flatten()
        lbl = label.numpy().flatten()

        mask = lbl != -1
        if mask.sum() > 0:
            all_probs.extend(out[mask])
            all_labels.extend(lbl[mask])

        if (i + 1) % 200 == 0:
            print(f"    Inference step [{i+1}/{len(loader)}]")

    return np.array(all_labels), np.array(all_probs)


# ==============================================================================
# Calibration
# ==============================================================================
def find_optimal_threshold(y_true, y_scores):
    """Find threshold maximizing Youden's J = sensitivity + specificity - 1."""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    j_scores = tpr - fpr
    ix = np.argmax(j_scores)
    return thresholds[ix]


def compute_metrics(y_true, y_scores, threshold):
    """Compute accuracy, sensitivity, specificity at a given threshold."""
    y_pred = (y_scores >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    sens = tp / (tp + fn + 1e-8)
    spec = tn / (tn + fp + 1e-8)
    return acc, sens, spec


def plot_distributions(y_true, y_scores, default_th, optimal_th, tissue, save_path):
    """Plot score distributions with threshold lines."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(y_scores[y_true == 0], bins=50, alpha=0.5, label='5mC-dominant', color='blue', density=True)
    ax.hist(y_scores[y_true == 1], bins=50, alpha=0.5, label='5hmC-enriched', color='red', density=True)
    ax.axvline(default_th, color='gray', ls='--', label=f'Default ({default_th})')
    ax.axvline(optimal_th, color='green', ls='-', lw=2, label=f'Optimal ({optimal_th:.3f})')

    acc_def = accuracy_score(y_true, (y_scores >= default_th).astype(int))
    acc_opt = accuracy_score(y_true, (y_scores >= optimal_th).astype(int))
    ax.set_title(f"{tissue}: Score Distribution\n"
                 f"Acc {acc_def*100:.1f}% → {acc_opt*100:.1f}% (Δ+{(acc_opt-acc_def)*100:.1f}%)")
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Plot saved: {save_path}")


# ==============================================================================
# Main
# ==============================================================================
def main():
    args = get_args()
    cfg = load_config(args.config)
    P, M, L, H = cfg['paths'], cfg['model'], cfg['labeling'], cfg['hardware']

    gpu = args.gpu_id if args.gpu_id is not None else H['gpu_id']
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print(f"Threshold Calibration | Tissue: {args.tissue} | Model: {args.model_path}")
    print("=" * 70)

    # Load data — target tissue only
    sample_map, sample_ids = prepare_sample_map(
        P['sample_tsv'], features=M['features'], target_tissue=args.tissue,
    )
    manifest_df = load_manifest(P['manifest'])
    target_df = pd.read_csv(P['target_csv'], index_col=0)

    ds = EpiSeq5hmCDataset(
        sample_ids, sample_map, manifest_df, target_df,
        P['fasta'], P['bigwig_dir'],
        th_hmc=L['th_5hmc'], th_mc=L['th_5mc'], th_hmc_neg=L['th_5hmc_neg'],
        seq_len=M['seq_len'], epi_window=M['epi_window'],
        epi_bin_size=M['epi_bin_size'], features=M['features'],
    )
    loader = DataLoader(ds, batch_size=H['batch_size'],
                        num_workers=H['num_workers'], shuffle=False)
    print(f"  Total items: {len(ds):,}")

    # Load model
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found: {args.model_path}")

    n_bins = (M['epi_window'] * 2) // M['epi_bin_size']
    model = EpiSeq5hmCModel(
        mode=args.mode, seq_len=M['seq_len'],
        n_features=len(M['features']), n_bins=n_bins,
    ).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    # Inference
    print(">>> Running inference...")
    y_true, y_scores = run_inference(model, loader, device)
    print(f"  Valid data points: {len(y_true):,} "
          f"(5hmC: {(y_true==1).sum():,} | 5mC: {(y_true==0).sum():,})")

    if len(y_true) == 0:
        print("ERROR: No valid labels found.")
        return

    # Calibration
    auroc = roc_auc_score(y_true, y_scores)
    optimal_th = find_optimal_threshold(y_true, y_scores)

    acc_def, sens_def, spec_def = compute_metrics(y_true, y_scores, 0.5)
    acc_opt, sens_opt, spec_opt = compute_metrics(y_true, y_scores, optimal_th)

    print()
    print("=" * 50)
    print(f"  Tissue          : {args.tissue}")
    print(f"  AUROC           : {auroc:.4f}")
    print(f"  Optimal Threshold: {optimal_th:.4f}")
    print("-" * 50)
    print(f"  Default  (0.50) : Acc {acc_def*100:.2f}% | "
          f"Sens {sens_def*100:.1f}% | Spec {spec_def*100:.1f}%")
    print(f"  Calibrated ({optimal_th:.2f}): Acc {acc_opt*100:.2f}% | "
          f"Sens {sens_opt*100:.1f}% | Spec {spec_opt*100:.1f}%")
    print(f"  Accuracy Gain   : +{(acc_opt-acc_def)*100:.2f}%")
    print("=" * 50)

    # Save results
    os.makedirs(P['output_dir'], exist_ok=True)
    result = {
        'tissue': args.tissue, 'auroc': auroc,
        'threshold_default': 0.5, 'threshold_optimal': optimal_th,
        'acc_default': acc_def, 'acc_calibrated': acc_opt,
        'sens_default': sens_def, 'sens_calibrated': sens_opt,
        'spec_default': spec_def, 'spec_calibrated': spec_opt,
    }
    csv_path = os.path.join(P['output_dir'], f"calibration_{args.tissue}.csv")
    pd.DataFrame([result]).to_csv(csv_path, index=False)
    print(f"  Results saved: {csv_path}")

    # Plot
    if not args.no_plot:
        plot_path = os.path.join(P['output_dir'], f"calibration_{args.tissue}.png")
        plot_distributions(y_true, y_scores, 0.5, optimal_th, args.tissue, plot_path)

    print(f">>> Calibration complete for {args.tissue}")


if __name__ == "__main__":
    main()
