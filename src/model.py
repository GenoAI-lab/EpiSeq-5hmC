import torch
import torch.nn as nn


class EpiSeq5hmCModel(nn.Module):
    """
    Dual-stream CNN for binary classification of 5hmC vs 5mC.

    Supports three modes for ablation study:
        'full' : Sequence + Chromatin (default)
        'seq'  : Sequence only
        'epi'  : Chromatin only

    Architecture:
        Sequence branch:  4 x seq_len  → Conv1D blocks → 256-dim
        Chromatin branch: n_features x n_bins → Conv1D blocks → 128-dim
        Fusion:           concatenation → Dense(128) → Dropout → Sigmoid
    """

    def __init__(self, mode='full', seq_len=1000, n_features=5, n_bins=40, dropout=0.5):
        super().__init__()
        assert mode in ('full', 'seq', 'epi'), f"Invalid mode: {mode}"
        self.mode = mode

        # --- Sequence branch ---
        if mode in ('full', 'seq'):
            self.seq_conv = nn.Sequential(
                nn.Conv1d(4, 64, kernel_size=12), nn.ReLU(), nn.MaxPool1d(4),
                nn.Conv1d(64, 128, kernel_size=6), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(128, 256, kernel_size=4), nn.ReLU(), nn.MaxPool1d(2),
                nn.Flatten(),
            )
            with torch.no_grad():
                seq_flat = self.seq_conv(torch.zeros(1, 4, seq_len)).shape[1]
            self.seq_fc = nn.Sequential(nn.Linear(seq_flat, 256), nn.ReLU())

        # --- Chromatin branch ---
        if mode in ('full', 'epi'):
            self.epi_conv = nn.Sequential(
                nn.Conv1d(n_features, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool1d(2),
                nn.Flatten(),
            )
            with torch.no_grad():
                epi_flat = self.epi_conv(torch.zeros(1, n_features, n_bins)).shape[1]
            self.epi_fc = nn.Sequential(nn.Linear(epi_flat, 128), nn.ReLU())

        # --- Fusion dimension ---
        if mode == 'full':
            fusion_dim = 256 + 128
        elif mode == 'seq':
            fusion_dim = 256
        else:
            fusion_dim = 128

        # --- Classifier ---
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 128), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1), nn.Sigmoid(),
        )

    def forward(self, seq, epi):
        """
        Args:
            seq: (B, 4, seq_len)       one-hot DNA sequence
            epi: (B, n_features, n_bins) binned chromatin signals
        Returns:
            (B, 1) 5hmC probability
        """
        parts = []
        if self.mode in ('full', 'seq'):
            parts.append(self.seq_fc(self.seq_conv(seq)))
        if self.mode in ('full', 'epi'):
            parts.append(self.epi_fc(self.epi_conv(epi)))
        return self.classifier(torch.cat(parts, dim=1) if len(parts) > 1 else parts[0])
