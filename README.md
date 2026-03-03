# EpiSeq-5hmC

**Discriminating 5-hydroxymethylcytosine from 5-methylcytosine at base resolution by integrating DNA sequence and chromatin features**

EpiSeq-5hmC is a dual-stream convolutional neural network that integrates DNA sequence context and tissue-matched chromatin features to discriminate 5-hydroxymethylcytosine (5hmC) from 5-methylcytosine (5mC) at base resolution. The model achieves AUROC 0.949 and accuracy 89.06% on high-confidence CpG sites across seven adult mouse tissues.

## Highlights

- **Multimodal architecture**: Dual-stream CNN combining 1 kb DNA sequence (one-hot encoded) with five chromatin marks (CTCF, DNase-seq, H3K4me1, H3K4me3, POLR2A) from a ±10 kb window
- **Ternary atlas ground truth**: High-confidence labels derived from base-resolution ternary methylome atlas (unmethylated C, 5mC, 5hmC)
- **Chromatin dominance**: Chromatin features alone achieve 85.94% accuracy; sequence alone 67.12%; combined 89.06%
- **Cross-tissue generalization**: Leave-one-tissue-out evaluation shows AUROC > 0.84 for four tissues, with tissue-specific calibration recovering accuracy in challenging tissues (e.g., liver: 45% → 85%)

## Repository Structure

```
EpiSeq-5hmC/
├── README.md                  # Project documentation
├── LICENSE                    # MIT License
├── environment.yml            # Conda environment specification
├── config.yaml                # Model training configuration
│
├── src/
│   ├── model.py               # EpiSeq5hmCModel (full / seq-only / epi-only)
│   ├── dataset.py             # Dataset class and data utilities
│   ├── train.py               # Cross-chromosomal training
│   ├── train_ablation.py      # Ablation study (seq / epi / full)
│   ├── train_loto.py          # Leave-one-tissue-out training
│   └── calibrate.py           # Tissue-specific threshold calibration
│
├── splits/
│   ├── chrom_splits.txt       # 100 chromosome partitions (pre-generated)
│   └── generate_splits.py     # Script to regenerate splits
│
├── models/
│   └── best_model_cv.pth      # Trained model weights
│
└── data/
    ├── README.md              # Data download and preparation guide
    ├── prepare_targets.py     # Generate ternary target labels
    ├── ternary_targets_sample.csv              # Sample output for verification
    └── output_unique_matched.tsv               # ENCODE sample metadata
```

## System Requirements

- **Python**: 3.10 or higher
- **GPU**: NVIDIA GPU with CUDA support (recommended; CPU training is possible but slow)
- **RAM**: 16 GB minimum (32 GB recommended for loading GEO matrices)
- **Disk**: 
  - ENCODE BigWig files: ~15 GB
  - GEO matrices: ~2 GB
  - mm10 reference genome: ~830 MB compressed (~2.6 GB uncompressed)

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/HyoeunBang/EpiSeq-5hmC.git
cd EpiSeq-5hmC
```

### 2. Create and activate conda environment
```bash
conda env create -f environment.yml
conda activate EpiSeq-5hmC
```

### 3. Install PyTorch
> **Note:** The `environment.yml` does not include a CUDA-specific PyTorch build,
> as the appropriate version depends on your hardware and driver setup.
> Follow the instructions below based on your environment.

#### GPU (recommended) — install matching your CUDA version
```bash
# CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121

# CUDA 12.4
pip install torch --index-url https://download.pytorch.org/whl/cu124

# CUDA 13.0
pip install torch --index-url https://download.pytorch.org/whl/cu130
```

Not sure which CUDA version you have? Run `nvidia-smi` to check.

#### CPU only

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

> For the full list of available PyTorch builds, visit:
> https://pytorch.org/get-started/locally/


### 4. Verify installation
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```


## Quick Start

### 1. Prepare data

Download external datasets following the instructions in [`data/README.md`](data/README.md), then generate target labels:

```bash
cd data
python prepare_targets.py
cd ..
```

### 2. Train the model

```bash
# Cross-chromosomal training (main result)
python src/train.py --split_idx 0 --config config.yaml

# Ablation study
python src/train_ablation.py --mode seq  --split_idx 0 --config config.yaml
python src/train_ablation.py --mode epi  --split_idx 0 --config config.yaml
python src/train_ablation.py --mode full --split_idx 0 --config config.yaml

# Leave-one-tissue-out
python src/train_loto.py --exclude_tissue Liver --split_idx 0 --config config.yaml
```

### 3. Calibrate thresholds

```bash
python src/calibrate.py --tissue Liver --model_path models/model_loto_no_Liver_cv0.pth --config config.yaml
```

## Data Sources

| Source | Description | Access |
|--------|-------------|--------|
| Ternary methylome atlas | Base-resolution 5mC/5hmC measurements (bACE-Infinium) | [GEO GSE290585](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE290585) |
| Sample metadata | ZD → BS/bACE array mapping, tissue, age | [Zenodo 17109847](https://zenodo.org/records/17109847) |
| Chromatin features | 35 BigWig signal tracks (5 marks × 7 tissues) | [ENCODE](https://www.encodeproject.org) |
| Probe manifest | MM285 array annotation (mm10, GENCODEvM25) | [InfiniumAnnotation](https://zwdzwd.github.io/InfiniumAnnotation) |
| Reference genome | mm10 (GRCm38) | [UCSC](https://hgdownload.soe.ucsc.edu/goldenPath/mm10/bigZips/) |

See [`data/README.md`](data/README.md) for detailed download instructions and file descriptions.

## Model Architecture

```
Sequence branch:   4 × 1000 (one-hot) → Conv1D blocks → 256-dim feature vector
Chromatin branch:  5 × 40   (binned)  → Conv1D blocks → 128-dim feature vector
Fusion:            384-dim concat → Dense(128) → Dropout(0.5) → Sigmoid → P(5hmC)
```

Total trainable parameters: ~4.3M

## Reproducing Manuscript Results

The main results reported in the paper were obtained from chromosome split index **0** (first line in `splits/chrom_splits.txt`), selected as the best-performing model based on validation loss across 100 random chromosome partitions.

To reproduce:

```bash
# Train with the same split
python src/train.py --split_idx 0 --config config.yaml

# Or load the pre-trained model directly for evaluation
```

The pre-trained weights are provided in `models/best_model_cv.pth`.

LOTO models are not included. To reproduce LOTO results, train each model individually:

```bash
for tissue in Cerebellum Heart Kidney Liver Lung Spleen Thymus; do
    python src/train_loto.py --exclude_tissue $tissue --split_idx 0 --config config.yaml
done
```

Then calibrate thresholds for each held-out tissue:

```bash
for tissue in Cerebellum Heart Kidney Liver Lung Spleen Thymus; do
    python src/calibrate.py --tissue $tissue \
        --model_path results/model_loto_no_${tissue}_cv0.pth \
        --config config.yaml
done
```

## Key Results

| Evaluation | AUROC | Accuracy |
|------------|-------|----------|
| Cross-chromosomal (global) | 0.949 | 89.06% |
| Sequence only | 0.734 | 67.12% |
| Chromatin only | 0.921 | 85.94% |

| LOTO Tissue | AUROC | Accuracy (default) | Accuracy (calibrated) |
|-------------|-------|--------------------|-----------------------|
| Lung | 0.943 | 87.23% | 87.19% |
| Heart | 0.933 | 83.52% | 84.94% |
| Kidney | 0.884 | 79.39% | 80.73% |
| Cerebellum | 0.866 | 66.01% | 76.02% |
| Liver | 0.827 | 45.66% | 76.55% |
| Spleen | 0.794 | 43.27% | 75.68% |
| Thymus | 0.709 | 37.63% | 70.09% |

## Citation

If you use EpiSeq-5hmC in your research, please cite:

```
Kim W, Bang H. EpiSeq-5hmC: Discriminating 5-hydroxymethylcytosine from 5-methylcytosine 
at base resolution by integrating DNA sequence and chromatin features (2026)
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Contact

- Hyoeun Bang — hebang@suwon.ac.kr
