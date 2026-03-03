# Data Download and Preparation

EpiSeq-5hmC requires three external data sources. This guide describes how to obtain and prepare them.

## Overview

| Source | Files | Size (approx.) | Purpose |
|--------|-------|-----------------|---------|
| GEO (GSE290585) | 2 processed matrix CSVs | ~1–2 GB each | Beta values (BS and bACE arrays, mixed) |
| Zenodo | MouseArrayMaster.xlsx | ~5 MB | Sample metadata (ZD → BS/bACE IDAT mapping, tissue, age) |
| ENCODE | 35 BigWig files | ~100–500 MB each | Chromatin signal tracks (5 marks × 7 tissues) |

After downloading, run `prepare_targets.py` to generate the ternary target label file used for training.

---

## 1. GEO — Ternary Methylome Matrices

Download two processed beta-value matrix files from [GSE290585](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE290585):

- `GSE290585_20250221_GEO_processed_matrix_GPL30650.csv.gz`
- `GSE290585_20250827_GEO_processed_matrix_GPL30650.csv.gz`

These are batch uploads at different dates, **not** separate BS/bACE files. Both contain mixed BS and bACE arrays identified by IDAT column headers. Which IDAT corresponds to BS or bACE is determined by the metadata file below.

Place the downloaded files in this directory:

```
data/
├── GSE290585_20250221_GEO_processed_matrix_GPL30650.csv.gz
└── GSE290585_20250827_GEO_processed_matrix_GPL30650.csv.gz
```

## 2. Zenodo — Sample Metadata

Download `MouseArrayMaster.xlsx` from the ternary methylome atlas Zenodo repository:

> https://zenodo.org/records/17109847

This file maps each biological sample (ZD) to its paired array identifiers:
- **`bACE` column = 0** → BS array (measures total methylation: 5mC + 5hmC)
- **`bACE` column = 1** → bACE array (measures 5hmC directly)

It also contains tissue type (`Tissue`) and age (`AgeInWeeks`) annotations.

Place the file in this directory:

```
data/
└── MouseArrayMaster.xlsx
```

## 3. ENCODE — Chromatin Signal Tracks

Download 35 BigWig signal track files (5 chromatin marks × 7 tissues) from the [ENCODE portal](https://www.encodeproject.org). All files are mm10-aligned, uniformly processed signal tracks.

### File list

| Tissue | CTCF | DNase-seq | H3K4me1 | H3K4me3 | POLR2A |
|--------|------|-----------|---------|---------|--------|
| Cerebellum | ENCFF876RKZ | ENCFF293XWM | ENCFF296IKH | ENCFF009XAS | ENCFF279CBP |
| Heart | ENCFF147IYR | ENCFF501KJH | ENCFF590RTF | ENCFF031YFX | ENCFF888EZW |
| Kidney | ENCFF826DBQ | ENCFF161VPP | ENCFF769TPY | ENCFF566SLT | ENCFF835XXX |
| Liver | ENCFF155SPJ | ENCFF100XUL | ENCFF989WVI | ENCFF310LUR | ENCFF779ZHR |
| Lung | ENCFF304KVI | ENCFF046WVP | ENCFF629UOT | ENCFF863AFJ | ENCFF035UNV |
| Spleen | ENCFF535WVL | ENCFF130MCM | ENCFF076HOK | ENCFF360QGB | ENCFF119UPE |
| Thymus | ENCFF669UPX | ENCFF516GQR | ENCFF275PZN | ENCFF676KTM | ENCFF802KMQ |

Each file can be downloaded from: `https://www.encodeproject.org/files/{ACCESSION}/@@download/{ACCESSION}.bigWig`

For example:
```bash
wget https://www.encodeproject.org/files/ENCFF876RKZ/@@download/ENCFF876RKZ.bigWig
```

Place all BigWig files in the `ENCODE/` subdirectory:

```
data/
└── ENCODE/
    ├── ENCFF876RKZ.bigWig
    ├── ENCFF293XWM.bigWig
    ├── ...
    └── ENCFF802KMQ.bigWig
```

## 4. Additional Required Files

### mm10 Reference Genome

Download the mouse mm10 reference genome FASTA and create an index:

```bash
wget https://hgdownload.soe.ucsc.edu/goldenPath/mm10/bigZips/mm10.fa.gz
gunzip mm10.fa.gz
samtools faidx mm10.fa
```

### MM285 Probe Manifest

Download the MM285 array manifest from the [Infinium Annotation page](https://zwdzwd.github.io/InfiniumAnnotation):

1. Navigate to **Mouse/MM285 Array**
2. Select genome build **GRCm38 / mm10**
3. Download the file under **Gene annotation (GENCODEvM25) and promoters**

Alternatively, download directly via command line:

```bash
wget https://github.com/zhou-lab/InfiniumAnnotationV1/raw/main/Anno/MM285/N296070/MM285.mm10.manifest.gencode.vM25.tsv.gz
```

Place the file as:

```
data/
└── MM285.mm10.manifest.gencode.vM25.tsv.gz
```

## 5. Generate Target Labels

After downloading all files, run:

```bash
python prepare_targets.py --data_dir data
```

This script performs three steps:

1. Parses `MouseArrayMaster.xlsx` to identify BS/bACE array pairs for each biological sample (ZD)
2. Filters to include only the 7 ZD_IDs present in `output_unique_matched.tsv` (ZD234, ZD19, ZD20, ZD21, ZD22, ZD23, ZD35)
3. Computes per-CpG 5hmC and 5mC fractions from the GEO beta-value matrices

Output:

```
data/
└── ternary_targets.csv
```

### Expected output format

A sample file (`ternary_targets_sample.csv`) is included in the repository for reference. The generated `ternary_targets.csv` should follow this structure:

- **Rows**: CpG probe IDs (e.g., `cg00101675_BC21`)
- **Columns**: `{ZD_ID}_5hmC` and `{ZD_ID}_5mC` pairs for the 7 tissues

```
ID_REF,ZD19_5hmC,ZD19_5mC,ZD20_5hmC,...
cg00101675_BC21,0.2436591579285699,0.49443849815375357,0.2644375185461219,...
cg00116289_BC21,0.0605216357770086,0.7634300436745906,0.0390125244970936,...
cg00211372_TC21,0.2901738937540525,0.6185385580305398,0.3093176985361246,...
...
```

| ZD_ID | Tissue |
|-------|--------|
| ZD234 | Cerebellum |
| ZD19 | Heart |
| ZD20 | Kidney |
| ZD21 | Liver |
| ZD22 | Lung |
| ZD23 | Spleen |
| ZD35 | Thymus |

5hmC values are direct bACE readouts; 5mC values are computed as BS − bACE (clipped to ≥ 0). You can verify your output against `ternary_targets_sample.csv` to confirm the pipeline ran correctly.

## Final Directory Structure

```
data/
├── README.md                          (this file)
├── mm10.fa                            (reference genome)
├── mm10.fa.fai                        (FASTA index)
├── MM285.mm10.manifest.gencode.vM25.tsv.gz
├── MouseArrayMaster.xlsx              (from Zenodo)
├── GSE290585_20250221_GEO_processed_matrix_GPL30650.csv.gz
├── GSE290585_20250827_GEO_processed_matrix_GPL30650.csv.gz
├── output_unique_matched.tsv          (ENCODE sample metadata)
├── ternary_targets_sample.csv         (sample output for verification)
├── ternary_targets.csv                (generated by prepare_targets.py)
│
└── ENCODE/
    ├── ENCFF876RKZ.bigWig
    ├── ENCFF293XWM.bigWig
    ├── ... (35 files total)
    └── ENCFF802KMQ.bigWig
```
