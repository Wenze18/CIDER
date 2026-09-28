# CIDER

**Cycle-consistent Integration of molecular Design and gene-Expression Response prediction**

CIDER is a closed-loop framework for gene-expression-guided molecular discovery. A forward generator proposes molecular structures from a target gene-expression signature and experimental context. A reverse predictor estimates their gene-expression responses, and cycle-consistency scoring prioritises candidates whose predicted responses agree with the target.

- **CIDER-Conservative** prioritises response recovery near known chemical space for analog-like optimisation.
- **CIDER-Exploratory** balances response recovery with chemical novelty, drug-likeness and synthetic accessibility.

This repository contains the training and inference code accompanying the CIDER manuscript. Data are downloaded from LINCS through GEO. Model checkpoints are produced by training and are not included in the repository.

## Installation

The code was tested on Linux with Python 3.9.16, PyTorch 2.8.0, CUDA 12.8 and RDKit 2025.09.2. A CUDA-capable GPU is recommended for training and molecular generation.

Run all commands from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[data]"
```

Install a PyTorch build compatible with the local CUDA environment first if a different CUDA runtime is required.

## LINCS Data

The study uses LINCS L1000 **Level 5** signatures from [GSE92742](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE92742) and [GSE70138](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE70138). The required files are publicly available from NCBI:

| File | GSE92742 | GSE70138 |
| --- | --- | --- |
| Level 5 expression matrix | [COMPZ.MODZ, 473,647 signatures](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx.gz) | [COMPZ, 118,050 signatures, 2017-03-06](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE70nnn/GSE70138/suppl/GSE70138_Broad_LINCS_Level5_COMPZ_n118050x12328_2017-03-06.gctx.gz) |
| Signature metadata | [sig_info](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/GSE92742_Broad_LINCS_sig_info.txt.gz) | [sig_info](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE70nnn/GSE70138/suppl/GSE70138_Broad_LINCS_sig_info_2017-03-06.txt.gz) |
| Signature quality | [sig_metrics](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/GSE92742_Broad_LINCS_sig_metrics.txt.gz) | [sig_metrics](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE70nnn/GSE70138/suppl/GSE70138_Broad_LINCS_sig_metrics_2017-03-06.txt.gz) |
| Compound structures | [pert_info](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/GSE92742_Broad_LINCS_pert_info.txt.gz) | [pert_info](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE70nnn/GSE70138/suppl/GSE70138_Broad_LINCS_pert_info_2017-03-06.txt.gz) |
| Landmark gene annotations | [gene_info](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/GSE92742_Broad_LINCS_gene_info.txt.gz) | [gene_info](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE70nnn/GSE70138/suppl/GSE70138_Broad_LINCS_gene_info_2017-03-06.txt.gz) |

Download these files and prepare the landmark-gene matrix:

```bash
python -m cider.lincs --download
python -m cider.prepare
```

Allow approximately 60 GB of free disk space for the compressed and decompressed source files, in addition to space for training outputs. Downloads are stored in `data/raw/level5/` and `data/raw/metadata/`. Existing completed files are reused. Files downloaded manually from the links above can be placed in these same directories; run `python -m cider.lincs` without `--download` to process them.

Preprocessing retains small-molecule signatures (`pert_type = trt_cp`) with transcriptional activity score (TAS) at least 0.2, joins compound structures and experimental metadata, and extracts the 978 landmark genes. Phase 2 genes are aligned to the Phase 1 matrix order. This produces `data/processed/lincs_cp_landmark_all.h5ad`.

`cider.prepare` encodes molecules as SELFIES sequences of at most 128 tokens, groups records by source SMILES for the training/validation/test split, and estimates response and log-dose normalisation statistics from the training partition. It writes `data/processed/cache.pt` and a metadata summary. Molecular feature caches are computed automatically during training; no additional feature downloads are required.

## Training

Train the reverse response predictors and forward generators:

```bash
python -m cider.train --stage reverse --device cuda:0
python -m cider.train --stage forward --device cuda:0
```

The training runner executes each stage in dependency order. Checkpoints are written directly to `checkpoints/reverse/` and `checkpoints/forward/`, where the inference configurations read them. Training summaries are written to `outputs/training/`. All directories are created by the relevant commands. Existing selected checkpoints are protected from accidental overwriting.

To inspect the commands before training or train a single component:

```bash
python -m cider.train --stage all --dry-run
python -m cider.train --recipe configs/training/forward_exploratory_cvae.json --device cuda:0
```

| Component | Recipes in `configs/training/` |
| --- | --- |
| Tabular response models | `reverse_tabular_01.json` through `reverse_tabular_09.json` |
| Transformer response model | `reverse_transformer.json` |
| Perceiver response model | `reverse_perceiver.json` |
| Prefix-conditioned generators | `forward_prefix.json`, `forward_wide_prefix.json` |
| Retrieval-conditioned generator | `forward_retrieval.json` |
| Response-neighbourhood distillation | `forward_neighbor.json` |
| Structure-aware multi-positive learning | `forward_multipositive_prefix_a.json`, `forward_multipositive_prefix_b.json`, `forward_multipositive_neighbor.json` |
| Conditional variational generator | `forward_exploratory_cvae.json` |
| Response-to-fingerprint prediction | `forward_fingerprint.json` |
| Learned structural ranking | `forward_structural_ranker.json` |

Each recipe specifies the architecture, loss, optimiser, seed and downstream checkpoint. Recipes run for at most ten epochs; early-stopping settings are specified per component. `--epochs` can override the epoch limit. Run independent jobs on separate GPUs when parallelising.

`configs/reverse.json` specifies the cross-family ensemble and its manuscript weights. For new training runs, fit ensemble weights on the validation partition:

```bash
python -m cider.training.ensemble \
  --manifest configs/reverse.json \
  --out_manifest checkpoints/reverse/ensemble.json \
  --device cuda:0
```

Use the fitted configuration with `--manifest checkpoints/reverse/ensemble.json` for reverse prediction and `--reverse_manifest checkpoints/reverse/ensemble.json` for molecular generation.

## Reverse Response Prediction

Predict responses for the test partition:

```bash
python -m cider.reverse \
  --split test --rows 9665 \
  --out_prefix outputs/reverse_test \
  --device cuda:0
```

External molecules can be supplied with `--input_csv data/molecules.csv`. The CSV must contain `smiles`, `dose`, `cell` and `time`, with an optional `id`. `dose` is the normalised log-dose; `cell` and `time` are integer IDs from the prepared cache vocabularies. Predictions use the cache's normalised gene-expression space.

Outputs comprise a response array (`.pred.npy`), row metadata (`.meta.csv`) and a summary (`.metrics.json`). Reference-based response metrics are included when observed target responses are available.

## Molecular Generation

Generate molecules for target profiles in the prepared cache:

```bash
python -m cider.generate \
  --version conservative --split test --rows 8 \
  --out outputs/conservative.csv --device cuda:0

python -m cider.generate \
  --version exploratory --split test --rows 8 \
  --out outputs/exploratory.csv --device cuda:0
```

Each mode returns up to eight selected molecular identities per condition. Use `--row_indices` for explicit cache rows. For the manuscript benchmark sampling settings, replace `--rows 8` with `--sample_rows --rows 200 --row_seed 20260507 --seed 42`.

Selection combines chemical utility with within-pool ranks of Pearson correlation, cosine similarity, mean squared error and cycle score. Lower cycle scores indicate better agreement. Resolved sampling and ranking settings are saved alongside the generated CSV. Run `python -m cider.generate --help` for all options.


