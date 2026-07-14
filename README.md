# Bilinear MLP Interpretability

This repository reproduces and extends ["Bilinear MLPs enable weight-based mechanistic interpretability"](https://arxiv.org/pdf/2410.08417) (Pearce et al., ICLR 2025). It targets:
- **Section 4 (Vision)**: MNIST/Fashion-MNIST bilinear MLP eigendecomposition
- **Section 5 (Language)**: Negation circuit discovery via SAE analysis
- **Extensions**: CP decomposition and cross-dataset robustness

The goal of this README is to make the **codebase reproducible without extra effort**, and to document the **report** and **presentation** build steps.

---

## Table of Contents

1. [Quick Start](#quick-start-reproducibility-smoke-tests)
2. [Environment Setup](#environment-setup)
3. [Notebooks (All Results)](#notebooks-all-results)
4. [Full Reproduction Paths](#full-reproduction-paths)
5. [Datasets](#datasets)
6. [Pre-trained Checkpoints](#pre-trained-checkpoints-and-results)
7. [Tests](#tests)
8. [Repository Structure](#repository-structure-high-level)

---

## Quick Start (Reproducibility Smoke Tests)

These commands verify the environment and run minimal tests.

```bash
# Vision (2 epochs, MPS-friendly)
./scripts/train/run_vision.sh test

# Language (quick tests)
./scripts/train/run_language.sh test

# Cross-dataset robustness (2 epochs, MNIST only)
./scripts/train/run_extension_cross_dataset.sh test
```

## Environment Setup

```bash
# Snellius (GPU)
conda env create -f environment.yml && conda activate bilinear_mlp

# Local (CPU/MPS)
conda env create -f environment_cpu.yml && conda activate bilinear_mlp_cpu
```

## Notebooks (All Results)

The `notebooks/` directory contains Jupyter notebooks that reproduce **all figures and results** from the report. Each notebook can be run end-to-end and automatically downloads required checkpoints/results from Google Drive.

| Notebook | Description | Report Sections |
|----------|-------------|-----------------|
| `01_reproduction_vision.ipynb` | **Vision experiments (Section 4)**: eigenspectrum analysis, eigenvector visualization, ablation studies. Verifies low-rank emergence, weight decay effectiveness, and interpretable eigenvectors. | Figures 1-7 |
| `02_reproduction_language.ipynb` | **Language experiments (Section 5)**: correlation analysis, negation circuits, SAE training effect. Verifies low-rank approximation, negation AND-gate structure, and SAE training correlation. | Figures 8-10 |
| `03a_extension_cross_dataset_robustness.ipynb` | **Extension 1: Cross-Dataset Robustness**: Tests whether regularized models learn universal "Platonic forms" (shape geometry) vs dataset-specific artifacts. Validates via EMNIST-Digits transfer, USPS domain shift, and geometric letter semantics (O→0, I→1). | Extension 1 |
| `03b_cp_extension.ipynb` | **Extension 2: CP Decomposition**: Analyzes architectural rank constraints (Fixed, Lambda, Gated CP) vs emergent low-rank from regularization. Compares accuracy-interpretability trade-offs and eigenvector quality. | Extension 2 |

### Running Notebooks

```bash
# Activate environment
conda activate bilinear_mlp_cpu  # or bilinear_mlp for GPU

# Start Jupyter
jupyter notebook notebooks/

# Or run specific notebook from command line
jupyter nbconvert --to notebook --execute notebooks/01_reproduction_vision.ipynb
```

**Note**: Checkpoints and results are automatically downloaded on first run. All plotting code is imported from `src/plot_utils/` modules.

---

## Full Reproduction Paths

### Vision (Section 4)

```bash
# Base configs (4 configs x 5 seeds x MNIST+Fashion)
./scripts/train/run_vision.sh train base

# Noise sweep (Figure 4)
./scripts/train/run_vision.sh train noise

# Model size sweep (Figure 5)
./scripts/train/run_vision.sh train size

# Challenge task (Figure 6)
./scripts/train/run_vision.sh train challenge

# Adversarial robustness (Figure 7) - uses existing checkpoints
./scripts/train/run_vision.sh figures  # Generates Figure 7 from noise sweep

# Generate all vision figures
./scripts/train/run_vision.sh figures

# Run all vision experiments
./scripts/train/run_vision.sh all
```

### Language (Section 5)

```bash
# Figure 8 negation circuit visualization
./scripts/train/run_language.sh figure8 --device mps

# Figure 9 correlation sweep (all 3 models)
./scripts/train/run_language.sh figure9

# Figure 10 SAE training time analysis
./scripts/train/run_language.sh figure10

# Negation discovery
./scripts/train/run_language.sh negation

# Interaction analysis
./scripts/train/run_language.sh interaction

# Generate all language figures
./scripts/train/run_language.sh figures

# Run all language experiments
./scripts/train/run_language.sh all
```

### Extension 1: Cross-Dataset Robustness

```bash
# Train all cross-dataset models (MNIST + EMNIST, CoM enabled)
./scripts/train/run_extension_cross_dataset.sh train all

# Generate all cross-dataset figures
./scripts/train/run_extension_cross_dataset.sh figures

# Run full pipeline (train + figures)
./scripts/train/run_extension_cross_dataset.sh all
```

### Extension 2: CP Decomposition

```bash
# Train CP models (rank sweep, all modes)
./scripts/train/run_extension_cp.sh train all

# Generate all CP figures
./scripts/train/run_extension_cp.sh figures

# Run full pipeline (train + figures)
./scripts/train/run_extension_cp.sh all

# Quick test (2 epochs)
./scripts/train/run_extension_cp.sh test
```

## Outputs and Expected Artifacts

- **Checkpoints**: `checkpoints/` (vision, extension_cross_dataset, CP)
- **Results JSONs**: `results/`
- **Figures for report**: `Report/figures/`
- **Interactive assets**: `Report/paper_hub_bundle/`, `results/interactive/`

## Datasets

All datasets are **automatically downloaded** via PyTorch's `torchvision` on first use:

| Dataset | Source | Usage |
|---------|--------|-------|
| MNIST | `torchvision.datasets.MNIST` | Vision (Section 4), baseline digit classification |
| Fashion-MNIST | `torchvision.datasets.FashionMNIST` | Vision (Section 4), more complex classification |
| EMNIST (Letters) | `torchvision.datasets.EMNIST` | Extension 1, cross-dataset robustness |
| USPS | `torchvision.datasets.USPS` | Extension 1, transfer learning validation |

**Language models** (Section 5) use pre-trained models from HuggingFace:
- `tdooms/ts-medium` (6L TinyStories)
- `tdooms/fw-small` (12L FineWeb)
- `tdooms/fw-medium` (16L FineWeb)

SAE checkpoints are also from HuggingFace (`tdooms/fw-medium-scope`, etc.).

**Data directory**: Downloaded datasets are cached in `data/` (gitignored). The first run may take a few minutes to download.

---

## Pre-trained Checkpoints and Results

Due to file size constraints, trained model checkpoints and full result files are hosted on Google Drive.

### Automatic Download (Recommended)

Artifacts are **downloaded automatically** when running figure generation scripts via `src/artifact_loader.py`. To manually trigger download:

```python
from src.artifact_loader import ensure_artifacts
ensure_artifacts()  # Downloads checkpoints.zip and results.zip if not present
```

Or via CLI:
```bash
python -m src.artifact_loader
```

### Manual Download (If Automatic Download Fails)

If automatic download fails (e.g., SSL issues, rate limiting), download manually:

1. **Download the zip files:**
   - [checkpoints.zip](https://drive.google.com/file/d/12RI9zXhgjXcrVO50qU3GGRvHNTyKw3ws/view?usp=sharing) (~3.5 GB)
   - [results.zip](https://drive.google.com/file/d/1iy3BfsCVOOIt43yi9h9KNe31wwGPVFgK/view?usp=sharing) (~200 MB)

2. **Place them in the project root and extract:**
   ```bash
   # Using Python (recommended)
   python -c "from src.artifact_loader import extract_zip; extract_zip('checkpoints.zip', 'checkpoints'); extract_zip('results.zip', 'results')"
   
   # Or using command line
   unzip checkpoints.zip
   unzip results.zip
   ```

3. **Verify the structure:**
   ```
   bilinear_mlp_reproduction/
   ├── checkpoints/
   │   ├── vision/
   │   │   ├── mnist/
   │   │   ├── fashion/
   │   │   ├── challenge/
   │   │   ├── noise_sweep/
   │   │   └── size_sweep/
   │   ├── extension_cross_dataset/
   │   └── extension_cp/
   └── results/
       ├── language/
       ├── extension_cross_dataset/
       └── ...
   ```

### File Checksums (for verification)

| File | Size | MD5 |
|------|------|-----|
| checkpoints.zip | ~3.6 GB | `a457f540b0acb7cbda9d9c94173456fb` |
| results.zip | ~187 MB | `b7b41d9af61750c7dfc9b6a5cb1a3324` |

Verify with: `md5 checkpoints.zip results.zip` (macOS) or `md5sum checkpoints.zip results.zip` (Linux)

These checkpoints enable full reproduction of all figures without retraining (~40 GPU hours).

## Reproducibility Requirements

These are enforced project conventions:

- **Original code**: `bilinear-decomposition-main/` contains the original paper code with **minor modifications for MPS compatibility** (Apple Silicon). Changes include CPU fallbacks for `torch.linalg.eigh()` which is unsupported on MPS.
- **Seeds**: experiments use `[42, 43, 44, 45, 46]`.
- **Logging**: every experiment must log to `wandb` and track CO2 via `codecarbon`.
- **Center-of-Mass (CoM)**: applied to raw tensors before normalization.
- **USPS upscaling**: 16x16 → 28x28 before CoM.
- **MPS caveat**: some `einsum` ops are forced to CPU on MPS (language).

## Known Constraints and Notes

- **Language on MPS is slow** (4–6 hours). Prefer GPU if available.
- **Figure 8 SAE limitation**: ts-medium lacks mlp-in SAE checkpoints. We used fw-medium (features 3834/751).

## Tests

```bash
python -m pytest tests/ -v
python -m pytest tests/ -v -k "test_effective_rank"
```

## Repository Structure (High-Level)

```
bilinear_mlp_reproduction/
├── notebooks/           # Jupyter notebooks with all results (run these!)
│   ├── 01_reproduction_vision.ipynb   # Vision experiments (Section 4)
│   ├── 02_reproduction_language.ipynb # Language experiments (Section 5)
│   ├── 03a_extension_cross_dataset_robustness.ipynb  # Extension 1: Cross-dataset transfer
│   └── 03b_cp_extension.ipynb         # Extension 2: CP decomposition analysis
├── src/                 # Core code (vision, language, models, plot_utils)
├── scripts/             # Experiment runners and figure generation
├── configs/             # YAML experiment configs
├── tests/               # Unit tests (pytest)
├── environment.yml      # GPU environment (CUDA)
├── environment_cpu.yml  # CPU/MPS environment (Mac/local)
└── bilinear-decomposition-main/  # Original paper code (wrapped, not modified)
```

### Key Code Organization

- **`src/`**: All reusable code (models, data loading, spectral analysis, plotting)
- **`scripts/`**: Shell scripts and Python runners for experiments
- **`notebooks/`**: Self-contained notebooks that generate all report figures
- **`tests/`**: Pytest unit tests for core functionality
