# Pan-Cancer Classification

This repository contains code and tools for pan-cancer data classification using deep learning architectures, SVM, and feature selection techniques. It includes preprocessing pipelines, channel expansion, neural network models, model comparison, and results analysis.

`py/ensemble` contains feature selection (with Chi2 and Random Forest) and classification (with SVM, MLP, CNN-1D, BaseKAN, EfficientKAN, FourierKAN, IT-KAN, DI-CNN+).

`py/fc_kan` contains the Fully Connected KAN classification pipeline.

`R/HMCC` contains the HMCC feature selection and classification pipeline.

## Project Structure

- `data/` - Raw, processed, and split datasets.
  - `processed` - _Place here .csv files from [zenodo.org/records/17993056](https://zenodo.org/records/17993056)_
  - `splits` - Per detaset predefined splits in .txt and .npy format
- `py/` – Python codebase and experiments.
  - `ensemble/` – Reusable code and runners for main methods.
    - `models/` – Model implementations: Autoencoder, BaseKAN, CNN-1D, CNN, EfficientKAN, FourierKAN, HybridKAN, MLP, ViT.
    - `scripts/` – Preprocessing and dataset-management helpers.
    - `utils/` – Utilities for channel expansion, feature selection, neighbor-informed processing, plotting, and CLI.
    - `evaluate_model.py` – Trains and evaluates classifiers (SVM and neural nets); saves logits and selected features; returns per-model loss, accuracy, and SEM.
    - `main.py` – Orchestrates end-to-end experiments: loads `configs.yaml` and datasets, sets up results and logs, runs cross-validated evaluation per dataset, and saves aggregated logits, metrics, ROC plots, and summary charts.
    - `configs.yaml` – Specifies datasets, feature counts, and per-model hyperparameters.
    - `run_seeds.sh` — Script that runs `main.py` using random seeds 42, 1000, and 12345.
  - `fc_kan/` – Fully Connected KAN classification pipeline. SE
- `R/` – R-based classifiers.
  - `renv.lock` – renv environment snapshot.
  - `HMCC/` – Ensemble feature selection and classification (Rahaman et al., 2025; [doi:10.1016/j.compbiomed.2025.110687](https://doi.org/10.1016/j.compbiomed.2025.110687)).
    - `feature_selection.r` – HMCC ensemble feature-selection implementation.
    - `classification.r` – Classification pipeline using HMCC-selected features; supports SVM and Random Forest.

## Results Placeholder
The following placeholders appear in results directory structures throughout this document:

| Placeholder | Description |
|---|---|
| `{dataset}` | Dataset name as defined in `configs.yaml` (e.g. `Stomach adenocarcinoma`) |
| `{model}` | Model name (e.g. `SVM`, `CNN-1D`, `MLP`, `BaseKAN`) |
| `{fold}` | Cross-validation fold index |
| `{timestamp}` | Run timestamp in `YYYYMMDD_HHMMSS` format |
| `{n_features}` | Number of selected features as set in `configs.yaml` |
| `{seed}` | Random seed passed via `--seed` (default: `42`) |
| `{mode}` | FC KAN feature mode (`10k` or `30`) |

## Requirements

- **Python 3.12** – See [requirements.txt](requirements.txt); PyTorch and Torchvision must be installed separately.
- **R** – See [renv.lock](R/renv.lock).

## Python Setup - for Ensemble and FC_KAN

1. Create and activate a virtual environment:
   ```sh
   python3.12 -m venv .venv
   source .venv/bin/activate
   ```

2. Install PyTorch (2.12.0) and Torchvision (0.27.0) for your OS and CUDA configuration via the [official PyTorch installer](https://pytorch.org/get-started/locally/). An NVIDIA GPU with CUDA support is strongly recommended.

3. Install remaining dependencies:
   ```sh
   pip3 install -r requirements.txt
   ```

## Ensemble - Usage
```sh
cd py/ensemble/
python main.py
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `-c, --configs` | str | `configs.yaml` | Path to YAML config file (datasets, feature counts, models). |
| `-f, --input_output_folder` | str | `../../` | Base project folder containing `data/`; results are saved here. |
| `-s, --seed` | int | `42` | Random seed for reproducibility. |

### Results

Results are saved to `{input_output_folder}/results/Ensemble_{n_features}_s{seed}_{timestamp}/`.

Structure:
- `charts/`
- `logits/`
  - `{dataset}/`
    - `{model}/`
      - `logits_{dataset}_{fold}_{model}.txt`
- `selected_features/`
  - `selected_features_{dataset}_{fold}.txt` - 0 indexed

## FC KAN - Usage

```sh
cd py/fc_kan/
python run_fc_kan_dog_bs_sum.py --all --mode 10k
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `--cancer` | str | — | Cancer type name (exact match to splits file). |
| `--gse` | str | — | GSE code (e.g. `GSE13507`). |
| `--mode` | `10k` \| `30` | `10k` | Feature mode. |
| `--all` | flag | `false` | Run all datasets. |
| `--splits-dir` | path | `../../data/splits/kfold` | Directory containing split files. |
| `--datasets-dir` | path | `../../data/processed` | Directory containing dataset CSV files. |
| `--selected-features-dir` | path | `selected_features` | Directory containing selected features files. |
| `--output-dir` | path | `../../results/` | Output directory. |
| `--epochs` | int | `50` | Number of training epochs. |
| `--batch-size` | int | `32` | Training batch size. |
| `--lr` | float | `1e-3` | Learning rate. |
| `--device` | str | `cuda` / `cpu` | Device for training; auto-detected from CUDA availability. |

### Results

Results are saved to the directory specified by `--output-dir`. By default, a folder named `FCKAN_f{mode}_{timestamp}` is created inside `results/`.

Structure:
- `logits/`
  - `{dataset}/`
      - `fold_{fold}.txt`
- `summary/`
  - `{dataset}_summary.json`

## HMCC - Setup and Usage

```sh
Rscript --vanilla -e "install.packages('renv', repos = 'https://cloud.r-project.org')"
Rscript -e "renv::init(bare = TRUE); renv::restore(lockfile = 'R/renv.lock')"
Rscript R/HMCC/classification.r
```

### Results
A folder named `HMCC_{timestamp}` is created inside `results/`.

Structure:
- `logits/`
  - `{dataset}/`
      - `RF`
        - `fold_{fold}.txt`
      - `SVMR`
        - `fold_{fold}.txt`
- `selected_features/`
  - `{dataset}`
    - `selected_features_{fold}.txt` - 0 indexed

## Test Environment
Experiments were run on the following hardware:
| | |
|---|---|
| **OS** | Ubuntu 24.04.4 LTS |
| **CPU** | Intel Xeon @ 2.30 GHz |
| **RAM** | 31.1 GB |
| **GPU** | NVIDIA Tesla T4 — 15.0 GB (driver 580.159.03) |
| **IDEs** | Spyder (dataset exploration, transposed matrices); VS Code (development, execution, navigation) |