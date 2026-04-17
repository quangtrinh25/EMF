# WCE Electromagnetic Localization Pipeline

This repository contains the unified Python machine learning physics pipeline for real-time **6-DoF localization of Wireless Capsule Endoscopes (WCE)** using multi-frequency electromagnetic induction tracking.

## Overview
The system mathematically models non-linear magnetic fields induced by a custom 3-transmitter (TX) and 3-receiver (RX) coil topological setup. A 9-Channel measurement array is fed into the unified `pipeline.py`, which algorithmically evaluates and resolves the dynamic capsule Cartesian position ($x, y, z$) and 3D rotational orientation ($roll, pitch, yaw$) utilizing Multi-Output ensemble regression logic (`AdaBoostRegressor` & `XGBRegressor`).

### Pipeline Features
* **Zero-Copy Data Ingestion**: Utilizes heavy Apache Arrow multithreaded architecture natively via `Polars` to ingest and process ~2.25 million-row continuous datastreams optimally within a rigid 32 GB RAM hardware limit.
* **Consolidated Framework**: Orchestrates dataset generation (ROI bounds), magnetic induction simulation ($\mathcal{B}$-fields computation), iterative cross-validation training, and graphic performance generation inside a single Python script (`pipeline.py`).
* **Regulated Grid Search**: Implements native robust `GridSearchCV` scaling independently optimized per-model across exactly bounded subsample constraints and capped hardware threading (preventing runtime OOM duplication crashes).

## Setup & Environment
The system enforces isolated environments utilizing Python 3.11+.

```bash
# 1. Initialize Pyenv automated virtual configuration limits
source setup.sh

# 2. Activate sandbox
source .venv/bin/activate

# 3. Pull required packages (Polars, XGBoost, Scikit-Learn, etc)
pip install -r requirements.txt
```

## Running the Pipeline

The entire framework natively executes through `pipeline.py` manipulating distinct command-line parameters based on your goals. Outputs (RMSE graphs, inference metrics, tracking grids) will seamlessly dump into `/runs/pipeline`.

### 1. Rapid Development Verification (Quick Test)
To safely run a lightning-fast logic test through the entire script structure, dynamically limit the maximum data row limit and entirely skip the hyperparameter search loop.
```bash
python pipeline.py --skip-search --max-samples 10000
```

### 2. Standard Conference Experiment (Full Evaluation)
Executes a heavy, overnight evaluation simulating the core manuscript findings. Natively executes rigorous cross-validation (5 randomized seeds), iterating over 80+ hyperparameter structures, while actively capped internally (`n_jobs=4`) avoiding operating system OOM failure.
```bash
python pipeline.py
```

### 3. Re-Simulate Datasets from Scratch
Delete the primary cached mathematical logs and enforce a raw recalculation of physical coordinates based on custom topological bounds. 
```bash
python pipeline.py --rebuild-data --n-samples 2250000
```

## Useful Arguments Reference
- `--skip-search`: Bypasses hours of deep grid search optimization. Adopts explicit hardware inference values immediately.
- `--max-samples <N>`: Truncate datastream loading dynamically allowing instant model prototyping.
- `--search-n-jobs <N>`: Thread multiplier limit. Do **not** apply `-1` concurrently over 1.5M records. Defaults linearly to `4`.
- `--rebuild-data`: Drops natively generated `roi_grid.csv` and `emf_data.csv` objects.
- `--test-ratio` & `--val-ratio`: Explicit dataset allocation modifiers (Default identically to `0.15`).
- `--search-verbose`: Adjusts explicit logging thresholds to natively print mathematical progression ticks (`[CV X/Y]`).
