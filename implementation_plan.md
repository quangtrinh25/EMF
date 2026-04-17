# Refactor EMF Localization Pipeline

This plan outlines the steps to build a unified, clean, and optimized machine learning pipeline for the EMF capsule localization project, focusing strictly on AdaBoost and XGBoost models.

## Proposed Changes

We will create a clean and comprehensive Python script named `pipeline.py`. This script will orchestrate everything from dataset loading to multi-seed execution and plotting.

### 1. Data Loading & Memory Management
- **Pipeline Strategy:** In line with your suggestion, the data will first be fully generated as CSV files (`roi_grid.csv` and `emf_data.csv`).
- **Memory (RAM) Feasibility:** 32GB RAM is more than enough! 2.25 million rows with 15 numerical columns (9 EMF + 6 labels) stored as `float32` takes less than 300 MB of native memory and roughly 1-2 GB when processed in Pandas.
- **Action:** `pipeline.py` will contain a function `load_and_preprocess_data()` that reads the CSV files wholly into memory, ensuring lightning-fast model training compared to disk-reading chunks.

### 2. Rigorous Data Splitting
- Instead of static splits, we will utilize Python's `train_test_split` with a dynamic seed.
- **Split Ratios:** Training (70%), Validation (15%), and Test (15%).
- A custom scaling pipeline (`EMFPreprocessor`) will be fitted on the Training split to compress the long-tail EMF distributions using signed-log transforms, then applied to Validation and Test splits to prevent data leakage.

### 3. Hyperparameter Optimization & Model Search
AdaBoost and XGBoost are fundamentally different models under the hood, so they possess different parameters for their grid searches. Because evaluating 2.25 million rows iteratively takes incredibly long, we will optimize parameters on a *subsampled cross-validation pool* (e.g., 100k points), and then apply the best hyperparameters to the full dataset.

- **AdaBoost Grid Options:**
  - Base Estimator Maximum Depth: `[15, 20, 30]`
  - Number of Estimators: `[50, 100]`
  - Learning Rate: `[0.05, 0.1, 0.5]`

- **XGBoost Grid Options:**
  - Maximum Depth: `[6, 12, 20]`
  - Number of Estimators: `[100, 200]`
  - Learning Rate: `[0.05, 0.1]`
  - Subsample / Colsample: `[0.8, 1.0]`

### 4. Multi-Seed Validation Loop (Mean ± Std)
To ensure the pipeline evaluates system stability exactly as requested:
- The script will loop over pre-defined random seeds (e.g., `[42, 43, 44, 45, 46]`).
- For each seed:
  1. The data is shuffled and split into the 70/15/15 chunks.
  2. The optimized AdaBoost and XGBoost models are trained on the 70% fraction.
  3. Predictions are made against the unseen Test split.
  4. Accuracy metrics (Position RMSE, Orientation RMSE) are logged.
- Once all seeds finish running, the pipeline calculates and prints the aggregated results strictly as `Mean ± Standard Deviation`.

### 5. Error Plotting & Visualization
The output stage will run functions derived from the known `evaluation.py` to automatically save:
- `scatter_3d_test.png`: 3D view of actual vs. predicted real-life spatial tracking.
- `pos_error_trend.png`: A line graph showcasing the error positioning discrepancy per sample points.
- `ori_error_trend.png`: A graph visualizing tracking error for orientations in degrees.

## User Review Required

Does this detailed layout of steps cover all parameters and actions you are envisioning for the unified `pipeline.py` script?
