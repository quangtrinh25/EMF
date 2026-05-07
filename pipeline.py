"""Unified end-to-end EMF localization pipeline.

This script consolidates the project into one conference-aligned workflow:
1. Generate the ROI grid if needed.
2. Simulate the 9-channel EMF dataset if needed.
3. Load the full CSV datasets into memory.
4. Split the data into train/validation/test using 70/15/15.
5. Fit a feature preprocessor on the training split only.
6. Tune and train the configured regressors.
7. Evaluate RMSE/R2/timing, save plots, and aggregate results across seeds.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import pickle
import time
import gc
import threading
import psutil
from dataclasses import asdict, dataclass
from functools import partial

os.environ.setdefault("MPLCONFIGDIR", os.path.join("/tmp", "matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone, RegressorMixin
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.metrics import make_scorer, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.validation import check_is_fitted
from scipy.stats import shapiro, norm

try:
    from xgboost import XGBRegressor
except ImportError:  # pragma: no cover
    XGBRegressor = None

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False

from tqdm import tqdm


class MemoryGuard:
    def __init__(self, threshold_percent: float = 90.0):
        self.threshold = threshold_percent
        self.active = False
        self.thread = None

    def _monitor(self):
        while self.active:
            if psutil.virtual_memory().percent > self.threshold:
                log_message(f"\n[Safety] FATAL: System RAM exceeded {self.threshold}%! Terminating to prevent hard crash.")
                os._exit(1)
            time.sleep(1)

    def start(self):
        self.active = True
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()

    def stop(self):
        self.active = False

MEMORY_GUARD = MemoryGuard(threshold_percent=92.0)


POSE_COLUMNS = ["x", "y", "z", "cos_roll", "cos_pitch", "cos_yaw"]
ANGLE_POSE_COLUMNS = ["x", "y", "z", "roll_deg", "pitch_deg", "yaw_deg"]
EMF_COLUMNS = [f"EMF_TX{tx_idx}_RX{rx_idx}" for tx_idx in range(1, 4) for rx_idx in range(1, 4)]
DEFAULT_SEEDS = [42, 43, 44, 45, 46]
DEFAULT_MODELS = ["adaboost", "randomforest", "decisiontree", "xgboost"]

PRESET_DEFAULTS = {
    "safe": {
        "models": ",".join(DEFAULT_MODELS),
        "seeds": "42",
        "search_samples": 10_000,
        "search_folds": 2,
        "search_n_jobs": 1,
        "search_verbose": 0,
        "skip_search": True,
        "model_n_jobs": -1,
        "skip_plots": True,
        "max_samples": 50_000,
        "max_samples_strategy": "random",
        "target_format": "angle",
    },
    "balanced": {
        "models": ",".join(DEFAULT_MODELS),
        "seeds": "42",
        "search_samples": 20_000,
        "search_folds": 2,
        "search_n_jobs": 1,
        "search_verbose": 0,
        "skip_search": False,
        "model_n_jobs": -1,
        "skip_plots": True,
        "max_samples": 100_000,
        "max_samples_strategy": "random",
        "target_format": "angle",
    },
    "full": {
        "models": ",".join(DEFAULT_MODELS),
        "seeds": ",".join(str(seed) for seed in DEFAULT_SEEDS),
        "search_samples": 10_000,
        "search_folds": 2,
        "search_n_jobs": 1,
        "search_verbose": 2,
        "skip_search": False,
        "model_n_jobs": -1,
        "skip_plots": False,
        "max_samples": -1,
        "max_samples_strategy": "random",
        "target_format": "angle",
    },
}


def log_message(message: str) -> None:
    tqdm.write(message)




# The paper explicitly states AdaBoost uses 30 estimators with linear loss and
# XGBoost uses depth 20 with 100 estimators. Remaining knobs follow the repo
# defaults when search is disabled.
CONFERENCE_REFERENCE_PARAMS = {
    "adaboost": {
        "estimator_depth": 15,
        "min_samples_split": 10,
        "min_samples_leaf": 5,
        "n_estimators": 30,
        "learning_rate": 0.1,
        "loss": "linear",
    },
    "randomforest": {
        "n_estimators": 50,
        "max_depth": 15,
        "min_samples_split": 10,
        "min_samples_leaf": 5,
        "max_samples": 0.2,
    },
    "decisiontree": {
        "max_depth": 15,
        "min_samples_split": 10,
        "min_samples_leaf": 5,
    },
    "xgboost": {
        "n_estimators": 100,
        "max_depth": 12,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5.0,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "gamma": 0.1,
    },
}

# Search spaces are based on implementation_plan.md, with the AdaBoost paper
# baseline of 30 estimators added to the candidate list.
PARAM_GRIDS = {
    "adaboost": {
        "estimator_depth": [15],
        "n_estimators": [30],
        "learning_rate": [0.05, 0.1],
        "loss": ["linear"],
    },
    "randomforest": {
        "n_estimators": [100],
        "max_depth": [10, 15],
        "min_samples_split": [10],
    },
    "decisiontree": {
        "max_depth": [10, 15],
        "min_samples_split": [10],
    },
    "xgboost": {
        "max_depth": [8, 12, 14],
        "n_estimators": [200],
        "learning_rate": [0.1],
        "subsample": [0.8],
        "colsample_bytree": [0.8],
        "min_child_weight": [5.0],
        "reg_alpha": [0.0],
        "reg_lambda": [1.0],
        "gamma": [0.0],
    },
}

TX_SPECS = [
    {
        "pos_m": np.array([8.50, 0.00, 7.25], dtype=np.float64) * 1e-2,
        "euler_deg": (0.0, -15.0, 0.0),
        "turns": 300,
        "freq_hz": 4000,
        "radius_m": 0.06,
    },
    {
        "pos_m": np.array([-4.25, 7.36, 7.25], dtype=np.float64) * 1e-2,
        "euler_deg": (15.0, 0.0, 30.0),
        "turns": 300,
        "freq_hz": 4500,
        "radius_m": 0.06,
    },
    {
        "pos_m": np.array([-4.25, -7.36, 7.25], dtype=np.float64) * 1e-2,
        "euler_deg": (15.0, 0.0, 120.0),
        "turns": 300,
        "freq_hz": 5000,
        "radius_m": 0.06,
    },
]

RX_SPECS = [
    {"turns": 250, "area_m2": 3.0 * np.pi * 1e-6, "axis0": np.array([1.0, 0.0, 0.0], dtype=np.float64)},
    {"turns": 200, "area_m2": 10.0 * 7.0 * 1e-6, "axis0": np.array([0.0, 1.0, 0.0], dtype=np.float64)},
    {"turns": 200, "area_m2": 10.0 * 10.0 * 1e-6, "axis0": np.array([0.0, 0.0, 1.0], dtype=np.float64)},
]

MU0 = 4.0 * np.pi * 1e-7


@dataclass(frozen=True)
class ROIConfig:
    x_min: float = -0.10
    x_max: float = 0.10
    y_min: float = 0.667882
    y_max: float = 0.867882
    z_min: float = 0.05879
    z_max: float = 0.20879
    angle_min_deg: float = 0.0
    angle_max_deg: float = 180.0
    nx: int = 15
    ny: int = 15
    nz: int = 10
    n_roll: int = 10
    n_pitch: int = 10
    n_yaw: int = 10

    @property
    def total_points(self) -> int:
        return self.nx * self.ny * self.nz * self.n_roll * self.n_pitch * self.n_yaw


class EMFPreprocessor:
    """Signed-log + standardization preprocessor for 9-channel EMF."""

    def __init__(self, use_signed_log: bool = True, use_standardize: bool = True, eps: float = 1e-12):
        self.use_signed_log = bool(use_signed_log)
        self.use_standardize = bool(use_standardize)
        self.eps = float(eps)
        self._fitted = False
        self.scale_: np.ndarray | None = None
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "EMFPreprocessor":
        Z = np.asarray(X, dtype=np.float64)
        if Z.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {Z.shape}")

        if self.use_signed_log:
            scale = np.median(np.abs(Z), axis=0)
            scale = np.maximum(scale, self.eps)
            self.scale_ = scale.astype(np.float64)
            Z = np.sign(Z) * np.log1p(np.abs(Z) / self.scale_)

        if self.use_standardize:
            mean = Z.mean(axis=0)
            std = Z.std(axis=0)
            std = np.maximum(std, self.eps)
            self.mean_ = mean.astype(np.float64)
            self.std_ = std.astype(np.float64)

        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("EMFPreprocessor.fit must be called before transform.")

        Z = np.asarray(X, dtype=np.float64)
        if Z.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {Z.shape}")

        if self.use_signed_log:
            Z = np.sign(Z) * np.log1p(np.abs(Z) / self.scale_)

        if self.use_standardize:
            Z = (Z - self.mean_) / self.std_

        return Z.astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class AdaBoostPoseRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        estimator_depth: int = 15,
        min_samples_split: int = 10,
        min_samples_leaf: int = 5,
        n_estimators: int = 30,
        learning_rate: float = 0.1,
        loss: str = "linear",
        random_state: int = 42,
        n_jobs: int = 1,
    ):
        self.estimator_depth = estimator_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.loss = loss
        self.random_state = random_state
        self.n_jobs = n_jobs

    def _make_inner(self) -> MultiOutputRegressor:
        base_tree = DecisionTreeRegressor(
            max_depth=self.estimator_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
        )
        booster_kwargs = {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "loss": self.loss,
            "random_state": self.random_state,
        }
        try:
            booster = AdaBoostRegressor(estimator=base_tree, **booster_kwargs)
        except TypeError:  # pragma: no cover
            booster = AdaBoostRegressor(base_estimator=base_tree, **booster_kwargs)
        return MultiOutputRegressor(booster, n_jobs=self.n_jobs)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AdaBoostPoseRegressor":
        self.model_ = self._make_inner()
        self.model_.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "model_")
        return self.model_.predict(X)


class XGBoostPoseRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 20,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: float = 1.0,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        gamma: float = 0.0,
        random_state: int = 42,
        n_jobs: int = 1,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.random_state = random_state
        self.n_jobs = n_jobs

    def _make_inner(self) -> MultiOutputRegressor:
        if XGBRegressor is None:  # pragma: no cover
            raise ImportError("xgboost is required to use the XGBoost model.")

        booster = XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            min_child_weight=self.min_child_weight,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            gamma=self.gamma,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbosity=0,
        )
        return MultiOutputRegressor(booster, n_jobs=1)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostPoseRegressor":
        self.model_ = self._make_inner()
        self.model_.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "model_")
        return self.model_.predict(X)


class RandomForestPoseRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        n_estimators: int = 50,
        max_depth: int | None = 15,
        min_samples_split: int = 10,
        min_samples_leaf: int = 5,
        max_samples: float | None = 0.2,
        random_state: int = 42,
        n_jobs: int = 1,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_samples = max_samples
        self.random_state = random_state
        self.n_jobs = n_jobs

    def _make_inner(self) -> MultiOutputRegressor:
        forest = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_samples=self.max_samples,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        return MultiOutputRegressor(forest, n_jobs=1)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestPoseRegressor":
        self.model_ = self._make_inner()
        self.model_.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "model_")
        return self.model_.predict(X)


class DecisionTreePoseRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        max_depth: int | None = 15,
        min_samples_split: int = 10,
        min_samples_leaf: int = 5,
        random_state: int = 42,
        n_jobs: int = 1,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.n_jobs = n_jobs

    def _make_inner(self) -> MultiOutputRegressor:
        tree = DecisionTreeRegressor(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
        )
        return MultiOutputRegressor(tree, n_jobs=self.n_jobs)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreePoseRegressor":
        self.model_ = self._make_inner()
        self.model_.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "model_")
        return self.model_.predict(X)


class ChunkEnsembleRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, base_estimator: BaseEstimator, n_chunks: int = 10):
        self.base_estimator = base_estimator
        self.n_chunks = n_chunks
        self.estimators_ = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ChunkEnsembleRegressor":
        self.estimators_ = []
        chunk_size = max(1, len(X) // self.n_chunks)
        with tqdm(total=self.n_chunks, desc=f"Chunk Ensembling ({self.n_chunks}x)", leave=False) as pbar:
            for i in range(self.n_chunks):
                start = i * chunk_size
                end = (i + 1) * chunk_size if i < self.n_chunks - 1 else len(X)
                X_chunk = X[start:end]
                y_chunk = y[start:end]
                
                est = clone(self.base_estimator)
                est.fit(X_chunk, y_chunk)
                self.estimators_.append(est)
                
                # Force memory flush after each chunk training
                gc.collect()
                pbar.update(1)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "estimators_")
        preds = np.array([est.predict(X) for est in self.estimators_])
        return np.mean(preds, axis=0)


def build_model(model_name: str, random_state: int, n_jobs: int, **params) -> BaseEstimator:
    name = model_name.strip().lower()
    shared = {"random_state": random_state, "n_jobs": n_jobs}
    if name == "adaboost":
        return AdaBoostPoseRegressor(**shared, **params)
    if name == "xgboost":
        return XGBoostPoseRegressor(**shared, **params)
    if name == "randomforest":
        return RandomForestPoseRegressor(**shared, **params)
    if name == "decisiontree":
        return DecisionTreePoseRegressor(**shared, **params)
    raise ValueError(f"Unsupported model: {model_name}")


def pose_to_angle_targets(y_pose: np.ndarray) -> np.ndarray:
    y_pose = np.asarray(y_pose, dtype=np.float64)
    angles = np.degrees(np.arccos(np.clip(y_pose[:, 3:], -1.0, 1.0)))
    return np.column_stack([y_pose[:, :3], angles]).astype(np.float32)


def pose_to_cosine_targets(y_pose: np.ndarray) -> np.ndarray:
    y_pose = np.asarray(y_pose, dtype=np.float64)
    cosines = np.cos(np.deg2rad(y_pose[:, 3:]))
    return np.column_stack([y_pose[:, :3], cosines]).astype(np.float32)


def to_target_format(y_pose: np.ndarray, target_format: str) -> np.ndarray:
    if target_format == "cosine":
        return np.asarray(y_pose, dtype=np.float32)
    if target_format == "angle":
        return pose_to_angle_targets(y_pose)
    raise ValueError(f"Unsupported target format: {target_format}")


def target_columns(target_format: str) -> list[str]:
    if target_format == "cosine":
        return POSE_COLUMNS
    if target_format == "angle":
        return ANGLE_POSE_COLUMNS
    raise ValueError(f"Unsupported target format: {target_format}")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, target_format: str) -> dict:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    pos_diff_mm = (y_true[:, :3] - y_pred[:, :3]) * 1000.0
    pos_error = np.linalg.norm(pos_diff_mm, axis=1)

    if target_format == "cosine":
        true_angles = np.degrees(np.arccos(np.clip(y_true[:, 3:], -1.0, 1.0)))
        pred_angles = np.degrees(np.arccos(np.clip(y_pred[:, 3:], -1.0, 1.0)))
        aux_diff = y_true[:, 3:] - y_pred[:, 3:]
        aux_mae_key = "cos_mae_per_axis"
        aux_rmse_key = "cos_rmse_per_axis"
    elif target_format == "angle":
        true_angles = y_true[:, 3:]
        pred_angles = y_pred[:, 3:]
        aux_diff = true_angles - pred_angles
        aux_mae_key = "angle_mae_per_axis_deg"
        aux_rmse_key = "angle_rmse_per_axis_deg"
    else:
        raise ValueError(f"Unsupported target format: {target_format}")

    angle_diff = true_angles - pred_angles
    ori_error = np.linalg.norm(angle_diff, axis=1)

    r2_targets = r2_score(y_true, y_pred, multioutput="raw_values")

    return {
        "mean_pos_error_mm": float(pos_error.mean()),
        "std_pos_error_mm": float(pos_error.std(ddof=0)),
        "rmse_pos_mm": float(np.sqrt(np.mean(pos_error**2))),
        "mean_ori_error_deg": float(ori_error.mean()),
        "std_ori_error_deg": float(ori_error.std(ddof=0)),
        "rmse_ori_deg": float(np.sqrt(np.mean(ori_error**2))),
        aux_mae_key: np.mean(np.abs(aux_diff), axis=0).astype(float),
        aux_rmse_key: np.sqrt(np.mean(aux_diff**2, axis=0)).astype(float),
        "r2_per_target": np.asarray(r2_targets, dtype=float),
        "r2_mean": float(np.mean(r2_targets)),
        "pos_error_per_sample": pos_error,
        "ori_error_per_sample": ori_error,
    }


def plot_error_figures(y_true: np.ndarray, y_pred: np.ndarray, metrics: dict, output_root: str) -> list[str]:
    output_dir = os.path.dirname(output_root) or "."
    stem = os.path.splitext(os.path.basename(output_root))[0]
    os.makedirs(output_dir, exist_ok=True)

    sample_index = np.arange(len(y_true))
    pos_true_mm = y_true[:, :3] * 1000.0
    pos_pred_mm = y_pred[:, :3] * 1000.0
    pos_error = metrics["pos_error_per_sample"]
    ori_error = metrics["ori_error_per_sample"]

    saved_paths: list[str] = []

    scatter_path = os.path.join(output_dir, f"{stem}_scatter.png")
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pos_true_mm[:, 0], pos_true_mm[:, 1], pos_true_mm[:, 2], s=8, alpha=0.55, label="Ground truth")
    ax.scatter(pos_pred_mm[:, 0], pos_pred_mm[:, 1], pos_pred_mm[:, 2], s=8, alpha=0.55, marker="^", label="Prediction")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title("Pose Prediction on Test Split")
    ax.legend(loc="upper left")
    ax.text2D(
        0.02,
        0.02,
        (
            f"Mean position error: {metrics['mean_pos_error_mm']:.4f} mm\n"
            f"RMSE position: {metrics['rmse_pos_mm']:.4f} mm\n"
            f"Mean orientation error: {metrics['mean_ori_error_deg']:.4f} deg\n"
            f"RMSE orientation: {metrics['rmse_ori_deg']:.4f} deg\n"
            f"Mean R2: {metrics['r2_mean']:.4f}"
        ),
        transform=ax.transAxes,
        fontsize=8.5,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#94A3B8", alpha=0.9),
    )
    fig.tight_layout()
    fig.savefig(scatter_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(scatter_path)

    pos_path = os.path.join(output_dir, f"{stem}_pos_error.png")
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(sample_index, pos_error, color="#2563EB", linewidth=0.9, alpha=0.85)
    ax.axhline(metrics["mean_pos_error_mm"], color="#F59E0B", linestyle="--", linewidth=1.4)
    ax.axhline(metrics["rmse_pos_mm"], color="#DC2626", linestyle=":", linewidth=1.4)
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Position error (mm)")
    ax.set_title("Position Error per Sample")
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    fig.savefig(pos_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(pos_path)

    ori_path = os.path.join(output_dir, f"{stem}_ori_error.png")
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(sample_index, ori_error, color="#7C3AED", linewidth=0.9, alpha=0.85)
    ax.axhline(metrics["mean_ori_error_deg"], color="#F59E0B", linestyle="--", linewidth=1.4)
    ax.axhline(metrics["rmse_ori_deg"], color="#DC2626", linestyle=":", linewidth=1.4)
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Orientation error (deg)")
    ax.set_title("Orientation Error per Sample")
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    fig.savefig(ori_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(ori_path)

    # Position Error Histogram
    pos_hist_path = os.path.join(output_dir, f"{stem}_pos_hist.png")
    fig, ax = plt.subplots(figsize=(8, 5))
    mu, std = norm.fit(pos_error)
    ax.hist(pos_error, bins=50, density=True, alpha=0.6, color="#3B82F6")
    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    ax.plot(x, p, 'k', linewidth=2)
    ax.axvline(mu, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mu:.2f} mm')
    ax.axvline(mu + 2*std, color='orange', linestyle='dashed', linewidth=2, label=f'+2σ: {mu+2*std:.2f} mm')
    ax.set_xlabel("Position error (mm)")
    ax.set_ylabel("Density")
    ax.set_title(f"Position Error Distribution (μ={mu:.2f}, σ={std:.2f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(pos_hist_path, dpi=150)
    plt.close(fig)
    saved_paths.append(pos_hist_path)

    # Orientation Error Histogram
    ori_hist_path = os.path.join(output_dir, f"{stem}_ori_hist.png")
    fig, ax = plt.subplots(figsize=(8, 5))
    mu_ori, std_ori = norm.fit(ori_error)
    ax.hist(ori_error, bins=50, density=True, alpha=0.6, color="#7C3AED")
    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu_ori, std_ori)
    ax.plot(x, p, 'k', linewidth=2)
    ax.axvline(mu_ori, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mu_ori:.2f} deg')
    ax.axvline(mu_ori + 2*std_ori, color='orange', linestyle='dashed', linewidth=2, label=f'+2σ: {mu_ori+2*std_ori:.2f} deg')
    ax.set_xlabel("Orientation error (deg)")
    ax.set_ylabel("Density")
    ax.set_title(f"Orientation Error Distribution (μ={mu_ori:.2f}, σ={std_ori:.2f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ori_hist_path, dpi=150)
    plt.close(fig)
    saved_paths.append(ori_hist_path)

    return saved_paths


def plot_trajectory(traj_name: str, y_true_pose: np.ndarray, y_pred_pose: np.ndarray, metrics: dict, output_dir: str, target_format: str = "angle") -> str:
    os.makedirs(output_dir, exist_ok=True)
    traj_path = os.path.join(output_dir, f"traj_{traj_name}.png")
    
    pos_true_mm = y_true_pose[:, :3] * 1000.0
    pos_pred_mm = y_pred_pose[:, :3] * 1000.0
    pos_error = metrics["pos_error_per_sample"]
    
    fig = plt.figure(figsize=(16, 6))
    
    # 3D Line Plot
    ax1 = fig.add_subplot(121, projection="3d")
    
    # Limits for shadows
    min_x, max_x = np.min(pos_true_mm[:, 0]), np.max(pos_true_mm[:, 0])
    min_y, max_y = np.min(pos_true_mm[:, 1]), np.max(pos_true_mm[:, 1])
    min_z, max_z = np.min(pos_true_mm[:, 2]), np.max(pos_true_mm[:, 2])
    span = max(10, max([max_x - min_x, max_y - min_y, max_z - min_z]))
    
    ax1.set_xlim(min_x - span*0.1, max_x + span*0.1)
    ax1.set_ylim(min_y - span*0.1, max_y + span*0.1)
    ax1.set_zlim(min_z - span*0.1, max_z + span*0.1)
    
    # Ground truth shadows
    ax1.plot(pos_true_mm[:, 0], pos_true_mm[:, 1], zs=min_z - span*0.1, zdir='z', color='gray', alpha=0.3, linestyle=':')
    ax1.plot(pos_true_mm[:, 0], pos_true_mm[:, 2], zs=max_y + span*0.1, zdir='y', color='gray', alpha=0.3, linestyle=':')
    ax1.plot(pos_true_mm[:, 1], pos_true_mm[:, 2], zs=min_x - span*0.1, zdir='x', color='gray', alpha=0.3, linestyle=':')
    
    ax1.plot(pos_true_mm[:, 0], pos_true_mm[:, 1], pos_true_mm[:, 2], color="black", label="Ground truth", linewidth=2)
    ax1.plot(pos_pred_mm[:, 0], pos_pred_mm[:, 1], pos_pred_mm[:, 2], color="#E83E8C", label="Prediction", linewidth=2, linestyle="--")
    
    # Orientation Tripods (Paper style)
    is_angle = (target_format == "angle")
    rot_true = build_rotation_batch(
        np.cos(np.deg2rad(y_true_pose[:, 3])) if is_angle else y_true_pose[:, 3],
        np.cos(np.deg2rad(y_true_pose[:, 4])) if is_angle else y_true_pose[:, 4],
        np.cos(np.deg2rad(y_true_pose[:, 5])) if is_angle else y_true_pose[:, 5]
    )
    
    step = max(1, len(pos_true_mm) // 20)
    length = span * 0.1
    
    for i in range(0, len(pos_true_mm), step):
        p = pos_true_mm[i]
        R = rot_true[i]
        ax1.quiver(p[0], p[1], p[2], R[0,0], R[1,0], R[2,0], color='r', length=length, normalize=True, alpha=0.6)
        ax1.quiver(p[0], p[1], p[2], R[0,1], R[1,1], R[2,1], color='g', length=length, normalize=True, alpha=0.6)
        ax1.quiver(p[0], p[1], p[2], R[0,2], R[1,2], R[2,2], color='b', length=length, normalize=True, alpha=0.6)
    ax1.set_xlabel("X (mm)")
    ax1.set_ylabel("Y (mm)")
    ax1.set_zlabel("Z (mm)")
    ax1.set_title(f"Trajectory: {traj_name}")
    ax1.legend()
    
    # Error along trajectory
    ax2 = fig.add_subplot(122)
    sample_index = np.arange(len(pos_error))
    mu, std = np.mean(pos_error), np.std(pos_error)
    ax2.plot(sample_index, pos_error, color="#3B82F6", linewidth=1.5)
    ax2.fill_between(sample_index, mu - std, mu + std, color="#3B82F6", alpha=0.2, label="±1σ")
    ax2.axhline(mu, color="red", linestyle="--", label=f"Mean: {mu:.2f} mm")
    ax2.set_xlabel("Sample index")
    ax2.set_ylabel("Position error (mm)")
    ax2.set_title("Position Error Along Trajectory")
    ax2.legend()
    
    fig.tight_layout()
    fig.savefig(traj_path, dpi=150)
    plt.close(fig)
    return traj_path


def plot_all_models_trajectory(traj_name: str, y_true_pose: np.ndarray, model_preds: dict[str, np.ndarray], output_dir: str, target_format: str = "angle") -> str:
    os.makedirs(output_dir, exist_ok=True)
    traj_path = os.path.join(output_dir, f"traj_conclusion_{traj_name}.png")
    
    pos_true_mm = y_true_pose[:, :3] * 1000.0
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")
    
    min_x, max_x = np.min(pos_true_mm[:, 0]), np.max(pos_true_mm[:, 0])
    min_y, max_y = np.min(pos_true_mm[:, 1]), np.max(pos_true_mm[:, 1])
    min_z, max_z = np.min(pos_true_mm[:, 2]), np.max(pos_true_mm[:, 2])
    span = max(10, max([max_x - min_x, max_y - min_y, max_z - min_z]))
    
    ax.set_xlim(min_x - span*0.1, max_x + span*0.1)
    ax.set_ylim(min_y - span*0.1, max_y + span*0.1)
    ax.set_zlim(min_z - span*0.1, max_z + span*0.1)
    
    # Ground truth shadows
    ax.plot(pos_true_mm[:, 0], pos_true_mm[:, 1], zs=min_z - span*0.1, zdir='z', color='gray', alpha=0.3, linestyle=':')
    ax.plot(pos_true_mm[:, 0], pos_true_mm[:, 2], zs=max_y + span*0.1, zdir='y', color='gray', alpha=0.3, linestyle=':')
    ax.plot(pos_true_mm[:, 1], pos_true_mm[:, 2], zs=min_x - span*0.1, zdir='x', color='gray', alpha=0.3, linestyle=':')
    
    # Ground truth
    ax.plot(pos_true_mm[:, 0], pos_true_mm[:, 1], pos_true_mm[:, 2], color="black", label="Ground truth", linewidth=3)
    
    colors = ["#EF4444", "#3B82F6", "#10B981", "#F59E0B", "#8B5CF6", "#EC4899"]
    for i, (model_name, y_pred_pose) in enumerate(model_preds.items()):
        pos_pred_mm = y_pred_pose[:, :3] * 1000.0
        color = colors[i % len(colors)]
        ax.plot(pos_pred_mm[:, 0], pos_pred_mm[:, 1], pos_pred_mm[:, 2], color=color, label=model_name, linewidth=2, linestyle="--", alpha=0.8)
        
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title(f"Conclusion: All Models vs Ground Truth ({traj_name})")
    ax.legend(loc="best")
    
    fig.tight_layout()
    fig.savefig(traj_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return traj_path


def _rot_x(angle_deg: float) -> np.ndarray:
    angle_rad = np.deg2rad(angle_deg)
    cval = np.cos(angle_rad)
    sval = np.sin(angle_rad)
    return np.array([[1.0, 0.0, 0.0], [0.0, cval, -sval], [0.0, sval, cval]], dtype=np.float64)


def _rot_y(angle_deg: float) -> np.ndarray:
    angle_rad = np.deg2rad(angle_deg)
    cval = np.cos(angle_rad)
    sval = np.sin(angle_rad)
    return np.array([[cval, 0.0, sval], [0.0, 1.0, 0.0], [-sval, 0.0, cval]], dtype=np.float64)


def _rot_z(angle_deg: float) -> np.ndarray:
    angle_rad = np.deg2rad(angle_deg)
    cval = np.cos(angle_rad)
    sval = np.sin(angle_rad)
    return np.array([[cval, -sval, 0.0], [sval, cval, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def build_tx_runtime() -> list[dict]:
    runtime_specs = []
    for spec in TX_SPECS:
        alpha_deg, beta_deg, gamma_deg = spec["euler_deg"]
        rotation = _rot_z(gamma_deg) @ _rot_y(beta_deg) @ _rot_x(alpha_deg)
        direction = rotation @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
        area_tx = np.pi * spec["radius_m"] ** 2
        moment_mag = spec["turns"] * 1.0 * area_tx
        runtime_specs.append(
            {
                "pos_m": spec["pos_m"],
                "m_vec": (moment_mag * direction).astype(np.float64),
                "omega": 2.0 * np.pi * spec["freq_hz"],
            }
        )
    return runtime_specs


def iter_roi_blocks(config: ROIConfig):
    x_values = np.linspace(config.x_min, config.x_max, config.nx, dtype=np.float32)
    y_values = np.linspace(config.y_min, config.y_max, config.ny, dtype=np.float32)
    z_values = np.linspace(config.z_min, config.z_max, config.nz, dtype=np.float32)
    roll_deg = np.linspace(config.angle_min_deg, config.angle_max_deg, config.n_roll, dtype=np.float32)
    pitch_deg = np.linspace(config.angle_min_deg, config.angle_max_deg, config.n_pitch, dtype=np.float32)
    yaw_deg = np.linspace(config.angle_min_deg, config.angle_max_deg, config.n_yaw, dtype=np.float32)

    cos_roll = np.cos(np.deg2rad(roll_deg)).astype(np.float32)
    cos_pitch = np.cos(np.deg2rad(pitch_deg)).astype(np.float32)
    cos_yaw = np.cos(np.deg2rad(yaw_deg)).astype(np.float32)

    for x_value in x_values:
        y_grid, z_grid, roll_grid, pitch_grid, yaw_grid = np.meshgrid(
            y_values,
            z_values,
            cos_roll,
            cos_pitch,
            cos_yaw,
            indexing="ij",
        )
        block_size = y_grid.size
        yield np.column_stack(
            [
                np.full(block_size, x_value, dtype=np.float32),
                y_grid.ravel(),
                z_grid.ravel(),
                roll_grid.ravel(),
                pitch_grid.ravel(),
                yaw_grid.ravel(),
            ]
        )


def save_roi_grid(output_path: str, config: ROIConfig) -> str:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    start_time = time.time()
    written_rows = 0
    with open(output_path, "w", newline="", buffering=1 << 20) as handle:
        writer = csv.writer(handle)
        writer.writerow(POSE_COLUMNS)
        with tqdm(total=config.total_points, desc="Building ROI grid") as pbar:
            for block in iter_roi_blocks(config):
                writer.writerows(block.tolist())
                written_rows += len(block)
                pbar.update(len(block))
    size_mb = os.path.getsize(output_path) / 1e6
    log_message(f"[ROI] saved {output_path}  rows={written_rows:,}  size={size_mb:.1f} MB")
    return output_path


def plot_roi_region(config: ROIConfig, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(42)

    # 1. Spatial ROI
    spatial_path = os.path.join(output_dir, "roi_spatial.png")
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    x_vals = np.linspace(config.x_min, config.x_max, config.nx)
    y_vals = np.linspace(config.y_min, config.y_max, config.ny)
    z_vals = np.linspace(config.z_min, config.z_max, config.nz)
    
    # Sample points for visualization
    X, Y, Z = np.meshgrid(x_vals, y_vals, z_vals, indexing="ij")
    pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    if len(pts) > 200:
        pts = rng.choice(pts, size=200, replace=False)
    
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=10, alpha=0.3, label="Sampled grid points", c="blue")
    
    for tx in TX_SPECS:
        ax.scatter(tx["pos_m"][0], tx["pos_m"][1], tx["pos_m"][2], marker="*", s=100, c="red", label="TX Coil")
    
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Spatial ROI Grid Coverage")
    fig.tight_layout()
    fig.savefig(spatial_path, dpi=150)
    plt.close(fig)

    # 2. Orientation coverage
    ori_path = os.path.join(output_dir, "roi_orientations.png")
    fig, ax = plt.subplots(figsize=(7, 6))
    roll_vals = np.linspace(config.angle_min_deg, config.angle_max_deg, config.n_roll)
    pitch_vals = np.linspace(config.angle_min_deg, config.angle_max_deg, config.n_pitch)
    yaw_vals = np.linspace(config.angle_min_deg, config.angle_max_deg, config.n_yaw)
    
    R, P, Y = np.meshgrid(roll_vals, pitch_vals, yaw_vals, indexing="ij")
    ori_pts = np.column_stack([R.ravel(), P.ravel(), Y.ravel()])
    
    sc = ax.scatter(ori_pts[:, 0], ori_pts[:, 1], c=ori_pts[:, 2], cmap="viridis", alpha=0.5, s=20)
    plt.colorbar(sc, label="Yaw (deg)")
    ax.set_xlabel("Roll (deg)")
    ax.set_ylabel("Pitch (deg)")
    ax.set_title("Orientation Grid Coverage in ROI")
    fig.tight_layout()
    fig.savefig(ori_path, dpi=150)
    plt.close(fig)
    
    log_message(f"[ROI] plots saved to {output_dir}")


def extract_pose_targets(labels_df: pd.DataFrame | "pl.DataFrame", target_format: str) -> np.ndarray:
    if all(col in labels_df.columns for col in POSE_COLUMNS):
        pose_targets = labels_df.select(POSE_COLUMNS).to_numpy().astype(np.float32)
    elif all(col in labels_df.columns for col in ANGLE_POSE_COLUMNS):
        angle_targets = labels_df.select(ANGLE_POSE_COLUMNS).to_numpy().astype(np.float32)
        pose_targets = pose_to_cosine_targets(angle_targets)
    elif len(labels_df.columns) == 6:
        pose_targets = labels_df.to_numpy().astype(np.float32)
    else:
        raise ValueError("Unexpected label format.")
    return to_target_format(pose_targets, target_format)


def compute_B(position_batch: np.ndarray, tx_runtime: dict) -> np.ndarray:
    relative = position_batch - tx_runtime["pos_m"]
    radius = np.linalg.norm(relative, axis=1, keepdims=True)
    unit = relative / radius
    magnetic_moment = tx_runtime["m_vec"]
    unit_dot_moment = (unit @ magnetic_moment).reshape(-1, 1)
    return (MU0 / (4.0 * np.pi)) / radius**3 * (3.0 * unit_dot_moment * unit - magnetic_moment)


def build_rotation_batch(cos_roll: np.ndarray, cos_pitch: np.ndarray, cos_yaw: np.ndarray) -> np.ndarray:
    roll = np.arccos(np.clip(cos_roll, -1.0, 1.0))
    pitch = np.arccos(np.clip(cos_pitch, -1.0, 1.0))
    yaw = np.arccos(np.clip(cos_yaw, -1.0, 1.0))

    sin_roll, cos_roll_vals = np.sin(roll), np.cos(roll)
    sin_pitch, cos_pitch_vals = np.sin(pitch), np.cos(pitch)
    sin_yaw, cos_yaw_vals = np.sin(yaw), np.cos(yaw)

    batch_size = len(roll)
    rotation = np.zeros((batch_size, 3, 3), dtype=np.float64)
    rotation[:, 0, 0] = cos_yaw_vals * cos_pitch_vals
    rotation[:, 0, 1] = cos_yaw_vals * sin_pitch * sin_roll - sin_yaw * cos_roll_vals
    rotation[:, 0, 2] = cos_yaw_vals * sin_pitch * cos_roll_vals + sin_yaw * sin_roll
    rotation[:, 1, 0] = sin_yaw * cos_pitch_vals
    rotation[:, 1, 1] = sin_yaw * sin_pitch * sin_roll + cos_yaw_vals * cos_roll_vals
    rotation[:, 1, 2] = sin_yaw * sin_pitch * cos_roll_vals - cos_yaw_vals * sin_roll
    rotation[:, 2, 0] = -sin_pitch
    rotation[:, 2, 1] = cos_pitch_vals * sin_roll
    rotation[:, 2, 2] = cos_pitch_vals * cos_roll_vals
    return rotation


def save_emf_dataset(
    roi_csv: str,
    output_path: str,
    snr_db: float,
    chunk_size: int,
    random_seed: int,
    noise_model: str = "per_channel",
    total_rows: int | None = None,
) -> str:
    tx_runtime = build_tx_runtime()
    rng = np.random.default_rng(random_seed)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    start_time = time.time()
    expected_total_rows = total_rows
    written_rows = 0
    with open(output_path, "w", newline="", buffering=1 << 20) as handle:
        with tqdm(total=expected_total_rows, desc="Simulating EMF dataset") as pbar:
            for chunk_idx, chunk in enumerate(pd.read_csv(roi_csv, chunksize=chunk_size), start=1):
                position = chunk[["x", "y", "z"]].to_numpy(dtype=np.float64)
                cos_roll = chunk["cos_roll"].to_numpy(dtype=np.float64)
                cos_pitch = chunk["cos_pitch"].to_numpy(dtype=np.float64)
                cos_yaw = chunk["cos_yaw"].to_numpy(dtype=np.float64)

                rotation_batch = build_rotation_batch(cos_roll, cos_pitch, cos_yaw)
                emf_matrix = np.zeros((len(chunk), 9), dtype=np.float64)

                col_idx = 0
                for tx_runtime_spec in tx_runtime:
                    magnetic_flux = compute_B(position, tx_runtime_spec)
                    for rx_spec in RX_SPECS:
                        rx_axis = rotation_batch @ rx_spec["axis0"]
                        dot_value = np.einsum("ni,ni->n", magnetic_flux, rx_axis)
                        emf_matrix[:, col_idx] = (
                            rx_spec["turns"] * tx_runtime_spec["omega"] * rx_spec["area_m2"] * dot_value
                        )
                        col_idx += 1

                if noise_model == "per_sample":
                    signal_power = np.mean(emf_matrix**2, axis=1, keepdims=True)
                elif noise_model == "per_channel":
                    signal_power = emf_matrix**2
                else:
                    raise ValueError(f"Unknown noise model: {noise_model}")

                noise_power = signal_power / (10.0 ** (snr_db / 10.0))
                noisy_emf = emf_matrix + np.sqrt(noise_power) * rng.standard_normal(emf_matrix.shape)

                pd.DataFrame(noisy_emf.astype(np.float32), columns=EMF_COLUMNS).to_csv(
                    handle,
                    index=False,
                    header=False,
                    float_format="%.8g",
                )

                written_rows += len(chunk)
                pbar.update(len(chunk))

    size_mb = os.path.getsize(output_path) / 1e6
    log_message(f"[EMF] saved {output_path}  rows={written_rows:,}  size={size_mb:.1f} MB")
    return output_path


def analyze_noise(
    roi_csv: str,
    snr_db: float,
    noise_model: str,
    simulation_seed: int,
    output_dir: str,
    n_samples: int = 10_000,
) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    log_message("[Noise] Analyzing simulated noise...")
    rng = np.random.default_rng(simulation_seed)
    
    chunk = next(pd.read_csv(roi_csv, chunksize=n_samples))
    position = chunk[["x", "y", "z"]].to_numpy(dtype=np.float64)
    cos_roll = chunk["cos_roll"].to_numpy(dtype=np.float64)
    cos_pitch = chunk["cos_pitch"].to_numpy(dtype=np.float64)
    cos_yaw = chunk["cos_yaw"].to_numpy(dtype=np.float64)

    rotation_batch = build_rotation_batch(cos_roll, cos_pitch, cos_yaw)
    tx_runtime = build_tx_runtime()
    emf_clean = np.zeros((len(chunk), 9), dtype=np.float64)

    col_idx = 0
    for tx_runtime_spec in tx_runtime:
        magnetic_flux = compute_B(position, tx_runtime_spec)
        for rx_spec in RX_SPECS:
            rx_axis = rotation_batch @ rx_spec["axis0"]
            dot_value = np.einsum("ni,ni->n", magnetic_flux, rx_axis)
            emf_clean[:, col_idx] = (
                rx_spec["turns"] * tx_runtime_spec["omega"] * rx_spec["area_m2"] * dot_value
            )
            col_idx += 1

    if noise_model == "per_sample":
        signal_power = np.mean(emf_clean**2, axis=1, keepdims=True)
    elif noise_model == "per_channel":
        signal_power = emf_clean**2
    else:
        raise ValueError(f"Unknown noise model: {noise_model}")

    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    emf_noisy = emf_clean + np.sqrt(noise_power) * rng.standard_normal(emf_clean.shape)
    residuals = emf_noisy - emf_clean

    report = {
        "target_snr_db": snr_db,
        "noise_model": noise_model,
        "channels": {}
    }

    # Plot SNR per channel
    realized_snr = np.zeros(9)
    for c in range(9):
        noise_mean = float(np.mean(residuals[:, c]))
        noise_std = float(np.std(residuals[:, c]))
        signal_mean = float(np.mean(np.abs(emf_clean[:, c])))
        r_snr = float(20 * np.log10(signal_mean / noise_std)) if noise_std > 0 else 0.0
        realized_snr[c] = r_snr
        
        subset = residuals[:5000, c]
        stat, p = shapiro(subset)
        
        report["channels"][EMF_COLUMNS[c]] = {
            "noise_mean": noise_mean,
            "noise_std": noise_std,
            "signal_mean": signal_mean,
            "realized_snr_db": r_snr,
            "shapiro_stat": float(stat),
            "shapiro_p": float(p)
        }

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(EMF_COLUMNS, realized_snr, color="#3B82F6")
    ax.axhline(snr_db, color="red", linestyle="--", label=f"Target SNR ({snr_db} dB)")
    ax.set_xticks(range(len(EMF_COLUMNS)))
    ax.set_xticklabels(EMF_COLUMNS, rotation=45, ha="right")
    ax.set_ylabel("Realized SNR (dB)")
    ax.set_title("Realized SNR per Channel")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "snr_per_channel.png"), dpi=150)
    plt.close(fig)

    # Plot Noise Histograms
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    for c in range(9):
        ax = axes[c]
        data = residuals[:, c]
        mu, std = norm.fit(data)
        ax.hist(data, bins=50, density=True, alpha=0.6, color="#10B981")
        xmin, xmax = ax.get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)
        ax.plot(x, p, 'k', linewidth=2)
        ax.set_title(f"{EMF_COLUMNS[c]}\nμ={mu:.2e}, σ={std:.2e}", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "noise_hist_grid.png"), dpi=150)
    plt.close(fig)
    
    with open(os.path.join(output_dir, "noise_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    log_message(f"[Noise] Analysis saved to {output_dir}")
    return report




def _add_trajectory_noise(emf: np.ndarray, snr_db: float, seed: int, noise_model: str) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if noise_model == "per_sample":
        signal_power = np.mean(emf**2, axis=1, keepdims=True)
    else:
        signal_power = emf**2
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    return emf + np.sqrt(noise_power) * rng.standard_normal(emf.shape).astype(np.float32)


def generate_trajectories(
    config: ROIConfig,
    snr_db: float | None = None,
    simulation_seed: int = 42,
    noise_model: str = "per_channel",
) -> dict:
    """Build synthetic evaluation trajectories outside the train/val/test splits."""
    trajectories = {}
    
    t = np.linspace(0, 4 * np.pi, 500)
    r = np.linspace(0, 0.08, 500)
    
    # 2D Square (Spiral equivalent mathematically in bounds)
    x_2d = r * np.cos(t)
    y_2d = config.y_min + (config.y_max - config.y_min) / 2.0 + r * np.sin(t)
    z_2d = np.full_like(t, config.z_min + (config.z_max - config.z_min) / 2.0)
    
    cos_roll = np.ones_like(t)
    cos_pitch = np.ones_like(t)
    cos_yaw = np.cos(t)
    y_2d_pose = np.column_stack([x_2d, y_2d, z_2d, cos_roll, cos_pitch, cos_yaw])
    
    # 3D Circular Helix
    x_3d = 0.08 * np.cos(t)
    y_3d = config.y_min + (config.y_max - config.y_min) / 2.0 + 0.08 * np.sin(t)
    z_3d = config.z_min + (config.z_max - config.z_min) * (t / (4 * np.pi))
    y_3d_pose = np.column_stack([x_3d, y_3d, z_3d, cos_roll, cos_pitch, cos_yaw])
    
    # 3D Conical Spiral (shrinks from bottom to top)
    t_conical = np.linspace(0, 20 * np.pi, 1000)
    r_conical = 0.08 * (1.0 - t_conical / (20 * np.pi))
    x_conical = r_conical * np.cos(t_conical)
    y_conical = config.y_min + (config.y_max - config.y_min) / 2.0 + r_conical * np.sin(t_conical)
    z_conical = config.z_min + (config.z_max - config.z_min) * (t_conical / (20 * np.pi))
    
    cos_roll_conical = np.ones_like(t_conical)
    cos_pitch_conical = np.ones_like(t_conical)
    cos_yaw_conical = np.cos(t_conical)
    
    y_conical_pose = np.column_stack([x_conical, y_conical, z_conical, cos_roll_conical, cos_pitch_conical, cos_yaw_conical])
    
    def pose_to_emf(pose_batch: np.ndarray) -> np.ndarray:
        position = pose_batch[:, :3]
        cr, cp, cy = pose_batch[:, 3], pose_batch[:, 4], pose_batch[:, 5]
        rotation_batch = build_rotation_batch(cr, cp, cy)
        tx_runtime = build_tx_runtime()
        emf_matrix = np.zeros((len(pose_batch), 9), dtype=np.float64)
        col_idx = 0
        for tx_spec in tx_runtime:
            magnetic_flux = compute_B(position, tx_spec)
            for rx_spec in RX_SPECS:
                rx_axis = rotation_batch @ rx_spec["axis0"]
                dot_value = np.einsum("ni,ni->n", magnetic_flux, rx_axis)
                emf_matrix[:, col_idx] = rx_spec["turns"] * tx_spec["omega"] * rx_spec["area_m2"] * dot_value
                col_idx += 1
        return emf_matrix.astype(np.float32)

    emf_2d = pose_to_emf(y_2d_pose)
    emf_3d = pose_to_emf(y_3d_pose)
    emf_conical = pose_to_emf(y_conical_pose)

    if snr_db is not None:
        emf_2d = _add_trajectory_noise(emf_2d, snr_db, simulation_seed, noise_model)
        emf_3d = _add_trajectory_noise(emf_3d, snr_db, simulation_seed + 1, noise_model)
        emf_conical = _add_trajectory_noise(emf_conical, snr_db, simulation_seed + 2, noise_model)

    trajectories["2d_spiral"] = {"X": emf_2d, "y_pose": y_2d_pose.astype(np.float32)}
    trajectories["3d_helix"] = {"X": emf_3d, "y_pose": y_3d_pose.astype(np.float32)}
    trajectories["3d_conical_spiral"] = {"X": emf_conical, "y_pose": y_conical_pose.astype(np.float32)}
    return trajectories


def load_dataset(
    emf_csv: str,
    roi_csv: str,
    max_samples: int = -1,
    sample_strategy: str = "head",
    target_format: str = "cosine",
) -> tuple[np.ndarray, np.ndarray]:
    """Load aligned EMF features and pose labels.

    `sample_strategy="head"` loads only the first `max_samples` rows for a true
    quick run. `sample_strategy="random"` preserves random subsampling behavior
    but must load the full dataset first.
    """
    import polars as pl

    n_rows = max_samples if max_samples > 0 and sample_strategy == "head" else None

    log_message(f"[Load] EMF  : {emf_csv} (Polars)")
    X = pl.read_csv(emf_csv, has_header=False, n_rows=n_rows).to_numpy().astype(np.float32)

    log_message(f"[Load] Pose : {roi_csv} (Polars)")
    labels_df = pl.read_csv(roi_csv, n_rows=n_rows)

    try:
        y = extract_pose_targets(labels_df, target_format=target_format)
    except ValueError as exc:
        raise ValueError(f"Unexpected label format in {roi_csv}.") from exc

    if X.shape[0] != y.shape[0]:
        raise ValueError(f"Mismatched rows: X={X.shape[0]} y={y.shape[0]}")
    if X.shape[1] != 9 or y.shape[1] != 6:
        raise ValueError(f"Unexpected shapes: X={X.shape} y={y.shape}")

    if max_samples > 0 and sample_strategy == "random" and len(X) > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X), size=max_samples, replace=False)
        idx.sort()
        X, y = X[idx], y[idx]
        log_message(f"[Load] Applied random subset: {max_samples:,} rows")
    elif max_samples > 0 and sample_strategy == "head":
        log_message(f"[Load] Applied head subset: {len(X):,} rows")

    log_message(f"[Load] X={X.shape}  y={y.shape}  target_format={target_format}")
    return X, y


def build_run_summary(args: argparse.Namespace, models: list[str], seeds: list[int], roi_config: ROIConfig) -> dict:
    sample_mode = "full dataset"
    if args.max_samples > 0:
        if args.max_samples_strategy == "head":
            sample_mode = f"first {args.max_samples:,} rows"
        else:
            sample_mode = f"random {args.max_samples:,} rows after full load"

    sample_note = None
    if args.max_samples > 0 and args.max_samples_strategy == "head":
        sample_note = "Fastest option, but it may be biased if the CSV rows are ordered by pose."
    elif args.max_samples > 0 and args.max_samples_strategy == "random":
        sample_note = "More representative than head sampling, but it requires a full dataset load."

    stages = [
        "Optionally rebuild ROI and simulated EMF CSV files.",
        "Generate two synthetic trajectory test paths (2d_spiral, 3d_helix).",
        "Load aligned EMF features and pose labels.",
        f"Convert pose targets to {args.target_format} representation.",
        "Split data into train/validation/test sets for each seed.",
        "Fit the EMF preprocessor on training data only.",
        "Optionally run GridSearchCV on a search subset.",
        "Train the final model on the full training split.",
        "Evaluate on validation, test, and synthetic trajectories.",
        "Save metrics, artifacts, and aggregate summaries.",
    ]
    if args.compare_only:
        stages = [
            "Reuse the existing ROI and EMF CSV files.",
            "Load a bounded dataset subset for quick comparison.",
            f"Convert pose targets to {args.target_format} representation.",
            "Split data into train/validation/test sets.",
            "Fit the preprocessor and train each model once.",
            "Compare models using validation and test metrics only.",
            "Write compact comparison CSV files.",
        ]

    return {
        "purpose": "Train regressors that map 9-channel EMF measurements to 6-DoF pose.",
        "inputs": {
            "roi_csv": args.roi_csv,
            "emf_csv": args.emf_csv,
            "sample_mode": sample_mode,
            "sample_note": sample_note,
            "target_format": args.target_format,
            "preset": args.preset,
            "allow_heavy_run": args.allow_heavy_run,
            "compare_only": args.compare_only,
        },
        "outputs": {
            "output_dir": args.output_dir,
            "artifacts": [
                "run_config.json",
                "per_seed_metrics.csv",
                "aggregate_metrics.csv",
                "seed_<seed>/<model>/artifact.pkl",
                "seed_<seed>/<model>/metrics.json",
            ],
        },
        "stages": stages,
        "models": models,
        "seeds": seeds,
        "search": {
            "enabled": not args.skip_search,
            "search_samples": args.search_samples,
            "search_folds": args.search_folds,
            "search_n_jobs": args.search_n_jobs,
        },
        "preprocessing": {
            "signed_log": not args.no_signed_log,
            "standardize": not args.no_standardize,
        },
        "roi_grid": asdict(roi_config),
    }


def print_run_summary(summary: dict) -> None:
    log_message("=" * 72)
    log_message("[Run] Purpose")
    log_message(f"  {summary['purpose']}")
    log_message("[Run] Inputs")
    log_message(f"  ROI CSV       : {summary['inputs']['roi_csv']}")
    log_message(f"  EMF CSV       : {summary['inputs']['emf_csv']}")
    log_message(f"  Sample mode   : {summary['inputs']['sample_mode']}")
    log_message(f"  Target format : {summary['inputs']['target_format']}")
    log_message(f"  Preset        : {summary['inputs']['preset']}")
    log_message(f"  Heavy override: {'on' if summary['inputs']['allow_heavy_run'] else 'off'}")
    log_message(f"  Compare only  : {'on' if summary['inputs']['compare_only'] else 'off'}")
    if summary["inputs"]["sample_note"]:
        log_message(f"  Sample note   : {summary['inputs']['sample_note']}")
    log_message("[Run] Models / Seeds")
    log_message(f"  Models        : {', '.join(summary['models'])}")
    log_message(f"  Seeds         : {', '.join(str(seed) for seed in summary['seeds'])}")
    log_message("[Run] Search")
    log_message(
        "  Grid search   : "
        f"{'enabled' if summary['search']['enabled'] else 'disabled'}"
        f" (samples={summary['search']['search_samples']}, "
        f"folds={summary['search']['search_folds']}, "
        f"jobs={summary['search']['search_n_jobs']})"
    )
    log_message("[Run] Preprocessing")
    log_message(
        "  Steps         : "
        f"signed_log={'on' if summary['preprocessing']['signed_log'] else 'off'}, "
        f"standardize={'on' if summary['preprocessing']['standardize'] else 'off'}"
    )
    log_message("[Run] Stages")
    for idx, stage in enumerate(summary["stages"], start=1):
        log_message(f"  {idx}. {stage}")
    log_message("[Run] Outputs")
    log_message(f"  Output dir    : {summary['outputs']['output_dir']}")
    log_message("=" * 72)


def split_dataset(
    X: np.ndarray,
    y: np.ndarray,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if val_ratio <= 0.0 or test_ratio <= 0.0 or val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio and test_ratio must be > 0 and sum to < 1.")

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=val_ratio + test_ratio,
        random_state=seed,
        shuffle=True,
    )
    val_fraction = val_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        train_size=val_fraction,
        random_state=seed,
        shuffle=True,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def pose_loss_for_search(y_true: np.ndarray, y_pred: np.ndarray, target_format: str) -> float:
    metrics = compute_metrics(y_true, y_pred, target_format=target_format)
    return metrics["rmse_pos_mm"] + metrics["rmse_ori_deg"]


def ensure_parent_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if hasattr(value, "__dataclass_fields__"):
        return to_jsonable(asdict(value))
    return value


def save_json(path: str, payload: dict) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(to_jsonable(payload), handle, indent=2, sort_keys=True)


def save_pickle(path: str, payload) -> None:
    ensure_parent_dir(path)
    with open(path, "wb") as handle:
        pickle.dump(payload, handle)


def subsample_for_search(X: np.ndarray, y: np.ndarray, max_samples: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if max_samples <= 0 or len(X) <= max_samples:
        return X, y

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(X), size=max_samples, replace=False)
    return X[indices], y[indices]


def save_predictions(path: str, y_true: np.ndarray, y_pred: np.ndarray, target_format: str) -> None:
    ensure_parent_dir(path)
    columns = target_columns(target_format)
    pred_columns = [f"pred_{column}" for column in columns]
    df = pd.DataFrame(np.column_stack([y_true, y_pred]), columns=columns + pred_columns)
    df.to_csv(path, index=False)


def maybe_build_data(args: argparse.Namespace, roi_config: ROIConfig) -> None:
    need_roi = args.rebuild_data or not os.path.exists(args.roi_csv)
    need_emf = args.rebuild_data or not os.path.exists(args.emf_csv)

    if need_roi:
        log_message(f"[Data] Building ROI grid -> {args.roi_csv}")
        save_roi_grid(args.roi_csv, roi_config)
    else:
        log_message(f"[Data] Reusing ROI grid   -> {args.roi_csv}")

    if need_emf:
        log_message(f"[Data] Building EMF CSV   -> {args.emf_csv}")
        save_emf_dataset(
            roi_csv=args.roi_csv,
            output_path=args.emf_csv,
            snr_db=args.snr_db,
            chunk_size=args.chunk_size,
            random_seed=args.simulation_seed,
            total_rows=roi_config.total_points,
        )
    else:
        log_message(f"[Data] Reusing EMF CSV    -> {args.emf_csv}")


def parse_csv_list(raw_value: str | list) -> list[str]:
    if isinstance(raw_value, list):
        # Already split by argparse via nargs="+"
        tokens = []
        for item in raw_value:
            tokens.extend(item.split(","))
        return [t.strip() for t in tokens if t.strip()]
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def parse_seeds(raw_value: str) -> list[int]:
    seeds = [int(token) for token in parse_csv_list(raw_value)]
    if not seeds:
        raise ValueError("At least one seed is required.")
    return seeds


def parse_models(raw_value: str) -> list[str]:
    models = [token.lower() for token in parse_csv_list(raw_value)]
    for model_name in models:
        if model_name not in PARAM_GRIDS:
            supported = ", ".join(sorted(PARAM_GRIDS))
            raise ValueError(f"Unsupported model '{model_name}'. Supported models: {supported}.")
    if not models:
        raise ValueError("At least one model is required.")
    if "xgboost" in models and XGBRegressor is None:
        raise ImportError("xgboost is not installed, but xgboost was requested in --models.")
    return models


def build_runtime_risk_messages(args: argparse.Namespace, roi_config: ROIConfig, models: list[str], seeds: list[int]) -> list[str]:
    risks: list[str] = []
    model_seed_runs = len(models) * len(seeds)
    total_grid_points = roi_config.total_points

    if args.rebuild_data and total_grid_points > 500_000:
        risks.append(
            f"--rebuild-data with ROI size {total_grid_points:,} will regenerate a very large simulated dataset."
        )
    if args.max_samples <= 0 and model_seed_runs >= 4:
        risks.append(
            f"full-dataset training across {len(models)} models x {len(seeds)} seeds can be RAM- and CPU-heavy."
        )
    if not args.skip_search and model_seed_runs >= 4 and args.max_samples <= 0:
        risks.append(
            f"GridSearchCV is enabled for {model_seed_runs} model/seed runs on the full dataset."
        )
    if args.max_samples_strategy == "random" and args.max_samples > 100_000:
        risks.append(
            "--max-samples-strategy=random with > 100k samples can be slow after full dataset load."
        )
    if args.search_n_jobs > 2:
        risks.append(f"--search-n-jobs={args.search_n_jobs} increases concurrent memory usage.")
    return risks


def validate_runtime_safety(args: argparse.Namespace, roi_config: ROIConfig, models: list[str], seeds: list[int]) -> None:
    risks = build_runtime_risk_messages(args, roi_config, models, seeds)
    if not risks:
        return
    if args.allow_heavy_run:
        for risk in risks:
            log_message(f"[Safety] warning: {risk}")
        return

    message = "Blocked a heavy configuration to avoid crashing the machine:\n"
    message += "\n".join(f"- {risk}" for risk in risks)
    message += "\nUse --preset safe, or pass --allow-heavy-run if you really want this."
    raise RuntimeError(message)


def apply_compare_only_mode(args: argparse.Namespace) -> None:
    if not args.compare_only:
        return
    args.rebuild_data = False
    args.skip_search = True
    args.skip_plots = True
    args.save_predictions = False
    args.search_n_jobs = 1
    args.search_verbose = 0
    if args.max_samples <= 0:
        args.max_samples = 50_000
        args.max_samples_strategy = "head"


def save_search_results(path: str, search: GridSearchCV) -> None:
    cv_df = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
    cv_df.to_csv(path, index=False)


def compact_metrics(metrics: dict) -> dict:
    return {key: value for key, value in metrics.items() if not key.endswith("_per_sample")}


def flatten_metric_row(model_name: str, seed: int, best_params: dict, result: dict) -> dict:
    row = {
        "model": model_name,
        "seed": seed,
        "search_time_s": result["search_time_s"],
        "training_time_s": result["training_time_s"],
        "inference_time_ms_per_sample": result["inference_time_ms_per_sample"],
        "search_pool_size": result["search_pool_size"],
        "best_params": json.dumps(to_jsonable(best_params), sort_keys=True),
    }
    for split_name in ("val_metrics", "test_metrics"):
        prefix = split_name.replace("_metrics", "")
        for metric_name in (
            "mean_pos_error_mm",
            "std_pos_error_mm",
            "rmse_pos_mm",
            "mean_ori_error_deg",
            "std_ori_error_deg",
            "rmse_ori_deg",
            "r2_mean",
        ):
            row[f"{prefix}_{metric_name}"] = result[split_name][metric_name]
    return row


def aggregate_summary(per_seed_df: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "search_time_s",
        "training_time_s",
        "inference_time_ms_per_sample",
        "val_rmse_pos_mm",
        "val_rmse_ori_deg",
        "val_r2_mean",
        "test_rmse_pos_mm",
        "test_rmse_ori_deg",
        "test_r2_mean",
    ]
    grouped = per_seed_df.groupby("model")[metric_columns].agg(["mean", "std"]).reset_index()
    grouped.columns = [
        "model" if column == ("model", "") else f"{column[0]}_{column[1]}"
        for column in grouped.columns.to_flat_index()
    ]
    return grouped.reset_index(drop=True)


def build_compare_ranking(aggregate_df: pd.DataFrame) -> pd.DataFrame:
    ranking_df = aggregate_df.copy()
    ranking_df["val_score_mean"] = (
        ranking_df["val_rmse_pos_mm_mean"] + ranking_df["val_rmse_ori_deg_mean"]
    )
    ranking_df = ranking_df.sort_values(
        ["val_score_mean", "test_rmse_pos_mm_mean", "test_rmse_ori_deg_mean", "model"]
    ).reset_index(drop=True)
    ranking_df.insert(0, "rank", np.arange(1, len(ranking_df) + 1))
    return ranking_df


def train_and_evaluate_model(
    trajectories: dict,
    model_name: str,
    seed: int,
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    preprocessor: EMFPreprocessor,
    args: argparse.Namespace,
    run_dir: str,
) -> tuple[dict, dict]:
    if args.skip_search:
        best_params = copy.deepcopy(CONFERENCE_REFERENCE_PARAMS[model_name])
        search_time_s = 0.0
        search_pool_size = 0
        log_message(f"[{model_name}][seed={seed}] using conference reference params: {best_params}")
    else:
        X_search, y_search = subsample_for_search(X_train, y_train, args.search_samples, seed)
        estimator = build_model(model_name, random_state=seed, n_jobs=args.model_n_jobs)
        grid = PARAM_GRIDS[model_name]
        pose_scorer = make_scorer(
            partial(pose_loss_for_search, target_format=args.target_format),
            greater_is_better=False,
        )
        search = GridSearchCV(
            estimator=estimator,
            param_grid=grid,
            scoring=pose_scorer,
            cv=args.search_folds,
            n_jobs=args.search_n_jobs,
            pre_dispatch="1*n_jobs",
            refit=True,
            verbose=args.search_verbose,
        )
        search_start = time.perf_counter()
        search.fit(X_search, y_search)
        search_time_s = time.perf_counter() - search_start
        search_pool_size = len(X_search)
        best_params = copy.deepcopy(search.best_params_)
        log_message(
            f"[{model_name}][seed={seed}] search best score={search.best_score_:.4f}"
            f" params={best_params} pool={search_pool_size:,}"
        )
        if not args.compare_only:
            save_search_results(os.path.join(run_dir, "search_results.csv"), search)

    estimator = build_model(
        model_name=model_name,
        random_state=seed,
        n_jobs=args.model_n_jobs,
        **best_params,
    )

    estimator = ChunkEnsembleRegressor(base_estimator=estimator, n_chunks=10)

    train_start = time.perf_counter()
    estimator.fit(X_train, y_train)
    training_time_s = time.perf_counter() - train_start

    val_pred = estimator.predict(X_val).astype(np.float64)
    test_start = time.perf_counter()
    test_pred = estimator.predict(X_test).astype(np.float64)
    inference_time_ms_per_sample = (time.perf_counter() - test_start) / len(X_test) * 1000.0

    val_metrics = compute_metrics(y_val, val_pred, target_format=args.target_format)
    test_metrics = compute_metrics(y_test, test_pred, target_format=args.target_format)

    traj_metrics = {}
    if not args.compare_only:
        for traj_name, traj_data in trajectories.items():
            X_traj_prep = preprocessor.transform(traj_data["X"])
            y_traj_pred = estimator.predict(X_traj_prep)
            y_traj_true = to_target_format(traj_data["y_pose"], args.target_format)
            traj_metrics[traj_name] = compute_metrics(y_traj_true, y_traj_pred, target_format=args.target_format)
    plot_paths = []
    if not args.compare_only and not args.skip_plots:
        plot_paths.extend(plot_error_figures(
            y_true=y_test,
            y_pred=test_pred,
            metrics=test_metrics,
            output_root=os.path.join(run_dir, "test"),
        ))
        traj_preds = {}
        for traj_name, metrics in traj_metrics.items():
            y_traj_true = to_target_format(trajectories[traj_name]["y_pose"], args.target_format)
            X_traj_prep = preprocessor.transform(trajectories[traj_name]["X"])
            y_traj_pred = estimator.predict(X_traj_prep)
            traj_preds[traj_name] = y_traj_pred
            traj_path = plot_trajectory(
                traj_name=traj_name,
                y_true_pose=y_traj_true,
                y_pred_pose=y_traj_pred,
                metrics=metrics,
                output_dir=run_dir,
                target_format=args.target_format,
            )
            plot_paths.append(traj_path)

    if not args.compare_only and args.save_predictions:
        save_predictions(
            os.path.join(run_dir, "test_predictions.csv"),
            y_test,
            test_pred,
            target_format=args.target_format,
        )

    if not args.compare_only:
        save_pickle(
            os.path.join(run_dir, "artifact.pkl"),
            {
                "seed": seed,
                "model_name": model_name,
                "best_params": best_params,
                "estimator": estimator,
                "preprocessor": preprocessor,
                "target_format": args.target_format,
                "training_time_s": training_time_s,
                "search_time_s": search_time_s,
                "inference_time_ms_per_sample": inference_time_ms_per_sample,
            },
        )

    result = {
        "search_time_s": search_time_s,
        "training_time_s": training_time_s,
        "inference_time_ms_per_sample": inference_time_ms_per_sample,
        "search_pool_size": search_pool_size,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "traj_metrics": traj_metrics,
        "traj_preds": traj_preds if not args.skip_plots and not args.compare_only else {},
        "plot_paths": plot_paths,
    }
    if not args.compare_only:
        save_json(
            os.path.join(run_dir, "metrics.json"),
            {
                "search_time_s": search_time_s,
                "training_time_s": training_time_s,
                "inference_time_ms_per_sample": inference_time_ms_per_sample,
                "search_pool_size": search_pool_size,
                "val_metrics": compact_metrics(val_metrics),
                "test_metrics": compact_metrics(test_metrics),
                "traj_metrics": {k: compact_metrics(v) for k, v in traj_metrics.items()},
                "plot_paths": plot_paths,
            },
        )
    return best_params, result


def run_pipeline(args: argparse.Namespace) -> None:
    roi_config = ROIConfig(
        x_min=args.x_min,
        x_max=args.x_max,
        y_min=args.y_min,
        y_max=args.y_max,
        z_min=args.z_min,
        z_max=args.z_max,
        angle_min_deg=args.angle_min_deg,
        angle_max_deg=args.angle_max_deg,
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        n_roll=args.n_roll,
        n_pitch=args.n_pitch,
        n_yaw=args.n_yaw,
    )

    seeds = parse_seeds(args.seeds)
    models = parse_models(args.models)
    validate_runtime_safety(args, roi_config, models, seeds)
    run_summary = build_run_summary(args, models, seeds, roi_config)

    os.makedirs(args.output_dir, exist_ok=True)
    print_run_summary(run_summary)
    save_json(
        os.path.join(args.output_dir, "run_config.json"),
        {
            "args": vars(args),
            "run_summary": run_summary,
            "roi_config": roi_config,
            "conference_reference_params": CONFERENCE_REFERENCE_PARAMS,
            "param_grids": PARAM_GRIDS,
        },
    )

    maybe_build_data(args, roi_config)
    
    if not args.skip_plots:
        plot_roi_region(roi_config, args.output_dir)
        
    if not args.skip_noise_analysis and not args.compare_only:
        analyze_noise(
            roi_csv=args.roi_csv,
            snr_db=args.snr_db,
            noise_model=args.noise_model,
            simulation_seed=args.simulation_seed,
            output_dir=os.path.join(args.output_dir, "noise_analysis"),
            n_samples=args.noise_samples,
        )

    trajectories = {} if args.compare_only else generate_trajectories(
        roi_config,
        snr_db=None if args.no_trajectory_noise else args.snr_db,
        simulation_seed=args.simulation_seed,
        noise_model=args.noise_model,
    )
    X, y = load_dataset(
        emf_csv=args.emf_csv,
        roi_csv=args.roi_csv,
        max_samples=args.max_samples,
        sample_strategy=args.max_samples_strategy,
        target_format=args.target_format,
    )

    per_seed_rows: list[dict] = []
    aggregate_traj_preds = {}
    MEMORY_GUARD.start()
    try:
        for seed in tqdm(seeds, desc="Seeds"):
            log_message("=" * 72)
            log_message(f"[Seed] {seed}")
            gc.collect()
            X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
                X=X,
                y=y,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                seed=seed,
            )
            log_message(
                f"[Split] train={len(X_train):,}  val={len(X_val):,}  test={len(X_test):,}"
                f"  ratios={(1.0 - args.val_ratio - args.test_ratio):.2f}/{args.val_ratio:.2f}/{args.test_ratio:.2f}"
            )

            preprocessor = EMFPreprocessor(
                use_signed_log=not args.no_signed_log,
                use_standardize=not args.no_standardize,
            )
            X_train_proc = preprocessor.fit_transform(X_train)
            X_val_proc = preprocessor.transform(X_val)
            X_test_proc = preprocessor.transform(X_test)

            seed_dir = os.path.join(args.output_dir, f"seed_{seed}")
            if not args.compare_only:
                save_pickle(
                    os.path.join(seed_dir, "preprocessor.pkl"),
                    {
                        "seed": seed,
                        "preprocessor": preprocessor,
                    },
                )

            for model_name in tqdm(models, desc=f"Models (Seed {seed})", leave=False):
                model_dir = os.path.join(seed_dir, model_name)
                if not args.compare_only:
                    os.makedirs(model_dir, exist_ok=True)

                if args.use_wandb:
                    if not WANDB_AVAILABLE:
                        raise ImportError("wandb is not installed. Please install it or remove --use-wandb")
                    wandb.init(
                        project=args.wandb_project,
                        group="emf-pipeline",
                        name=f"{model_name}-seed{seed}",
                        config={
                            "model": model_name,
                            "seed": seed,
                            "target_format": args.target_format,
                            "noise_model": args.noise_model,
                            "snr_db": args.snr_db,
                            "max_samples": args.max_samples,
                            "max_samples_strategy": args.max_samples_strategy,
                        },
                        reinit=True
                    )

                best_params, result = train_and_evaluate_model(
                    trajectories=trajectories,
                    model_name=model_name,
                    seed=seed,
                    X_train=X_train_proc,
                    X_val=X_val_proc,
                    X_test=X_test_proc,
                    y_train=y_train.astype(np.float32),
                    y_val=y_val.astype(np.float32),
                    y_test=y_test.astype(np.float32),
                    preprocessor=preprocessor,
                    args=args,
                    run_dir=model_dir,
                )
                per_seed_rows.append(flatten_metric_row(model_name, seed, best_params, result))
                
                if seed == args.seeds[0] and result.get("traj_preds"):
                    for traj_name, preds in result["traj_preds"].items():
                        if traj_name not in aggregate_traj_preds:
                            aggregate_traj_preds[traj_name] = {}
                        aggregate_traj_preds[traj_name][model_name] = preds
                
                if args.use_wandb:
                    wandb.config.update({"best_params": best_params})
                    wandb.log({"val": result["val_metrics"], "test": result["test_metrics"]})
                    for traj_name, metrics in result.get("traj_metrics", {}).items():
                        wandb.log({f"traj_{traj_name}": metrics})
                    
                    if result.get("plot_paths"):
                        images = {os.path.basename(path): wandb.Image(path) for path in result["plot_paths"]}
                        wandb.log(images)
                        
                    wandb.finish()



                log_message(
                    f"[{model_name}][seed={seed}] "
                    f"val_rmse=({result['val_metrics']['rmse_pos_mm']:.4f} mm, {result['val_metrics']['rmse_ori_deg']:.4f} deg)  "
                    f"test_rmse=({result['test_metrics']['rmse_pos_mm']:.4f} mm, {result['test_metrics']['rmse_ori_deg']:.4f} deg)  "
                    f"R2={result['test_metrics']['r2_mean']:.4f}  "
                    f"train={result['training_time_s']:.2f}s  "
                    f"infer={result['inference_time_ms_per_sample']:.4f} ms/sample"
                )
                gc.collect()
                gc.collect()
                
        # Generate conclusion image
        if not args.skip_plots and not args.compare_only and aggregate_traj_preds:
            for traj_name, model_preds in aggregate_traj_preds.items():
                y_true_pose = to_target_format(trajectories[traj_name]["y_pose"], args.target_format)
                con_path = plot_all_models_trajectory(traj_name, y_true_pose, model_preds, args.output_dir, args.target_format)
                log_message(f"[{traj_name}] Saved conclusion plot to {con_path}")
                if args.use_wandb:
                    try:
                        wandb.init(project="emf_localization", name=f"conclusion_{traj_name}", reinit=True)
                        wandb.log({f"conclusion_{traj_name}": wandb.Image(con_path)})
                        wandb.finish()
                    except Exception:
                        pass
    finally:
        MEMORY_GUARD.stop()

    per_seed_df = pd.DataFrame(per_seed_rows)
    per_seed_path = os.path.join(args.output_dir, "per_seed_metrics.csv")
    per_seed_df.to_csv(per_seed_path, index=False)

    aggregate_df = aggregate_summary(per_seed_df)
    aggregate_path = os.path.join(args.output_dir, "aggregate_metrics.csv")
    aggregate_df.to_csv(aggregate_path, index=False)
    compare_ranking_df = build_compare_ranking(aggregate_df)
    compare_ranking_path = os.path.join(args.output_dir, "model_ranking.csv")
    compare_ranking_df.to_csv(compare_ranking_path, index=False)
    save_json(
        os.path.join(args.output_dir, "aggregate_metrics.json"),
        {
            "per_seed_csv": per_seed_path,
            "aggregate_csv": aggregate_path,
            "model_ranking_csv": compare_ranking_path,
            "aggregate_rows": aggregate_df.to_dict(orient="records"),
        },
    )

    if args.compare_only:
        log_message("=" * 72)
        log_message("[Summary] Model ranking")
        for _, row in compare_ranking_df.iterrows():
            log_message(
                f"#{int(row['rank'])} {row['model']}: "
                f"val_score={row['val_score_mean']:.4f} "
                f"(val_pos={row['val_rmse_pos_mm_mean']:.4f}, val_ori={row['val_rmse_ori_deg_mean']:.4f}) "
                f"test_pos={row['test_rmse_pos_mm_mean']:.4f} "
                f"test_ori={row['test_rmse_ori_deg_mean']:.4f}"
            )
    else:
        log_message("=" * 72)
        log_message("[Summary] Mean ± Std across seeds")
        for _, row in aggregate_df.iterrows():
            log_message(
                f"{row['model']}: "
                f"test_pos_rmse={row['test_rmse_pos_mm_mean']:.4f}±{row['test_rmse_pos_mm_std']:.4f} mm, "
                f"test_ori_rmse={row['test_rmse_ori_deg_mean']:.4f}±{row['test_rmse_ori_deg_std']:.4f} deg, "
                f"R2={row['test_r2_mean_mean']:.4f}±{row['test_r2_mean_std']:.4f}, "
                f"infer={row['inference_time_ms_per_sample_mean']:.4f}±{row['inference_time_ms_per_sample_std']:.4f} ms/sample"
            )
    log_message(f"[Summary] per-seed metrics -> {per_seed_path}")
    log_message(f"[Summary] aggregate metrics -> {aggregate_path}")
    log_message(f"[Summary] model ranking -> {compare_ranking_path}")


def build_arg_parser(defaults: dict | None = None) -> argparse.ArgumentParser:
    defaults = defaults or PRESET_DEFAULTS["safe"]
    parser = argparse.ArgumentParser(description="Unified EMF localization pipeline.")

    parser.add_argument("--roi-csv", type=str, default="roi_grid.csv")
    parser.add_argument("--emf-csv", type=str, default="emf_data.csv")
    parser.add_argument("--output-dir", type=str, default="runs/pipeline")
    parser.add_argument("--rebuild-data", action="store_true", help="Regenerate ROI and EMF CSV files.")
    parser.add_argument(
        "--compare-only",
        action="store_true",
        help="Skip rebuild/search/plots/trajectory extras and just compare model metrics quickly.",
    )
    parser.add_argument(
        "--allow-heavy-run",
        action="store_true",
        help="Bypass the safety guard for expensive full-data configurations.",
    )
    parser.add_argument("--snr-db", type=float, default=40.0)
    parser.add_argument("--chunk-size", type=int, default=50_000)
    parser.add_argument("--simulation-seed", type=int, default=42)

    parser.add_argument("--models", nargs="+", default=defaults["models"])
    parser.add_argument("--seeds", nargs="+", default=defaults["seeds"])
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--search-samples", type=int, default=defaults["search_samples"])
    parser.add_argument("--search-folds", type=int, default=defaults["search_folds"])
    parser.add_argument("--search-n-jobs", type=int, default=defaults["search_n_jobs"])
    parser.add_argument("--search-verbose", type=int, default=defaults["search_verbose"])
    parser.add_argument("--skip-search", action="store_true", default=defaults["skip_search"])
    parser.add_argument("--model-n-jobs", type=int, default=defaults["model_n_jobs"])
    parser.add_argument("--no-signed-log", action="store_true")
    parser.add_argument("--no-standardize", action="store_true")
    parser.add_argument("--skip-plots", action="store_true", default=defaults["skip_plots"])
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=defaults["max_samples"],
        help="Limit the dataset size for quick experiments. With --max-samples-strategy=head this loads only the first N rows.",
    )
    parser.add_argument(
        "--max-samples-strategy",
        choices=("head", "random"),
        default=defaults["max_samples_strategy"],
        help="Subset strategy for --max-samples. 'head' is fastest; 'random' preserves random sampling but loads the full dataset first.",
    )
    parser.add_argument(
        "--target-format",
        choices=("cosine", "angle"),
        default=defaults["target_format"],
        help="Orientation target representation used for training/evaluation. 'angle' is easier to interpret; 'cosine' matches the original label encoding.",
    )
    
    parser.add_argument("--noise-model", choices=("per_sample", "per_channel"), default="per_channel")
    parser.add_argument("--no-trajectory-noise", action="store_true")
    parser.add_argument("--noise-samples", type=int, default=10000)
    parser.add_argument("--skip-noise-analysis", action="store_true")
    
    parser.add_argument("--use-wandb", action="store_true", help="Enable tracking to Weights & Biases")
    parser.add_argument("--wandb-project", type=str, default="emf_localization", help="W&B project name")

    parser.add_argument("--x_min", type=float, default=ROIConfig.x_min)
    parser.add_argument("--x_max", type=float, default=ROIConfig.x_max)
    parser.add_argument("--y_min", type=float, default=ROIConfig.y_min)
    parser.add_argument("--y_max", type=float, default=ROIConfig.y_max)
    parser.add_argument("--z_min", type=float, default=ROIConfig.z_min)
    parser.add_argument("--z_max", type=float, default=ROIConfig.z_max)
    parser.add_argument("--angle_min_deg", type=float, default=ROIConfig.angle_min_deg)
    parser.add_argument("--angle_max_deg", type=float, default=ROIConfig.angle_max_deg)
    parser.add_argument("--nx", type=int, default=ROIConfig.nx)
    parser.add_argument("--ny", type=int, default=ROIConfig.ny)
    parser.add_argument("--nz", type=int, default=ROIConfig.nz)
    parser.add_argument("--n_roll", type=int, default=ROIConfig.n_roll)
    parser.add_argument("--n_pitch", type=int, default=ROIConfig.n_pitch)
    parser.add_argument("--n_yaw", type=int, default=ROIConfig.n_yaw)

    return parser


def parse_args_with_presets() -> argparse.Namespace:
    preset_parser = argparse.ArgumentParser(add_help=False)
    preset_parser.add_argument("--preset", choices=tuple(PRESET_DEFAULTS), default="safe")
    preset_args, _ = preset_parser.parse_known_args()
    parser = build_arg_parser(defaults=PRESET_DEFAULTS[preset_args.preset])
    parser.add_argument(
        "--preset",
        choices=tuple(PRESET_DEFAULTS),
        default=preset_args.preset,
        help="Runtime preset. 'safe' is the default and is designed not to overload the machine.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args_with_presets()
    apply_compare_only_mode(args)
    run_pipeline(args)


if __name__ == "__main__":
    main()
