"""Unified end-to-end EMF localization pipeline.

This script consolidates the project into one conference-aligned workflow:
1. Generate the ROI grid if needed.
2. Simulate the 9-channel EMF dataset if needed.
3. Load the full CSV datasets into memory.
4. Split the data into train/validation/test using 70/15/15.
5. Fit a feature preprocessor on the training split only.
6. Tune and train AdaBoost and XGBoost regressors.
7. Evaluate RMSE/R2/timing, save plots, and aggregate results across seeds.

The implementation reuses the same core logic as the existing conference code
while restricting the modeling stage to AdaBoost and XGBoost only.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import pickle
import time
from dataclasses import asdict, dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import make_scorer, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.validation import check_is_fitted

try:
    from xgboost import XGBRegressor
except ImportError:  # pragma: no cover
    XGBRegressor = None


POSE_COLUMNS = ["x", "y", "z", "cos_roll", "cos_pitch", "cos_yaw"]
EMF_COLUMNS = [f"EMF_TX{tx_idx}_RX{rx_idx}" for tx_idx in range(1, 4) for rx_idx in range(1, 4)]
DEFAULT_SEEDS = [42, 43, 44, 45, 46]

# The paper explicitly states AdaBoost uses 30 estimators with linear loss and
# XGBoost uses depth 20 with 100 estimators. Remaining knobs follow the repo
# defaults when search is disabled.
CONFERENCE_REFERENCE_PARAMS = {
    "adaboost": {
        "estimator_depth": 20,
        "n_estimators": 30,
        "learning_rate": 0.1,
        "loss": "linear",
    },
    "xgboost": {
        "n_estimators": 100,
        "max_depth": 20,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1.0,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "gamma": 0.0,
    },
}

# Search spaces are based on implementation_plan.md, with the AdaBoost paper
# baseline of 30 estimators added to the candidate list.
PARAM_GRIDS = {
    "adaboost": {
        "estimator_depth": [15, 20, 30],
        "n_estimators": [30, 50, 100],
        "learning_rate": [0.05, 0.1, 0.5],
        "loss": ["linear"],
    },
    "xgboost": {
        "max_depth": [6, 12, 20],
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "min_child_weight": [1.0],
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
        estimator_depth: int = 20,
        n_estimators: int = 30,
        learning_rate: float = 0.1,
        loss: str = "linear",
        random_state: int = 42,
        n_jobs: int = 1,
    ):
        self.estimator_depth = estimator_depth
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.loss = loss
        self.random_state = random_state
        self.n_jobs = n_jobs

    def _make_inner(self) -> MultiOutputRegressor:
        base_tree = DecisionTreeRegressor(
            max_depth=self.estimator_depth,
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


def build_model(model_name: str, random_state: int, n_jobs: int, **params) -> BaseEstimator:
    name = model_name.strip().lower()
    shared = {"random_state": random_state, "n_jobs": n_jobs}
    if name == "adaboost":
        return AdaBoostPoseRegressor(**shared, **params)
    if name == "xgboost":
        return XGBoostPoseRegressor(**shared, **params)
    raise ValueError(f"Unsupported model: {model_name}")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    pos_diff_mm = (y_true[:, :3] - y_pred[:, :3]) * 1000.0
    pos_error = np.linalg.norm(pos_diff_mm, axis=1)

    true_angles = np.degrees(np.arccos(np.clip(y_true[:, 3:], -1.0, 1.0)))
    pred_angles = np.degrees(np.arccos(np.clip(y_pred[:, 3:], -1.0, 1.0)))
    angle_diff = true_angles - pred_angles
    ori_error = np.linalg.norm(angle_diff, axis=1)

    cos_diff = y_true[:, 3:] - y_pred[:, 3:]
    r2_targets = r2_score(y_true, y_pred, multioutput="raw_values")

    return {
        "mean_pos_error_mm": float(pos_error.mean()),
        "std_pos_error_mm": float(pos_error.std(ddof=0)),
        "rmse_pos_mm": float(np.sqrt(np.mean(pos_error**2))),
        "mean_ori_error_deg": float(ori_error.mean()),
        "std_ori_error_deg": float(ori_error.std(ddof=0)),
        "rmse_ori_deg": float(np.sqrt(np.mean(ori_error**2))),
        "cos_mae_per_axis": np.mean(np.abs(cos_diff), axis=0).astype(float),
        "cos_rmse_per_axis": np.sqrt(np.mean(cos_diff**2, axis=0)).astype(float),
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

    return saved_paths


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
        for block_index, block in enumerate(iter_roi_blocks(config), start=1):
            writer.writerows(block.tolist())
            written_rows += len(block)
            elapsed = time.time() - start_time
            pct = 100.0 * written_rows / config.total_points
            eta = elapsed * (100.0 - pct) / pct if pct > 0 else 0.0
            print(
                f"[ROI] block={block_index:>3}  rows={written_rows:>12,}/{config.total_points:,}"
                f"  ({pct:5.1f}%)  {elapsed:.1f}s  ETA={eta:.0f}s",
                flush=True,
            )
    size_mb = os.path.getsize(output_path) / 1e6
    print(f"[ROI] saved {output_path}  rows={written_rows:,}  size={size_mb:.1f} MB")
    return output_path


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


def save_emf_dataset(roi_csv: str, output_path: str, snr_db: float, chunk_size: int, random_seed: int) -> str:
    tx_runtime = build_tx_runtime()
    rng = np.random.default_rng(random_seed)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    start_time = time.time()
    total_rows = 0
    with open(output_path, "w", newline="", buffering=1 << 20) as handle:
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

            signal_power = np.mean(emf_matrix**2, axis=1, keepdims=True)
            noise_power = signal_power / (10.0 ** (snr_db / 10.0))
            noisy_emf = emf_matrix + np.sqrt(noise_power) * rng.standard_normal(emf_matrix.shape)

            pd.DataFrame(noisy_emf.astype(np.float32), columns=EMF_COLUMNS).to_csv(
                handle,
                index=False,
                header=False,
                float_format="%.8g",
            )

            total_rows += len(chunk)
            elapsed = time.time() - start_time
            print(f"[EMF] chunk={chunk_idx:>4}  rows={total_rows:>12,}  elapsed={elapsed:.1f}s", flush=True)

    size_mb = os.path.getsize(output_path) / 1e6
    print(f"[EMF] saved {output_path}  rows={total_rows:,}  size={size_mb:.1f} MB")
    return output_path


def load_dataset(emf_csv: str, roi_csv: str, chunk_size: int = 100_000) -> tuple[np.ndarray, np.ndarray]:
    import polars as pl
    
    print(f"[Load] EMF  : {emf_csv} (Polars)")
    X = pl.read_csv(emf_csv, has_header=False).to_numpy().astype(np.float32)

    print(f"[Load] Pose : {roi_csv} (Polars)")
    labels_df = pl.read_csv(roi_csv)
    
    if all(col in labels_df.columns for col in POSE_COLUMNS):
        y = labels_df.select(POSE_COLUMNS).to_numpy().astype(np.float32)
    elif len(labels_df.columns) == 6:
        y = labels_df.to_numpy().astype(np.float32)
    else:
        raise ValueError(f"Unexpected label format in {roi_csv}.")

    if X.shape[0] != y.shape[0]:
        raise ValueError(f"Mismatched rows: X={X.shape[0]} y={y.shape[0]}")
    if X.shape[1] != 9 or y.shape[1] != 6:
        raise ValueError(f"Unexpected shapes: X={X.shape} y={y.shape}")

    print(f"[Load] X={X.shape}  y={y.shape}")
    return X, y


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


def pose_loss_for_search(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    metrics = compute_metrics(y_true, y_pred)
    return metrics["rmse_pos_mm"] + metrics["rmse_ori_deg"]


POSE_SCORER = make_scorer(pose_loss_for_search, greater_is_better=False)


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


def save_predictions(path: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    ensure_parent_dir(path)
    pred_columns = [f"pred_{column}" for column in POSE_COLUMNS]
    df = pd.DataFrame(np.column_stack([y_true, y_pred]), columns=POSE_COLUMNS + pred_columns)
    df.to_csv(path, index=False)


def maybe_build_data(args: argparse.Namespace, roi_config: ROIConfig) -> None:
    need_roi = args.rebuild_data or not os.path.exists(args.roi_csv)
    need_emf = args.rebuild_data or not os.path.exists(args.emf_csv)

    if need_roi:
        print(f"[Data] Building ROI grid -> {args.roi_csv}")
        save_roi_grid(args.roi_csv, roi_config)
    else:
        print(f"[Data] Reusing ROI grid   -> {args.roi_csv}")

    if need_emf:
        print(f"[Data] Building EMF CSV   -> {args.emf_csv}")
        save_emf_dataset(
            roi_csv=args.roi_csv,
            output_path=args.emf_csv,
            snr_db=args.snr_db,
            chunk_size=args.chunk_size,
            random_seed=args.simulation_seed,
        )
    else:
        print(f"[Data] Reusing EMF CSV    -> {args.emf_csv}")


def parse_csv_list(raw_value: str) -> list[str]:
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
            raise ValueError(f"Unsupported model '{model_name}'. Use adaboost and/or xgboost.")
    if not models:
        raise ValueError("At least one model is required.")
    if "xgboost" in models and XGBRegressor is None:
        raise ImportError("xgboost is not installed, but xgboost was requested in --models.")
    return models


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


def train_and_evaluate_model(
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
        print(f"[{model_name}][seed={seed}] using conference reference params: {best_params}")
    else:
        X_search, y_search = subsample_for_search(X_train, y_train, args.search_samples, seed)
        estimator = build_model(model_name, random_state=seed, n_jobs=args.model_n_jobs)
        grid = PARAM_GRIDS[model_name]
        search = GridSearchCV(
            estimator=estimator,
            param_grid=grid,
            scoring=POSE_SCORER,
            cv=args.search_folds,
            n_jobs=args.search_n_jobs,
            refit=True,
            verbose=args.search_verbose,
        )
        search_start = time.perf_counter()
        search.fit(X_search, y_search)
        search_time_s = time.perf_counter() - search_start
        search_pool_size = len(X_search)
        best_params = copy.deepcopy(search.best_params_)
        print(
            f"[{model_name}][seed={seed}] search best score={search.best_score_:.4f}"
            f" params={best_params} pool={search_pool_size:,}"
        )
        save_search_results(os.path.join(run_dir, "search_results.csv"), search)

    estimator = build_model(
        model_name=model_name,
        random_state=seed,
        n_jobs=args.model_n_jobs,
        **best_params,
    )

    train_start = time.perf_counter()
    estimator.fit(X_train, y_train)
    training_time_s = time.perf_counter() - train_start

    val_pred = estimator.predict(X_val).astype(np.float64)
    test_start = time.perf_counter()
    test_pred = estimator.predict(X_test).astype(np.float64)
    inference_time_ms_per_sample = (time.perf_counter() - test_start) / len(X_test) * 1000.0

    val_metrics = compute_metrics(y_val, val_pred)
    test_metrics = compute_metrics(y_test, test_pred)
    plot_paths = []
    if not args.skip_plots:
        plot_paths = plot_error_figures(
            y_true=y_test,
            y_pred=test_pred,
            metrics=test_metrics,
            output_root=os.path.join(run_dir, "test"),
        )

    if args.save_predictions:
        save_predictions(os.path.join(run_dir, "test_predictions.csv"), y_test, test_pred)

    save_pickle(
        os.path.join(run_dir, "artifact.pkl"),
        {
            "seed": seed,
            "model_name": model_name,
            "best_params": best_params,
            "estimator": estimator,
            "preprocessor": preprocessor,
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
        "plot_paths": plot_paths,
    }
    save_json(
        os.path.join(run_dir, "metrics.json"),
        {
            "search_time_s": search_time_s,
            "training_time_s": training_time_s,
            "inference_time_ms_per_sample": inference_time_ms_per_sample,
            "search_pool_size": search_pool_size,
            "val_metrics": compact_metrics(val_metrics),
            "test_metrics": compact_metrics(test_metrics),
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

    os.makedirs(args.output_dir, exist_ok=True)
    save_json(
        os.path.join(args.output_dir, "run_config.json"),
        {
            "args": vars(args),
            "roi_config": roi_config,
            "conference_reference_params": CONFERENCE_REFERENCE_PARAMS,
            "param_grids": PARAM_GRIDS,
        },
    )

    maybe_build_data(args, roi_config)
    X, y = load_dataset(args.emf_csv, args.roi_csv)
    if args.max_samples > 0 and len(X) > args.max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X), size=args.max_samples, replace=False)
        X, y = X[idx], y[idx]

    per_seed_rows: list[dict] = []
    for seed in tqdm(seeds, desc="Seeds Execution", position=0):
        print("=" * 72)
        print(f"[Seed] {seed}")
        X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
            X=X,
            y=y,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=seed,
        )
        print(
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
        save_pickle(
            os.path.join(seed_dir, "preprocessor.pkl"),
            {
                "seed": seed,
                "preprocessor": preprocessor,
            },
        )

        for model_name in tqdm(models, desc=f"Models (Seed {seed})", leave=False, position=1):
            model_dir = os.path.join(seed_dir, model_name)
            os.makedirs(model_dir, exist_ok=True)

            best_params, result = train_and_evaluate_model(
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

            print(
                f"[{model_name}][seed={seed}] "
                f"val_rmse=({result['val_metrics']['rmse_pos_mm']:.4f} mm, {result['val_metrics']['rmse_ori_deg']:.4f} deg)  "
                f"test_rmse=({result['test_metrics']['rmse_pos_mm']:.4f} mm, {result['test_metrics']['rmse_ori_deg']:.4f} deg)  "
                f"R2={result['test_metrics']['r2_mean']:.4f}  "
                f"train={result['training_time_s']:.2f}s  "
                f"infer={result['inference_time_ms_per_sample']:.4f} ms/sample"
            )

    per_seed_df = pd.DataFrame(per_seed_rows)
    per_seed_path = os.path.join(args.output_dir, "per_seed_metrics.csv")
    per_seed_df.to_csv(per_seed_path, index=False)

    aggregate_df = aggregate_summary(per_seed_df)
    aggregate_path = os.path.join(args.output_dir, "aggregate_metrics.csv")
    aggregate_df.to_csv(aggregate_path, index=False)
    save_json(
        os.path.join(args.output_dir, "aggregate_metrics.json"),
        {
            "per_seed_csv": per_seed_path,
            "aggregate_csv": aggregate_path,
            "aggregate_rows": aggregate_df.to_dict(orient="records"),
        },
    )

    print("=" * 72)
    print("[Summary] Mean ± Std across seeds")
    for _, row in aggregate_df.iterrows():
        print(
            f"{row['model']}: "
            f"test_pos_rmse={row['test_rmse_pos_mm_mean']:.4f}±{row['test_rmse_pos_mm_std']:.4f} mm, "
            f"test_ori_rmse={row['test_rmse_ori_deg_mean']:.4f}±{row['test_rmse_ori_deg_std']:.4f} deg, "
            f"R2={row['test_r2_mean_mean']:.4f}±{row['test_r2_mean_std']:.4f}, "
            f"infer={row['inference_time_ms_per_sample_mean']:.4f}±{row['inference_time_ms_per_sample_std']:.4f} ms/sample"
        )
    print(f"[Summary] per-seed metrics -> {per_seed_path}")
    print(f"[Summary] aggregate metrics -> {aggregate_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified EMF localization pipeline using AdaBoost and XGBoost.")

    parser.add_argument("--roi-csv", type=str, default="roi_grid.csv")
    parser.add_argument("--emf-csv", type=str, default="emf_data.csv")
    parser.add_argument("--output-dir", type=str, default="runs/pipeline")
    parser.add_argument("--rebuild-data", action="store_true", help="Regenerate ROI and EMF CSV files.")
    parser.add_argument("--snr-db", type=float, default=40.0)
    parser.add_argument("--chunk-size", type=int, default=50_000)
    parser.add_argument("--simulation-seed", type=int, default=42)

    parser.add_argument("--models", type=str, default="adaboost,xgboost")
    parser.add_argument("--seeds", type=str, default="42,43,44,45,46")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--search-samples", type=int, default=100_000)
    parser.add_argument("--search-folds", type=int, default=3)
    parser.add_argument("--search-n-jobs", type=int, default=4)
    parser.add_argument("--search-verbose", type=int, default=3)
    parser.add_argument("--skip-search", action="store_true")
    parser.add_argument("--model-n-jobs", type=int, default=1)
    parser.add_argument("--no-signed-log", action="store_true")
    parser.add_argument("--no-standardize", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--max-samples", type=int, default=-1, help="Max total samples to load for quick testing")

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


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
