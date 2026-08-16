import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from calibration.report_utils import COLS, extract_emf_and_pose


def parse_named_csvs(items):
    parsed = []
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Expected name=path CSV argument, got: {item}")
        name, path = item.split("=", 1)
        if not name or not path:
            raise ValueError(f"Expected name=path CSV argument, got: {item}")
        parsed.append((name, path))
    return parsed


def target_from_pose_deg(poses):
    target = np.zeros((len(poses), 6), dtype=np.float32)
    target[:, :3] = poses[:, :3]
    target[:, 3:] = np.cos(np.radians(poses[:, 3:]))
    return target


def build_past_windows(emf_v, window_size):
    if window_size < 1:
        raise ValueError("window_size must be at least 1")
    emf_v = np.asarray(emf_v, dtype=np.float32)
    if emf_v.ndim != 2 or emf_v.shape[1] != 9:
        raise ValueError(f"Expected EMF shape (N, 9), got {emf_v.shape}")
    if len(emf_v) == 0:
        return np.empty((0, window_size * 9), dtype=np.float32)

    pad = np.repeat(emf_v[:1], window_size - 1, axis=0)
    padded = np.vstack([pad, emf_v])
    windows = np.empty((len(emf_v), window_size * 9), dtype=np.float32)
    for i in range(len(emf_v)):
        windows[i] = padded[i:i + window_size].reshape(-1)
    return windows


def frame_to_temporal_arrays(df, window_size):
    emf_v, poses = extract_emf_and_pose(df)
    return build_past_windows(emf_v, window_size), target_from_pose_deg(poses)


def load_temporal_csvs(named_csvs, window_size):
    xs = []
    ys = []
    counts = []
    for name, path in parse_named_csvs(named_csvs):
        df = pd.read_csv(path, header=None, names=COLS)
        x, y = frame_to_temporal_arrays(df, window_size)
        xs.append(x)
        ys.append(y)
        counts.append({"name": name, "path": path, "rows": len(df)})
    if not xs:
        raise ValueError("No CSV files provided")
    return np.vstack(xs).astype(np.float32), np.vstack(ys).astype(np.float32), counts
