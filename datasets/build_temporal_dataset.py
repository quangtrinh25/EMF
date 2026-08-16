import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.temporal_windows import load_temporal_csvs


def augment_windows(windows, rng, copies, noise_std_mV, gain_drift):
    batches = [windows.astype(np.float32)]
    for _ in range(copies):
        gains = rng.uniform(1.0 - gain_drift, 1.0 + gain_drift, size=windows.shape)
        noise = rng.normal(0.0, noise_std_mV * 1e-3, size=windows.shape)
        batches.append(np.maximum(windows * gains + noise, 0.0))
    return np.vstack(batches).astype(np.float32)


def write_counts(path, counts, role):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["role", "name", "path", "rows"])
        if f.tell() == 0:
            writer.writeheader()
        for row in counts:
            writer.writerow({"role": role, **row})


def main():
    parser = argparse.ArgumentParser(description="Build past-window temporal train/val datasets from real CSVs.")
    parser.add_argument("--train_csvs", nargs="+", required=True, help="Training CSVs as name=path.")
    parser.add_argument("--val_csvs", nargs="+", required=True, help="Validation CSVs as name=path.")
    parser.add_argument("--out_dir", default="data/temporal_balanced_easy")
    parser.add_argument("--window_size", type=int, default=5)
    parser.add_argument("--augment_copies", type=int, default=200)
    parser.add_argument("--noise_std_mV", type=float, default=0.5)
    parser.add_argument("--gain_drift", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    train_x, train_y, train_counts = load_temporal_csvs(args.train_csvs, args.window_size)
    val_x, val_y, val_counts = load_temporal_csvs(args.val_csvs, args.window_size)

    train_x_aug = augment_windows(train_x, rng, args.augment_copies, args.noise_std_mV, args.gain_drift)
    train_y_aug = np.tile(train_y, (args.augment_copies + 1, 1)).astype(np.float32)
    idx = rng.permutation(len(train_y_aug))

    np.savez(os.path.join(args.out_dir, "train.npz"), emf=train_x_aug[idx], target=train_y_aug[idx])
    np.savez(os.path.join(args.out_dir, "val.npz"), emf=val_x.astype(np.float32), target=val_y.astype(np.float32))
    np.savez(os.path.join(args.out_dir, "test.npz"), emf=val_x.astype(np.float32), target=val_y.astype(np.float32))

    counts_path = os.path.join(args.out_dir, "temporal_counts.csv")
    if os.path.exists(counts_path):
        os.remove(counts_path)
    write_counts(counts_path, train_counts, "train")
    write_counts(counts_path, val_counts, "val")

    print(f"Saved temporal train/val to {args.out_dir}")
    print(f"Train base rows: {len(train_y)} augmented rows: {len(train_y_aug)}")
    print(f"Val rows: {len(val_y)} input_dim={train_x.shape[1]}")


if __name__ == "__main__":
    main()
