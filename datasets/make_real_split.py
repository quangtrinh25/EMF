import argparse
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from calibration.report_utils import COLS, FILE_MAP


def read_raw(data_dir, key):
    return pd.read_csv(os.path.join(data_dir, FILE_MAP[key]), header=None, names=COLS)


def write_split(df, out_dir, role, key, source_rows, metadata_rows):
    role_dir = Path(out_dir) / role
    role_dir.mkdir(parents=True, exist_ok=True)
    out_path = role_dir / f"{key}.csv"
    df.to_csv(out_path, header=False, index=False)
    for source_row in source_rows:
        metadata_rows.append({
            "role": role,
            "dataset_key": key,
            "source_filename": FILE_MAP[key],
            "source_row_index": int(source_row),
            "split_filename": str(out_path),
        })
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Create deterministic real CSV train/dev/private splits.")
    parser.add_argument("--raw_dir", default="data/raw_calibration")
    parser.add_argument("--out_dir", default="data/real_splits/noleak_balanced_easy")
    parser.add_argument("--train_files", nargs="+", default=["set10_cyl_no_rot", "set11_con_spi_rot"], choices=list(FILE_MAP.keys()))
    parser.add_argument("--private_files", nargs="+", default=["set13_con_spi_no_rot", "set12_cyl_spi_rot"], choices=list(FILE_MAP.keys()))
    parser.add_argument("--dev_mod", type=int, default=5)
    parser.add_argument("--dev_remainder", type=int, default=0)
    args = parser.parse_args()

    if args.dev_mod <= 1:
        raise ValueError("--dev_mod must be greater than 1")
    if not 0 <= args.dev_remainder < args.dev_mod:
        raise ValueError("--dev_remainder must be in [0, dev_mod)")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata_rows = []
    counts = []

    for key in args.train_files:
        df = read_raw(args.raw_dir, key)
        source_index = pd.Series(range(len(df)))
        dev_mask = (source_index % args.dev_mod) == args.dev_remainder
        train_df = df.loc[~dev_mask].reset_index(drop=True)
        dev_df = df.loc[dev_mask].reset_index(drop=True)
        train_rows = source_index.loc[~dev_mask].to_list()
        dev_rows = source_index.loc[dev_mask].to_list()
        write_split(train_df, out_dir, "calib_train", key, train_rows, metadata_rows)
        write_split(dev_df, out_dir, "real_dev", key, dev_rows, metadata_rows)
        counts.append({"role": "calib_train", "dataset_key": key, "rows": len(train_df)})
        counts.append({"role": "real_dev", "dataset_key": key, "rows": len(dev_df)})

    for key in args.private_files:
        df = read_raw(args.raw_dir, key)
        rows = list(range(len(df)))
        write_split(df, out_dir, "private_test", key, rows, metadata_rows)
        counts.append({"role": "private_test", "dataset_key": key, "rows": len(df)})

    metadata = pd.DataFrame(metadata_rows)
    metadata.to_csv(out_dir / "metadata.csv", index=False)
    count_df = pd.DataFrame(counts)
    count_df.to_csv(out_dir / "split_counts.csv", index=False)

    print(count_df.to_string(index=False))
    print(f"Saved metadata to {out_dir / 'metadata.csv'}")


if __name__ == "__main__":
    main()
