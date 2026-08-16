import argparse

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Check that real split roles do not share source rows.")
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--roles", nargs="+", default=["calib_train", "real_dev", "private_test"])
    args = parser.parse_args()

    df = pd.read_csv(args.metadata)
    df = df[df["role"].isin(args.roles)].copy()
    key_cols = ["source_filename", "source_row_index"]
    duplicated = df[df.duplicated(key_cols, keep=False)].sort_values(key_cols + ["role"])
    if not duplicated.empty:
        print("Leakage detected: the same source row appears in multiple split roles.")
        print(duplicated.to_string(index=False))
        raise SystemExit(1)

    print("No split leakage detected.")
    print(df.groupby("role").size().to_string())


if __name__ == "__main__":
    main()
