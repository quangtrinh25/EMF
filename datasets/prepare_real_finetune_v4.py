"""Create a real-only fine-tune view that retains a hybrid model's domain layout."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Prepare staged real-only v4 fine-tuning data.")
    parser.add_argument("--real_data_dir", required=True)
    parser.add_argument("--hybrid_data_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    real_dir = Path(args.real_data_dir)
    hybrid_dir = Path(args.hybrid_data_dir)
    with open(real_dir / "protocol.json") as stream:
        real_protocol = json.load(stream)
    with open(hybrid_dir / "protocol.json") as stream:
        hybrid_protocol = json.load(stream)
    for key in ("candidate", "candidate_config", "fold", "packed_input_dim"):
        if real_protocol.get(key) != hybrid_protocol.get(key):
            raise ValueError(f"Real and hybrid protocols differ for {key}")
    if real_protocol.get("sealed_test_opened") is not False or hybrid_protocol.get("sealed_test_opened") is not False:
        raise RuntimeError("Fine-tune builder refuses opened test data")
    with np.load(real_dir / "train.npz") as loaded:
        real_count = len(loaded["target"])
    protocol = dict(hybrid_protocol)
    hybrid_info = dict(protocol.get("hybrid_training", {}))
    hybrid_info.update({
        "fine_tune_real_only": True,
        "fine_tune_real_samples": int(real_count),
        "parent_hybrid_data_dir": str(hybrid_dir),
        "test_labels_loaded": False,
    })
    protocol["hybrid_training"] = hybrid_info
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(real_dir / "train.npz", out_dir / "train.npz")
    if (real_dir / "val.npz").exists():
        shutil.copy2(real_dir / "val.npz", out_dir / "val.npz")
    with open(out_dir / "protocol.json", "w") as stream:
        json.dump(protocol, stream, indent=2)
    print(json.dumps(hybrid_info, indent=2))


if __name__ == "__main__":
    main()
