"""Freeze a selected v4 candidate before the one-time sealed evaluation."""

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.v4_common import sha256_file


def main():
    parser = argparse.ArgumentParser(description="Freeze v4 model/config/protocol after quality-gate acceptance.")
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--quality_gate", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--protocol_config", default="configs/new_calib_v4.yaml")
    parser.add_argument("--training_config", default="configs/training_v4.yaml")
    parser.add_argument(
        "--v32_manifest",
        default="checkpoints/new_calib_v3_temporal_deployment/deployment_manifest.json",
    )
    parser.add_argument("--hybrid_position_weight", type=float, default=0.20)
    parser.add_argument("--orientation_rule", default="equal_rotation6d_mean_then_gram_schmidt")
    parser.add_argument("--additional_files", nargs="*", default=[])
    args = parser.parse_args()
    checkpoint_dir = Path(args.checkpoint_dir)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    if out_dir.exists():
        raise FileExistsError(f"Frozen candidate already exists: {out_dir}")
    with open(data_dir / "protocol.json") as stream:
        protocol = json.load(stream)
    if protocol.get("sealed_test_opened") is not False or (data_dir / "sealed_test.npz").exists():
        raise RuntimeError("Cannot freeze after sealed test material has been opened")
    with open(args.quality_gate) as stream:
        gate = json.load(stream)
    if gate.get("accepted") is not True:
        raise RuntimeError("Quality gate did not accept this candidate")
    with open(args.v32_manifest) as stream:
        v32_manifest = json.load(stream)
    if v32_manifest.get("test_opened") is not False:
        raise RuntimeError("Cannot freeze with a v3.2 bundle that opened the test")
    for item in v32_manifest.get("models", []):
        observed = sha256_file(Path(item["path"]) / "best.pt")
        if observed != item.get("best_pt_sha256"):
            raise RuntimeError(f"v3.2 checkpoint hash changed: {item['path']}")
    out_dir.mkdir(parents=True)
    copied = []
    for source in (
        checkpoint_dir / "best.pt",
        checkpoint_dir / "model_metadata.json",
        checkpoint_dir / "feature_mean.npy",
        checkpoint_dir / "feature_std.npy",
        checkpoint_dir / "pose_mean.npy",
        checkpoint_dir / "pose_std.npy",
        data_dir / "protocol.json",
        Path(args.protocol_config),
        Path(args.training_config),
        Path(args.quality_gate),
        *[Path(path) for path in args.additional_files],
    ):
        destination = out_dir / (
            "v32_deployment_manifest.json" if source == Path(args.v32_manifest) else source.name
        )
        shutil.copy2(source, destination)
        copied.append(destination)
    v32_destination = out_dir / "v32_deployment_manifest.json"
    shutil.copy2(args.v32_manifest, v32_destination)
    copied.append(v32_destination)
    manifest = {
        "schema_version": 4,
        "frozen": True,
        "test_opened_at_freeze": False,
        "model_sha256": sha256_file(checkpoint_dir / "best.pt"),
        "bundle": {
            "v32_manifest_sha256": sha256_file(args.v32_manifest),
            "v32_model_sha256": {
                item["path"]: item["best_pt_sha256"] for item in v32_manifest["models"]
            },
            "hybrid_v4_checkpoint": str(checkpoint_dir),
            "hybrid_position_weight": args.hybrid_position_weight,
            "orientation_rule": args.orientation_rule,
            "absolute_timestamp_used_as_feature": False,
        },
        "protocol_config_sha256": sha256_file(args.protocol_config),
        "sealed_source_sha256": protocol["sealed_test"]["sha256"],
        "files_sha256": {path.name: sha256_file(path) for path in copied},
        "quality_gate": gate,
    }
    with open(out_dir / "freeze_manifest.json", "w") as stream:
        json.dump(manifest, stream, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
