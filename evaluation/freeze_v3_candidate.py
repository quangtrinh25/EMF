import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Freeze a dev-selected v3 checkpoint before opening test.")
    parser.add_argument('--checkpoint_dir', required=True)
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--val_metrics', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--protocol_config', default='configs/new_calib_v3.yaml')
    parser.add_argument('--training_config', default='configs/training_v3.yaml')
    parser.add_argument('--selection_reason', required=True)
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    data_dir = Path(args.data_dir)
    val_metrics = Path(args.val_metrics)
    out_dir = Path(args.out_dir)
    if out_dir.exists():
        raise FileExistsError(f"Frozen candidate already exists: {out_dir}")
    if (val_metrics.parent / 'test_metrics.csv').exists() or (val_metrics.parent / 'test_predictions.csv').exists():
        raise RuntimeError("Refusing to freeze: test artifacts already exist beside validation results")
    protocol_path = data_dir / 'protocol.json'
    with open(protocol_path) as stream:
        protocol = json.load(stream)
    if protocol.get('test_opened') is not False:
        raise RuntimeError("Protocol does not explicitly record test_opened=false")

    pooled = pd.read_csv(val_metrics)
    pooled = pooled.loc[pooled['scope'] == 'pooled']
    if len(pooled) != 1:
        raise ValueError("Validation metrics must contain exactly one pooled row")
    metrics = pooled.iloc[0].to_dict()

    out_dir.mkdir(parents=True)
    copied = []
    for filename in (
        'best.pt', 'resnet_best.pt', 'model_metadata.json',
        'emf_mean.npy', 'emf_std.npy', 'pose_mean.npy', 'pose_std.npy',
        'best_validation_metrics.json', 'training_history.csv',
    ):
        source = checkpoint_dir / filename
        if not source.exists():
            raise FileNotFoundError(source)
        destination = out_dir / filename
        shutil.copy2(source, destination)
        copied.append(destination)
    for source, filename in (
        (protocol_path, 'data_protocol.json'),
        (Path(args.protocol_config), 'new_calib_v3.yaml'),
        (Path(args.training_config), 'training_v3.yaml'),
        (val_metrics, 'validation_metrics.csv'),
    ):
        destination = out_dir / filename
        shutil.copy2(source, destination)
        copied.append(destination)

    manifest = {
        'schema_version': 1,
        'frozen': True,
        'test_opened_at_freeze': False,
        'source_checkpoint_dir': str(checkpoint_dir),
        'source_data_dir': str(data_dir),
        'selection_reason': args.selection_reason,
        'pooled_validation_metrics': metrics,
        'files_sha256': {path.name: sha256(path) for path in copied},
    }
    with open(out_dir / 'freeze_manifest.json', 'w') as stream:
        json.dump(manifest, stream, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
