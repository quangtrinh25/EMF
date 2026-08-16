"""Shared checkpoint helpers for v4 evaluation and realtime inference."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from models.localizer_v4 import ResidualLocalizerV4, TemporalGRULocalizerV4


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_device(requested):
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device("cuda" if requested in {"auto", "cuda"} and torch.cuda.is_available() else "cpu")


def load_model_v4(checkpoint_dir, device):
    checkpoint_dir = Path(checkpoint_dir)
    with open(checkpoint_dir / "model_metadata.json") as stream:
        metadata = json.load(stream)
    if metadata.get("schema_version") != 4:
        raise ValueError(f"Not a v4 checkpoint: {checkpoint_dir}")
    candidate = metadata["candidate_config"]
    if metadata["model_type"] == "gru":
        model = TemporalGRULocalizerV4(
            window_size=int(metadata["window_size"]), output_dim=9,
            hidden=int(metadata.get("hidden", 128)),
            slope=float(metadata.get("leaky_relu_slope", 0.01)),
        )
    else:
        model = ResidualLocalizerV4(
            window_size=int(metadata["window_size"]),
            feature_mode=candidate["feature_mode"],
            generation_count=len(metadata["generation_to_index"]),
            use_calibration_adapter=bool(candidate.get("calibration_adapter", False)),
            adapter_channel_std=metadata.get("adapter_channel_std"),
            output_dim=9,
            n_blocks=int(metadata["residual_blocks"]),
            hidden=int(metadata["hidden"]),
            slope=float(metadata.get("leaky_relu_slope", 0.01)),
        )
    state = torch.load(checkpoint_dir / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    return model, metadata


def checkpoint_signature(checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir)
    return {
        "best_pt_sha256": sha256_file(checkpoint_dir / "best.pt"),
        "metadata_sha256": sha256_file(checkpoint_dir / "model_metadata.json"),
    }

