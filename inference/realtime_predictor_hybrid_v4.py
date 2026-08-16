"""Realtime fixed-row predictor for the selected v3.2.2 + hybrid-v4 rule.

Timestamp is transport metadata for this calibration: row spacing is fixed by
the robot program, so absolute time never enters the network features.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.evaluate_v3 import load_model
from evaluation.v4_common import load_model_v4, resolve_device
from physics.rotation_repr import so3_geodesic_deg, target_v3_to_pose_deg


def fixed_row_features(emf_buffer):
    """Build the four W3 feature layouts from a causal V-valued row buffer."""
    if not 1 <= len(emf_buffer) <= 3:
        raise ValueError("fixed_row_features expects one to three causal rows")
    padded = [emf_buffer[0]] * (3 - len(emf_buffer)) + list(emf_buffer)
    window = np.stack(padded).astype(np.float32)
    raw = window.reshape(-1)
    current = window[-1]
    logratio = np.log((window[:-1] + 1e-4) / (current[None, :] + 1e-4))
    logratio = np.concatenate([
        current, np.clip(logratio, -5.0, 5.0).reshape(-1),
    ]).astype(np.float32)
    first_real = 3 - len(emf_buffer)
    row_ids = np.maximum(np.arange(3) - first_real, 0)
    dt_ratio = np.diff(row_ids).astype(np.float32)
    rowtime = np.concatenate([raw, dt_ratio]).astype(np.float32)
    return raw, rowtime, logratio, rowtime.copy()


class HybridRuntimeModule(torch.nn.Module):
    """One traced module containing all six networks and the selected rule."""

    def __init__(
        self, raw_position_models, rowtime_position_model, orientation_model,
        hybrid_model, hybrid_generation_index, hybrid_position_weight,
        hybrid_orientation_weight,
    ):
        super().__init__()
        self.raw_position_models = torch.nn.ModuleList(raw_position_models)
        self.rowtime_position_model = rowtime_position_model
        self.orientation_model = orientation_model
        self.hybrid_model = hybrid_model
        self.register_buffer(
            "hybrid_generation_index",
            torch.tensor([hybrid_generation_index], dtype=torch.long),
        )
        self.hybrid_position_weight = float(hybrid_position_weight)
        self.hybrid_orientation_weight = float(hybrid_orientation_weight)

    def forward(self, raw_features, rowtime_features, logratio_features, hybrid_features):
        position_members = [
            model(raw_features, return_normalized=False)
            for model in self.raw_position_models
        ]
        position_members.append(
            self.rowtime_position_model(rowtime_features, return_normalized=False)
        )
        position_stack = torch.stack(position_members)
        v32_orientation = self.orientation_model(
            logratio_features, return_normalized=False,
        )
        v32 = torch.cat([
            position_stack[:, :, :3].mean(dim=0), v32_orientation[:, 3:],
        ], dim=1)
        generation = self.hybrid_generation_index.expand(hybrid_features.shape[0])
        hybrid = self.hybrid_model(
            hybrid_features, generation, return_normalized=False,
        )
        alpha = self.hybrid_position_weight
        beta = self.hybrid_orientation_weight
        # The direct 6D mean is projected by the common pose decoder using
        # Gram-Schmidt.  Development metrics are numerically equivalent to the
        # more expensive two-rotation chordal/SVD mean.
        output = torch.cat([
            (1.0 - alpha) * v32[:, :3] + alpha * hybrid[:, :3],
            (1.0 - beta) * v32[:, 3:] + beta * hybrid[:, 3:],
        ], dim=1)
        # Pack all runtime diagnostics into one tensor so CUDA performs a
        # single device-to-host synchronization per row.
        return torch.cat([
            output, v32, hybrid,
            position_stack[:, :, :3].permute(1, 0, 2).reshape(output.shape[0], -1),
        ], dim=1)


class HybridCudaGraphRunner:
    def __init__(self, module, input_dims, device):
        self.module = module.eval()
        self.static_inputs = tuple(
            torch.zeros((1, dim), dtype=torch.float32, device=device) for dim in input_dims
        )
        side_stream = torch.cuda.Stream()
        side_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side_stream), torch.inference_mode():
            for _ in range(5):
                self.module(*self.static_inputs)
        torch.cuda.current_stream().wait_stream(side_stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph), torch.inference_mode():
            self.static_output = self.module(*self.static_inputs)

    def __call__(self, inputs):
        for target, source in zip(self.static_inputs, inputs):
            target.copy_(source)
        self.graph.replay()
        return self.static_output.clone()


class RealtimeHybridPosePredictorV4:
    """Transport-neutral row-time predictor ready for a future TCP/IP layer."""

    def __init__(
        self, v32_manifest, hybrid_checkpoint_dir, hybrid_position_weight=0.20,
        generation="real_current", device="auto", backend="auto", input_unit="mV",
        cpu_threads=None, maximum_timestamp_gap_ns=None,
        hybrid_orientation_weight=0.50,
    ):
        if cpu_threads is not None:
            torch.set_num_threads(int(cpu_threads))
        self.device = resolve_device(device)
        self.input_unit = input_unit
        self.position_weight = float(hybrid_position_weight)
        if not 0.0 <= self.position_weight <= 1.0:
            raise ValueError("hybrid_position_weight must be in [0,1]")
        self.orientation_weight = float(hybrid_orientation_weight)
        if not 0.0 <= self.orientation_weight <= 1.0:
            raise ValueError("hybrid_orientation_weight must be in [0,1]")
        self.maximum_timestamp_gap_ns = (
            None if maximum_timestamp_gap_ns is None else int(maximum_timestamp_gap_ns)
        )
        with open(v32_manifest) as stream:
            manifest = json.load(stream)
        if manifest.get("test_opened") is not False:
            raise RuntimeError("v3.2 manifest is not test-locked")

        raw_models, raw_metadata = [], []
        rowtime_model = rowtime_metadata = None
        orientation_model = orientation_metadata = None
        for item in manifest["models"]:
            model, metadata = load_model(item["path"], self.device)
            if item["role"] == "position_member":
                raw_models.append(model)
                raw_metadata.append(metadata)
            elif item["role"] == "position_member_unit_row_time":
                rowtime_model, rowtime_metadata = model, metadata
            elif item["role"] == "orientation":
                orientation_model, orientation_metadata = model, metadata
        if len(raw_models) != 3 or rowtime_model is None or orientation_model is None:
            raise ValueError("Expected the frozen three-raw + rowtime + logratio v3.2 bundle")
        if any(int(row["temporal_window"]) != 3 for row in [*raw_metadata, rowtime_metadata, orientation_metadata]):
            raise ValueError("Hybrid runtime currently requires the selected causal W3 bundle")
        hybrid_model, hybrid_metadata = load_model_v4(hybrid_checkpoint_dir, self.device)
        if generation not in hybrid_metadata["generation_to_index"]:
            raise ValueError(f"Unknown hybrid generation {generation!r}")
        if int(hybrid_metadata["window_size"]) != 3:
            raise ValueError("Hybrid runtime requires a W3 checkpoint")
        self.hybrid_model = hybrid_model
        self.hybrid_metadata = hybrid_metadata
        self.generation = generation
        generation_index = int(hybrid_metadata["generation_to_index"][generation])
        module = HybridRuntimeModule(
            raw_models, rowtime_model, orientation_model, hybrid_model,
            generation_index, self.position_weight, self.orientation_weight,
        ).to(self.device).eval()
        self.backend = (
            "cuda_graph" if backend == "auto" and self.device.type == "cuda"
            else ("torchscript" if backend == "auto" else backend)
        )
        input_dims = (27, 29, 27, int(hybrid_metadata["packed_input_dim"]))
        examples = tuple(
            torch.zeros((1, dim), dtype=torch.float32, device=self.device) for dim in input_dims
        )
        if self.backend == "torchscript":
            traced = torch.jit.trace(module, examples, check_trace=False)
            self.runner = torch.jit.freeze(traced.eval())
        elif self.backend == "cuda_graph":
            if self.device.type != "cuda":
                raise ValueError("cuda_graph requires CUDA")
            self.runner = HybridCudaGraphRunner(module, input_dims, self.device)
        elif self.backend == "eager":
            self.runner = module
        else:
            raise ValueError(f"Unsupported backend {self.backend!r}")
        self.v3_feature_stats = [
            (model.emf_mean.detach().cpu().numpy(), model.emf_std.detach().cpu().numpy())
            for model in [*raw_models, rowtime_model, orientation_model]
        ]
        self.reset()

    def reset(self):
        self.emf_buffer = []
        self.last_timestamp_ns = None

    def _features(self, emf):
        self.emf_buffer.append(emf)
        if len(self.emf_buffer) > 3:
            self.emf_buffer.pop(0)
        warmup = len(self.emf_buffer) < 3
        return fixed_row_features(self.emf_buffer), warmup

    def _run(self, features):
        inputs = tuple(
            torch.as_tensor(value[None, :], dtype=torch.float32, device=self.device)
            for value in features
        )
        with torch.inference_mode():
            return self.runner(inputs) if self.backend == "cuda_graph" else self.runner(*inputs)

    def _ood_fraction(self, features):
        v3_values = [features[0], features[0], features[0], features[1], features[2]]
        fractions = [
            float(np.mean(np.abs((values - mean) / (std + 1e-8)) > 3.0))
            for values, (mean, std) in zip(v3_values, self.v3_feature_stats)
        ]
        packed = torch.as_tensor(features[3][None, :], dtype=torch.float32, device=self.device)
        generation = torch.tensor(
            [self.hybrid_metadata["generation_to_index"][self.generation]],
            dtype=torch.long, device=self.device,
        )
        with torch.inference_mode():
            values = self.hybrid_model.build_features(packed, generation)
            z = torch.abs(
                (values - self.hybrid_model.feature_mean)
                / (self.hybrid_model.feature_std + 1e-8)
            )
        fractions.append(float(torch.mean((z > 3.0).float()).item()))
        return max(fractions)

    def predict(self, timestamp_ns, reset, emf):
        values = np.asarray(emf, dtype=np.float32)
        if values.shape != (9,) or not np.isfinite(values).all():
            raise ValueError("emf must contain nine finite values")
        if self.input_unit == "mV":
            values = values * 1e-3
        timestamp = None if timestamp_ns is None else int(timestamp_ns)
        automatic_reset = False
        if timestamp is not None and self.last_timestamp_ns is not None:
            delta = timestamp - self.last_timestamp_ns
            automatic_reset = delta <= 0 or (
                self.maximum_timestamp_gap_ns is not None
                and delta > self.maximum_timestamp_gap_ns
            )
        if reset or automatic_reset:
            self.reset()
        features, warmup = self._features(values)
        self.last_timestamp_ns = timestamp
        packed_output = self._run(features).detach().cpu().numpy()[0]
        output = packed_output[:9]
        v32 = packed_output[9:18]
        hybrid = packed_output[18:27]
        position_members = packed_output[27:].reshape(4, 3)
        pose = target_v3_to_pose_deg(output[None, :])[0]
        position_dispersion = float(np.sqrt(np.mean(np.sum(
            (np.vstack([position_members, hybrid[None, :3]]) - output[None, :3]) ** 2,
            axis=1,
        ))))
        orientation_dispersion = float(so3_geodesic_deg(
            v32[None, 3:], hybrid[None, 3:],
        )[0])
        ood_fraction = self._ood_fraction(features)
        confidence = float(np.clip(np.exp(
            -position_dispersion / 2.0 - orientation_dispersion / 2.0 - 2.0 * ood_fraction
        ), 0.0, 1.0))
        return {
            "timestamp_ns": timestamp,
            "pose": {
                "x_mm": float(pose[0]), "y_mm": float(pose[1]), "z_mm": float(pose[2]),
                "roll_deg": float(pose[3]), "pitch_deg": float(pose[4]), "yaw_deg": float(pose[5]),
            },
            "confidence": confidence,
            "dispersion": {
                "position_rms_mm": position_dispersion,
                "orientation_v32_vs_v4_deg": orientation_dispersion,
                "ood_feature_fraction_max_model": ood_fraction,
            },
            "warmup": bool(warmup),
            "state_reset": bool(reset or automatic_reset),
            "time_feature_policy": "fixed_robot_row_delta_absolute_timestamp_excluded",
            "blend_rule": {
                "hybrid_position_weight": self.position_weight,
                "hybrid_orientation_weight": self.orientation_weight,
                "orientation_representation": "weighted_rotation6d_then_gram_schmidt",
            },
            "device": str(self.device),
            "backend": self.backend,
        }


def main():
    parser = argparse.ArgumentParser(description="Run the selected hybrid v4 realtime JSONL interface.")
    parser.add_argument("--v32_manifest", default="checkpoints/new_calib_v3_temporal_deployment/deployment_manifest.json")
    parser.add_argument("--hybrid_checkpoint_dir", required=True)
    parser.add_argument("--hybrid_position_weight", type=float, default=0.20)
    parser.add_argument("--hybrid_orientation_weight", type=float, default=0.50)
    parser.add_argument("--generation", default="real_current")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--backend", choices=["auto", "eager", "torchscript", "cuda_graph"], default="auto")
    parser.add_argument("--input_unit", choices=["mV", "V"], default="mV")
    parser.add_argument("--cpu_threads", type=int, default=None)
    parser.add_argument("--maximum_timestamp_gap_ns", type=int, default=None)
    args = parser.parse_args()
    predictor = RealtimeHybridPosePredictorV4(
        args.v32_manifest,
        args.hybrid_checkpoint_dir,
        hybrid_position_weight=args.hybrid_position_weight,
        hybrid_orientation_weight=args.hybrid_orientation_weight,
        generation=args.generation,
        device=args.device,
        backend=args.backend,
        input_unit=args.input_unit,
        cpu_threads=args.cpu_threads,
        maximum_timestamp_gap_ns=args.maximum_timestamp_gap_ns,
    )
    for line in sys.stdin:
        if not line.strip():
            continue
        request = json.loads(line)
        response = predictor.predict(
            request.get("timestamp_ns"), request.get("reset", False), request["emf"],
        )
        print(json.dumps(response), flush=True)


if __name__ == "__main__":
    main()
