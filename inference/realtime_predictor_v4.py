"""Stateful causal v4 predictor prepared for a future TCP/IP transport layer."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.v4_common import load_model_v4, resolve_device
from physics.rotation_repr import mean_rotation_6d, so3_geodesic_deg, target_v3_to_pose_deg


class FixedGenerationWrapper(torch.nn.Module):
    def __init__(self, model, generation_index):
        super().__init__()
        self.model = model
        self.register_buffer("generation_index", torch.tensor([generation_index], dtype=torch.long))

    def forward(self, packed):
        generation = self.generation_index.expand(packed.shape[0])
        return self.model(packed, generation, return_normalized=False)


class CudaGraphRunner:
    def __init__(self, wrapper, input_dim, device):
        if device.type != "cuda":
            raise ValueError("CUDA Graph requires a CUDA device")
        self.static_input = torch.zeros((1, input_dim), dtype=torch.float32, device=device)
        self.wrapper = wrapper.eval()
        side_stream = torch.cuda.Stream()
        side_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side_stream), torch.inference_mode():
            for _ in range(5):
                self.wrapper(self.static_input)
        torch.cuda.current_stream().wait_stream(side_stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph), torch.inference_mode():
            self.static_output = self.wrapper(self.static_input)

    def __call__(self, packed):
        self.static_input.copy_(packed)
        self.graph.replay()
        return self.static_output.clone()


class RealtimePosePredictorV4:
    """Transport-neutral interface: timestamp/reset/EMF in, pose/confidence out."""

    def __init__(
        self, checkpoint_dirs, generation="new_2026_08_03", device="auto",
        backend="auto", input_unit="mV", cpu_threads=None,
    ):
        if not checkpoint_dirs:
            raise ValueError("At least one checkpoint is required")
        if cpu_threads is not None:
            torch.set_num_threads(int(cpu_threads))
        self.device = resolve_device(device)
        self.input_unit = input_unit
        self.models = []
        self.metadata = []
        reference = None
        for checkpoint_dir in checkpoint_dirs:
            model, metadata = load_model_v4(checkpoint_dir, self.device)
            signature = {
                "candidate_config": metadata["candidate_config"],
                "packed_input_dim": metadata["packed_input_dim"],
                "generation_to_index": metadata["generation_to_index"],
                "window_size": metadata["window_size"],
            }
            if reference is None:
                reference = signature
            elif signature != reference:
                raise ValueError("Realtime ensemble checkpoints are feature-incompatible")
            self.models.append(model)
            self.metadata.append(metadata)
        self.reference = self.metadata[0]
        if generation not in self.reference["generation_to_index"]:
            raise ValueError(f"Unknown deployment generation {generation!r}")
        self.generation = generation
        self.generation_index = int(self.reference["generation_to_index"][generation])
        self.window_size = int(self.reference["window_size"])
        self.input_dim = int(self.reference["packed_input_dim"])
        self.median_dt = float(self.reference["train_dt_median"][generation])
        self.time_mode = self.reference.get("generation_time_mode", {}).get(
            generation, "hardware_timestamp",
        )
        self.fixed_dt_seconds = self.reference.get("generation_fixed_dt_seconds", {}).get(generation)
        self.dt_clip = float(self.reference["dt_ratio_clip"])
        self.gap_reset_multiple = float(self.reference["gap_reset_multiple"])
        self.backend = (
            "cuda_graph" if backend == "auto" and self.device.type == "cuda"
            else ("torchscript" if backend == "auto" else backend)
        )
        if self.backend == "cuda_graph" and self.device.type != "cuda":
            raise ValueError("cuda_graph backend requires CUDA")
        self.runners = []
        example = torch.zeros((1, self.input_dim), dtype=torch.float32, device=self.device)
        for model in self.models:
            wrapper = FixedGenerationWrapper(model, self.generation_index).to(self.device).eval()
            if self.backend == "torchscript":
                traced = torch.jit.trace(wrapper, example, check_trace=False)
                self.runners.append(torch.jit.freeze(traced.eval()))
            elif self.backend == "cuda_graph":
                self.runners.append(CudaGraphRunner(wrapper, self.input_dim, self.device))
            elif self.backend == "eager":
                self.runners.append(wrapper)
            else:
                raise ValueError(f"Unsupported backend {self.backend!r}")
        self.reset()

    def reset(self):
        self.emf_buffer = []
        self.time_buffer = []
        self.last_wall_timestamp_ns = None

    def _packed_features(self, timestamp_ns, reset, emf):
        emf = np.asarray(emf, dtype=np.float32)
        if emf.shape != (9,) or not np.isfinite(emf).all():
            raise ValueError("emf must contain nine finite values")
        timestamp_value = None
        if timestamp_ns is not None:
            try:
                timestamp_value = int(timestamp_ns)
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError("timestamp_ns must be an integer nanosecond timestamp") from error
        if self.time_mode == "hardware_timestamp" and timestamp_value is None:
            raise ValueError("This checkpoint requires timestamp_ns for realtime inference")
        automatic_reset = False
        if timestamp_value is not None and self.last_wall_timestamp_ns is not None:
            wall_delta_s = (timestamp_value - self.last_wall_timestamp_ns) * 1e-9
            automatic_reset = wall_delta_s <= 0
            if self.time_mode == "hardware_timestamp":
                automatic_reset |= wall_delta_s > self.gap_reset_multiple * self.median_dt
            elif self.fixed_dt_seconds is not None:
                automatic_reset |= wall_delta_s > self.gap_reset_multiple * float(self.fixed_dt_seconds)
        if reset or automatic_reset:
            self.reset()
        if self.input_unit == "mV":
            emf = emf * 1e-3
        feature_time = (
            timestamp_value
            if self.time_mode == "hardware_timestamp"
            else (0 if not self.time_buffer else int(self.time_buffer[-1]) + 1)
        )
        self.emf_buffer.append(emf)
        self.time_buffer.append(feature_time)
        self.last_wall_timestamp_ns = timestamp_value
        if len(self.emf_buffer) > self.window_size:
            self.emf_buffer.pop(0)
            self.time_buffer.pop(0)
        warmup = len(self.emf_buffer) < self.window_size
        padded_emf = [self.emf_buffer[0]] * (self.window_size - len(self.emf_buffer)) + self.emf_buffer
        padded_time = [self.time_buffer[0]] * (self.window_size - len(self.time_buffer)) + self.time_buffer
        time_scale = 1e-9 if self.time_mode == "hardware_timestamp" else 1.0
        dt_ratio = np.clip(
            np.diff(np.asarray(padded_time, dtype=np.int64)) * time_scale / self.median_dt,
            0.0, self.dt_clip,
        )
        packed = np.concatenate([np.stack(padded_emf).reshape(-1), dt_ratio]).astype(np.float32)
        return packed, warmup, automatic_reset

    def _predict_targets(self, packed):
        tensor = torch.as_tensor(packed[None, :], dtype=torch.float32, device=self.device)
        outputs = []
        with torch.inference_mode():
            for runner in self.runners:
                outputs.append(runner(tensor).detach().cpu().numpy()[0])
        return np.stack(outputs)

    def _ood_fraction(self, packed):
        model = self.models[0]
        tensor = torch.as_tensor(packed[None, :], dtype=torch.float32, device=self.device)
        generation = torch.tensor([self.generation_index], dtype=torch.long, device=self.device)
        with torch.inference_mode():
            if hasattr(model, "build_features"):
                values = model.build_features(tensor, generation)
                z = torch.abs((values - model.feature_mean) / (model.feature_std + 1e-8))
            else:
                raw = tensor[:, :self.window_size * 9].reshape(1, self.window_size, 9)
                z = torch.abs((raw - model.raw_mean) / (model.raw_std + 1e-8))
        return float(torch.mean((z > 3.0).float()).item())

    def predict(self, timestamp_ns, reset, emf):
        packed, warmup, automatic_reset = self._packed_features(timestamp_ns, bool(reset), emf)
        members = self._predict_targets(packed)
        if len(members) == 1:
            ensemble = members[0]
        else:
            position = members[:, :3].mean(axis=0)
            orientation = mean_rotation_6d(members[:, None, 3:]) [0]
            ensemble = np.concatenate([position, orientation])
        pose = target_v3_to_pose_deg(ensemble[None, :])[0]
        position_dispersion = float(np.sqrt(np.mean(np.sum((members[:, :3] - ensemble[:3]) ** 2, axis=1))))
        orientation_reference = np.repeat(ensemble[None, 3:], len(members), axis=0)
        orientation_dispersion = float(np.sqrt(np.mean(
            so3_geodesic_deg(members[:, 3:], orientation_reference) ** 2
        )))
        ood_fraction = self._ood_fraction(packed)
        confidence = float(np.clip(np.exp(
            -position_dispersion / 2.0 - orientation_dispersion / 2.0 - 2.0 * ood_fraction
        ), 0.0, 1.0))
        return {
            "timestamp_ns": None if timestamp_ns is None else int(timestamp_ns),
            "pose": {
                "x_mm": float(pose[0]), "y_mm": float(pose[1]), "z_mm": float(pose[2]),
                "roll_deg": float(pose[3]), "pitch_deg": float(pose[4]), "yaw_deg": float(pose[5]),
            },
            "confidence": confidence,
            "dispersion": {
                "position_rms_mm": position_dispersion,
                "orientation_rms_deg": orientation_dispersion,
                "ood_feature_fraction": ood_fraction,
            },
            "warmup": bool(warmup),
            "state_reset": bool(reset or automatic_reset),
            "device": str(self.device),
            "backend": self.backend,
        }


def main():
    parser = argparse.ArgumentParser(description="Run the transport-neutral v4 realtime JSONL interface.")
    parser.add_argument("--checkpoint_dirs", nargs="+", required=True)
    parser.add_argument("--generation", default="new_2026_08_03")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--backend", choices=["auto", "eager", "torchscript", "cuda_graph"], default="auto")
    parser.add_argument("--input_unit", choices=["mV", "V"], default="mV")
    parser.add_argument("--cpu_threads", type=int, default=None)
    args = parser.parse_args()
    predictor = RealtimePosePredictorV4(
        args.checkpoint_dirs, generation=args.generation, device=args.device,
        backend=args.backend, input_unit=args.input_unit, cpu_threads=args.cpu_threads,
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
