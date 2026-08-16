"""Stateful JSONL runtime for the selected five-model v3.2.2 localizer."""

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
from evaluation.v4_common import resolve_device, sha256_file
from inference.realtime_predictor_hybrid_v4 import fixed_row_features
from physics.rotation_repr import target_v3_to_pose_deg


class SelectedV32RuntimeModule(torch.nn.Module):
    def __init__(self, raw_position_models, rowtime_position_model, orientation_model):
        super().__init__()
        self.raw_position_models = torch.nn.ModuleList(raw_position_models)
        self.rowtime_position_model = rowtime_position_model
        self.orientation_model = orientation_model

    def forward(self, raw_features, rowtime_features, logratio_features):
        members = [
            model(raw_features, return_normalized=False)
            for model in self.raw_position_models
        ]
        members.append(
            self.rowtime_position_model(rowtime_features, return_normalized=False)
        )
        position_stack = torch.stack(members)
        orientation = self.orientation_model(
            logratio_features, return_normalized=False,
        )
        output = torch.cat([
            position_stack[:, :, :3].mean(dim=0), orientation[:, 3:],
        ], dim=1)
        return torch.cat([
            output,
            position_stack[:, :, :3].permute(1, 0, 2).reshape(output.shape[0], -1),
        ], dim=1)


class V32CudaGraphRunner:
    def __init__(self, module, device):
        self.module = module.eval()
        self.static_inputs = (
            torch.zeros((1, 27), dtype=torch.float32, device=device),
            torch.zeros((1, 29), dtype=torch.float32, device=device),
            torch.zeros((1, 27), dtype=torch.float32, device=device),
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


class RealtimePosePredictorV32:
    """Fixed-delta-row predictor; timestamp is never a learned feature."""

    def __init__(
        self,
        deployment_manifest="checkpoints/new_calib_v3_temporal_deployment/deployment_manifest.json",
        device="auto",
        backend="auto",
        input_unit="mV",
        cpu_threads=None,
        maximum_timestamp_gap_ns=None,
    ):
        if cpu_threads is not None:
            torch.set_num_threads(int(cpu_threads))
        self.device = resolve_device(device)
        self.input_unit = input_unit
        self.maximum_timestamp_gap_ns = (
            None if maximum_timestamp_gap_ns is None else int(maximum_timestamp_gap_ns)
        )
        with open(deployment_manifest) as stream:
            manifest = json.load(stream)
        raw_models, rowtime_model, orientation_model = [], None, None
        for item in manifest["models"]:
            if sha256_file(Path(item["path"]) / "best.pt") != item.get("best_pt_sha256"):
                raise RuntimeError(f"Selected checkpoint hash changed: {item['path']}")
            model, _ = load_model(item["path"], self.device)
            if item["role"] == "position_member":
                raw_models.append(model)
            elif item["role"] == "position_member_unit_row_time":
                rowtime_model = model
            elif item["role"] == "orientation":
                orientation_model = model
        if len(raw_models) != 3 or rowtime_model is None or orientation_model is None:
            raise ValueError("Deployment manifest is not the selected v3.2.2 bundle")
        module = SelectedV32RuntimeModule(
            raw_models, rowtime_model, orientation_model,
        ).to(self.device).eval()
        self.feature_stats = [
            (model.emf_mean.detach().cpu().numpy(), model.emf_std.detach().cpu().numpy())
            for model in [*raw_models, rowtime_model, orientation_model]
        ]
        self.backend = (
            "cuda_graph" if backend == "auto" and self.device.type == "cuda"
            else ("torchscript" if backend == "auto" else backend)
        )
        examples = (
            torch.zeros((1, 27), dtype=torch.float32, device=self.device),
            torch.zeros((1, 29), dtype=torch.float32, device=self.device),
            torch.zeros((1, 27), dtype=torch.float32, device=self.device),
        )
        if self.backend == "torchscript":
            traced = torch.jit.trace(module, examples, check_trace=False)
            self.runner = torch.jit.freeze(traced.eval())
        elif self.backend == "cuda_graph":
            if self.device.type != "cuda":
                raise ValueError("cuda_graph requires CUDA")
            self.runner = V32CudaGraphRunner(module, self.device)
        elif self.backend == "eager":
            self.runner = module
        else:
            raise ValueError(f"Unsupported backend {self.backend!r}")
        self.reset()

    def reset(self):
        self.emf_buffer = []
        self.last_timestamp_ns = None

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
        self.emf_buffer.append(values)
        if len(self.emf_buffer) > 3:
            self.emf_buffer.pop(0)
        warmup = len(self.emf_buffer) < 3
        raw, rowtime, logratio, _ = fixed_row_features(self.emf_buffer)
        feature_values = [raw, raw, raw, rowtime, logratio]
        inputs = tuple(
            torch.as_tensor(value[None, :], dtype=torch.float32, device=self.device)
            for value in (raw, rowtime, logratio)
        )
        with torch.inference_mode():
            packed = (
                self.runner(inputs) if self.backend == "cuda_graph"
                else self.runner(*inputs)
            ).detach().cpu().numpy()[0]
        output = packed[:9]
        position_members = packed[9:].reshape(4, 3)
        pose = target_v3_to_pose_deg(output[None, :])[0]
        position_dispersion = float(np.sqrt(np.mean(np.sum(
            (position_members - output[None, :3]) ** 2, axis=1,
        ))))
        ood_fraction = max(
            float(np.mean(np.abs((feature - mean) / (std + 1e-8)) > 3.0))
            for feature, (mean, std) in zip(feature_values, self.feature_stats)
        )
        confidence = float(np.clip(np.exp(
            -position_dispersion / 2.0 - 2.0 * ood_fraction
        ), 0.0, 1.0))
        self.last_timestamp_ns = timestamp
        return {
            "timestamp_ns": timestamp,
            "pose": {
                "x_mm": float(pose[0]), "y_mm": float(pose[1]), "z_mm": float(pose[2]),
                "roll_deg": float(pose[3]), "pitch_deg": float(pose[4]), "yaw_deg": float(pose[5]),
            },
            "confidence": confidence,
            "dispersion": {
                "position_member_rms_mm": position_dispersion,
                "ood_feature_fraction_max_model": ood_fraction,
            },
            "warmup": bool(warmup),
            "state_reset": bool(reset or automatic_reset),
            "time_feature_policy": "fixed_robot_row_delta_absolute_timestamp_excluded",
            "model": "v3.2.2-selected-deployment",
            "device": str(self.device),
            "backend": self.backend,
        }


def main():
    parser = argparse.ArgumentParser(description="Run selected v3.2.2 realtime JSONL predictor.")
    parser.add_argument("--deployment_manifest", default="checkpoints/new_calib_v3_temporal_deployment/deployment_manifest.json")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--backend", choices=["auto", "eager", "torchscript", "cuda_graph"], default="auto")
    parser.add_argument("--input_unit", choices=["mV", "V"], default="mV")
    parser.add_argument("--cpu_threads", type=int, default=None)
    parser.add_argument("--maximum_timestamp_gap_ns", type=int, default=None)
    args = parser.parse_args()
    predictor = RealtimePosePredictorV32(
        args.deployment_manifest, args.device, args.backend, args.input_unit,
        args.cpu_threads, args.maximum_timestamp_gap_ns,
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
