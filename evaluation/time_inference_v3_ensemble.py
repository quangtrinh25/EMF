import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.evaluate_v3 import load_model


class InferenceWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, features):
        return self.model(features, return_normalized=False)


class EnsembleInferenceWrapper(torch.nn.Module):
    def __init__(self, position_models, aux_position_models, orientation_model):
        super().__init__()
        self.position_models = torch.nn.ModuleList(position_models)
        self.aux_position_models = torch.nn.ModuleList(aux_position_models)
        self.orientation_model = orientation_model

    def forward(self, position_features, orientation_features, aux_position_features):
        base_predictions = [
            model(position_features, return_normalized=False)
            for model in self.position_models
        ]
        aux_predictions = [
            model(aux_position_features[index], return_normalized=False)
            for index, model in enumerate(self.aux_position_models)
        ]
        base_stacked = torch.stack(base_predictions)
        position_stacked = (
            torch.cat([base_stacked, torch.stack(aux_predictions)], dim=0)
            if aux_predictions else base_stacked
        )
        orientation_prediction = self.orientation_model(
            orientation_features, return_normalized=False,
        )
        ensemble = torch.cat([
            position_stacked[:, :, :3].mean(dim=0),
            orientation_prediction[:, 3:],
        ], dim=1)
        return ensemble, base_stacked, position_stacked, orientation_prediction


def prepare_backend(model, example, backend):
    if backend == 'eager':
        return model
    with torch.inference_mode():
        traced = torch.jit.trace(InferenceWrapper(model), example, check_trace=False)
        traced = torch.jit.freeze(traced.eval())
        if backend == 'torchscript_opt':
            traced = torch.jit.optimize_for_inference(traced)
    return traced


def main():
    parser = argparse.ArgumentParser(description="Benchmark batch-one v3 ensemble inference without file I/O.")
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--checkpoint_dirs', nargs='+', required=True)
    parser.add_argument('--aux_position_data_dirs', nargs='*', default=[])
    parser.add_argument('--aux_position_checkpoint_dirs', nargs='*', default=[])
    parser.add_argument('--orientation_data_dir', required=True)
    parser.add_argument('--orientation_checkpoint_dir', required=True)
    parser.add_argument('--split', choices=['train', 'val'], default='val')
    parser.add_argument('--iterations', type=int, default=2000)
    parser.add_argument('--warmup', type=int, default=100)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--interop_threads', type=int, default=1)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='cpu')
    parser.add_argument(
        '--input_residency', choices=['device', 'host_pinned'], default='device',
        help='Use host_pinned on CUDA to include per-row CPU-to-GPU feature transfer.',
    )
    parser.add_argument(
        '--backend',
        choices=[
            'eager', 'torchscript', 'torchscript_opt',
            'torchscript_ensemble', 'cuda_graph',
        ],
        default='eager',
    )
    parser.add_argument('--out', default=None)
    args = parser.parse_args()
    if len(args.aux_position_data_dirs) != len(args.aux_position_checkpoint_dirs):
        raise ValueError("Auxiliary position data/checkpoint lists must have equal length")

    if args.threads < 1 or args.interop_threads < 1:
        raise ValueError("Thread counts must be positive")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.interop_threads)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but PyTorch cannot access a CUDA device. "
            "Check the NVIDIA driver, or use --device cpu/auto."
        )
    device = torch.device(
        'cuda' if args.device in {'auto', 'cuda'} and torch.cuda.is_available() else 'cpu'
    )
    if args.input_residency == 'host_pinned' and device.type != 'cuda':
        raise ValueError("host_pinned input residency requires --device cuda")
    feature_storage_device = torch.device(
        'cpu' if args.input_residency == 'host_pinned' else device
    )
    filename = 'train.npz' if args.split == 'train' else 'val.npz'
    position_features = torch.tensor(
        np.load(Path(args.data_dir) / filename)['emf'].astype(np.float32),
        device=feature_storage_device,
    )
    orientation_features = torch.tensor(
        np.load(Path(args.orientation_data_dir) / filename)['emf'].astype(np.float32),
        device=feature_storage_device,
    )
    if len(position_features) != len(orientation_features):
        raise ValueError("Position and orientation benchmark datasets are not aligned")
    aux_position_features = [
        torch.tensor(
            np.load(Path(path) / filename)['emf'].astype(np.float32),
            device=feature_storage_device,
        )
        for path in args.aux_position_data_dirs
    ]
    if args.input_residency == 'host_pinned':
        position_features = position_features.pin_memory()
        orientation_features = orientation_features.pin_memory()
        aux_position_features = [values.pin_memory() for values in aux_position_features]
    if any(len(values) != len(position_features) for values in aux_position_features):
        raise ValueError("Auxiliary position benchmark datasets are not aligned")

    position_models = [load_model(path, device)[0] for path in args.checkpoint_dirs]
    aux_position_models = [load_model(path, device)[0] for path in args.aux_position_checkpoint_dirs]
    orientation_model = load_model(args.orientation_checkpoint_dir, device)[0]

    def runtime_batch(values, index):
        sample = values[index:index + 1]
        return (
            sample.to(device, non_blocking=True)
            if sample.device.type == 'cpu' and device.type == 'cuda'
            else sample
        )

    position_example = runtime_batch(position_features, 0)
    orientation_example = runtime_batch(orientation_features, 0)
    aux_position_examples = [runtime_batch(values, 0) for values in aux_position_features]
    parameter_count = sum(
        sum(parameter.numel() for parameter in model.parameters())
        for model in [*position_models, *aux_position_models, orientation_model]
    )
    ensemble_model = None
    if args.backend == 'cuda_graph' and device.type != 'cuda':
        raise ValueError("cuda_graph backend requires --device cuda")
    if args.backend in {'torchscript_ensemble', 'cuda_graph'}:
        with torch.inference_mode():
            ensemble_model = torch.jit.trace(
                EnsembleInferenceWrapper(
                    position_models, aux_position_models, orientation_model,
                ),
                (
                    position_example, orientation_example,
                    tuple(aux_position_examples),
                ),
                check_trace=False,
            )
            ensemble_model = torch.jit.freeze(ensemble_model.eval())
    else:
        position_models = [
            prepare_backend(model, position_example, args.backend)
            for model in position_models
        ]
        aux_position_models = [
            prepare_backend(model, values, args.backend)
            for model, values in zip(aux_position_models, aux_position_examples)
        ]
        orientation_model = prepare_backend(
            orientation_model, orientation_example, args.backend,
        )

    def forward(model, features):
        return (
            model(features, return_normalized=False)
            if args.backend == 'eager'
            else model(features)
        )

    cuda_graph = None
    static_position_features = None
    static_orientation_features = None
    static_aux_position_features = None
    static_ensemble_outputs = None
    if args.backend == 'cuda_graph':
        static_position_features = position_example.clone()
        static_orientation_features = orientation_example.clone()
        static_aux_position_features = tuple(values.clone() for values in aux_position_examples)
        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream), torch.inference_mode():
            for _ in range(3):
                ensemble_model(
                    static_position_features,
                    static_orientation_features,
                    static_aux_position_features,
                )
        torch.cuda.current_stream().wait_stream(warmup_stream)
        cuda_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(cuda_graph), torch.inference_mode():
            static_ensemble_outputs = ensemble_model(
                static_position_features,
                static_orientation_features,
                static_aux_position_features,
            )

    def infer(index):
        with torch.inference_mode():
            if cuda_graph is not None:
                static_position_features.copy_(
                    position_features[index:index + 1], non_blocking=True,
                )
                static_orientation_features.copy_(
                    orientation_features[index:index + 1], non_blocking=True,
                )
                for target, source in zip(static_aux_position_features, aux_position_features):
                    target.copy_(source[index:index + 1], non_blocking=True)
                cuda_graph.replay()
                return static_ensemble_outputs[0].cpu().numpy()
            if ensemble_model is not None:
                return ensemble_model(
                    runtime_batch(position_features, index),
                    runtime_batch(orientation_features, index),
                    tuple(runtime_batch(values, index) for values in aux_position_features),
                )[0].cpu().numpy()
            position_predictions = [
                forward(model, runtime_batch(position_features, index))[:, :3]
                for model in position_models
            ]
            position_predictions.extend(
                forward(model, runtime_batch(values, index))[:, :3]
                for model, values in zip(aux_position_models, aux_position_features)
            )
            orientation_prediction = forward(
                orientation_model, runtime_batch(orientation_features, index),
            )[:, 3:]
            output = torch.cat([
                torch.stack(position_predictions).mean(dim=0),
                orientation_prediction,
            ], dim=1)
        return output.cpu().numpy()

    for index in range(args.warmup):
        infer(index % len(position_features))
    if device.type == 'cuda':
        torch.cuda.synchronize()
    durations_ms = []
    for index in range(args.iterations):
        start = time.perf_counter_ns()
        infer(index % len(position_features))
        if device.type == 'cuda':
            torch.cuda.synchronize()
        durations_ms.append((time.perf_counter_ns() - start) / 1e6)

    values = np.asarray(durations_ms)
    report = {
        'device': str(device),
        'backend': args.backend,
        'input_residency': args.input_residency,
        'host_to_device_input_transfer_included': args.input_residency == 'host_pinned',
        'torch_threads': args.threads,
        'torch_interop_threads': args.interop_threads,
        'batch_size': 1,
        'iterations': args.iterations,
        'models': len(position_models) + len(aux_position_models) + 1,
        'position_checkpoint_dirs': args.checkpoint_dirs,
        'aux_position_checkpoint_dirs': args.aux_position_checkpoint_dirs,
        'orientation_checkpoint_dir': args.orientation_checkpoint_dir,
        'total_parameters': parameter_count,
        'mean_ms': float(np.mean(values)),
        'p50_ms': float(np.percentile(values, 50)),
        'p95_ms': float(np.percentile(values, 95)),
        'p99_ms': float(np.percentile(values, 99)),
        'max_ms': float(np.max(values)),
        'p95_rate_hz': float(1000.0 / np.percentile(values, 95)),
        'aggregation': 'tensor_fast_path_no_redundant_so3_svd',
        'cuda_graph_capture_excluded_from_timing': args.backend == 'cuda_graph',
        'scope': 'network_forward_and_ensemble_only_no_acquisition_or_tcp_io',
    }
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as stream:
            json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
