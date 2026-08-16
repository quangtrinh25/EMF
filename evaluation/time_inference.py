import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.residual_net import ResidualNet


def load_model(checkpoint_dir, device):
    model = ResidualNet().to(device)
    ckpt_path = os.path.join(checkpoint_dir, 'resnet_best.pt')
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(checkpoint_dir, 'best.pt')
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.set_normalization(
        np.load(os.path.join(checkpoint_dir, 'emf_mean.npy')),
        np.load(os.path.join(checkpoint_dir, 'emf_std.npy')),
        np.load(os.path.join(checkpoint_dir, 'pose_mean.npy')),
        np.load(os.path.join(checkpoint_dir, 'pose_std.npy')),
    )
    model.eval()
    return model, ckpt_path


def measure(model, device, batch_size, warmup, runs):
    x = torch.randn(batch_size, 9, dtype=torch.float32, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            model(x, return_normalized=False)
        if device.type == 'cuda':
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(runs):
            model(x, return_normalized=False)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

    ms_per_batch = elapsed / runs * 1000.0
    ms_per_sample = ms_per_batch / batch_size
    samples_per_sec = batch_size * runs / elapsed
    return ms_per_batch, ms_per_sample, samples_per_sec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', default='checkpoints/capsule_hybrid_final')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--warmup', type=int, default=200)
    parser.add_argument('--runs', type=int, default=5000)
    parser.add_argument('--out_csv', default='results/capsule_hybrid_final/inference_time.csv')
    args = parser.parse_args()

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    model, ckpt_path = load_model(args.checkpoint_dir, device)
    ms_batch, ms_sample, samples_per_sec = measure(
        model,
        device,
        args.batch_size,
        args.warmup,
        args.runs,
    )

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, 'w') as f:
        f.write('checkpoint,device,batch_size,warmup,runs,ms_per_batch,ms_per_sample,samples_per_sec\n')
        f.write(
            f'{ckpt_path},{device.type},{args.batch_size},{args.warmup},{args.runs},'
            f'{ms_batch:.6f},{ms_sample:.6f},{samples_per_sec:.3f}\n'
        )

    print(f"Checkpoint: {ckpt_path}")
    print(f"Device: {device.type}")
    print(f"Batch size: {args.batch_size}")
    print(f"Latency per sample: {ms_sample:.6f} ms")
    print(f"Latency per batch: {ms_batch:.6f} ms")
    print(f"Throughput: {samples_per_sec:.3f} samples/sec")
    print(f"Saved: {args.out_csv}")


if __name__ == '__main__':
    main()
