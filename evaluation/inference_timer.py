import os
import torch
import time
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.residual_net import ResidualNet

def measure_latency_ms(model, device='cuda', n_warmup=100, n_runs=1000):
    x = torch.randn(1, 9).to(device)
    model.eval()
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(x)
        if device == 'cuda':
            torch.cuda.synchronize()
        
        t0 = time.perf_counter()
        for _ in range(n_runs):
            _ = model(x)
        if device == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
    return elapsed / n_runs * 1000   # ms

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Profiling on device: {device}")

    model = ResidualNet().to(device)
    
    # Run latency check
    latency = measure_latency_ms(model, device=device.type)
    print("\n" + "="*40)
    print("      INFERENCE LATENCY PROFILE")
    print("="*40)
    print(f"Avg Latency per Sample: {latency:.4f} ms")
    print(f"Throughput            : {1000.0/latency:.1f} FPS")
    print(f"Target (< 1.0 ms)     : {'MET (Success)' if latency < 1.0 else 'FAILED'}")
    print("="*40)

if __name__ == '__main__':
    main()
