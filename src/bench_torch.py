"""PyTorch conv3d of the same 27-point stencil, timed against the C++ reference.

This is the CPU comparison. The brief's 1.5–2× against cuSPARSE and the roofline
are measured by src/stencil_27.cu on the NVIDIA card, not here.
"""

from __future__ import annotations

import math
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
N = 48
H = 1.0


def weights() -> torch.Tensor:
    kernel = torch.zeros(1, 1, 3, 3, 3)
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                hops = abs(dx) + abs(dy) + abs(dz)
                value = {0: -14.0, 1: 1.0, 2: 0.5, 3: 0.25}[hops]
                kernel[0, 0, dz + 1, dy + 1, dx + 1] = value
    return kernel / (6.0 * H * H)


def field(n: int) -> torch.Tensor:
    axis = torch.arange(n)
    z, y, x = torch.meshgrid(axis, axis, axis, indexing="ij")
    values = torch.sin(2 * math.pi * x / n) * torch.cos(2 * math.pi * y / n) * torch.cos(4 * math.pi * z / n)
    return values[None, None].float()


def main() -> None:
    binary = ROOT / "runs" / "stencil_cpu"
    runs = ROOT / "runs"
    runs.mkdir(exist_ok=True)
    if not binary.exists():
        subprocess.check_call(["c++", "-O3", "-std=c++17", "src/stencil_cpu.cpp", "-o", str(binary)], cwd=ROOT)
    completed = subprocess.run([str(binary)], cwd=ROOT, check=False, capture_output=True, text=True)
    print(completed.stdout, end="")
    raw = np.fromfile(runs / "cpu_float.bin", dtype=np.float32).reshape(N, N, N)

    volume = field(N)
    kernel = weights()
    padded = F.pad(volume, (1, 1, 1, 1, 1, 1), mode="circular")
    with torch.no_grad():
        conv = F.conv3d(padded, kernel)
    gap = float((conv[0, 0] - torch.from_numpy(raw)).abs().max())
    print(f"torch_vs_cpu_inf {gap:.9e}")

    def once():
        with torch.no_grad():
            F.conv3d(padded, kernel)

    once()
    started = time.perf_counter()
    repeats = 5
    for _ in range(repeats):
        once()
    torch_s = (time.perf_counter() - started) / repeats
    print(f"torch_seconds {torch_s:.6f}")
    print("gpu_bandwidth_and_cusparse not_measured_on_this_machine")
    if completed.returncode != 0 or gap > 1e-6:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
