"""Synthetic training data with KernelSynth (algorithm 2 in the Chronos paper)."""
import argparse
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

PERIODS = [4, 6, 7, 12, 14, 24, 30, 48, 52, 60, 96, 168]  # in time steps


def kernel_bank(length, device):
    x = torch.linspace(0, 1, length, dtype=torch.float64, device=device)
    d = (x[:, None] - x[None, :]).abs()
    eye = torch.eye(length, dtype=torch.float64, device=device)
    return torch.stack(
        [torch.ones_like(d)]                                                     # constant
        + [s * eye for s in (0.1, 1.0)]                                           # white noise
        + [s**2 + torch.outer(x, x) for s in (0.0, 1.0, 10.0)]                    # linear (dot product)
        + [torch.exp(-d**2 / (2 * l**2)) for l in (0.1, 1.0, 10.0)]               # RBF
        + [(1 + d**2 / (2 * a)) ** -a for a in (0.1, 1.0, 10.0)]                  # rational quadratic
        # periodic, p in time steps
        + [torch.exp(-2 * torch.sin(torch.pi * d * (length - 1) / p) ** 2) for p in PERIODS]
    )


def random_kernel(bank, rng):
    # 1 to 5 kernels combined with random + or *
    kernel = bank[rng.randrange(len(bank))]
    for _ in range(rng.randrange(5)):
        other = bank[rng.randrange(len(bank))]
        kernel = kernel + other if rng.random() < 0.5 else kernel * other
    return kernel


def kernel_synth(n, length=1024, seed=0, batch=128):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng, gen = random.Random(seed), torch.Generator(device).manual_seed(seed)
    bank = kernel_bank(length, device)
    eye = torch.eye(length, dtype=torch.float64, device=device)
    chunks, done = [], 0
    while done < n:
        K = torch.stack([random_kernel(bank, rng) for _ in range(min(batch, n - done))])
        # jitter for cholesky, drop the draws that still fail
        jitter = 1e-6 * K.diagonal(dim1=1, dim2=2).mean(1)[:, None, None]
        L, info = torch.linalg.cholesky_ex(K + jitter * eye)
        z = torch.randn(len(K), length, 1, dtype=torch.float64, device=device, generator=gen)
        x = (L @ z)[info == 0, :, 0]
        chunks.append(x.float().cpu())
        done += len(x)
    return torch.cat(chunks).numpy()


def plot(series, path="assets/synthetic_samples.png"):
    fig, axes = plt.subplots(4, 2, figsize=(12, 7))
    for ax, s in zip(axes.flat, series):
        ax.plot(s, color="#2a78d6", lw=1)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("KernelSynth samples: Gaussian-process draws with random composite kernels")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150)
    print(f"saved {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=100_000, help="number of series")
    p.add_argument("--length", type=int, default=1024)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--plot", action="store_true", help="also plot 8 series to assets/synthetic_samples.png")
    args = p.parse_args()

    t0 = time.perf_counter()
    x = kernel_synth(args.n, args.length, args.seed)
    os.makedirs("data", exist_ok=True)
    np.save("data/synth.npy", x)
    print(f"{len(x):,} series of length {args.length} -> data/synth.npy ({time.perf_counter() - t0:.0f}s)")
    if args.plot:
        plot(x[:8])


if __name__ == "__main__":
    main()
