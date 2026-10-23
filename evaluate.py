"""Zero-shot MASE and WQL on ETTh1 and ETTm1, against seasonal naive and naive."""
import argparse
import json
import os
import urllib.request

import matplotlib.pyplot as plt
import numpy as np
import torch

from model import ModelConfig, TinyChronos

URL = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/{}.csv"
DATASETS = {"ETTh1": 24, "ETTm1": 96}  # season = one day
QUANTILES = np.arange(1, 10) / 10
METHODS = ["tiny-chronos", "seasonal naive", "naive"]


def load(name):
    path = f"data/{name}.csv"
    if not os.path.exists(path):
        os.makedirs("data", exist_ok=True)
        urllib.request.urlretrieve(URL.format(name), path + ".part")
        os.replace(path + ".part", path)  # no half-downloaded csv if it gets interrupted
    with open(path) as f:
        columns = f.readline().strip().split(",")[1:]  # skip the date column
    return columns, np.loadtxt(path, delimiter=",", skiprows=1, usecols=range(1, len(columns) + 1)).T


def windows(y, ctx_len, h, n):
    # n evenly spaced windows in the last 20%, no overlap
    start = int(0.8 * len(y))
    stride = (len(y) - start) // n
    assert stride >= h, "too many windows to fit in the test region without overlap"
    origins = start + stride * np.arange(n)
    return np.stack([y[o - ctx_len:o] for o in origins]), np.stack([y[o:o + h] for o in origins])


# baselines and metrics take (windows, time) arrays
def seasonal_naive(context, season, horizon):
    # repeat the last season
    return np.tile(context[:, -season:], -(-horizon // season))[:, :horizon]


def naive(context, horizon):
    return np.repeat(context[:, -1:], horizon, 1)


def mase(y, point, context, season):
    # MAE / in-sample MAE of seasonal naive on the context
    return np.abs(y - point).mean(1) / np.abs(context[:, season:] - context[:, :-season]).mean(1)


def wql(y, quantiles):
    # weighted quantile loss, quantiles[i] is the forecast for QUANTILES[i]
    q = QUANTILES[:, None, None]
    pinball = np.maximum(q * (y - quantiles), (q - 1) * (y - quantiles))
    return 2 * pinball.sum(2).mean(0) / np.abs(y).sum(1)


def plot(name, examples, path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.5))
    for ax, (column, context, future, samples) in zip(axes.flat, examples):
        h = len(future)
        tail = context[-2 * h:]
        lo, median, hi = np.quantile(samples, [0.1, 0.5, 0.9], 0)
        ax.plot(np.arange(-len(tail), h), np.r_[tail, future], color="#2a78d6", lw=1.2, label="actual")
        ax.fill_between(np.arange(h), lo, hi, color="#eb6834", alpha=0.25, lw=0, label="10-90% interval")
        ax.plot(np.arange(h), median, color="#eb6834", lw=1.5, label="median forecast")
        ax.axvline(0, color="#898781", lw=0.8, ls="--")
        ax.set_title(column)
        ax.spines[["top", "right"]].set_visible(False)
    axes.flat[0].legend(frameon=False)
    fig.suptitle(f"{name}: zero-shot forecasts (time steps from the forecast start)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="out/ckpt.pt")
    p.add_argument("--horizon", type=int, default=64)
    p.add_argument("--windows", type=int, default=10, help="non-overlapping forecast windows per column")
    p.add_argument("--samples", type=int, default=20, help="sampled paths per forecast")
    p.add_argument("--quick", action="store_true", help="2 windows and 5 samples, for a CPU smoke run")
    args = p.parse_args()
    if args.quick:
        args.windows, args.samples = 2, 5

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    torch.set_float32_matmul_precision("high")
    ckpt = torch.load(args.ckpt, map_location=device)
    model = TinyChronos(ModelConfig(**ckpt["config"])).to(device).eval()
    model.load_state_dict(ckpt["model"])
    h = args.horizon
    ctx_len = model.config.max_len - h  # same context as forecast() uses
    # MASE needs more than one season of context
    assert ctx_len > max(DATASETS.values()), f"--horizon {h} leaves only {ctx_len} context points, not more than one season"

    results = {}
    table = ["| dataset | method | MASE | WQL | rel. MASE | rel. WQL |", "|---|---|---|---|---|---|"]
    for name, season in DATASETS.items():
        columns, data = load(name)
        scores, examples = {m: [] for m in METHODS}, []
        for column, y in zip(columns, data):
            context, future = windows(y, ctx_len, h, args.windows)
            samples = model.forecast(torch.tensor(context, dtype=torch.float32, device=device), h, args.samples)
            samples = samples.cpu().numpy()
            snaive, last = seasonal_naive(context, season, h), naive(context, h)
            forecasts = {  # method: (point, quantiles)
                "tiny-chronos": (np.median(samples, 1), np.quantile(samples, QUANTILES, 1)),
                "seasonal naive": (snaive, snaive[None]),
                "naive": (last, last[None]),
            }
            for m, (point, quantiles) in forecasts.items():
                scores[m].append(np.stack([mase(future, point, context, season), wql(future, quantiles)], 1))
            examples.append((column, context[0], future[0], samples[0]))

        scores = {m: np.concatenate(s) for m, s in scores.items()}  # (all windows, [mase, wql])
        results[name] = {}
        for m in METHODS:
            mean = scores[m].mean(0)
            # geometric mean of the ratio to seasonal naive, like in the Chronos paper
            rel = np.exp(np.log(scores[m] / scores["seasonal naive"]).mean(0))
            results[name][m] = {"mase": float(mean[0]), "wql": float(mean[1]),
                                "rel_mase": float(rel[0]), "rel_wql": float(rel[1])}
            table.append(f"| {name} | {m} | {mean[0]:.3f} | {mean[1]:.3f} | {rel[0]:.3f} | {rel[1]:.3f} |")
        plot(name, examples[::2], f"assets/forecast_{name.lower()}.png")

    table = "\n".join(table)
    print(f"\nzero-shot, horizon {h}, {args.windows} windows per column, {args.samples} samples; "
          "rel. = geometric mean of the per-window ratio to seasonal naive (lower is better)\n")
    print(table)
    os.makedirs("out", exist_ok=True)
    with open("out/eval.json", "w") as f:
        json.dump({"ckpt": args.ckpt, "horizon": h, "windows": args.windows, "samples": args.samples,
                   "results": results, "table": table}, f, indent=2)
    print("saved out/eval.json")


if __name__ == "__main__":
    main()
