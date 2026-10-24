"""Pretrain on real series from real.py, mixed with TSMixup, and on KernelSynth series from synth.py."""
import argparse
import contextlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass, fields

import numpy as np
import torch

from model import ModelConfig, TinyChronos, encode, mean_scale


# 38 min with --compile on an RTX 5070 Ti
@dataclass
class TrainConfig:
    dim: int = 384
    n_layers: int = 8
    n_heads: int = 6
    ctx_len: int = 512
    horizon: int = 64
    batch_size: int = 64
    max_steps: int = 30000
    lr: float = 5e-4
    min_lr: float = 5e-5
    warmup_steps: int = 1000
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    synth_share: float = 0.1  # KernelSynth windows, the rest are real windows mixed with TSMixup
    log_every: int = 100
    eval_every: int = 1000
    eval_batches: int = 20
    data_dir: str = "data"
    out: str = "out/ckpt.pt"
    compile: bool = False
    seed: int = 0


def split(values, lengths, dataset, device):
    """Concatenated series, grouped by dataset -> (train, val) pools of (values, offsets, lengths, first, count).

    In a pool, dataset d has the series first[d] to first[d] + count[d] - 1.
    """
    count = np.bincount(dataset)
    first = np.cumsum(count) - count
    n_val = np.maximum(1, count // 50)  # hold out 2% of the series of each dataset
    assert (count > n_val).all(), "each dataset needs at least 2 series, one goes to validation"
    t = lambda a: torch.as_tensor(a, device=device)
    shared = t(values), t(np.cumsum(lengths) - lengths), t(lengths)
    return (*shared, t(first + n_val), t(count - n_val)), (*shared, t(first), t(n_val))


def randint_below(high, generator=None):
    # torch.randint takes a single high, here every row has its own
    return (torch.rand(high.shape, device=high.device, generator=generator) * high).long()


def sample_windows(pool, n, win, generator=None):
    # a dataset uniformly, then one of its series, then a start, so small datasets aren't drowned by traffic
    values, offsets, lengths, first, count = pool
    d = torch.randint(len(first), (n,), device=values.device, generator=generator)
    series = first[d] + randint_below(count[d], generator)
    starts = offsets[series] + randint_below(lengths[series] - win + 1, generator)
    return values[starts[:, None] + torch.arange(win, device=values.device)]


def tsmixup(x, ctx_len, generator=None):
    """(B, 3, win) windows -> (B, win) mixes of the first k windows of each row, k in {1, 2, 3}."""
    k = torch.randint(1, 4, (len(x), 1), device=x.device, generator=generator)
    # dirichlet(1.5) weights are normalized gamma(1.5) draws, torch.distributions can't take a generator
    w = torch._standard_gamma(torch.full((len(x), 3), 1.5, device=x.device), generator=generator)
    w = w * (torch.arange(3, device=x.device) < k)
    w = w / w.sum(1, keepdim=True)
    return (w[..., None] * x / mean_scale(x[..., :ctx_len])).sum(1)


def get_batch(synth, real, cfg, generator=None):
    win, B = cfg.ctx_len + cfg.horizon, cfg.batch_size
    x = tsmixup(sample_windows(real, 3 * B, win, generator).view(B, 3, win), cfg.ctx_len, generator)
    is_synth = torch.rand(B, 1, device=x.device, generator=generator) < cfg.synth_share
    x = torch.where(is_synth, sample_windows(synth, B, win, generator), x)
    # scale with the context part only, like at forecast time
    tokens = encode(x, mean_scale(x[:, :cfg.ctx_len]))
    return tokens[:, :-1], tokens[:, 1:]


def lr_at(step, cfg):
    # linear warmup, then cosine decay down to min_lr
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    return cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model, synth, real, cfg, autocast):
    model.eval()
    gen = torch.Generator(real[0].device).manual_seed(0)  # same val windows every time
    losses = []
    for _ in range(cfg.eval_batches):
        with autocast:
            losses.append(model(*get_batch(synth, real, cfg, gen))[1].item())
    model.train()
    return sum(losses) / len(losses)


def train(cfg: TrainConfig):
    torch.manual_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    cuda = device == "cuda"
    torch.set_float32_matmul_precision("high")  # tf32
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if cuda else contextlib.nullcontext()

    win = cfg.ctx_len + cfg.horizon
    synth = np.load(os.path.join(cfg.data_dir, "synth.npy"))
    real = np.load(os.path.join(cfg.data_dir, "real.npz"))
    assert min(synth.shape[1], real["lengths"].min()) >= win, "series are shorter than a training window"
    synth_train, synth_val = split(synth.ravel(), np.full(len(synth), synth.shape[1]), np.zeros(len(synth), int), device)
    real_train, real_val = split(real["values"], real["lengths"], real["dataset"], device)

    mcfg = ModelConfig(cfg.dim, cfg.n_layers, cfg.n_heads, win)
    model = TinyChronos(mcfg).to(device)
    fwd = torch.compile(model) if cfg.compile and cuda else model
    tokens_per_step = cfg.batch_size * (win - 1)
    print(f"{model.num_params() / 1e6:.2f}M params on {device}, {real_train[-1].sum():,} real series from "
          f"{len(real['names'])} datasets and {synth_train[-1].sum():,} synthetic ones for training, "
          f"{tokens_per_step * cfg.max_steps / 1e6:.1f}M training tokens")

    # no weight decay on biases and norms
    params = list(model.parameters())
    opt = torch.optim.AdamW([
        {"params": [p for p in params if p.dim() >= 2], "weight_decay": cfg.weight_decay},
        {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0},
    ], lr=cfg.lr, betas=(0.9, 0.95))

    os.makedirs(os.path.dirname(cfg.out) or ".", exist_ok=True)
    metrics_path = os.path.splitext(cfg.out)[0] + "_metrics.json"
    metrics, loss_sum, logged = [], 0.0, 0
    start = t0 = time.perf_counter()
    for step in range(1, cfg.max_steps + 1):
        lr = lr_at(step - 1, cfg)
        for g in opt.param_groups:
            g["lr"] = lr
        with autocast:
            # loss on every position, context included
            loss = fwd(*get_batch(synth_train, real_train, cfg))[1]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        opt.zero_grad(set_to_none=True)
        loss_sum += loss.detach()

        last = step == cfg.max_steps
        if step % cfg.log_every == 0 or last:
            n = step - logged
            metrics.append({"step": step, "tokens": step * tokens_per_step, "train_loss": loss_sum.item() / n,
                            "lr": lr, "tok_s": n * tokens_per_step / (time.perf_counter() - t0)})
            eta = (cfg.max_steps - step) * tokens_per_step / metrics[-1]["tok_s"] / 60
            print(f"step {step:5d} | loss {metrics[-1]['train_loss']:.3f} | lr {lr:.1e} | "
                  f"{metrics[-1]['tok_s']:,.0f} tok/s | eta {eta:.0f} min")
            loss_sum, logged, t0 = 0.0, step, time.perf_counter()
        if step % cfg.eval_every == 0 or last:
            t = time.perf_counter()
            val = evaluate(fwd, synth_val, real_val, cfg, autocast)
            metrics.append({"step": step, "tokens": step * tokens_per_step, "val_loss": val})
            print(f"step {step:5d} | val loss {val:.3f}")
            torch.save({"model": model.state_dict(), "config": asdict(mcfg)}, cfg.out)
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=1)
            t0 += time.perf_counter() - t  # don't count eval time in tok/s

    print(f"saved {cfg.out} and {metrics_path} ({(time.perf_counter() - start) / 60:.1f} min)")
    return metrics


def parse_args():
    p = argparse.ArgumentParser()
    for f in fields(TrainConfig):
        if isinstance(f.default, bool):
            p.add_argument(f"--{f.name}", action=argparse.BooleanOptionalAction, default=f.default)
        else:
            p.add_argument(f"--{f.name}", type=type(f.default), default=f.default)
    return TrainConfig(**vars(p.parse_args()))


if __name__ == "__main__":
    train(parse_args())
