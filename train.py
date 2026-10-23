"""Pretrain on the KernelSynth series from synth.py."""
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


# ~7 min with --compile on an RTX 5070 Ti
@dataclass
class TrainConfig:
    dim: int = 256
    n_layers: int = 6
    n_heads: int = 8
    ctx_len: int = 256
    horizon: int = 64
    batch_size: int = 64
    max_steps: int = 20000
    lr: float = 1e-3
    min_lr: float = 1e-4
    warmup_steps: int = 500
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    log_every: int = 100
    eval_every: int = 1000
    eval_batches: int = 20
    data_dir: str = "data"
    out: str = "out/ckpt.pt"
    compile: bool = False
    seed: int = 0


def get_batch(series, cfg, generator=None):
    # scale with the context part only, like at forecast time
    win = cfg.ctx_len + cfg.horizon
    rows = torch.randint(len(series), (cfg.batch_size, 1), device=series.device, generator=generator)
    starts = torch.randint(series.size(1) - win + 1, (cfg.batch_size, 1), device=series.device, generator=generator)
    x = series[rows, starts + torch.arange(win, device=series.device)]
    tokens = encode(x, mean_scale(x[:, :cfg.ctx_len]))
    return tokens[:, :-1], tokens[:, 1:]


def lr_at(step, cfg):
    # linear warmup, then cosine decay down to min_lr
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    return cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model, series, cfg, autocast):
    model.eval()
    gen = torch.Generator(series.device).manual_seed(0)  # same val windows every time
    losses = []
    for _ in range(cfg.eval_batches):
        with autocast:
            losses.append(model(*get_batch(series, cfg, gen))[1].item())
    model.train()
    return sum(losses) / len(losses)


def train(cfg: TrainConfig):
    torch.manual_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    cuda = device == "cuda"
    torch.set_float32_matmul_precision("high")  # tf32
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if cuda else contextlib.nullcontext()

    series = torch.from_numpy(np.load(os.path.join(cfg.data_dir, "synth.npy"))).to(device)
    assert series.size(1) >= cfg.ctx_len + cfg.horizon, "series are shorter than a training window"
    n_val = max(1, len(series) // 50)  # hold out 2% of the series
    train_series, val_series = series[n_val:], series[:n_val]

    mcfg = ModelConfig(cfg.dim, cfg.n_layers, cfg.n_heads, cfg.ctx_len + cfg.horizon)
    model = TinyChronos(mcfg).to(device)
    fwd = torch.compile(model) if cfg.compile and cuda else model
    tokens_per_step = cfg.batch_size * (cfg.ctx_len + cfg.horizon - 1)
    print(f"{model.num_params() / 1e6:.2f}M params on {device}, {len(train_series):,} training series, "
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
            loss = fwd(*get_batch(train_series, cfg))[1]
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
            print(f"step {step:5d} | loss {metrics[-1]['train_loss']:.3f} | lr {lr:.1e} | {metrics[-1]['tok_s']:,.0f} tok/s")
            loss_sum, logged, t0 = 0.0, step, time.perf_counter()
        if step % cfg.eval_every == 0 or last:
            t = time.perf_counter()
            val = evaluate(fwd, val_series, cfg, autocast)
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
