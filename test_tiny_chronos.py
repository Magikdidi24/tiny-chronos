import contextlib

import numpy as np
import torch

from evaluate import mase, naive, seasonal_naive, windows, wql
from model import LIMIT, N_BINS, ModelConfig, TinyChronos, decode, encode, mean_scale
from synth import PERIODS, kernel_bank, kernel_synth
from train import TrainConfig, evaluate, get_batch, train


def test_tokenizer_roundtrip():
    x = torch.randn(4, 200) * torch.tensor([[1e-3], [1.0], [50.0], [1e4]])
    scale = mean_scale(x)
    err = (decode(encode(x, scale), scale) - x).abs()
    assert (err <= (LIMIT / N_BINS + 1e-5) * scale).all()  # half a bin width, in data units
    assert encode(torch.tensor([[-1e9, 1e9]]), torch.ones(1, 1)).tolist() == [[0, N_BINS - 1]]


def test_model_is_causal():
    torch.manual_seed(0)
    m = TinyChronos(ModelConfig(dim=32, n_layers=2, n_heads=4, max_len=24))
    x = torch.randint(0, N_BINS, (2, 24))
    y = x.clone()
    y[:, -1] = (x[:, -1] + 1) % N_BINS  # change only the last token
    for training in (True, False):  # eval + no_grad takes pytorch's fast path
        m.train(training)
        with torch.no_grad():
            assert torch.allclose(m(x)[0][:, :-1], m(y)[0][:, :-1], atol=1e-5)


def test_forecast_rows_are_windows_and_values_are_rescaled():
    m = TinyChronos(ModelConfig(dim=8, n_layers=1, n_heads=2, max_len=16))
    m.forward = lambda idx: (torch.nn.functional.one_hot(torch.full_like(idx, 700), N_BINS) * 1e4, None)  # always bin 700
    context = torch.arange(1.0, 13.0) * torch.tensor([[1.0], [100.0]])  # 2 windows of 12 points, 100x apart
    out = m.forecast(context, horizon=8, n_samples=3)
    # rows stay separate, each scaled by its own last 8 points
    assert torch.allclose(out, decode(torch.full((2, 3, 8), 700), context[:, -8:].abs().mean(1)[:, None, None]))


def test_kernel_synth():
    x = kernel_synth(16, length=64)
    assert x.shape == (16, 64) and x.dtype == np.float32 and np.isfinite(x).all()
    bank = kernel_bank(200, "cpu")
    assert len(bank) == 24 and torch.allclose(bank, bank.transpose(1, 2))
    # periodic kernels are back to 1 after p steps
    assert all(abs(k[0, p] - 1) < 1e-9 for k, p in zip(bank[-len(PERIODS):], PERIODS))


def test_mase():
    rng = np.random.default_rng(0)
    context, y = rng.normal(size=(3, 48)), rng.normal(size=(3, 8))
    assert np.allclose(mase(y, y, context, 24), 0)
    # slope 1 line: naive errors are 1, 2, 3 and the lag-3 in-sample error is 3
    line = np.arange(13.0)[None]
    assert np.allclose(mase(line[:, 10:], naive(line[:, :10], 3), line[:, :10], 3), 2 / 3)


def test_wql():
    rng = np.random.default_rng(0)
    y = rng.normal(size=(3, 8))
    assert np.allclose(wql(y, y[None]), 0)
    # with a point forecast WQL is just WAPE
    point = y + rng.normal(size=y.shape)
    assert np.allclose(wql(y, point[None]), np.abs(y - point).sum(1) / np.abs(y).sum(1))


def test_seasonal_naive_on_periodic_series():
    t = np.arange(150)
    y = (np.sin(2 * np.pi * t / 24) + 0.5 * np.cos(2 * np.pi * t / 8))[None]
    context, future = y[:, :120], y[:, 120:]  # a horizon longer than one season
    forecast = seasonal_naive(context, 24, future.shape[1])
    assert np.allclose(forecast, future)  # MASE would be 0/0 here
    assert np.allclose(wql(future, forecast[None]), 0)


def test_eval_windows_follow_their_context():
    context, future = windows(np.arange(1000.0), ctx_len=50, h=8, n=10)  # on a ramp, each value is its time index
    assert context.shape == (10, 50) and future.shape == (10, 8)
    assert (future[:, 0] == context[:, -1] + 1).all()  # future starts right after its context
    assert future.min() >= 800 and (np.diff(future[:, 0]) >= 8).all()  # last 20% only, windows don't overlap


def test_batch_is_scaled_by_its_context_only():
    cfg = TrainConfig(ctx_len=6, horizon=2, batch_size=4)
    series = torch.tensor([[1.0] * 6 + [10.0] * 2])  # a single window: context 1s, horizon 10s
    x, y = get_batch(series, cfg)
    assert torch.equal(x[:, 1:], y[:, :-1])
    assert torch.equal(y[:, -2:], encode(torch.full((4, 2), 10.0), torch.ones(4, 1)))


def test_train_end_to_end(tmp_path):
    rng = np.random.default_rng(0)
    # random-phase sines, easy enough to learn something in 40 steps
    series = (np.sin(2 * np.pi * np.arange(48) / 12 + rng.uniform(0, 6.3, (64, 1))) + 3).astype(np.float32)
    np.save(tmp_path / "synth.npy", series)
    cfg = TrainConfig(dim=32, n_layers=2, n_heads=4, ctx_len=24, horizon=8, batch_size=16, max_steps=40,
                      warmup_steps=5, log_every=10, eval_every=40, eval_batches=2,
                      data_dir=str(tmp_path), out=str(tmp_path / "ckpt.pt"))
    metrics = train(cfg)
    assert metrics[-1]["val_loss"] < metrics[0]["train_loss"] - 0.5
    assert (tmp_path / "ckpt_metrics.json").exists()

    ckpt = torch.load(cfg.out)
    model = TinyChronos(ModelConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["model"])
    s, no_autocast = torch.from_numpy(series), contextlib.nullcontext()
    assert evaluate(model, s, cfg, no_autocast) == evaluate(model, s, cfg, no_autocast)  # same val windows
    samples = model.eval().forecast(s[:3], horizon=8, n_samples=5)
    assert samples.shape == (3, 5, 8) and torch.isfinite(samples).all()
