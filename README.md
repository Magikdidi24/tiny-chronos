# tiny-chronos

[![tests](https://github.com/Magikdidi24/tiny-chronos/actions/workflows/tests.yml/badge.svg)](https://github.com/Magikdidi24/tiny-chronos/actions/workflows/tests.yml)

A small version of [Chronos](https://arxiv.org/abs/2403.07815), the model that does time series forecasting as language modeling: you scale the series, cut the values into bins and train a transformer to predict the next bin. Mine has 17.6M params and is trained on real and synthetic series. I test it zero-shot on ETTh1 and ETTm1.

This is v2. v1 was trained only on synthetic data and lost to seasonal naive. v2 adds real data and a bigger model, and the results of both are below.

## What changed in v2

v1 predicted flat lines and missed the sharp daily dips of ETTh1. Gaussian processes don't really make spikes or sudden level changes like these, so real data was the first thing to add. Most windows now come from 5 real datasets, mixed with TSMixup, the augmentation Chronos uses to get more variety out of a limited set of datasets. KernelSynth stays, but only for 10% of the windows like in the paper.

The rest brings it closer to the Chronos setup. The context goes from 256 to 512 points, the length Chronos uses. On ETTm1, 256 points were less than 3 days and 512 is more than 5. The bins go from 1024 to 4096 (Chronos has 4094 plus 2 special tokens), so a bin is under 1% of the context's mean absolute value instead of 3%. With more varied data and longer windows I made the model bigger too: 8 layers of width 384 instead of 6 of width 256, so 17.6M params instead of 5.35M. That's a bit less than Chronos-T5 Mini (20M). It trains for 30k steps instead of 20k, with lr 5e-4 instead of 1e-3.

The eval doesn't change, so the two versions can be compared directly. The model now sees 512 points, but the MASE scale is still computed on the last 256 like in v1. Otherwise the baseline scores would move just because the context got longer.

## How it works

Most of the training data comes from 5 datasets of the [Monash forecasting archive](https://forecastingdata.org/), all CC BY 4.0: electricity use per hour, traffic occupancy per hour, electricity demand in Australia every 30 min, pedestrian counts in Melbourne per hour and solar power every 10 min. `real.py` downloads them from Zenodo and keeps the series with no missing values that are at least one training window long. None of them is ETT, so the test stays zero-shot like in the paper.

A training window is picked in 3 steps: a dataset uniformly, then a series in it, then a start. Traffic has 862 series and the Australian demand only 5, so if I picked series directly 62% of the windows would be traffic and only 0.4% Australian demand.

The real windows go through TSMixup. For each window I draw k from 1 to 3, take k windows from random series, scale each one by the mean absolute value of its context and mix them with weights from a Dirichlet(1.5). With k = 1 it's just a normal window. This makes new series out of the few datasets I have, like the daily cycle of one with the trend of another.

The other 10% of the windows come from KernelSynth, the generator from the Chronos paper. Each series is a sample from a Gaussian process with a random kernel, made of 1 to 5 basic kernels (linear, RBF, periodic, noise...) combined with `+` or `*`. `synth.py` makes 100k series of 1024 points, so a window of 576 fits in them.

![KernelSynth samples](assets/synthetic_samples.png)

To tokenize a window I divide it by the mean absolute value of its context and cut [-15, 15] into 4096 uniform bins. The model is a small decoder-only transformer (8 layers, width 384) instead of the T5 used in Chronos, trained with plain cross-entropy on windows of 512 context points + 64 to predict. 30k steps of 64 windows, so 1.1B tokens, which took 38 min on my RTX 5070 Ti (about 500k tokens/s with `torch.compile`).

To forecast it samples the next 64 tokens one by one, 20 paths per series. The median of the paths is the point forecast and the quantiles give the probabilistic one.

## Results

Evaluated on the last 20% of ETTh1 (hourly) and ETTm1 (every 15 min), 10 windows per column, horizon 64. The baselines are seasonal naive (repeat the last day) and naive (repeat the last value). The rel. columns are the geometric mean of the ratio to seasonal naive, like in the paper, lower is better. The eval is small and doesn't follow the paper's protocol, so these numbers can't be compared with the ones in the Chronos paper.

v1 is the 5.35M param model (6 layers, width 256, 1024 bins, 256 context points) trained only on KernelSynth, 20k steps in about 7 min.

| dataset | method | MASE | WQL | rel. MASE | rel. WQL |
|---|---|---|---|---|---|
| ETTh1 | tiny-chronos v2 | 1.290 | 0.331 | 1.173 | 0.937 |
| ETTh1 | tiny-chronos v1 | 1.405 | 0.345 | 1.257 | 1.006 |
| ETTh1 | seasonal naive | 1.114 | 0.335 | 1.000 | 1.000 |
| ETTh1 | naive | 1.746 | 0.544 | 1.558 | 1.558 |
| ETTm1 | tiny-chronos v2 | 1.359 | 0.337 | 1.127 | 0.917 |
| ETTm1 | tiny-chronos v1 | 1.329 | 0.333 | 1.220 | 0.962 |
| ETTm1 | seasonal naive | 1.070 | 0.316 | 1.000 | 1.000 |
| ETTm1 | naive | 1.796 | 0.584 | 1.469 | 1.469 |

v2 is better than v1 in both rel. columns on both datasets. On WQL it now beats seasonal naive window by window (rel. WQL 0.94 and 0.92), and on ETTh1 even the plain average WQL is a bit lower (0.331 against 0.335). The point forecasts still lose: rel. MASE is 1.17 and 1.13, so 13 to 17% worse than repeating the last day. On ETTm1 the plain averages didn't move much (MASE 1.359 against 1.329, WQL 0.337 against 0.333), the gain there only shows in the per-window ratios.

![ETTh1 forecasts of v2](assets/forecast_etth1.png)

ETTh1 has these sharp dips every day. v2 now follows the daily cycle of LUFL, but on HUFL and MUFL it still predicts something smooth and the dips fall outside its 10-90% band, where seasonal naive just copies them from the day before.

![ETTm1 forecasts of v2](assets/forecast_ettm1.png)

On ETTm1, v1 got the MUFL forecast completely wrong (around -10 while the real value stays near 10). v2 gets that one right. The v1 plots are in [assets/v1](assets/v1).

## Running it

It needs a GPU, on CPU the full training would take forever. I run it on my 5070 Ti.

```bash
pip install -r requirements.txt
python real.py              # downloads the 5 Monash datasets (about 50 MB), writes data/real.npz
python synth.py --plot      # data/synth.npy and assets/synthetic_samples.png
python train.py --compile   # out/ckpt.pt, 38 min on a 5070 Ti
python evaluate.py          # downloads ETT, writes out/eval.json and the forecast plots
```

The tests don't need any data (`pytest -q`). For a quick CPU run with a tiny model and only the smallest dataset:

```bash
python real.py --datasets solar_10_minutes_dataset
python synth.py --n 2000
python train.py --dim 64 --n_layers 2 --n_heads 4 --ctx_len 128 --max_steps 50 --warmup_steps 10 --eval_every 25
python evaluate.py --quick
```
