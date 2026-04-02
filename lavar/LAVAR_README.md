# LAVAR: Latent Autoencoder VAR for Forecasting

Languages: [English](LAVAR_README.md) | [한국어](LAVAR_README_KOR.md) | [日本語](LAVAR_README_JPN.md)

[![paper](https://img.shields.io/badge/paper-TBD-lightgrey)](#citation)
[![python](https://img.shields.io/badge/python-%3E%3D3.12-blue)](#installation)
[![pytorch](https://img.shields.io/badge/PyTorch-%3E%3D2.9.1-ee4c2c)](https://pytorch.org/)
[![pydantic](https://img.shields.io/badge/Pydantic-%3E%3D2.0-e92063)](https://docs.pydantic.dev/)
[![uv](https://img.shields.io/badge/uv-locked-6f42c1)](https://github.com/astral-sh/uv)

`lavar` is a compact, research-oriented PyTorch package for multi-step healthcare supply forecasting.
It implements a two-stage pipeline:

1. Stage 1 learns a latent representation and latent dynamics from non-supply covariates.
2. Stage 2 freezes the latent backbone and trains density-split supply heads on future supply targets.

## Public API

```python
from lavar import LAVARForecaster, LAVARConfig, RollingEvaluator, EvaluationResults
```

The main entry points are:

- `LAVARConfig`: pydantic config for model, training, and inference behavior
- `LAVARForecaster`: sklearn-style wrapper for `fit()`, `fit_heads()`, `predict()`, `save()`, and `load()`
- `RollingEvaluator`: rolling-origin evaluation helper
- `EvaluationResults`: summary object with `to_dataframe()` and `plot()`

## Current Architecture

### Stage 1 backbone

- Observation model: MLP encoder and MLP decoder
- Latent dynamics: configurable `VAR` or `GRU`
- Optional encoder augmentation: concatenate supply history to the Stage 1 input when `stage1_use_supply_history=True`

### Stage 2 supply modeling

- Stage 2 modes:
  - `baseline`: latent rollout only
  - `supply_history_latent`: latent rollout with supply-history-augmented encoder input
- Target splitting: dense / sparse / ultra-sparse buckets based on per-target nonzero rate
- Supported head families:
  - deterministic delta regression via MLP
  - deterministic delta regression via GRU decoder
  - probabilistic NB head
  - probabilistic ZINB head

### Default pipeline mapping

The current `LAVARForecaster` training pipeline uses density-split Stage 2 heads as follows:

| Bucket | Default head type | Notes |
| --- | --- | --- |
| Dense | `delta_mse` | Deterministic delta forecast |
| Sparse | `delta_mse` | Deterministic delta forecast |
| Ultra-sparse | `zinb` | Probabilistic count head |

`nb` is supported by the low-level Stage 2 model and trainer APIs, even though the default dispatcher currently routes the ultra-sparse bucket to `zinb`.

## Supported Head Matrix

| Head option | Class path | Backend | Output | Current usage |
| --- | --- | --- | --- | --- |
| `delta_mse` | `SupplyHeadMSE` | MLP | per-step `delta` | Default dense/sparse heads |
| `delta_mse` with `stage2_head_type="gru"` | `SupplyHeadGRU` via `LAVARWithSupplyGRU` | GRU | sequence of `delta` values | Available for baseline-mode delta heads |
| `nb` | `SupplyHeadNB` | MLP | `mu`, `theta` | Supported low-level option |
| `zinb` | `SupplyHeadZINB` | MLP | `pi`, `mu`, `theta` | Default ultra-sparse head |

Important implementation details:

- `stage2_head_type="gru"` only affects `delta_mse` heads.
- `LAVARWithSupplyGRU` is only used for deterministic delta heads.
- NB and ZINB heads remain pointwise MLP heads.
- `supply_history_latent` requires both `use_supply_history=True` and `stage1_use_supply_history=True`.

## Training Flow

1. `fit_stage1_shared(X, y)`
   - scales `X`
   - optionally scales `y` when Stage 1 consumes supply history
   - trains `LAVAR` with reconstruction loss and latent dynamics loss
   - optional multi-step latent supervision compares latent rollouts against encoded future observations
2. `fit_stage2_private(X, y)`
   - freezes Stage 1 weights
   - computes target density buckets on training windows only
   - trains bucket-specific supply heads
3. `predict(X_recent, y_recent)`
   - rolls out future latent states
   - applies the trained supply heads bucket-wise
   - runs prediction guardrails, optional zero gating, and optional integer rounding

For delta heads, Stage 2 predicts raw increments and integrates them from the last observed supply value `y0`.
For probabilistic heads, the default point forecast is the predicted mean `mu`.

## Quickstart

```python
import numpy as np

from lavar import LAVARConfig, LAVARForecaster, RollingEvaluator

cfg = LAVARConfig(
    device="cpu",
    dyn_p=7,
    horizon=14,
    latent_dim=8,
    latent_dynamics_type="gru",
    stage2_mode="baseline",
    stage2_head_type="gru",
    epochs_lavar=5,
    epochs_supply=5,
)

T, Dx, Dy = 512, 20, 6
X = np.random.randn(T, Dx).astype("float32")
y = np.abs(np.random.randn(T, Dy).astype("float32"))

model = LAVARForecaster(cfg)
model.fit(X, y)

X_recent = X[-cfg.dyn_p :]
y_recent = y[-cfg.dyn_p :]
forecast = model.predict(X_recent, y_recent=y_recent)

model.save("lavar_model.pth")
loaded = LAVARForecaster.load("lavar_model.pth")

results = RollingEvaluator(cfg).evaluate(X, y, fold_step=14, verbose=False)
print(results.summary)
```

Builder-style configuration is also available:

```python
cfg = (
    LAVARConfig.builder()
    .device("cpu")
    .latent(dim=8, encoder=[32, 16], decoder=[16, 32])
    .horizon(h=14, history=7)
    .build()
)
```

## Configuration Highlights

| Field | Default | Meaning |
| --- | --- | --- |
| `device` | `"mps"` | Training/inference device |
| `dyn_p` | `7` | Latent history length / dynamics order |
| `horizon` | `14` | Forecast horizon |
| `latent_dim` | `8` | Latent state dimension |
| `latent_dynamics_type` | `"var"` | Stage 1 latent dynamics: `var` or `gru` |
| `use_supply_history` | `False` | Concatenate supply history to Stage 2 encoder input |
| `stage1_use_supply_history` | `False` | Concatenate supply history during Stage 1 training |
| `stage2_mode` | `"baseline"` | Stage 2 training mode |
| `stage2_head_type` | `"mlp"` | Baseline-mode delta-head backend: `mlp` or `gru` |
| `dense_nonzero_rate_thr` | `0.70` | Dense bucket threshold |
| `ultra_nonzero_rate_thr` | `0.005` | Ultra-sparse bucket threshold |
| `horizon_loss_weight` | `"uniform"` | Baseline-mode delta loss weighting |
| `stage2_delta_nonneg_mode` | `"clamp"` | Nonnegative projection for integrated delta forecasts |
| `pred_guardrail_quantile` | `0.995` | Upper-bound quantile for prediction clipping |
| `zero_gate_k` | `7` | Zero-gating lookback window |
| `forecast_round_to_int` | `True` | Round forecasts to integers at inference |

`stage2_use_explicit_lag_coeff` exists in the config as a reserved experimental flag and is not currently consumed by the package.

## Data Contract and Shapes

Training expects aligned time series:

- `X`: shape `(T, Dx)`
- `y`: shape `(T, Dy)`

Public method shapes:

- `fit(X, y)`: `X: (T, Dx)`, `y: (T, Dy)`
- `fit_heads(X, y)`: same shapes as `fit()`
- `predict(X_recent, y_recent)`:
  - `X_recent`: `(dyn_p, Dx)` or `(dyn_p + 1, Dx)`
  - `y_recent`: same time length when supply history is enabled; otherwise it is still recommended because delta heads use the most recent observed supply as the integration baseline
  - returns `np.ndarray` with shape `(horizon, Dy)`

Window datasets:

- `RollingXYDataset`: `(x_past, x_future, y_future)`
- `RollingXYDatasetWithY0`: `(x_past, x_future, y0, y_future)`

When `use_supply_history=True`, the datasets concatenate `y` columns onto `x_past` and `x_future` so the encoder sees `Dx + Dy` features.

## Package Layout

```text
lavar/
├── __init__.py
├── config.py
├── forecaster.py
├── evaluation.py
├── losses.py
├── _core/
│   ├── dynamics.py
│   ├── heads.py
│   └── model.py
├── _data/
│   ├── dataset.py
│   └── scaler.py
└── _training/
    ├── stage1.py
    └── stage2/
        ├── __init__.py
        ├── common.py
        ├── stage2_test_baseline.py
        └── stage2_test_supply_history_latent.py
```

## Installation

From the repository root:

### Option A: `uv` (recommended)

```bash
uv sync
```

### Option B: editable install with `pip`

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

## Checkpoints and Outputs

Default training helpers write these checkpoint names when a save path is provided:

- `lavar_best.pth`
- `lavar_supply_dense_best.pth`
- `lavar_supply_sparse_best.pth`
- `lavar_supply_ultra_best.pth`

`LAVARForecaster.save(path)` stores the config, scalers, Stage 1 weights, Stage 2 head weights, bucket indices, and prediction guardrail bounds in a single artifact.

## Notes and Constraints

- Changing `latent_dim` requires refitting both stages.
- Changing `dyn_p` requires refitting because the latent dynamics shape changes.
- Do not unfreeze the Stage 1 `LAVAR` weights during Stage 2 head training.
- Density bucket indices are part of the trained Stage 2 contract and must stay aligned with the saved head weights.

## Citation

If you use this package, please cite the associated paper:

```bibtex
@article{lavar_tbd,
  title   = {LAVAR: Latent Autoencoder VAR for Forecasting},
  author  = {TBD},
  journal = {TBD},
  year    = {TBD}
}
```

## Acknowledgements

Built with PyTorch and pydantic, with a deliberately small code surface for experimentation on rolling healthcare demand and supply forecasting.
