# LAVAR — Latent Autoencoder VAR for Health-Econ Supply Forecasting

## Mandatory Patch Plan Logging

Before any implementation or editing task, the agent must create a patch-plan record in `/history` using:

`/history/NNN_<title>.md`

Rules:
1. Start from `001_...` and increment sequentially.
2. Create the history record first, before any code edits.
3. Include objective, scope, file list, implementation steps, and validation criteria.
4. Do not proceed with implementation without this record.

## Architecture

Two-stage pipeline for multi-target supply forecasting:

```
Stage 1: LAVAR (latent dynamics)
  x_t → Encoder → z_t       (nonlinear observation → latent)
  z_t = A·z_{t-1:t-p}       (linear VAR(p) dynamics in latent space)
  z_t → Decoder → x̂_t       (reconstruction for regularization)

Stage 2: Supply Heads (frozen LAVAR encoder)
  z_{t+1:t+H} = rollout(z_history)   (latent rollout via learned VAR)
  z_h → SupplyHead → ŷ_h            (per-bucket count prediction)
```

Targets are split into density buckets by nonzero rate:
- **Dense** (nonzero_rate >= 0.70): delta-MSE head (predicts Δy, integrates with y0)
- **Sparse** (between thresholds): delta-MSE head
- **Ultra-sparse** (nonzero_rate <= 0.005): ZINB head (zero-inflated negative binomial)

### Stage 2 Modes

Stage 2 training is dispatched by `cfg.stage2_mode` through `stage2.py`:

| Mode | File | Description |
|------|------|-------------|
| `ts_latent` (default) | `_training/ts_latent.py` | Latent rollout only |
| `ts_supply_ts_latent` | `_training/ts_supply_ts_latent.py` | Supply-history augmented encoder + latent rollout |

**Config constraint:** `stage2_mode="ts_supply_ts_latent"` requires `use_supply_history=True` (enforced at config construction via pydantic `model_validator`).

## Data Shapes

| Tensor | Shape | Description |
|--------|-------|-------------|
| `X` | `(T, Dx)` | Non-supply features (time × feature dim) |
| `y` | `(T, Dy)` | Supply targets (time × target dim) |
| `x_past` | `(B, p+1, Dx)` | Encoder input window |
| `x_past` (augmented) | `(B, p+1, Dx+Dy)` | When `use_supply_history=True` |
| `z_history` | `(B, p, k)` | Latent history for VAR rollout |
| `z_future` | `(B, H, k)` | Rolled-out latent states |
| `y_hat` | `(H, Dy)` | Forecast output |

Key dimensions: `p` = `dyn_p` (VAR order, default 7), `H` = `horizon` (default 14), `k` = `latent_dim` (default 8).

## Entry Points

```python
from lavar import LAVARForecaster, LAVARConfig, RollingEvaluator

# Configure
cfg = LAVARConfig.builder().device("mps").latent(dim=8).horizon(h=14, history=7).build()

# Fit full pipeline (Stage 1 + Stage 2)
model = LAVARForecaster(cfg)
model.fit(X, y)

# Or run explicitly as shared/global Stage 1 + private Stage 2
model.fit_stage1_shared(X, y=None)   # default shared style uses X only
model.fit_stage2_private(X, y)       # hospital-private heads

# Retrain only supply heads (Stage 2), LAVAR frozen
model.fit_heads(X_new, y_new)

# Forecast
y_hat = model.predict(X_recent, y_recent=y_recent)  # → np.ndarray (H, Dy)

# Save / load
model.save("model.pth")
model = LAVARForecaster.load("model.pth")

# Rolling evaluation with independent refit cadences
evaluator = RollingEvaluator(cfg)
results = evaluator.evaluate(X, y, lavar_retrain_cadence=90, heads_retrain_cadence=30)
df = results.to_dataframe()
```

### Supply-augmented mode example

```python
cfg = LAVARConfig(stage2_mode="ts_supply_ts_latent", use_supply_history=True,
                  device="cpu", epochs_lavar=2, epochs_supply=2)
model = LAVARForecaster(cfg)
model.fit(X, y)
y_hat = model.predict(X[-7:], y_recent=y[-7:])  # y_recent required
```

## Refit Strategy

Stage 1 (LAVAR) trains the encoder, decoder, and VAR dynamics — expensive, changes latent space.
Stage 2 (supply heads) trains lightweight count heads on frozen latent representations — fast.

In production rolling evaluation:
- **Full refit** (`model.fit`): every `lavar_retrain_cadence` days (default 90), or on quality trigger
- **Heads refit** (`model.fit_heads`): every `heads_retrain_cadence` days between full refits
- **Quality triggers**: skill vs naive < -0.5, or 3 consecutive guardrail activations → force full refit

## Package Layout

```
lavar/
├── __init__.py          # Public API: LAVARConfig, LAVARForecaster, RollingEvaluator
├── config.py            # LAVARConfig (pydantic) + LAVARConfigBuilder
├── forecaster.py        # LAVARForecaster: fit / fit_heads / predict / save / load
├── evaluation.py        # RollingEvaluator + FoldResult / EvaluationResults
├── losses.py            # NB and ZINB negative log-likelihood losses
├── _core/
│   ├── dynamics.py      # VARDynamics (linear latent transition)
│   ├── heads.py         # MLP, SupplyHeadNB, SupplyHeadZINB, SupplyHeadMSE
│   └── model.py         # LAVAR, LAVARWithSupply
├── _data/
│   ├── dataset.py       # RollingXYDataset, RollingXYDatasetWithY0
│   └── scaler.py        # StandardScalerTorch
└── _training/
    ├── stage1.py            # train_lavar()
    ├── stage2.py            # train_supply_heads() dispatcher, density splitting
    ├── ts_latent.py         # ts_latent mode training loop
    └── ts_supply_ts_latent.py # ts_supply_ts_latent mode training loop
```

## Dev Workflow

```bash
uv pip install -e .
python -c "from lavar import LAVARForecaster, LAVARConfig; print('OK')"
```

Quick smoke test:
```python
import numpy as np
from lavar import LAVARForecaster, LAVARConfig

cfg = LAVARConfig(device="cpu", epochs_lavar=2, epochs_supply=2, latent_dim=4,
                  early_stop_patience_lavar=None, early_stop_patience_supply=None)
X = np.random.randn(120, 10).astype(np.float32)
y = np.abs(np.random.randn(120, 5).astype(np.float32))
model = LAVARForecaster(cfg)
model.fit(X, y)
y_hat = model.predict(X[-7:], y_recent=y[-7:])
assert y_hat.shape == (14, 5)
```
