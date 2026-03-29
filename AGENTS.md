# LAVAR — Agent Reference

## Mandatory Patch Plan Logging

Before any implementation or editing task, the agent must create a patch-plan record in `/history` using:

`/history/NNN_<title>.md`

Rules:
1. Start from `001_...` and increment sequentially.
2. Create the history record first, before any code edits.
3. Include objective, scope, file list, implementation steps, and validation criteria.
4. Do not proceed with implementation without this record.

## Public API

```python
from lavar import LAVARForecaster, LAVARConfig, RollingEvaluator, EvaluationResults
```

### LAVARConfig

Pydantic model. All fields have defaults. Key fields:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `device` | `"cpu"\|"mps"\|"cuda"` | `"cpu"` | Compute device |
| `dyn_p` | `int` | `7` | VAR order (history length) |
| `horizon` | `int` | `14` | Forecast horizon |
| `latent_dim` | `int` | `8` | Latent space dimensionality |
| `encoder_hidden` | `List[int]` | `[32, 16]` | Encoder MLP layers |
| `decoder_hidden` | `List[int]` | `[16, 32]` | Decoder MLP layers |
| `use_supply_history` | `bool` | `False` | Concatenate y to encoder input |
| `stage1_use_supply_history` | `bool` | `False` | Stage 1 encoder augmentation toggle |
| `stage2_mode` | `"ts_latent"\|"ts_supply_ts_latent"` | `"ts_latent"` | Stage 2 training mode |
| `stage2_use_explicit_lag_coeff` | `bool` | `False` | Experiment: explicit lag coefficients |
| `epochs_lavar` | `int` | `100` | Stage 1 training epochs |
| `epochs_supply` | `int` | `100` | Stage 2 training epochs |
| `lr_lavar` | `float` | `1e-3` | Stage 1 learning rate |
| `lr_supply` | `float` | `1e-3` | Stage 2 learning rate |
| `batch_size` | `int` | `64` | Training batch size |
| `dense_nonzero_rate_thr` | `float` | `0.70` | Dense bucket threshold |
| `ultra_nonzero_rate_thr` | `float` | `0.005` | Ultra-sparse bucket threshold |

Builder pattern:
```python
cfg = LAVARConfig.builder().device("mps").latent(dim=8).horizon(h=14, history=7).build()
```

### LAVARForecaster

```python
class LAVARForecaster:
    def __init__(self, config: LAVARConfig | None = None)
    def fit_stage1_shared(self, X: ndarray|Tensor, y: ndarray|Tensor | None = None) -> LAVARForecaster
    def fit_stage2_private(self, X: ndarray|Tensor, y: ndarray|Tensor) -> LAVARForecaster
    def fit(self, X: ndarray|Tensor, y: ndarray|Tensor) -> LAVARForecaster
    def fit_heads(self, X: ndarray|Tensor, y: ndarray|Tensor) -> LAVARForecaster
    def predict(self, X_recent: ndarray|Tensor, y_recent: ndarray|Tensor | None = None) -> ndarray
    def save(self, path: str) -> None
    @classmethod
    def load(cls, path: str) -> LAVARForecaster
    @property
    def is_fitted(self) -> bool
    @property
    def is_heads_fitted(self) -> bool
```

**Input shapes:**
- `fit(X, y)`: `X: (T, Dx)`, `y: (T, Dy)`
- `fit_heads(X, y)`: same as fit; LAVAR weights frozen
- `predict(X_recent, y_recent)`: `X_recent: (p, Dx)` or `(p+1, Dx)`, `y_recent`: same length, required when `use_supply_history=True`
- **Returns:** `np.ndarray` of shape `(H, Dy)`

### RollingEvaluator

```python
class RollingEvaluator:
    def __init__(self, config: LAVARConfig)
    def evaluate(
        self, X, y,
        lavar_retrain_cadence: int = 90,
        heads_retrain_cadence: int = 90,
        quality_triggers: bool = True,
        fold_step: int = 14,
        verbose: bool = True,
    ) -> EvaluationResults
```

### EvaluationResults

```python
@dataclass
class EvaluationResults:
    folds: List[FoldResult]       # per-fold metrics + guardrail meta
    summary: Dict[str, float]     # mean_mse, median_mse, mean_skill, median_skill, n_folds
    def to_dataframe(self) -> pd.DataFrame   # columns: fold_id, t_end, mse, rmse, mae, ...
    def plot(self) -> None                   # matplotlib MSE + skill plot
```

## Tensor Shape Conventions

| Symbol | Meaning | Typical Value |
|--------|---------|---------------|
| `T` | Total time steps | varies |
| `B` | Batch size | 64 |
| `Dx` | Non-supply feature dim | varies |
| `Dy` | Supply target dim | varies |
| `p` | VAR order (`dyn_p`) | 7 |
| `H` | Forecast horizon | 14 |
| `k` | Latent dim (`latent_dim`) | 8 |

Augmented input dim when `use_supply_history=True`: `Dx + Dy`.

## Stage 2 Modes

Stage 2 training is dispatched by `cfg.stage2_mode` through `stage2.py`:

| Mode | File | Description |
|------|------|-------------|
| `ts_latent` (default) | `ts_latent.py` | Latent rollout only |
| `ts_supply_ts_latent` | `ts_supply_ts_latent.py` | Supply-history augmented encoder + latent rollout |

**Config constraint:** `stage2_mode="ts_supply_ts_latent"` requires `use_supply_history=True`. This is enforced by a pydantic `model_validator` at config construction time.

**Experiment flag:** `stage2_use_explicit_lag_coeff` (default `False`) is reserved for future explicit lag-coefficient experiments. No consumer exists yet.

## Pipeline Flow

1. `fit()` → fit x_scaler (and y_scaler if augmented) → instantiate LAVAR → Stage 1 training → Stage 2 mode dispatch → density split + head training
2. `fit_heads()` → re-scale with existing scalers → Stage 2 mode dispatch only (LAVAR frozen)
3. `predict()` → scale → augment if needed → per-bucket forward → guardrails → round → numpy

## Do Not Touch

- Do not change `latent_dim` without refitting both stages (encoder/decoder dimensions are baked in)
- Do not change `dyn_p` without refitting (VAR dynamics matrix shape depends on it)
- Do not unfreeze LAVAR parameters during Stage 2 training
- Do not modify density bucket indices after Stage 2 — they map target columns to heads
- Supply heads share a reference to the same LAVAR instance — do not replace it independently

## Package Layout

```
lavar/
├── __init__.py            # LAVARConfig, LAVARForecaster, RollingEvaluator, EvaluationResults
├── config.py              # LAVARConfig + LAVARConfigBuilder
├── forecaster.py          # LAVARForecaster
├── evaluation.py          # RollingEvaluator, FoldResult, EvaluationResults
├── losses.py              # negative_binomial_nll, zinb_nll
├── _core/
│   ├── dynamics.py        # VARDynamics
│   ├── heads.py           # MLP, SupplyHeadNB, SupplyHeadZINB, SupplyHeadMSE
│   └── model.py           # LAVAR, LAVARWithSupply
├── _data/
│   ├── dataset.py         # RollingXYDataset, RollingXYDatasetWithY0
│   └── scaler.py          # StandardScalerTorch
└── _training/
    ├── stage1.py              # train_lavar()
    ├── stage2.py              # train_supply_heads() dispatcher, density splitting
    ├── ts_latent.py           # ts_latent mode training loop
    └── ts_supply_ts_latent.py # ts_supply_ts_latent mode training loop
```
