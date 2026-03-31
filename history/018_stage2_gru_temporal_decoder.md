# 018 — Stage 2 GRU Temporal Decoder

## Objective

Replace the pointwise MLP supply head in Stage 2 with a GRU-based sequential decoder.
The current architecture maps each rolled-out latent `z_h` independently to `ŷ_h` via an MLP,
discarding all cross-horizon temporal structure. A GRU decoder processes the full
latent trajectory `z_{1:H}` as a sequence, allowing each prediction to condition on
prior horizon steps.

## Problem Statement

From experiment 010 diagnostics:
- S1 baseline skill degrades unevenly across horizons (positive at h=4,10,11; negative at h=1,14)
- The supply head sees a single 8-dim latent vector per step — no positional or sequential context
- Dense targets (H02AB02, N02AX02) dominate MSE and would benefit most from temporal coherence

## Scope

### Files to modify
- `lavar/_core/heads.py` — add `SupplyHeadGRU` class
- `lavar/_core/model.py` — add `LAVARWithSupplyGRU` (or extend `LAVARWithSupply`) to use GRU head
- `lavar/config.py` — add `stage2_head_type` config field
- `lavar/_training/stage2/__init__.py` — wire GRU head type into dispatcher
- `lavar/_training/stage2/stage2_test_baseline.py` — handle GRU model in training loop

### Files to create
- `notebooks/011_hecon.ipynb` — experiment comparing S1 (MLP baseline) vs S2 (GRU decoder)

### Files NOT modified
- `lavar/_core/dynamics.py` — VAR dynamics unchanged
- `lavar/_training/stage1.py` — Stage 1 unchanged
- `lavar/_data/` — datasets unchanged

## Implementation Checklist

### Phase 1: GRU Supply Head (`heads.py`)

- [ ] **1.1** Create `SupplyHeadGRU` class in `lavar/_core/heads.py`
  - Input: `(B, H, latent_dim)` — full latent trajectory (NOT flattened)
  - Architecture:
    ```
    z_future (B, H, k) → GRU(input_size=k, hidden_size=gru_hidden) → h_seq (B, H, gru_hidden)
    h_seq → Linear(gru_hidden, output_dim) → delta (B, H, Dy_sel)
    ```
  - Constructor params: `input_dim`, `output_dim`, `gru_hidden_dim` (default 32), `num_layers` (default 1), `dropout` (default 0.0)
  - Forward signature: `forward(self, z_seq: Tensor) -> Tensor` returns `(B, H, output_dim)`
  - GRU processes the sequence left-to-right (h=1 → h=H), so each step conditions on all prior steps
  - No bidirectional — causal structure matters for forecasting

- [ ] **1.2** Create `SupplyHeadGRUNB` variant for probabilistic NB output (if needed for ultra-sparse)
  - Same GRU backbone, but split final layer into `mu_head` and `theta_head`
  - Returns dict `{"mu": ..., "theta": ...}` shaped `(B, H, Dy_sel)`
  - Or: keep ZINB head as-is for ultra-sparse (MLP), only use GRU for delta-MSE buckets
  - **Decision: start with GRU for delta-MSE only (dense + sparse). Keep ZINB MLP for ultra-sparse.**
    Ultra-sparse targets have near-zero signal; GRU won't help there.

### Phase 2: Model Integration (`model.py`)

- [ ] **2.1** Add `LAVARWithSupplyGRU` class (or add a branch in `LAVARWithSupply`)
  - **Preferred approach**: add a new class `LAVARWithSupplyGRU` to keep existing code untouched
  - Constructor: same as `LAVARWithSupply` but accepts `gru_hidden_dim`, `gru_num_layers`
  - Instantiates `SupplyHeadGRU` instead of `SupplyHeadMSE` when `supply_head_type="delta_mse"`

- [ ] **2.2** Implement `forward()` for GRU variant
  - Key difference from MLP path: do NOT flatten `z_future` to `(B*H, k)`
  - Instead pass full `z_future (B, H, k)` directly to `SupplyHeadGRU`
  - Delta integration and nonneg projection remain the same:
    ```python
    delta = self.supply_head(z_future)           # (B, H, Dy_sel)
    y_raw = y0.unsqueeze(1) + cumsum(delta, dim=1)
    y_hat = self._project_nonnegative(y_raw)
    ```
  - `return_delta=True` path works the same way

- [ ] **2.3** Ensure `return_params=True` still works for ZINB/NB heads
  - GRU model will only be used for delta-MSE buckets
  - ZINB/NB buckets still use the existing `LAVARWithSupply` with MLP heads
  - No change needed to probabilistic head path

### Phase 3: Config (`config.py`)

- [ ] **3.1** Add config fields:
  ```python
  stage2_head_type: Literal["mlp", "gru"] = "mlp"   # default preserves existing behavior
  gru_hidden_dim: int = 32
  gru_num_layers: int = 1
  gru_dropout: float = 0.0
  ```

- [ ] **3.2** No new validators needed — GRU head is orthogonal to `stage2_mode` and `use_supply_history`

### Phase 4: Training Loop (`_training/stage2/`)

- [ ] **4.1** Update `stage2_test_baseline.py::train_supply_head_indexed()`
  - When `cfg.stage2_head_type == "gru"` and `head_type == "delta_mse"`:
    - Instantiate `LAVARWithSupplyGRU` instead of `LAVARWithSupply`
    - Pass `gru_hidden_dim`, `gru_num_layers` from config
  - When `head_type in {"nb", "zinb"}` (ultra-sparse): always use MLP regardless of `stage2_head_type`
  - Training loop body is unchanged — `model(x_past, y0=y0_sel, return_delta=True)` works the same
  - Optimizer targets `model.supply_head.parameters()` — works for GRU too
  - Grad clipping unchanged

- [ ] **4.2** Update `__init__.py` dispatcher
  - No structural change needed — the dispatcher already delegates to `train_supply_head_indexed`
  - The head_type switch happens inside that function based on `cfg.stage2_head_type`

- [ ] **4.3** Update `forecaster.py` if needed
  - Check `fit_stage2_private()` — it calls `train_supply_heads()` which returns models
  - The returned models are stored as `self._dense_model`, `self._sparse_model`, etc.
  - `predict()` calls `model(X_r, y0=...)` — interface is the same for GRU variant
  - **Verify**: `predict()` passes `(B, p+1, D)` shaped input → `forward()` slices to `(B, p, D)` → works

### Phase 5: Experiment Notebook (`011_hecon.ipynb`)

- [ ] **5.1** Minimal 2-scenario comparison:
  - S1_mlp_baseline: `stage2_head_type="mlp"` (identical to 009 S1)
  - S2_gru_decoder: `stage2_head_type="gru"`, `gru_hidden_dim=32`

- [ ] **5.2** Same eval setup as 009/010:
  - 130 folds, fold_step=14, retrain_cadence=90
  - Report: eval summary, manual rolling, per-fold CSV, true-vs-pred plots
  - Save to `report/011/`

- [ ] **5.3** If S2 shows improvement, extend with hyperparameter variants:
  - S3_gru_hidden64: `gru_hidden_dim=64`
  - S4_gru_layers2: `gru_num_layers=2`
  - S5_gru_lr_low: `gru_hidden_dim=32`, `lr_supply=1e-4`

- [ ] **5.4** Diagnostic comparison (reuse 010 analysis pattern):
  - Per-bucket MSE: dense/sparse/ultra
  - Horizon decomposition: does GRU fix the h=1 and h=14 weakness?
  - Per-target breakdown for top-5 error contributors

## Architecture Diagram

```
Current (MLP — pointwise):
  z_future (B, H, k) → reshape(B*H, k) → MLP → reshape(B, H, Dy) → cumsum → ŷ

Proposed (GRU — sequential):
  z_future (B, H, k) → GRU(k → gru_hidden) → Linear(gru_hidden → Dy) → cumsum → ŷ
                         ↑                ↑
                    hidden state flows h=1 → h=H
                    (each step sees prior context)
```

## Key Design Decisions

1. **GRU over LSTM**: GRU has fewer parameters (2 gates vs 3), trains faster, and performs comparably for short sequences (H=14). Can revisit if GRU plateaus.

2. **GRU for delta-MSE only**: Ultra-sparse targets use ZINB head (MLP). GRU won't help on targets with <0.5% nonzero rate. This keeps the change surgical.

3. **New class vs branch**: `LAVARWithSupplyGRU` as a separate class avoids touching `LAVARWithSupply` internals. The training loop picks which class to instantiate based on config.

4. **No attention (yet)**: GRU is the simplest sequential model that adds temporal memory. If it shows promise, attention over the latent trajectory is a natural follow-up.

## Validation Criteria

- [ ] Smoke test passes: `LAVARForecaster(cfg_gru).fit(X, y)` completes without error
- [ ] `predict()` returns correct shape `(H, Dy)`
- [ ] S2_gru_decoder median_skill > S1_mlp_baseline median_skill (primary metric)
- [ ] Horizon decomposition shows improved skill at h=1 and h=14 (the current weak points)
- [ ] No regression on ultra-sparse bucket (still uses MLP/ZINB)

## Risk / Rollback

- If GRU overfits (worse mean_mse from catastrophic folds): try dropout, reduce gru_hidden_dim
- If GRU shows no improvement: the MLP default is preserved via `stage2_head_type="mlp"`
- All existing tests and notebooks continue to work unchanged (new config field has default "mlp")
