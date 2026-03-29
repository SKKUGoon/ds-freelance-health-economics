# 001 - Stage2 Mode Split + Mandatory Patch-Plan Policy

## Objective
1. Keep `lavar/_training/stage2.py` as the container/dispatcher.
2. Split mode logic into:
   - `lavar/_training/ts_latent.py`
   - `lavar/_training/ts_supply_ts_latent.py`
3. Enforce shape-explicit comments in torch tensor manipulation blocks.
4. Make `/history/001_<title>.md` style patch-plan logging mandatory before implementation.
5. Add Stage 2 experiment toggle for explicit lag coefficients:
   - `stage2_use_explicit_lag_coeff: bool = False`

## Scope
- In scope:
  - `lavar/config.py` (`stage2_use_explicit_lag_coeff` boolean option, default `False`)
  - `lavar/_training/stage2.py` (dispatcher only)
  - `lavar/_training/ts_latent.py` (current latent-rollout mode)
  - `lavar/_training/ts_supply_ts_latent.py` (supply-history + latent timeseries mode)
  - `AGENTS.md` policy update
  - `CLAUDE.md` policy update
- Out of scope:
  - `lavar/forecaster.py`
  - `lavar/evaluation.py`
  - static-latent modes

## Mode Contract
- `stage2_mode = "ts_latent"` (default)
- `stage2_mode = "ts_supply_ts_latent"` requires `use_supply_history=True`
- `stage2_use_explicit_lag_coeff: bool = False` (experiment flag)
  - `False`: baseline Stage 2 behavior.
  - `True`: enable explicit lag-coefficient path in Stage 2 experiment logic.

## Shape Comment Standard
All torch-heavy paths must annotate shapes, for example:
- `x_past: (B, p+1, D_in)`
- `y0: (B, Dy)`
- `y_future: (B, H, Dy)`
- `y_sel: (B, H, Dy_sel)`
- `delta_true/delta_hat: (B, H, Dy_sel)`

## Deliverables
1. `stage2.py` dispatcher preserving `train_supply_heads(...)` interface.
2. Separate mode files:
   - `ts_latent.py`
   - `ts_supply_ts_latent.py`
3. Config experiment toggle:
   - `stage2_use_explicit_lag_coeff: bool = False`
4. Mandatory patch-plan policy text inserted into:
   - `AGENTS.md`
   - `CLAUDE.md`

## Validation
- Import path compatibility unchanged:
  - `from lavar._training.stage2 import train_supply_heads`
- Existing return contract preserved:
  - dense/sparse/ultra indices + model handles
- Unknown `stage2_mode` raises clear `ValueError`.
- `stage2_use_explicit_lag_coeff=False` keeps baseline behavior unchanged.
- `stage2_use_explicit_lag_coeff=True` activates experimental explicit lag-coefficient path.
