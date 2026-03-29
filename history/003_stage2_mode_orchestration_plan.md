# 003 - Stage2 Mode Orchestration Plan

## Objective
Implement a configuration-driven Stage 2 process that cleanly selects between:
- `ts_latent` (latent-rollout mode)
- `ts_supply_ts_latent` (supply-history + latent-rollout mode)

while keeping `LAVARForecaster` as the orchestration owner and preserving existing public APIs.

## Scope
- In scope:
  - `lavar/forecaster.py` (orchestration and mode selection ownership)
  - `lavar/_training/stage2.py` (container/dispatcher entrypoint)
  - `lavar/_training/ts_latent.py` (mode-specific training implementation)
  - `lavar/_training/ts_supply_ts_latent.py` (mode-specific training implementation)
  - `lavar/config.py` (mode contract and experimental flags)
- Out of scope:
  - `lavar/evaluation.py` public API changes
  - Stage 1 model architecture changes
  - static-latent Stage 2 modes

## Files
- `lavar/config.py`
- `lavar/forecaster.py`
- `lavar/_training/stage2.py`
- `lavar/_training/ts_latent.py`
- `lavar/_training/ts_supply_ts_latent.py`

## Mode Contract
- `stage2_mode`: `"ts_latent" | "ts_supply_ts_latent"`
- `stage2_use_explicit_lag_coeff`: `bool`, default `False`

Rules:
1. Unknown `stage2_mode` raises `ValueError` with allowed values.
2. `stage2_mode="ts_supply_ts_latent"` requires `use_supply_history=True`.
3. Default behavior with `stage2_use_explicit_lag_coeff=False` must preserve baseline behavior.

## Implementation Steps
1. Confirm and freeze return contract for `train_supply_heads(...)`:
   - `dense_indices`, `sparse_indices`, `ultra_indices`
   - `dense_model`, `sparse_model`, `ultra_model`
2. Keep `lavar/_training/stage2.py` as the only external Stage 2 entrypoint.
3. Ensure `stage2.py` dispatches strictly by `cfg.stage2_mode` to mode modules.
4. Keep mode-specific tensor logic in:
   - `ts_latent.py`
   - `ts_supply_ts_latent.py`
5. Keep orchestration ownership in `LAVARForecaster`:
   - scaler lifecycle
   - Stage 1/Stage 2 sequencing
   - config-driven training process
6. Keep `stage2_use_explicit_lag_coeff` as an experiment switch:
   - no behavior change when `False`
   - add explicit lag-coefficient path only when `True`
7. Add/maintain explicit shape comments for torch tensor operations:
   - `x_past: (B, p+1, D_in)`
   - `y0: (B, Dy)`
   - `y_future: (B, H, Dy)`
   - `y_sel: (B, H, Dy_sel)`
   - `delta_true/delta_hat: (B, H, Dy_sel)`

## Validation Criteria
- Import compatibility unchanged:
  - `from lavar._training.stage2 import train_supply_heads`
- Mode selection works from config for both supported modes.
- Guardrails for invalid configuration work:
  - `ts_supply_ts_latent` without `use_supply_history=True` fails clearly.
- Baseline parity:
  - `stage2_use_explicit_lag_coeff=False` preserves existing behavior.
- Smoke checks pass:
  - `fit/predict` with `ts_latent`
  - `fit_heads` with `ts_latent`
  - `fit/predict` with `ts_supply_ts_latent` + `use_supply_history=True`

## Risks and Mitigations
- Risk: mode logic divergence creates duplication.
  - Mitigation: keep shared policy in `stage2.py`; isolate only true mode differences.
- Risk: config mismatch causes silent misuse.
  - Mitigation: explicit validation and fail-fast errors.
- Risk: experiment flag adds regressions.
  - Mitigation: default-off flag, parity checks, and targeted smoke tests.
