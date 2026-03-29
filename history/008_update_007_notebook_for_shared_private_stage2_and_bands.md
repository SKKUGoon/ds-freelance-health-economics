# 008 - Update Notebook 007 for Shared/Private Stage2 and Uncertainty Bands

## Objective
Update `notebooks/007_hecon.ipynb` to:
1. Use the updated Stage 1/Stage 2 configuration contract.
2. Add an academic-style True vs Pred plot with two uncertainty bands and nonnegative lower bounds.

## Scope
- In scope:
  - notebook import/config updates for new Stage 1 flag behavior.
  - add manual rolling collection for prediction arrays.
  - add two-band uncertainty visualization with lower bound clipped at 0.
- Out of scope:
  - package source code changes under `lavar/`.

## Files
- `notebooks/007_hecon.ipynb`

## Implementation Steps
1. Update imports to include `LAVARForecaster` and `compute_fold_metrics`.
2. Update mode configs:
   - `ts_latent`: `stage1_use_supply_history=False`
   - `ts_supply_ts_latent`: `stage1_use_supply_history=True`
3. Add manual rolling function that collects:
   - `all_pred_times`, `all_pred_values`, `all_true_values`
4. Add academic-style plot cell with two bands:
   - overlap-dispersion band
   - calibrated residual interval
5. Enforce `>= 0` lower bounds for both confidence bands.

## Validation Criteria
- Notebook JSON remains valid.
- `ts_supply_ts_latent` run does not fail config validation.
- Plot cell runs when prediction arrays are populated.
- Both band lower bounds are clamped to 0.
