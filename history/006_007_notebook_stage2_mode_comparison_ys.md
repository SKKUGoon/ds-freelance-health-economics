# 006 - Notebook 007 Stage2 Mode Comparison (ys only)

## Objective
Create `notebooks/007_hecon.ipynb` to compare Stage 2 modes on the small-target set (`ys`) only:
- `ts_latent` with `use_supply_history=False`
- `ts_supply_ts_latent` with `use_supply_history=True`

## Scope
- In scope:
  - Build a runnable notebook for data loading, two mode runs, summary comparison, and plots.
  - Use `LAVARConfig` + `RollingEvaluator` from the package API.
  - Restrict experiments to `ys` only.
- Out of scope:
  - `yl` experiments
  - package code changes under `lavar/`
  - training algorithm changes

## Files
- `notebooks/007_hecon.ipynb`

## Implementation Steps
1. Add notebook title/intent markdown cell.
2. Add imports + plotting theme setup.
3. Load and align `X` and `ys` parquet files to daily index.
4. Build shared config dict and evaluation settings.
5. Run Experiment A (`ts_latent`).
6. Run Experiment B (`ts_supply_ts_latent`).
7. Build side-by-side summary DataFrame.
8. Plot fold-level MSE and skill curves for both modes.
9. Add concise conclusion cell template.

## Validation Criteria
- Notebook JSON is valid and opens in Jupyter.
- Both runs use identical evaluation settings except mode/supply-history switch.
- Summary table includes: `mean_mse`, `median_mse`, `mean_skill`, `median_skill`, `n_folds`.
- Fold-level comparison plots render without relying on package internals.
