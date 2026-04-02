# Patch Plan — 010 Ensemble (Drug-to-Drug, Seeded CI)

## Objective

Create `notebooks/010_ensemble.ipynb` that performs a full ensemble workflow using real per-drug predictions (not aggregate-only), runs across 5 random seeds, and produces true-vs-ensemble plots with 95% empirical confidence intervals bounded to `[0, inf)`.

## Scope

- Implement notebook-only changes for experiment execution and reporting.
- Reuse existing 008/009 checkpoint artifacts for inference reconstruction.
- Produce reproducible outputs under `notebooks/report/010_ensemble/hana/`.
- No changes to core `lavar/` package API or training loops.

## Files

- Create: `notebooks/010_ensemble.ipynb`
- Output dir used by notebook: `notebooks/report/010_ensemble/hana/`

## Implementation Steps

1. Set up notebook imports, paths, seed list (`[42, 123, 456, 789, 1024]`), and blend weight default (`w=0.55`).
2. Load unified Hana data (`lavar_ready_x.parquet`, `lavar_ready_y.parquet`) and build run plan for S11/S04 per seed.
3. Implement stage2 checkpoint listing + checkpoint selection helper by `fit_t_end <= t_end`.
4. Rebuild forecasters from checkpoint payloads (stage1 + stage2 heads) and generate per-fold, per-horizon, per-drug predictions for both S11/S04 per seed.
5. Build seeded blend predictions with non-negativity clipping: `y_blend = clip(w*y_s11 + (1-w)*y_s04, 0, inf)`.
6. Aggregate seeded blend predictions across 5 seeds per point (`seed, fold_id, h, abs_t, target`) to compute:
   - point estimate (mean)
   - lower/upper bounds via 95% empirical CI (2.5%, 97.5%), clipped to `[0, inf)`.
7. Compute metrics:
   - per-seed blend metrics (mean/median MSE, MAE, skill, p95/max fold MSE)
   - ensemble-CI point estimate metrics.
8. Produce graph(s): true vs ensemble predicted with shaded 95% CI, per-drug multi-panel visualization.
9. Save all CSV/PNG/MD artifacts to `notebooks/report/010_ensemble/hana/`.

## Validation Criteria

- Notebook executes end-to-end without missing-path errors (assuming existing 008 seed checkpoints are present).
- Generated predictions include both scenarios and all 5 seeds.
- CI output exists and satisfies `lower >= 0`, `upper >= 0`, `pred >= 0`.
- Plot artifact clearly overlays true values, ensemble point predictions, and CI bands.
- Report folder contains expected artifacts:
  - `ensemble_seed_predictions_per_drug.csv`
  - `ensemble_ci_per_drug.csv`
  - `ensemble_metrics_by_seed.csv`
  - `ensemble_metrics_overall.csv`
  - `ensemble_true_vs_pred_ci_per_drug.png`
  - `ensemble_summary.md`
