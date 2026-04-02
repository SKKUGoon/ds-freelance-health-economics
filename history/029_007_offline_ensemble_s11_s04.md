# 029 — 007 Offline Ensemble: S11 + S04 Blend

## Objective
Build an offline ensemble experiment using existing fold-level predictions from S11_LATENT16_E05 and S04_GRU_H32_L1_E02 (no retraining required). Evaluate static convex blend weights with temporal train/eval split for weight selection.

## Scope
- Notebook `007_ensemble.ipynb`
- Report output: `notebooks/report/007_ensemble/hana/`

## Data Sources
- `notebooks/report/006_graph/hana/top4_predictions_per_drug.csv` — per-drug per-fold predictions (349k rows)
- `notebooks/report/006_graph/hana/top4_fold_metrics_reconstructed.csv` — fold-level metrics (520 rows)

## Implementation Steps

### Part 1: Static Blend Grid Search
1. Load per-drug predictions for S11 and S04
2. Align on (fold_id, h, abs_t, target, target_idx) keys
3. For w in [0.00, 0.05, ..., 1.00]:
   - Compute y_blend = w * y_S11 + (1-w) * y_S04
   - Compute per-fold MSE, MAE vs y_true
   - Compute naive MSE and skill from fold metrics
   - Record: mean_mse, mean_mae, median_skill, max_fold_mse, p95_fold_mse, p90_fold_mse

### Part 2: Temporal Split for Weight Selection
1. Split 130 folds by t_end: first 70% (folds 0-90, ~t_end <= ~2000) = tune, last 30% (folds 91-129) = eval
2. Select w* on tune set (optimize for mean_mse or composite objective)
3. Report eval-set metrics at w* and compare to pure S11 / pure S04

### Part 3: Visualization
1. Weight vs metric curves (mean_mse, p95_mse, max_mse, mean_mae)
2. Foldwise MSE comparison: blend vs S11 vs S04
3. Summary table

## Validation Criteria
- Blend at w=1.0 matches pure S11 metrics; w=0.0 matches pure S04
- Temporal split eval is strictly out-of-sample for weight selection
- All fold-level metrics are recomputed from per-drug predictions (no stale cached values)
