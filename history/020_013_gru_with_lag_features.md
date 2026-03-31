# 020 — Experiment 013: GRU Decoder + Drug Lag Features

## Objective

Test whether feeding lagged drug history into the encoder (alongside X_base) improves GRU decoder performance. Motivation: 012 diagnostics showed drugs with zero recent history get non-zero predictions because the model has no per-target activity signal. Lag features provide this signal through the encoder → latent → GRU path.

Prior work:
- 009: MLP + lag features showed marginal improvement (S4 y_large_lag was best non-baseline)
- 011: GRU h=32 achieved first positive mean skill (+0.086) without lag features
- Hypothesis: GRU's temporal memory can exploit lag features better than MLP could

## Scenarios

| Scenario | Head | Input | Dims | Description |
|----------|------|-------|------|-------------|
| S1_gru_baseline | GRU h=32 | X_base | 35 | Control (= 011 S2) |
| S2_gru_ysmall_lag | GRU h=32 | X_base + y_small lags(1..14) | 707 | Drug-level history |
| S3_gru_ylarge_lag | GRU h=32 | X_base + y_large lags(1..14) | 539 | Category-level history |

All scenarios: stage2_mode="baseline", 130 folds, fold_step=14, retrain_cadence=90, quality_triggers=True, stage2_head_type="gru", gru_hidden_dim=32, gru_num_layers=1.

## Files

| File | Action |
|------|--------|
| `history/020_013_gru_with_lag_features.md` | Create (this file) |
| `notebooks/013_hecon.ipynb` | Create — experiment notebook |

## Implementation Steps

- [x] Create history record
- [ ] Cell 0: Markdown header
- [ ] Cell 1: Imports, setup, data loading, lag feature construction
- [ ] Cell 2: Define 3 scenarios with configs
- [ ] Cell 3: RollingEvaluator loop for all scenarios
- [ ] Cell 4: Summary table + fold-level CSV
- [ ] Cell 5: Manual rolling for per-fold comparison
- [ ] Cell 6: Summary comparison table
- [ ] Cell 7: Conclusion markdown

## Validation

- Run all cells in `013_hecon.ipynb`
- Compare S2/S3 vs S1 on mean/median MSE and skill
- Check whether drugs with zero recent history get suppressed predictions
- Output files in `report/013/`

## Expected Outputs

- `013_eval_mode_compare_summary.csv`
- `013_eval_mode_compare_folds.csv`
- `013_manual_S1_gru_baseline_summary.csv` (+ folds)
- `013_manual_S2_gru_ysmall_lag_summary.csv` (+ folds)
- `013_manual_S3_gru_ylarge_lag_summary.csv` (+ folds)
