## Objective

Update `notebooks/006_graph.ipynb` to stop using aggregate (`sum`) trajectories and instead produce per-drug prediction-vs-true outputs and plots.

## Scope

- Modify reconstruction output schema from aggregate totals to long-form per-drug rows.
- Replace aggregate trajectory visualization with per-drug plots for each selected run.
- Keep existing checkpoint-loading inference flow (no training).

## File List

- `notebooks/006_graph.ipynb`

## Implementation Steps

1. In prediction reconstruction loop, iterate over all target columns and store one row per `(run, fold, horizon step, drug)`.
2. Save long-form prediction CSV with fields including `target`, `y_true`, and `y_pred`.
3. Replace aggregate trajectory plot cell with per-drug subplot grids per run.
4. Keep fold-wise metric plots and leakage diagnostics unchanged.

## Validation Criteria

- Prediction CSV contains per-drug rows with non-empty `target` column.
- No reliance on `y_true_sum`/`y_pred_sum` remains.
- Per-drug true-vs-pred plots render for each top run.
