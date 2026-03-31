# 016 - 009 Notebook Scenario Matrix with Continuous History

## Objective
Create `notebooks/009_hecon.ipynb` implementing the corrected scenario matrix where S2-S9 use explicit contiguous history streams (lag 1..14) over `y_small` or `y_large`, with lr and mollified variants.

## Scope
- In scope:
  - Build new notebook `009_hecon.ipynb`.
  - Define scenarios S1..S9 with consistent metadata and exports.
  - Use contiguous lag history features (`t-1..t-14`) for y-history scenarios.
  - Keep target as `y_small`.
- Out of scope:
  - Core model/trainer code changes.
  - Modifying existing `008` notebook.

## Files
- `notebooks/009_hecon.ipynb`

## Implementation Steps
1. Add notebook scaffold and imports.
2. Load `X`, `y_small`, `y_large` and reindex daily.
3. Build lag-feature blocks for `y_small` and `y_large` using lags 1..14.
4. Construct scenario config matrix S1..S9 (baseline, lr_low, mollified variants).
5. Run evaluator across scenario-specific `(X, y)` data.
6. Export eval summary/folds to `report/009`.
7. Run manual rolling collector and export per-scenario CSVs.
8. Save per-scenario true-vs-pred uncertainty plots.

## Validation Criteria
- Notebook JSON parses correctly.
- Scenario names S1..S9 present.
- S2/S3/S6/S7 use `X_base + lag(y_small, 1..14)`.
- S4/S5/S8/S9 use `X_base + lag(y_large, 1..14)`.
- `report/009` export paths are defined for eval + manual artifacts.
