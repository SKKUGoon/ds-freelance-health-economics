# 010 - Fix 007 Notebook Export Consistency

## Objective
Fix `notebooks/007_hecon.ipynb` so saved CSV artifacts match the run source and avoid mixing evaluator vs manual rolling outputs.

## Scope
- In scope:
  - Rename evaluator-export CSV outputs to explicit `eval` filenames.
  - Add manual-run CSV exports with explicit `manual` filenames and mode suffix.
  - Add run-type metadata columns for traceability.
- Out of scope:
  - Model/training logic changes.

## Files
- `notebooks/007_hecon.ipynb`

## Implementation Steps
1. Change existing export cell to save:
   - `007_eval_mode_compare_summary.csv`
   - `007_eval_mode_compare_folds.csv`
2. Add `run_type="eval"` to evaluator exports.
3. Extend manual rolling cell to save:
   - `007_manual_<mode>_folds.csv`
   - `007_manual_<mode>_summary.csv`
4. Add `run_type="manual"` and `mode` columns to manual exports.

## Validation Criteria
- Evaluator and manual outputs are saved to different filenames.
- Console logs can be traced to the matching manual CSV.
- No ambiguity remains about which artifact is used for reporting.
