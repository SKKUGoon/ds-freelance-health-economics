# 005 - Legacy Cleanup in Stage2 and Evaluation

## Objective
Remove confirmed unused legacy code from `lavar/` without changing behavior.

## Scope
- In scope:
  - Remove unused helper `stitch_bucket_predictions` from `lavar/_training/stage2.py`.
  - Remove unused import `LAVARWithSupply` from `lavar/evaluation.py`.
- Out of scope:
  - Any Stage 1/Stage 2 training logic changes.
  - Any public API/signature changes.
  - Any config behavior changes.

## Files
- `lavar/_training/stage2.py`
- `lavar/evaluation.py`

## Implementation Steps
1. Delete the unused `stitch_bucket_predictions(...)` function in `stage2.py`.
2. Delete the unused `LAVARWithSupply` import in `evaluation.py`.
3. Run a lightweight import smoke check for package integrity.

## Validation Criteria
- No references to `stitch_bucket_predictions(...)` remain.
- `lavar/evaluation.py` has no unused `LAVARWithSupply` import.
- Smoke check passes:
  - `from lavar import LAVARConfig, LAVARForecaster`
  - `from lavar._training.stage2 import train_supply_heads`
