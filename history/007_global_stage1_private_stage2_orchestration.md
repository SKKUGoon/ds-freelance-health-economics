# 007 - Global Stage1 + Hospital-Private Stage2 Orchestration

## Objective
Refactor forecaster orchestration so Stage 1 can be trained as a shared/global backbone and Stage 2 can be retrained privately per hospital.

## Scope
- In scope:
  - Add explicit Stage 1 and Stage 2 orchestration methods in `lavar/forecaster.py`.
  - Add config flag for Stage 1 supply-history usage to support shared default behavior.
  - Keep backward-compatible `fit()` and `fit_heads()` APIs.
- Out of scope:
  - Redesign of Stage 2 model internals.
  - New Stage 2 head architectures.
  - Changes to `lavar/evaluation.py` interfaces.

## Files
- `lavar/config.py`
- `lavar/forecaster.py`

## Implementation Steps
1. Add `stage1_use_supply_history: bool = False` to config.
2. Add/adjust config validator for mode compatibility with current architecture.
3. In `LAVARForecaster`, split orchestration into:
   - `fit_stage1_shared(X, y=None)`
   - `fit_stage2_private(X, y)`
4. Keep `fit(X, y)` as convenience wrapper that runs both stages.
5. Keep `fit_heads(X, y)` as Stage 2-only retrain path (private head refresh).
6. Ensure Stage 2 private retraining reuses shared Stage 1 backbone and `x_scaler`.

## Validation Criteria
- `fit_stage1_shared(...)` trains Stage 1 without requiring proprietary usage history by default.
- `fit_stage2_private(...)` trains Stage 2 heads with fixed Stage 1 weights.
- `fit(...)` and `fit_heads(...)` remain functional.
- Existing save/load flow remains functional.
