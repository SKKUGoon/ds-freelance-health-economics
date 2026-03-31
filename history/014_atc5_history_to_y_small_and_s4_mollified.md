# 014 - ATC5 History to y_small + S4 Mollified Stage2

## Objective
1. Implement the missing S4 mollified behavior so the configured scenario differs from baseline supply-history training in actual optimization and inference behavior.
2. Add Scenario 5 in `notebooks/008_hecon.ipynb` that uses past `y_large` (ATC5 aggregates) as auxiliary history features to predict `y_small` (individual drug-level demand).

## Scope
- In scope:
  - Add configurable Stage2 nonnegative continuation modes for delta-MSE heads.
  - Wire annealed softplus behavior for S4 through config + Stage2 training loop.
  - Update notebook `008` to include scenario-specific inputs and S5 (ATC5-history-assisted y_small forecasting).
  - Ensure artifact exports include S5 and scenario metadata.
- Out of scope:
  - Any redesign of Stage1 architecture.
  - Changes to evaluator API surface beyond notebook-side scenario routing.

## Files
- `lavar/config.py`
- `lavar/_core/model.py`
- `lavar/_training/stage2/stage2_test_baseline.py`
- `lavar/_training/stage2/stage2_test_supply_history_latent.py`
- `lavar/forecaster.py` (if load path needs mode wiring)
- `notebooks/008_hecon.ipynb`

## Implementation Steps
1. Add config fields for Stage2 delta nonnegative mode and softplus annealing bounds.
2. Implement nonnegative projection helper in `LAVARWithSupply` and expose runtime setters used by Stage2 training loops.
3. In Stage2 trainers, apply per-epoch beta schedule when `stage2_delta_nonneg_mode="softplus_annealed"` for `delta_mse` heads.
4. Add S5 notebook scenario with scenario-specific `(X, y)` routing:
   - S1-S4: existing `X -> y_small`.
   - S5: `concat(X, lagged ATC5(y_large)) -> y_small`.
5. Add S5 artifact naming and metadata columns (`target_granularity`, `aux_source`) in notebook exports.
6. Validate notebook JSON structure and key imports.

## Validation Criteria
- S4 configured as mollified is no longer behaviorally identical to S2.
- S5 runs evaluator/manual paths using ATC5 lag features with no future leakage.
- `008` exports include S5 per-scenario artifacts and remain eval/manual separated.
- Existing scenarios S1-S4 continue to run with unchanged target (`y_small`).
