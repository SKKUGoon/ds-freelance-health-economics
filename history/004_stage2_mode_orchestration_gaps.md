# 004 - Stage2 Mode Orchestration: Gap Closure

## Objective
Close the three remaining gaps from the 003 orchestration plan that were not addressed by the 001 execution.

## Scope
- In scope:
  - `lavar/config.py` (pydantic model_validator for mode/supply-history coupling)
  - Smoke tests for untested paths
- Out of scope:
  - `stage2_use_explicit_lag_coeff=True` behavior (intentionally deferred; no consumer exists yet)
  - `lavar/forecaster.py` structural changes
  - `lavar/_training/stage2.py` changes
  - `lavar/evaluation.py`

## Gap 1: Fail-Fast Config Validation

**Problem:** `stage2_mode="ts_supply_ts_latent"` with `use_supply_history=False` only errors deep inside the training loop (`ts_supply_ts_latent.py:39`). This violates the 003 plan's "explicit validation and fail-fast errors" requirement.

**Fix:** Add a pydantic `model_validator` on `LAVARConfig` that rejects this combination at construction time.

```python
@model_validator(mode="after")
def _validate_stage2_mode(self) -> "LAVARConfig":
    if self.stage2_mode == "ts_supply_ts_latent" and not self.use_supply_history:
        raise ValueError(
            "stage2_mode='ts_supply_ts_latent' requires use_supply_history=True."
        )
    return self
```

**File:** `lavar/config.py`

## Gap 2: Missing Smoke Tests

**Problem:** The 003 validation criteria list three smoke checks. Only one was run.

| Check | Status |
|-------|--------|
| `fit/predict` with `ts_latent` | Done |
| `fit_heads` with `ts_latent` | **Missing** |
| `fit/predict` with `ts_supply_ts_latent` + `use_supply_history=True` | **Missing** |

**Fix:** Run both missing paths after the config validator is in place.

## Gap 3: Dead Flag Acknowledgement

**Problem:** `stage2_use_explicit_lag_coeff` exists in config but nothing reads it.

**Decision:** This is intentionally deferred. No code change needed. The flag is a placeholder for future experiment work. When a consumer is added, a new history record (005+) should accompany it.

## Deliverables
1. Pydantic `model_validator` in `lavar/config.py`.
2. Passing smoke tests:
   - `fit_heads` with default `ts_latent` mode.
   - `fit/predict` with `ts_supply_ts_latent` + `use_supply_history=True`.
3. Config validator rejects `ts_supply_ts_latent` without `use_supply_history=True`.

## Validation
- `LAVARConfig(stage2_mode="ts_supply_ts_latent", use_supply_history=False)` raises `ValueError`.
- `LAVARConfig(stage2_mode="ts_supply_ts_latent", use_supply_history=True)` succeeds.
- `LAVARConfig()` succeeds (defaults unchanged).
- Smoke test: `fit` then `fit_heads` with `ts_latent` produces `(H, Dy)` output.
- Smoke test: `fit/predict` with `ts_supply_ts_latent` + `use_supply_history=True` produces `(H, Dy)` output.
