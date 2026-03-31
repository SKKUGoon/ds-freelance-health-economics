# 015 - S5 ATC5 Lag Stream Patch

## Objective
Fix Scenario 5 ATC5 auxiliary feature construction to use a contiguous historical stream instead of sparse snapshots.

## Scope
- Update `notebooks/008_hecon.ipynb` S5 feature generation from sparse lags `[1, 7, 14]` to contiguous lags `1..14`.
- Keep strict causal shifting (no future leakage).

## Files
- `notebooks/008_hecon.ipynb`

## Implementation Steps
1. Replace the S5 lag list with `range(1, 15)`.
2. Keep the same ATC5 lag feature concatenation pipeline.
3. Add a small print for transparency on number of ATC5 lag features created.

## Validation Criteria
- Notebook JSON remains valid.
- S5 feature block now represents continuous lag history (`t-1` through `t-14`).
