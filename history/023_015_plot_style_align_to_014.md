# 023 - Align 015 Plot Style to 014

## Objective
Make `notebooks/015_hecon.ipynb` plotting style match `notebooks/014_hecon.ipynb` for per-scenario true-vs-pred visualization.

## Scope
- Replace the simplified top-N trajectory plot in 015 with the academic two-band uncertainty plot used in 014.
- Keep output filename pattern under `report/015/`.

## Files
- `notebooks/015_hecon.ipynb`

## Implementation Steps
1. Copy the `plot_scenario_true_vs_pred(...)` function implementation from 014 into 015.
2. Update 015 plot loop call signature to match 014 usage (`plot_round_mode`, `ci_alpha`, `max_targets`, `save_path`).
3. Preserve 015 filename prefix (`015_...`) for saved images.

## Validation Criteria
- Notebook JSON remains valid.
- 015 contains the same uncertainty-band plotting code path as 014.
- Plot generation cell uses explicit `save_path` and prints saved path.
