## Objective

Create `notebooks/009_seed.ipynb` to run cross-seed blend robustness checks for S11+S04 using existing `008_seed` artifacts/checkpoints (no retraining), and save outputs under `notebooks/report/009_seed/hana/`.

## Scope

- New notebook: `notebooks/009_seed.ipynb`
- Offline inference/reconstruction only from saved seed checkpoints.
- Evaluate blend weights (fixed and grid) per seed.
- Export summary CSVs, plots, and a short markdown note.

## File List

- `notebooks/009_seed.ipynb`

## Implementation Steps

1. Add setup/config cells for data, checkpoint, and report paths.
2. Implement checkpoint reconstruction helpers for seed runs.
3. Reconstruct per-drug predictions for `S11_LATENT16` and `S04_GRU_H32_L1` across seeds.
4. Compute fold-level blend metrics for `w=0.50`, `w=0.55`, plus `w` grid `0.00..1.00`.
5. Build per-seed robustness summaries versus pure S11 baseline.
6. Save outputs to `notebooks/report/009_seed/hana/` and generate diagnostic plots.

## Validation Criteria

- Notebook uses existing checkpoints; no `fit`/training calls.
- Blend robustness outputs are written to `report/009_seed/hana/`.
- Outputs include per-seed win counts and weight stability signals.
