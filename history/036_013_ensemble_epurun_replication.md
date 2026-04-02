## Objective

Replicate `010_ensemble.ipynb` into `013_ensemble.ipynb` for epurun using the same ensemble recipe and reporting workflow.

## Scope

- Create `notebooks/013_ensemble.ipynb` from `notebooks/010_ensemble.ipynb`.
- Point dependencies to epurun data and `011_seed` artifacts/checkpoints.
- Keep constants unchanged: seeds, blend weight (`w=0.55`), forecast horizon, and empirical 95% CI.
- Keep non-negativity clipping and per-drug export/graph generation logic identical.

## File List

- `notebooks/013_ensemble.ipynb` (new)

## Implementation Steps

1. Copy `010_ensemble.ipynb` to `013_ensemble.ipynb`.
2. Replace path roots:
   - `data/hana` -> `data/epurun`
   - `report/008_seed/hana` -> `report/011_seed/epurun`
   - `checkpoints/hana/stage1_seed` -> `checkpoints/epurun/stage1_seed`
   - `checkpoints/hana/stage2_seed` -> `checkpoints/epurun/stage2_seed`
   - `report/010_ensemble/hana` -> `report/013_ensemble/epurun`
3. Update notebook title/intro labels from `010`/Hana to `013`/Epurun.
4. Execute notebook logic to generate ensemble metrics, CI tables, and plots.

## Validation Criteria

- `notebooks/013_ensemble.ipynb` exists and references epurun inputs/outputs.
- Outputs are generated under `notebooks/report/013_ensemble/epurun/`.
- Ensemble artifacts include per-drug predictions, CI table, metrics CSVs, plot, and summary markdown.
