## Objective

Replicate `009_seed.ipynb` into `012_seed.ipynb` for epurun to perform the same cross-seed blend robustness analysis using the new `011_seed` epurun artifacts.

## Scope

- Create `notebooks/012_seed.ipynb` from `notebooks/009_seed.ipynb`.
- Update input dependencies to epurun `011_seed` reports/checkpoints.
- Keep blend analysis settings identical (including fixed weights and grid behavior).
- Generate outputs in epurun-specific report directory.

## File List

- `notebooks/012_seed.ipynb` (new)

## Implementation Steps

1. Copy `009_seed.ipynb` to `012_seed.ipynb`.
2. Replace path roots:
   - `data/hana` -> `data/epurun`
   - `report/008_seed/hana` -> `report/011_seed/epurun`
   - `report/009_seed/hana` -> `report/012_seed/epurun`
   - `checkpoints/hana/stage1_seed` -> `checkpoints/epurun/stage1_seed`
   - `checkpoints/hana/stage2_seed` -> `checkpoints/epurun/stage2_seed`
3. Update notebook title/intro to reflect `012_seed` + epurun.
4. Execute notebook logic and save blend robustness artifacts.

## Validation Criteria

- `notebooks/012_seed.ipynb` exists and references epurun inputs/outputs.
- Output artifacts are produced under `notebooks/report/012_seed/epurun/`.
- Metrics/plots are generated without referencing hana paths.
