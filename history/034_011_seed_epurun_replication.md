## Objective

Replicate the `008_seed.ipynb` workflow into `011_seed.ipynb` for the epurun dataset, preserving all modeling/training/evaluation configuration while isolating outputs and checkpoints under epurun-specific paths.

## Scope

- Create `notebooks/011_seed.ipynb` from `notebooks/008_seed.ipynb`.
- Switch all dataset, report, and checkpoint roots from `hana` to `epurun`.
- Keep scenario definitions, seeds, horizon, `dyn_p`, epochs, and retrain cadence unchanged.
- Ensure run signature payload uses `hospital="epurun"` to avoid cross-dataset checkpoint collisions.

## File List

- `notebooks/011_seed.ipynb` (new)

## Implementation Steps

1. Copy `008_seed.ipynb` to `011_seed.ipynb`.
2. Replace path roots:
   - `data/hana` -> `data/epurun`
   - `report/008_seed/hana` -> `report/011_seed/epurun`
   - `checkpoints/hana/stage1_seed` -> `checkpoints/epurun/stage1_seed`
   - `checkpoints/hana/stage2_seed` -> `checkpoints/epurun/stage2_seed`
3. Replace signature payload field `hospital` value from `hana` to `epurun`.
4. Execute notebook logic (via `uv run python` notebook-cell runner) to generate epurun seed checkpoints and reports.

## Validation Criteria

- `notebooks/011_seed.ipynb` exists and contains epurun paths.
- Training completes for all planned runs/seeds or reports clear failures.
- Output artifacts are produced under `notebooks/report/011_seed/epurun/`.
- Stage1/Stage2 checkpoints are written only under `notebooks/checkpoints/epurun/`.
