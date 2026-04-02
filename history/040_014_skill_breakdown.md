# 040 - 014 Skill Breakdown

## Objective
Add per-target naive error outputs to fold evaluation and create `notebooks/014_skill.ipynb` to explain aggregate skill differences with per-drug and density-bucket skill breakdowns for both HANA and Epurun.

## Scope
- Extend `compute_fold_metrics(...)` to expose per-target naive MSE and per-target skill.
- Add a new notebook that reads existing ensemble prediction artifacts, reconstructs naive baselines, computes per-drug and per-bucket skill, and writes separate reports under `notebooks/report/014_skill/hana` and `notebooks/report/014_skill/epurun`.
- Keep the change minimal and analysis-focused; no retraining pipeline changes.

## Files
- `lavar/evaluation.py`
- `notebooks/014_skill.ipynb`

## Implementation Steps
1. Update `compute_fold_metrics(...)` to compute `naive_mse_by_target` and `skill_vs_naive_by_target` alongside the existing aggregate metrics.
2. Create `notebooks/014_skill.ipynb` with reusable helpers for loading saved ensemble prediction artifacts, reconstructing naive predictions from `lavar_ready_y.parquet`, computing density buckets, and exporting summary tables and plots.
3. Configure the notebook to process both datasets and save outputs independently under `notebooks/report/014_skill/hana` and `notebooks/report/014_skill/epurun`.
4. Add concise markdown report generation summarizing aggregate, bucket, and per-drug skill behavior for each dataset.

## Validation Criteria
- `lavar/evaluation.py` remains importable and returns the new per-target metrics without changing the existing aggregate metric semantics.
- `notebooks/014_skill.ipynb` is valid notebook JSON.
- The notebook writes dataset-specific outputs under `notebooks/report/014_skill/hana` and `notebooks/report/014_skill/epurun`.
- Recomputed aggregate skill from notebook artifacts is consistent with the saved ensemble summaries within normal floating-point tolerance.
