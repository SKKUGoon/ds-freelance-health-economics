# 017 - 010 Diagnostics Notebook (S1 vs S8)

## Objective
Create `notebooks/010_hecon.ipynb` focused on diagnostics for the S1 baseline and S8 challenger, covering:
1) per-fold failure overlap and timing,
2) per-target density-bucket behavior,
3) simple S1+S8 ensembling,
4) horizon-wise error/skill decomposition.

## Scope
- In scope:
  - Build a diagnostics-first notebook consuming `report/009` artifacts.
  - Add executable code for fold-level analysis directly from eval fold CSV.
  - Add executable code paths to compute per-target / horizon diagnostics from S1/S8 rolling predictions.
  - Include notebook-native interpretation markdown.
- Out of scope:
  - Modifying model/training code.
  - Replacing production scenario defaults.

## Files
- `notebooks/010_hecon.ipynb`

## Implementation Steps
1. Load `report/009` eval summary/folds and source parquet files.
2. Implement fold catastrophe overlap diagnostics between S1 and S8 (including holiday proximity).
3. Add helper to generate/load cached manual rolling predictions for S1/S8.
4. Compute per-target metrics and summarize by density bucket.
5. Evaluate simple blend grid for S1+S8.
6. Compute horizon-wise MSE and skill for S1/S8.
7. Write concise analysis markdown in notebook.

## Validation Criteria
- Notebook JSON parses.
- Fold diagnostics execute without missing file errors.
- Cache logic can produce/load S1/S8 prediction tensors.
- Per-target, ensemble, and horizon sections are executable and save CSV outputs under `report/010`.
