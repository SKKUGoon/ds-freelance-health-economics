## Objective

Add Epurun hospital replication results (011_seed, 012_seed, 013_ensemble) to PAPER_ENG.md and PAPER_KOR.md, mirroring the existing Hana Stages E–G.

## Scope

- Summarize epurun seed stability (011_seed), cross-seed blend robustness (012_seed), and production ensemble (013_ensemble) results.
- Append new sections to both English and Korean papers.
- Update executive summaries, conclusions, limitations, figures, and artifacts lists.

## File List

- `notebooks/PAPER_ENG.md` (edit)
- `notebooks/PAPER_KOR.md` (edit)

## Key Epurun Results

- 248 ATC targets, 52 folds, horizon=14
- S04 slightly better mean_mse than S11 (0.957 vs 0.962), opposite of Hana
- S11 has better tail behavior (max_fold_mse 3.45 vs 4.57)
- Blend (w=0.55) improves mean MSE by -4.38% over S11
- Blend wins 5/5 seeds; w* range 0.45–0.55
- Per-drug: top 5 of 248 drugs account for ~20% of MSE

## Validation Criteria

- Both papers contain epurun sections with correct numbers from CSV reports.
- Figures and artifacts lists are updated.
