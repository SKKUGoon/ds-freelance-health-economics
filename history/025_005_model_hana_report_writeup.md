# Objective
Write a consolidated markdown report for completed `005_model` Hana experiments (85 runs) using generated CSV artifacts.

# Scope
- Read and synthesize results from:
  - `notebooks/report/005_model/hana/leaderboard.csv`
  - `notebooks/report/005_model/hana/progress_summary.csv`
  - `notebooks/report/005_model/hana/progress_fold_metrics.csv`
  - `notebooks/report/005_model/hana/scenario_epoch_group_summary.csv`
  - `notebooks/report/005_model/hana/worst_folds_by_mse.csv`
  - `notebooks/report/005_model/hana/worst_folds_by_skill.csv`
  - `notebooks/report/005_model/hana/run_status.json`
- Create one markdown report file in `notebooks/report/005_model/hana/`.

# File List
- `history/025_005_model_hana_report_writeup.md` (this plan record)
- `notebooks/report/005_model/hana/005_model_hana_report.md` (new report)

# Implementation Steps
1. Confirm run completion and protocol metadata.
2. Extract leaderboard highlights (best/worst runs).
3. Summarize scenario-family behavior and epoch-cap effects.
4. Summarize fold-level failure patterns from worst-fold tables.
5. Write recommendations and next actions.

# Validation Criteria
- Report reflects completed `85/85` status.
- Report uses metrics consistent with source CSVs.
- Report is saved as markdown under `notebooks/report/005_model/hana/`.
