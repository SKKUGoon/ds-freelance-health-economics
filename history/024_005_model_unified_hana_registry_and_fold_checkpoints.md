# Objective
Implement `notebooks/005_model.ipynb` as a unified experiment notebook for the new single-target Hana dataset, covering full scenario registry, non-overlapping rolling evaluation, fold-level analysis, and per-fold checkpoint persistence with Stage 1 reuse.

# Scope
- Create a complete runnable notebook at `notebooks/005_model.ipynb`.
- Use only new unified data paths under `notebooks/data/hana/`.
- Enforce `X -> y` only (no target-history or supply-history input to X).
- Include explicit scenario registry labeled `S1...Sk` and run all active scenarios.
- Add epoch-cap ablation matrix and execute 85 total runs (`17 scenarios x 5 epoch profiles`).
- Save checkpoints per fold to:
  - `notebooks/checkpoints/hana/stage1`
  - `notebooks/checkpoints/hana/stage2`
- Reuse Stage 1 checkpoints when fold window + Stage 1 signature matches.
- Keep no-overlap protocol (`horizon=14`, `fold_step=14`).

# File List
- `notebooks/005_model.ipynb` (new content)
- `history/024_005_model_unified_hana_registry_and_fold_checkpoints.md` (this plan record)

# Implementation Steps
1. Build notebook scaffolding with clear sections: setup, data load, scenario registry, epoch profiles, rolling runner, analysis, exports.
2. Encode active scenario list (`S01..S17`) and dropped scenario list (`D01..`) with reasons.
3. Add a cross-product run plan (`S x E`) targeting 85 runs.
4. Implement custom rolling evaluator in-notebook:
   - retrain cadence and quality trigger behavior
   - no overlap folds
   - fold metrics (MSE/RMSE/MAE/skill)
5. Implement Stage 1 checkpoint signature and reuse logic:
   - signature includes fold end, X schema fingerprint, and stage1-relevant hyperparameters
   - load existing stage1 checkpoint if available; otherwise train and save
6. Implement Stage 2 per-fold checkpoint save after head training.
7. Persist progress continuously:
   - run-level fold CSVs
   - global progress summary CSV
   - status JSON for resumability
8. Add analysis outputs in same notebook (leaderboard + fold diagnostics).

# Validation Criteria
- Notebook reads `notebooks/data/hana/lavar_ready_x.parquet` and `notebooks/data/hana/lavar_ready_y.parquet`.
- All active runs use `use_supply_history=False` and no lagged-y feature inputs.
- Rolling eval uses `horizon=14`, `fold_step=14`.
- Stage1 checkpoints are written in `notebooks/checkpoints/hana/stage1` and reused on signature hits.
- Stage2 checkpoints are written in `notebooks/checkpoints/hana/stage2` by run and fold.
- Progress artifacts are periodically written under `notebooks/report/005_model/hana`.
