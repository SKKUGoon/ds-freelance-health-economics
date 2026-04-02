# 030 — 008 Seed Stability Gate for S11 and S04

## Objective
Verify that the S11 vs S04 ranking (mean MSE gap ~0.19) is stable across random seeds. If the gap is within seed-to-seed variance, the ranking is noise and blend weight selection from 007 is unreliable.

## Scope
- Notebook `008_seed.ipynb`
- Report output: `notebooks/report/008_seed/hana/`
- Checkpoints: `notebooks/checkpoints/hana/stage1_seed/`, `stage2_seed/`

## Run Matrix
- Scenarios: S04_GRU_H32_L1 (E02), S11_LATENT16 (E05)
- Seeds: [42, 123, 456, 789, 1024]
- Total: 2 × 5 = 10 runs × 130 folds each

## Implementation Steps
1. Seed-aware signature: include seed in Stage 1 checkpoint hash to force fresh training per seed
2. `set_seed(seed)` called before each run_single_experiment
3. Rolling eval identical to 005: horizon=14, fold_step=14, retrain_cadence=90, quality_triggers
4. Analysis: per-seed metrics, cross-seed std, gap-vs-spread test, foldwise correlation
5. Decision rule: gap > 2 * max(std_S11, std_S04) → ranking reliable

## Validation Criteria
- Seed 42 metrics match 005 results (within float tolerance)
- 130 folds per run
- Clear summary of whether ranking is stable
