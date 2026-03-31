# 013 - Stage2 Folder Refactor and 008 Experiment Plan

## Objective
Refactor Stage 2 into a plugin-style package under `lavar/_training/stage2/`, rename mode naming to `baseline` and `supply_history_latent`, and create `notebooks/008_hecon.ipynb` with four experiment scenarios including a mollified training variant.

## Scope
- In scope:
  - Stage 2 package/file reorganization and dispatcher wiring.
  - Stage 2 mode rename + backward-compatible aliases.
  - `008_hecon.ipynb` creation with per-scenario plots and exports.
  - Artifact output standardization under `notebooks/report/008/`.
  - Documentation sync in `AGENTS.md` and `CLAUDE.md`.
- Out of scope:
  - Stage 1 model architecture changes.
  - Evaluation API redesign.

## Files (planned)
- `lavar/_training/stage2/__init__.py`
- `lavar/_training/stage2/common.py`
- `lavar/_training/stage2/stage2_test_baseline.py`
- `lavar/_training/stage2/stage2_test_supply_history_latent.py`
- `lavar/config.py`
- `lavar/forecaster.py` (only if mode/flag plumbing requires minimal updates)
- `lavar/evaluation.py` (only if import path update is required)
- `AGENTS.md`
- `CLAUDE.md`
- `notebooks/008_hecon.ipynb`

## Mode Contract (target)
- Primary modes:
  - `baseline`
  - `supply_history_latent`
- Backward-compatible aliases:
  - `ts_latent -> baseline`
  - `ts_supply_ts_latent -> supply_history_latent`

Validation constraints:
1. `supply_history_latent` requires `use_supply_history=True`.
2. Current fixed-input-dim constraint with Stage 1 remains enforced.

## Stage2 Dispatcher Design
- Keep container entrypoint: `from lavar._training.stage2 import train_supply_heads`.
- Dispatcher map (planned):
  - `baseline -> stage2_test_baseline.train_supply_head_indexed`
  - `supply_history_latent -> stage2_test_supply_history_latent.train_supply_head_indexed`
- Keep existing return contract unchanged:
  - `dense_indices`, `sparse_indices`, `ultra_indices`
  - `dense_model`, `sparse_model`, `ultra_model`

## 008 Notebook Experiment Design

### Dataset
- Use `ys` path consistent with 007.

### Scenarios
1. `S1_baseline`
   - `stage2_mode="baseline"`
   - `use_supply_history=False`
2. `S2_supply_history_latent`
   - `stage2_mode="supply_history_latent"`
   - `use_supply_history=True`
3. `S3_supply_history_latent_lr_low`
   - same as S2 with lower `lr_supply`
4. `S4_supply_history_latent_mollified`
   - same as S2 + smooth nonnegative continuation (softplus-style annealing)

### Plot requirement per scenario
- Draw True vs Pred graph with two uncertainty bands.
- Force nonnegative supply lower bounds:
  - both interval lower bounds clipped to `0`.

### Output artifacts
Save under `notebooks/report/008/`:
- `008_eval_mode_compare_summary.csv`
- `008_eval_mode_compare_folds.csv`
- `008_manual_<scenario>_summary.csv`
- `008_manual_<scenario>_folds.csv`
- `008_<scenario>_true_vs_pred.png`

## Implementation Steps
1. Create Stage 2 package folder + move mode-specific trainers.
2. Update dispatcher imports/map in package `__init__`.
3. Update config mode literals + alias mapping + validators.
4. Update docs (`AGENTS.md`, `CLAUDE.md`) mode names and stage2 package references.
5. Create `008_hecon.ipynb` with experimentation plan section, scenario runner, per-scenario plots, and exports.
6. Validate JSON for notebook and import compatibility for dispatcher path.

## Validation Criteria
- `train_supply_heads` import path remains stable.
- New mode names work; aliases resolve correctly.
- `008_hecon.ipynb` runs four scenarios and saves scenario-separated artifacts.
- Per-scenario plot lower confidence bounds are nonnegative.
