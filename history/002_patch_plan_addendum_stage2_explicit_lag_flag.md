# 002 - Patch Plan Addendum: Stage2 Explicit Lag-Coefficient Flag

## Objective
Document an experiment toggle for Stage 2 that enables explicit lag-coefficient handling.

## Scope
- In scope: patch-plan documentation only.
- Out of scope: source code edits.

## Requested Addition
- Add config option:
  - `stage2_use_explicit_lag_coeff: bool = False`
- Purpose:
  - Enable/disable explicit lag-coefficient behavior in Stage 2 experiments.
- Default behavior:
  - `False` keeps baseline behavior unchanged.

## Validation Criteria
- `history/001_stage2_mode_split_and_patch_policy.md` explicitly includes:
  - the new boolean flag name,
  - default value (`False`),
  - experimental intent,
  - impact on Stage 2 mode planning.
