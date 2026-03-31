# 022 — Experiment 015: Nonlinear Latent Dynamics + Regularized Latent16

## Objective

Two orthogonal improvements to Stage 1 latent quality, plus shipping zero gating:

1. **Ship zero gating** — integrate post-hoc zero gating (k=7) into `forecaster.py` as default post-processing
2. **Nonlinear latent dynamics** — replace linear VAR(p) with GRU-based dynamics in Stage 1
3. **Regularized latent_dim=16** — wider latent with encoder dropout + weight decay to reduce variance
4. **Combined** — GRU dynamics + regularized latent16

## Motivation

From experiment 014:
- Zero gating is a free lunch (+0.012 skill, improves every scenario)
- Stage 1 latent quality is the ceiling — horizon-weighted loss (neutral) and lag features (harmful) both failed
- latent_dim=16 showed better median but higher variance — needs regularization
- h=13-14 weakness is driven by VAR rollout drift → nonlinear dynamics may help

## Scenarios

| Scenario | latent_dynamics_type | latent_dim | encoder_hidden | encoder_dropout | weight_decay_supply | Description |
|----------|---------------------|------------|----------------|-----------------|---------------------|-------------|
| S1 | var | 8 | [32,16] | 0.0 | 0.0 | Control (014 S1 + zero gating) |
| S2 | gru | 8 | [32,16] | 0.0 | 0.0 | Nonlinear dynamics only |
| S3 | var | 16 | [64,32] | 0.1 | 1e-4 | Regularized latent16 |
| S4 | gru | 16 | [64,32] | 0.1 | 1e-4 | Combined |

All scenarios: `stage2_head_type="gru"`, `gru_hidden_dim=32`, `zero_gate_k=7`

## Files Modified

- `lavar/config.py` — add `zero_gate_k`, `latent_dynamics_type`, `dynamics_gru_hidden_dim`, `encoder_dropout`, `weight_decay_supply`
- `lavar/_core/dynamics.py` — add `GRUDynamics` class
- `lavar/_core/model.py` — config-driven dynamics selection in `LAVAR.__init__`
- `lavar/_core/heads.py` — add dropout support to `MLP`
- `lavar/_training/stage1.py` — handle GRU dynamics in re-init block
- `lavar/_training/stage2/stage2_test_baseline.py` — AdamW with weight decay
- `lavar/forecaster.py` — zero gating in `predict()`, pass dynamics config to LAVAR

## Files Created

- `history/022_015_nonlinear_dynamics_reg_latent16.md` — this file
- `notebooks/015_hecon.ipynb` — experiment notebook

## Validation

1. Smoke test: zero gating clamps zero-history targets
2. Smoke test: GRU dynamics produces valid predictions
3. Run 015_hecon.ipynb with all 4 scenarios
4. Compare mean_skill, median_skill, positive-skill folds across scenarios
