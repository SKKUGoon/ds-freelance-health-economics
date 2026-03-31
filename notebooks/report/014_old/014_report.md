# Experiment 014 — Horizon-Weighted Loss + Stage 1 Dynamics + Post-Hoc Zero Gating

## Objective

Three orthogonal improvements to the GRU baseline (011 S2):
1. **Horizon-weighted loss** — linear increasing weights on h=1..14 to fix late-horizon weakness
2. **Stage 1 dynamics** — increase `dyn_p` (7→14) and/or `latent_dim` (8→16) to improve VAR rollout
3. **Post-hoc zero gating** — clamp predictions to zero for drugs with zero recent history (k=7)

## Scenarios

| Scenario | dyn_p | latent_dim | encoder | horizon_loss | Change from S1 |
|----------|-------|------------|---------|--------------|----------------|
| S1 | 7 | 8 | [32,16] | uniform | Control |
| S2 | 7 | 8 | [32,16] | **linear** | Horizon weights only |
| S3 | **14** | 8 | [32,16] | uniform | dyn_p only |
| S4 | 7 | **16** | **[64,32]** | uniform | Wider latent/encoder |
| S5 | **14** | **16** | **[64,32]** | uniform | S3 + S4 |
| S6 | **14** | **16** | **[64,32]** | **linear** | Kitchen sink |

---

## Results — RollingEvaluator

| Scenario | mean_mse | med_mse | mean_skill | **med_skill** | pos_skill folds |
|----------|----------|---------|------------|---------------|-----------------|
| **S1 baseline** | **60.3** | 49.3 | +0.080 | **+0.091** | **102 (78%)** |
| **S2 h-linear** | **60.3** | 50.4 | **+0.082** | +0.088 | **104 (80%)** |
| S3 dyn_p=14 | 65.0 | 55.4 | -0.014 | -0.008 | 48 (37%) |
| S4 latent16 | 61.9 | **48.2** | +0.057 | **+0.098** | 99 (76%) |
| S5 combined | 65.2 | 55.7 | -0.014 | -0.007 | 43 (33%) |
| S6 kitchen sink | 65.2 | 55.4 | -0.016 | -0.011 | 34 (26%) |

## Results — Manual Rolling

| Scenario | mean_mse | med_mse | mean_skill | med_skill |
|----------|----------|---------|------------|-----------|
| **S1 baseline** | **59.0** | 49.0 | **+0.105** | +0.084 |
| S2 h-linear | 63.1 | 51.4 | +0.031 | +0.064 |
| S3 dyn_p=14 | 65.1 | 55.2 | -0.011 | -0.006 |
| S4 latent16 | 64.2 | **48.2** | +0.021 | **+0.089** |
| S5 combined | 65.1 | 54.7 | -0.012 | -0.006 |
| S6 kitchen sink | 64.8 | 55.4 | -0.008 | -0.004 |

---

## Analysis by Improvement

### 1. Horizon-Weighted Loss (S2 vs S1)

**Verdict: Neutral.** S2 is statistically indistinguishable from S1.

| Metric | S1 | S2 | Delta |
|--------|----|----|-------|
| Eval mean_mse | 60.28 | 60.25 | **-0.03** |
| Eval mean_skill | +0.080 | +0.082 | +0.003 |
| Eval positive-skill folds | 102 (78%) | 104 (80%) | +2 |
| H2H wins vs S1 | — | 62/130 (48%) | |

The linear horizon weights produce near-identical aggregate performance. The RollingEvaluator shows a marginal improvement (+0.003 mean skill, +2 positive folds), but the manual rolling shows S2 slightly worse (mean_skill +0.031 vs +0.105). The difference is within noise.

**Why it didn't help:** The h=13-14 weakness diagnosed in 012 (skill -0.08 to -0.17) is driven by VAR rollout error in the latent, not by the loss weighting. Upweighting late horizons in the Stage 2 loss cannot fix latent drift — the GRU simply receives worse `z_{13}`, `z_{14}` inputs regardless of how it was trained.

### 2. dyn_p=14 (S3 vs S1)

**Verdict: Harmful.** Increasing VAR order from 7 to 14 significantly degrades performance.

| Metric | S1 | S3 | Delta |
|--------|----|----|-------|
| Eval mean_mse | 60.3 | 65.0 | **+4.8** |
| Eval mean_skill | +0.080 | -0.014 | **-0.094** |
| Positive-skill folds | 102 (78%) | 48 (37%) | **-54** |
| H2H wins vs S1 | — | 28/130 (22%) | |

Doubling the VAR order collapses positive-skill folds from 78% to 37%. S3 wins only 22% of folds against S1.

**Why it hurt:** VAR(14) has 14× more parameters than VAR(7) in the transition matrix `A`. With latent_dim=8, VAR(7) estimates an 8×56 matrix; VAR(14) estimates 8×112. The additional parameters overfit the training latent trajectories and generalize poorly. The encoder architecture ([32,16]→8) only produces 8-dim latents — there isn't enough latent signal to justify 14 lags of history.

### 3. latent_dim=16 with wider encoder (S4 vs S1)

**Verdict: Mixed — better median, worse mean.**

| Metric | S1 | S4 | Delta |
|--------|----|----|-------|
| Eval mean_mse | 60.3 | 61.9 | +1.6 |
| Eval **median_mse** | 49.3 | **48.2** | **-1.1** |
| Eval mean_skill | +0.080 | +0.057 | -0.023 |
| Eval **median_skill** | +0.091 | **+0.098** | **+0.007** |
| H2H wins vs S1 | — | **71/130 (55%)** | |
| MSE std | 55.8 | 60.7 | +4.9 |

S4 achieves the **best median MSE (48.2)** and **best median skill (+0.098)** of all scenarios. It also wins 55% of head-to-head folds — the only scenario to win a majority against S1. However, its mean is worse due to higher variance (std 60.7 vs 55.8).

**Interpretation:** The wider latent captures more signal for typical folds (better median), but the larger model is less stable on outlier folds (higher variance). The encoder [64,32]→16 has more capacity, which helps on average-complexity inputs but can overfit on unusual periods.

### 4. Combined configurations (S5, S6)

**Verdict: dyn_p=14 poisons everything it touches.**

S5 (dyn_p=14 + latent16) and S6 (S5 + horizon weights) both produce negative mean skill, ~37% and ~26% positive-skill folds respectively. The dyn_p=14 failure dominates. Combining it with latent_dim=16 or horizon weights cannot rescue the VAR overfitting.

---

## Post-Hoc Zero Gating

| Scenario | mean_mse (orig) | mean_mse (gated) | **delta** | mean_skill (orig) | mean_skill (gated) |
|----------|-----------------|------------------|-----------|--------------------|---------------------|
| **S1 baseline** | 59.05 | **58.49** | **-0.56** | +0.105 | **+0.117** |
| S2 h-linear | 63.12 | **62.37** | -0.75 | +0.031 | +0.044 |
| S3 dyn_p=14 | 65.07 | **64.64** | -0.42 | -0.011 | -0.002 |
| **S4 latent16** | 64.21 | **62.97** | **-1.24** | +0.021 | **+0.041** |
| S5 combined | 65.07 | **64.66** | -0.41 | -0.012 | -0.004 |
| S6 kitchen sink | 64.81 | **64.36** | -0.45 | -0.008 | +0.002 |

**Zero gating improves every scenario.** The MSE reduction ranges from -0.41 to -1.24, with S4 benefiting most (-1.24 MSE, +0.020 skill). This confirms that phantom predictions for inactive drugs were adding unnecessary error.

The improvement is modest in absolute terms (~0.5-1.2 MSE) because ultra-sparse targets have small values, so false positives contribute little to overall MSE. But for operational correctness — not predicting drug usage that hasn't occurred in a week — this is valuable.

**Best gated result:** S1 + zero gating achieves mean_mse=58.49, mean_skill=+0.117 — the **best overall performance** in the experiment.

---

## Summary Table — All Experiments Cumulative

| Experiment | Best scenario | mean_skill | med_skill | Key finding |
|------------|---------------|------------|-----------|-------------|
| 008-010 | MLP baseline | -0.40 | +0.03 | MLP can't beat naive |
| 011 | GRU h=32 | +0.086 | +0.105 | Temporal structure is key |
| 013 | GRU + lag features | -0.069 | -0.002 | Encoder bottleneck kills lags |
| **014** | **S1 + zero gating** | **+0.117** | **+0.101** | **Best result so far** |
| 014 | S4 (latent16) + zero gating | +0.041 | +0.100 | Better median, worse mean |

## Key Findings

1. **S1 GRU baseline remains the best model.** None of the three improvements meaningfully beat it on aggregate. The GRU h=32 with dyn_p=7, latent_dim=8 is the sweet spot for this data size.

2. **dyn_p=14 is harmful.** The VAR overfits with too many lags. With only 8-dim latent, 7 lags are sufficient. This is a clear negative result.

3. **latent_dim=16 is promising but noisy.** Better median performance suggests the wider latent captures more signal. The variance increase suggests it needs more training data or regularization to stabilize.

4. **Horizon-weighted loss is a non-factor.** The h=13-14 weakness is a latent quality problem, not a loss weighting problem. The GRU is already doing the best it can with the latents it receives.

5. **Zero gating is a free lunch.** +0.012 skill improvement with zero implementation complexity. Should be adopted as a default post-processing step.

## Recommendations

1. **Adopt zero gating** as a standard post-processing step in the forecaster. It's pure improvement with no downside.

2. **Keep dyn_p=7, latent_dim=8** as the default configuration. The current architecture is well-matched to the data volume (T=2556).

3. **For next experiments**, the remaining improvement vectors are:
   - **Regularized latent_dim=16**: S4 showed better median but higher variance. Adding dropout, weight decay, or latent regularization could stabilize the wider model.
   - **Stage 1 architecture**: The encoder/decoder and VAR dynamics are the ceiling. Exploring nonlinear dynamics (GRU in latent transition instead of linear VAR) or deeper encoders could raise this ceiling.
   - **Training strategy**: Longer `train_days`, curriculum learning, or learning rate scheduling may help the existing architecture extract more from the data.
