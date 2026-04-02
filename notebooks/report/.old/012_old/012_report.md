# Experiment 012 — Diagnostics: S1 MLP Baseline vs S2 GRU Decoder

## Objective

Experiment 011 established S2 (GRU h=32) as the first model with positive mean skill (+0.086, median +0.105). This notebook diagnoses **where** and **how** the GRU improves over the MLP baseline across four dimensions: fold stability, density buckets, ensembling, and horizon position.

---

## 1) Per-Fold Diagnostics

### Catastrophic Folds

| Metric | S1 MLP | S2 GRU |
|--------|--------|--------|
| Catastrophic folds (MSE>500) | **2** | **1** |
| Only in this model | 1 (fold 8) | 0 |
| Shared | 1 (fold 0) | 1 (fold 0) |

The single catastrophic fold that S1 had exclusively — fold 8 (2020-04-23) — is the most dramatic difference in the entire dataset: **S1 MSE = 3,976 vs S2 MSE = 77**. This single fold accounts for the majority of S1's mean/median divergence. The GRU reduced it by **98%**. The shared catastrophic fold (2020-01-02, New Year) is nearly identical in both models (~535), suggesting it is a data artifact rather than a model deficiency.

### Head-to-Head Win Rate

| Metric | Value |
|--------|-------|
| S2 wins (lower MSE) | **85/130 (65%)** |
| S1 wins (lower MSE) | 45/130 (35%) |
| Mean improvement when S2 wins | **56.1 MSE** |
| Mean regression when S1 wins | 8.0 MSE |
| Oracle min(S1,S2) mean MSE | 57.1 |
| Oracle min(S1,S2) median MSE | 46.3 |

S2's wins are **7x larger** than its losses. When S2 wins, it improves by 56 MSE on average; when S1 wins, S1 only improves by 8. This extreme asymmetry is primarily driven by the elimination of fold 8's catastrophe, but the pattern holds even excluding outliers — S2 is broadly better, not just safer.

### Stability

| Metric | S1 MLP | S2 GRU |
|--------|--------|--------|
| Mean MSE | 93.8 | **59.8** |
| Median MSE | 51.6 | **48.5** |
| MSE std | 347.5 | **55.8** |
| Max MSE | 3,976 | **534** |

S1's variance is **6.2x** that of S2, entirely driven by tail behavior. The GRU's max MSE (534) is 7.4x lower than S1's max (3,976).

### Skill Distribution

| Metric | S1 MLP | S2 GRU |
|--------|--------|--------|
| Mean skill | -0.404 | **+0.086** |
| Median skill | +0.028 | **+0.105** |
| 25th percentile | -0.043 | **+0.022** |
| 75th percentile | +0.080 | **+0.154** |
| Positive-skill folds | 79/130 (61%) | **107/130 (82%)** |

S2's 25th percentile skill is positive (+0.022) — even its bad folds marginally beat naive. S1's worst quartile is negative.

### Holiday Proximity

S1's two catastrophic folds: 2020-01-02 (0 days from day-after-holiday) and 2020-04-23 (7 days). S2's single catastrophic fold: 2020-01-02 only. The GRU eliminates the non-holiday catastrophe entirely.

---

## 2) Per-Target Density-Bucket Breakdown

### Bucket-Level Summary

| Bucket | S1 MSE | S2 MSE | Delta | % Change |
|--------|--------|--------|-------|----------|
| **Dense** | 666.5 | **333.4** | -333.1 | **-50.0%** |
| Sparse | 61.1 | **51.2** | -9.9 | -16.1% |
| Ultra-sparse | 0.104 | **0.031** | -0.073 | -70.2% |

The GRU halves dense-bucket MSE. This is the single largest improvement source. Dense targets are the highest-volume drugs (nonzero_rate >= 70%) and dominate the overall MSE, so a 50% reduction here drives most of the aggregate improvement.

### Top Dense Target Improvements

| Target | S1 MSE | S2 MSE | % Change |
|--------|--------|--------|----------|
| **H02AB02** (hydrocortisone) | 1,571.7 | **707.3** | **-55.0%** |
| **N02AX02** (tramadol) | 610.7 | **216.6** | **-64.5%** |
| M01AB05 (diclofenac) | 335.3 | **295.2** | -12.0% |
| N01BB52 (lidocaine combo) | 148.4 | **114.5** | -22.8% |

H02AB02 and N02AX02 were identified in experiment 010 as the two worst targets — the GRU dramatically improves both. These two targets alone account for ~1,258 MSE reduction out of the ~1,332 dense-bucket total improvement (94%).

### Top Sparse Target Improvements

| Target | S1 MSE | S2 MSE | % Change |
|--------|--------|--------|----------|
| J01GB07 | 371.9 | **291.0** | -21.7% |
| J01DC05 | 389.9 | **342.7** | -12.1% |
| B05BB02 | 531.1 | **469.6** | -11.6% |
| J01DC09 | 44.4 | **16.2** | -63.4% |
| R06AB04 | 57.7 | **36.6** | -36.5% |

The high-MSE sparse targets (aminoglycosides, cephalosporins, electrolytes) all benefit, though less dramatically than the dense targets.

### Targets Where S2 Regresses

| Target | Bucket | S1 MSE | S2 MSE | Delta |
|--------|--------|--------|--------|-------|
| M01AB15 | sparse | 10.2 | 11.2 | +0.95 |
| A11EA | sparse | 5.4 | 5.5 | +0.05 |

Regressions are minor — the largest is +0.95 MSE (M01AB15), which is negligible compared to the improvements. No dense target regresses.

### Target Win Rate

S2 achieves lower MSE on **33/48 targets (69%)**. Combined with the magnitude asymmetry (large improvements vs tiny regressions), the GRU is a **broad-based improvement**, not driven by a few lucky targets.

---

## 3) Ensemble S1 + S2

| Alpha (S2 weight) | MSE | MAE |
|-------------------|-----|-----|
| 0.00 (pure S1) | 93.76 | 2.584 |
| 0.50 | 66.95 | 2.473 |
| **0.95** | **59.64** | **2.449** |
| 1.00 (pure S2) | 59.81 | 2.451 |

**Optimal blend: alpha = 0.95 (95% S2, 5% S1), MSE = 59.64.**

The improvement over pure S2 is marginal: 59.64 vs 59.81, a **0.3% reduction**. The blend curve is monotonically decreasing from S1 toward S2 with only a tiny uptick at alpha=1.0. This means:

1. **S2 is near-strictly dominant over S1** — there is almost no complementary information in S1's predictions.
2. **Ensembling is not worth the complexity.** The 0.17 MSE improvement from adding 5% S1 is not operationally meaningful.
3. Unlike experiment 010 where the optimal S8 blend was alpha=0.15 (85% S1, 15% S8), here the relationship is inverted — the new model is the strong one.

**Recommendation: Use pure S2 without blending.**

---

## 4) Horizon Decomposition (h=1..14)

This is the most revealing diagnostic. The MLP's per-horizon skill profile was catastrophically uneven:

| Horizon | S1 Skill | S2 Skill | Delta | Note |
|---------|----------|----------|-------|------|
| h=1 | **-1.865** | +0.022 | +1.888 | S1 catastrophic, **GRU fixes** |
| h=2 | **-0.636** | -0.007 | +0.629 | S1 catastrophic, **GRU fixes** |
| h=3 | **-2.901** | +0.064 | +2.965 | S1 worst horizon, **GRU fixes** |
| h=4 | +0.168 | +0.043 | -0.125 | S1 better |
| h=5 | **-0.999** | +0.215 | +1.214 | S1 catastrophic, **GRU fixes** |
| h=6 | +0.042 | -0.006 | -0.048 | ~tied |
| h=7 | -0.118 | -0.038 | +0.080 | Both weak |
| h=8 | +0.016 | +0.010 | -0.006 | ~tied |
| h=9 | -0.001 | -0.054 | -0.053 | S1 slightly better |
| h=10 | +0.138 | +0.189 | +0.051 | Both good |
| h=11 | +0.133 | **+0.273** | +0.140 | S2 best horizon |
| h=12 | -0.099 | **+0.113** | +0.212 | S2 flips negative to positive |
| h=13 | +0.032 | -0.085 | -0.117 | S2 regresses |
| h=14 | **-0.450** | -0.175 | +0.275 | Both negative, S2 much less bad |

### Key Findings

**The GRU eliminates all four catastrophic horizons (h=1,2,3,5).** The MLP had skill < -0.5 at these positions, meaning it was **worse than just repeating yesterday's value**. The GRU brings all four to near-zero or positive. This is the core mechanism of the aggregate improvement.

**Why did the MLP fail at h=1,3,5 specifically?** The MLP processes each horizon independently: `z_h → MLP → ŷ_h`. The delta-MSE head predicts `Δy` and integrates from `y_0`. At h=1, the first cumulative step has no error absorption — any bias in the MLP's single-step prediction propagates directly. At odd horizons (h=1,3,5), the latent rollout `z_h = A·z_{h-1}` may have systematic phase misalignment that the pointwise MLP cannot correct because it has no memory of what it predicted at h-1. The GRU, by conditioning each prediction on all prior hidden states, can learn to compensate for rollout drift.

**S2's remaining weaknesses:**
- h=7: skill = -0.038 (mild)
- h=9: skill = -0.054 (mild)
- h=13: skill = -0.085 (moderate)
- h=14: skill = -0.175 (worst remaining)

The late-horizon degradation (h=13-14) is expected: VAR rollout error accumulates and even the GRU cannot fully correct 2-week-ahead latent drift. However, S2's h=14 skill (-0.175) is **2.6x less negative** than S1's (-0.450).

**Horizons with positive skill:** S1: 6/14, S2: **8/14**. The GRU adds 2 net positive-skill horizons.

**Mean skill across horizons:** S1: -0.467, S2: **+0.040**. The GRU moves the average from deeply negative to marginally positive.

---

## Summary

| Dimension | Finding |
|-----------|---------|
| **Fold stability** | GRU eliminates 1 of 2 catastrophic folds; reduces MSE variance 6x |
| **Win rate** | S2 wins 65% of folds, with 7x larger wins than losses |
| **Dense targets** | -50% MSE; H02AB02 (-55%) and N02AX02 (-65%) drive most of it |
| **Sparse targets** | -16% MSE; broad improvement across high-volume antibiotics |
| **Ensemble** | Optimal alpha=0.95; blending adds <0.3% — not worth complexity |
| **Horizon h=1,3,5** | MLP skill < -1.0 (catastrophic); GRU brings to ~0 or positive |
| **Horizon h=14** | Both negative, but GRU is 2.6x less bad (-0.175 vs -0.450) |

## Interpretation

The GRU's improvement is not a generic capacity increase — it is a **structural fix for the MLP's inability to maintain temporal coherence across the forecast horizon**. The evidence:

1. **The improvement concentrates at the exact horizons where the MLP catastrophically failed** (h=1,3,5), not uniformly across all horizons.
2. **Dense targets benefit most** because they have the highest variance and the most temporal structure to exploit. The GRU's sequential conditioning lets it track intra-horizon trends that the pointwise MLP discards.
3. **Ensembling is nearly useless** because S2 is not just complementary — it strictly dominates on the failure modes that made S1 bad.

## Recommendations

1. **Adopt S2 (GRU h=32) as the default Stage 2 head.** There is no operational scenario where S1 is meaningfully preferred.

2. **Do not ensemble.** Pure S2 is within 0.3% of the optimal blend, and S1 adds no complementary signal.

3. **Next experiment targets:**
   - **h=13-14 degradation**: The remaining weakness. Options: (a) reduce effective horizon to 12, (b) horizon-weighted loss emphasizing late steps, (c) bidirectional or attention-based decoder.
   - **Dense-target-specific tuning**: H02AB02 and N02AX02 still have MSE >200. These two targets alone dominate overall MSE. Target-specific heads or per-target GRU hidden dims may help.
   - **Combine GRU with lag features**: Experiment 009's S4 (y_large lag) showed some complementary information. Testing GRU + lag features could compound the improvements.
