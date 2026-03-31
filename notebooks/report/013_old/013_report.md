# Experiment 013 — GRU Decoder + Drug Lag Features

## Objective

Test whether feeding lagged drug history into the encoder improves GRU decoder performance. Motivation from experiment 012: drugs with zero recent history still receive non-zero predictions because the encoder sees no per-target activity signal. Lag features provide this signal through the encoder → latent → GRU path.

## Scenarios

| Scenario | Head | Input | Input dims | Description |
|----------|------|-------|------------|-------------|
| S1_gru_baseline | GRU h=32 | X_base | 35 | Control (= 011 S2) |
| S2_gru_ysmall_lag | GRU h=32 | X_base + y_small lags(1..14) | 707 | Drug-level lag history |
| S3_gru_ylarge_lag | GRU h=32 | X_base + y_large lags(1..14) | 539 | Category-level lag history |

All scenarios: 130 folds, fold_step=14, retrain_cadence=90, quality_triggers=True, stage2_head_type="gru", gru_hidden_dim=32, gru_num_layers=1.

## Results — RollingEvaluator

| Scenario | mean_mse | med_mse | mean_mae | med_mae | mean_skill | **med_skill** |
|----------|----------|---------|----------|---------|------------|---------------|
| **S1 GRU baseline** | **60.3** | **49.3** | **2.43** | **2.33** | **+0.080** | **+0.091** |
| S2 GRU + y_small lag | 65.4 | 57.1 | 2.57 | 2.55 | -0.014 | -0.001 |
| S3 GRU + y_large lag | 70.8 | 56.9 | 2.59 | 2.51 | -0.136 | +0.002 |

## Results — Manual Rolling

| Scenario | mean_mse | med_mse | mean_skill | med_skill |
|----------|----------|---------|------------|-----------|
| **S1 GRU baseline** | **61.1** | **48.8** | **+0.067** | **+0.082** |
| S2 GRU + y_small lag | 67.1 | 56.8 | -0.069 | -0.002 |
| S3 GRU + y_large lag | 67.6 | 55.8 | -0.055 | +0.004 |

## Fold-Level Analysis

### Stability

| Scenario | Catastrophic (>500) | Positive-skill folds | MSE std | Max MSE |
|----------|---------------------|---------------------|---------|---------|
| **S1 baseline** | 1 | **102/130 (78%)** | 55.8 | 541 |
| S2 y_small lag | 1 | 56/130 (43%) | 56.2 | 542 |
| S3 y_large lag | **2** | 69/130 (53%) | **86.6** | **826** |

S3 adds a second catastrophic fold and increases MSE variance by 55%. S2 maintains the same tail behavior as S1 but drops positive-skill folds from 78% to 43%.

### Head-to-Head vs S1

| Challenger | Wins vs S1 | Mean MSE delta | Median MSE delta |
|------------|-----------|----------------|------------------|
| S2 y_small lag | 30/130 (23%) | +5.1 | +4.2 |
| S3 y_large lag | 35/130 (27%) | +10.6 | +3.6 |

Both lag variants lose to S1 on over 3/4 of folds. S3 is worse — its losses are larger (mean delta +10.6) and it wins even less consistently.

### Skill Distribution

| Scenario | p25 | p50 | p75 |
|----------|-----|-----|-----|
| **S1 baseline** | **+0.011** | **+0.091** | **+0.180** |
| S2 y_small lag | -0.013 | -0.001 | +0.006 |
| S3 y_large lag | -0.008 | +0.002 | +0.020 |

S1's entire interquartile range is positive. S2 and S3 collapse to near-zero skill across the distribution — the lag features don't just fail to help, they actively compress the model's ability to differentiate from naive.

## Diagnosis: Why Lag Features Hurt

**1. Encoder bottleneck.** The encoder architecture is `[32, 16]` hidden layers, mapping input to 8-dim latent. S1 feeds 35 features — a 35→32→16→8 compression that preserves signal. S2 feeds 707 features — a 707→32→16→8 compression (88:1 ratio) that destroys it. The encoder simply cannot represent 707 dimensions of drug-specific history through a 32-unit bottleneck. The latent quality degrades, and the GRU decoder receives worse inputs.

**2. Signal dilution.** The 48×14 = 672 lag features are mostly zero (ultra-sparse drugs contribute 14 lag columns of zeros each). The encoder must learn to ignore the majority of its input, which is harder than learning from a compact, dense input.

**3. VAR dynamics corruption.** Stage 1's VAR operates in latent space: `z_t = A·z_{t-1:p}`. When the encoder is fed noisy high-dimensional input, the learned latent dynamics become less stable. S3's increased MSE variance (std=86.6 vs 55.8) and new catastrophic fold suggest the VAR rollout is more fragile with corrupted latents.

**4. Not a GRU-specific issue — same failure as 009.** Experiment 009 tested lag features with MLP heads and found similarly weak results (S2 y_small_lag median_skill=+0.016, S4 y_large_lag=+0.039). The GRU cannot rescue what the encoder fails to encode.

## True-vs-Pred Plots

Visual comparison across all three scenarios shows nearly identical prediction patterns. The phantom non-zero predictions for inactive drugs are **not visibly reduced** in S2 or S3 — the lag features do not successfully gate zero-history targets. This confirms the encoder bottleneck: the drug-specific activity signal is lost before it reaches the decoder.

## Key Findings

1. **Lag features hurt GRU performance.** Both y_small and y_large lag variants produce worse results than the GRU baseline. This is consistent across both evaluation methods (RollingEvaluator and manual rolling).

2. **The encoder is the bottleneck, not the decoder.** The GRU can exploit temporal structure in the latent (proven in 011), but it cannot compensate for a degraded latent caused by encoder overload.

3. **y_large lags are worse than y_small lags.** Despite having fewer dimensions (539 vs 707), y_large adds a catastrophic fold and higher variance. Category-level aggregation doesn't help — it loses drug-specific signal without reducing dimensionality enough.

4. **Phantom predictions remain.** The original motivation — suppressing predictions for zero-history drugs — is not achieved through this approach.

## Recommendation

**Do not use lag features with the current encoder architecture.** The path to incorporating drug history requires either:

- **(A) Wider encoder**: Scale `encoder_hidden` to handle high-dim input (e.g., `[256, 64]` or `[512, 128]`). Risk: overfitting with T=2556 training samples.
- **(B) Separate encoding paths**: Encode X_base and drug history through separate encoders, then fuse in latent space. Avoids bottleneck without scaling the main encoder.
- **(C) Supply-history-augmented mode**: Use the existing `stage2_mode="supply_history_latent"` which concatenates drug history at the Stage 2 level (after encoding), bypassing the encoder bottleneck entirely.
- **(D) Post-hoc zero gating**: Apply a simple rule-based gate at prediction time: if a drug's last-k-day sum is zero, clamp prediction to zero. No model changes needed.

Option D is the simplest path to fix the phantom prediction problem. Options B and C address the deeper question of whether drug history can improve forecasting.
