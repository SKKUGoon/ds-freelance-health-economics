# 007 Experiment Report (`ys` only)

## Experiment goal
Compare Stage 2 modes in `notebooks/007_hecon.ipynb` on the small-target set (`ys`):

- `ts_latent` (`use_supply_history=False`)
- `ts_supply_ts_latent` (`use_supply_history=True`)

## Correct source artifacts

### Evaluator A/B artifacts
- `notebooks/007_eval_mode_compare_summary.csv`
- `notebooks/007_eval_mode_compare_folds.csv`

### Manual rolling artifact (single mode)
- `notebooks/007_manual_ts_supply_ts_latent_summary.csv`
- `notebooks/007_manual_ts_supply_ts_latent_folds.csv`

## Evaluator A/B result (official mode comparison)

| mode | n_folds | mean_mse | median_mse | mean_mae | median_mae | mean_skill | median_skill |
|---|---:|---:|---:|---:|---:|---:|---:|
| `ts_latent` | 130 | 93.7581 | 51.5915 | 2.5836 | 2.3862 | -0.4043 | 0.0275 |
| `ts_supply_ts_latent` | 130 | 487.8708 | 56.2693 | 3.8473 | 2.5007 | -6.2047 | 0.0021 |

## Manual rolling result (`ts_supply_ts_latent` only)

| mode | run_type | n_folds | mean_mse | median_mse | mean_skill | median_skill |
|---|---|---:|---:|---:|---:|---:|
| `ts_supply_ts_latent` | manual | 130 | 389.0352 | 56.2872 | -4.8604 | 0.0053 |

## Cross-artifact comparison

- The manual `ts_supply_ts_latent` run improves over evaluator `ts_supply_ts_latent` on mean metrics:
  - mean MSE: `487.87 -> 389.04`
  - mean skill: `-6.2047 -> -4.8604`
- Median metrics for `ts_supply_ts_latent` are nearly unchanged between evaluator and manual runs.
- Relative to evaluator `ts_latent`, the manual `ts_supply_ts_latent` run is still worse on overall metrics:
  - mean MSE remains higher (`389.04` vs `93.76`)
  - median MSE remains higher (`56.29` vs `51.59`)
  - mean and median skill remain lower.

## Decision for current `ys` setup
Use `ts_latent` as the default baseline for now.

`ts_supply_ts_latent` remains a valid experimental branch and showed improvement in manual rerun vs its own evaluator run, but it has not yet surpassed `ts_latent` on the current official A/B comparison artifacts.

## Hypotheses for Improvement (Experimental)

The current CSV-backed decision above remains unchanged. The points below are candidate explanations and interventions to test in future runs.

### Observed pattern
- Center metrics for `ts_supply_ts_latent` can be close to baseline, but a subset of folds drives large errors.
- Manual reruns improved aggregate means versus evaluator `ts_supply_ts_latent`, suggesting tuning/process sensitivity.

### Candidate mechanism (to validate)
- In the delta integration path, nonnegative projection can reduce useful gradient signal near the zero boundary in some regimes.
- This may increase sensitivity to fold-level distribution shifts when supply-history inputs are active.

### Candidate interventions
1. Replace hard nonnegative projection with a smooth approximation (e.g., softplus with annealed sharpness).
2. Add supply-history input-channel dropout.
3. Use a bottleneck encoder for supply-history branch before fusion.
4. Re-run with fixed seed/fold boundaries and compare tail-aware metrics (`p95`, max fold MSE, catastrophic-fold count).

### Promotion gate for `ts_supply_ts_latent`

Promote only if all conditions hold against baseline (`ts_latent`):
1. Median MSE is at least comparable.
2. Catastrophic-fold frequency is not worse.
3. Tail skill-collapse frequency is not worse.

## Process note
From this point, use these naming rules to avoid result mixing:

- Evaluator comparison outputs: `007_eval_*`
- Manual rolling outputs: `007_manual_*`

and cite only one run family at a time in conclusions.
