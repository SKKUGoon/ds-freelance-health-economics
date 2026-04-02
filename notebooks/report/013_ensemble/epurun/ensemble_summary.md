# 013 Ensemble Report (Epurun)

## Setup
- Seeds: [42, 123, 456, 789, 1024]
- Ensemble: y = 0.55 * S11 + 0.45 * S04
- Prediction bounds: clip to [0, inf)
- CI: empirical 95% (q2.5, q97.5) across seeded blend predictions

## Key Result (vs pure S11)
- Mean-MSE delta: -0.0421 (-4.38%)
- Blend mean-of-p95-fold-MSE: 1.3070
- S11 mean-of-p95-fold-MSE: 1.3365

## Artifacts
- ensemble_seed_predictions_per_drug.csv
- ensemble_ci_per_drug.csv
- ensemble_metrics_by_seed.csv
- ensemble_metrics_overall.csv
- ensemble_true_vs_pred_ci_per_drug.png