# 021 — Experiment 014: Horizon-Weighted Loss + Stage 1 Dynamics + Zero Gating

## Objective

Three orthogonal improvements to the GRU baseline (011 S2, mean skill +0.086):

1. **Horizon-weighted loss** — upweight later horizons in Stage 2 delta-MSE loss to fix h=13-14 weakness (skill -0.08 to -0.17)
2. **Stage 1 dynamics** — increase `dyn_p` and/or `latent_dim` to improve VAR rollout quality (the ceiling for all downstream improvements)
3. **Post-hoc zero gating** — clamp predictions to zero for drugs with zero recent history (fixes phantom predictions diagnosed in 012)

## Scenario Matrix

| Scenario | dyn_p | latent_dim | encoder_hidden | horizon_loss | zero_gate | Description |
|----------|-------|------------|----------------|--------------|-----------|-------------|
| S1 | 7 | 8 | [32,16] | uniform | no | GRU baseline (control) |
| S2 | 7 | 8 | [32,16] | linear | no | Horizon-weighted loss only |
| S3 | 14 | 8 | [32,16] | uniform | no | Longer VAR history |
| S4 | 7 | 16 | [64,32] | uniform | no | Wider latent + encoder |
| S5 | 14 | 16 | [64,32] | uniform | no | Combined dyn_p + latent |
| S6 | 14 | 16 | [64,32] | linear | no | S5 + horizon-weighted loss |

Zero gating is applied **post-hoc** to all scenarios' predictions (not a separate training run). Reported as a separate results table.

All scenarios: stage2_head_type="gru", gru_hidden_dim=32, gru_num_layers=1, X_base (35 dims), 130 folds, fold_step=14, retrain_cadence=90.

Note on S4/S5/S6 encoder: latent_dim=16 with encoder_hidden=[32,16] would mean the last hidden layer (16) maps to 16 outputs — it works but is very tight. Widening to [64,32] gives more capacity for the larger latent space.

## Code Changes Required

### Phase 1: Config field

**File:** `lavar/config.py`

Add field:
```python
horizon_loss_weight: Literal["uniform", "linear"] = "uniform"
```

- `"uniform"`: all horizons weighted equally (current behavior)
- `"linear"`: weights = [1, 2, 3, ..., H], normalized to mean=1.0

### Phase 2: Training loop — apply horizon weights to delta-MSE loss

**File:** `lavar/_training/stage2/stage2_test_baseline.py`

1. Before the epoch loop, compute `horizon_weights` tensor based on `cfg.horizon_loss_weight`:
   - `"uniform"`: `torch.ones(H)`
   - `"linear"`: `torch.arange(1, H+1, dtype=float)`, then normalize so mean=1.0
   - Reshape to `(1, H, 1)` for broadcasting against `(B, H, Dy_sel)`

2. Replace both train and val delta-MSE loss lines:
   ```python
   # Before:
   loss = torch.mean((delta_hat - delta_true) ** 2)
   # After:
   loss = torch.mean(horizon_weights * (delta_hat - delta_true) ** 2)
   ```

3. Only apply to `head_type == "delta_mse"` branches. NB/ZINB heads unchanged.

### Phase 3: Notebook 014_hecon.ipynb

Structure (following 011 pattern):

| Cell | Type | Content |
|------|------|---------|
| 0 | md | Title, hypothesis, scenario table |
| 1 | code | Imports, setup, report dir |
| 2 | code | Load data (X_base, y_small only — no lag features) |
| 3 | code | Define 6 scenarios with configs |
| 4 | code | RollingEvaluator loop for all 6 |
| 5 | code | Summary comparison table |
| 6 | code | Fold-level plots (MSE + skill) |
| 7 | code | Save eval CSVs |
| 8 | md | Manual rolling section header |
| 9 | code | `run_manual_rolling_collect()` function |
| 10 | code | Manual rolling loop + per-scenario CSV saves |
| 11 | code | Manual summary comparison table |
| 12 | code | `plot_scenario_true_vs_pred()` function |
| 13 | code | Generate True-vs-Pred plots |
| 14 | md | Post-hoc zero gating section header |
| 15 | code | Apply zero gating to all scenarios' manual rolling predictions; compute gated metrics; comparison table |
| 16 | md | Conclusion |

### Zero gating logic (cell 15)

```python
def apply_zero_gate(pred_values, true_values, y_tensor, pred_times, k=7):
    """Clamp prediction to 0 for targets where last-k-day sum is 0."""
    gated = []
    for fold_idx, (pred, times) in enumerate(zip(pred_values, pred_times)):
        t_end = int(times[0])
        y_recent = y_tensor[max(0, t_end-k):t_end].numpy()  # (k, Dy)
        recent_sum = y_recent.sum(axis=0)  # (Dy,)
        mask = (recent_sum > 0).astype(float)  # 1 if active, 0 if inactive
        gated.append(pred * mask[None, :])  # broadcast (H, Dy)
    return gated
```

Then recompute per-fold MSE/skill for each scenario with gating applied.

## Validation

- [ ] Smoke test: `LAVARConfig(horizon_loss_weight="linear")` constructs without error
- [ ] Smoke test: fit + predict works with `horizon_loss_weight="linear"`
- [ ] Smoke test: fit + predict works with `dyn_p=14, latent_dim=16, encoder_hidden=[64,32], decoder_hidden=[32,64]`
- [ ] Run all 6 scenarios in 014_hecon.ipynb
- [ ] Compare S2 vs S1 to isolate horizon weighting effect
- [ ] Compare S3 vs S1 to isolate dyn_p effect
- [ ] Compare S4 vs S1 to isolate latent_dim effect
- [ ] Apply zero gating post-hoc, check if phantom predictions reduced
- [ ] All artifacts saved to `report/014/`

## Expected Outputs

- `014_eval_mode_compare_summary.csv`
- `014_eval_mode_compare_folds.csv`
- `014_manual_{scenario}_summary.csv` (× 6)
- `014_manual_{scenario}_folds.csv` (× 6)
- `014_{scenario}_true_vs_pred.png` (× 6)
- `014_zero_gating_comparison.csv`
