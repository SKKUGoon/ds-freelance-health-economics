# 005 Model Report (Hana)

## Run Status
- Completed runs: **85 / 85** (`run_status.json`)
- Device: **mps**
- Protocol: `horizon=14`, `fold_step=14` (no overlap), `lavar_retrain_cadence=90`, `quality_triggers=true`
- Folds per run: **130**

## Top Results

### Best by median MSE (leaderboard order)
1. `S16_REG_LATENT16 + E02_CAP_200_200`  
   - mean_mse: `6.1618`, median_mse: `3.7671`, median_skill: `0.1523`
2. `S16_REG_LATENT16 + E05_CAP_200_300`  
   - mean_mse: `5.7449`, median_mse: `3.8579`, median_skill: `0.1919`
3. `S04_GRU_H32_L1 + E02_CAP_200_200`  
   - mean_mse: `5.4762`, median_mse: `3.8869`, median_skill: `0.1689`

### Best by mean MSE
1. `S11_LATENT16 + E05_CAP_200_300` -> **5.2780**
2. `S04_GRU_H32_L1 + E01_CAP_100_100` -> `5.4678`
3. `S04_GRU_H32_L1 + E02_CAP_200_200` -> `5.4762`
4. `S11_LATENT16 + E02_CAP_200_200` -> `5.4766`

## Key Findings

### 1) Legacy baseline family is unstable
- `S01`, `S02`, `S03` are the weakest group by average metrics and tail behavior.
- Catastrophic spikes are concentrated in these scenarios:
  - folds with `mse >= 200`: **18** total
  - distribution: `S03: 7`, `S02: 6`, `S01: 5`
- Worst events cluster around `t_end=1080` and `t_end=1528`.

### 2) Strongest family on unified Hana
- Best practical cluster is `S04/S08/S11/S14/S16`.
- `S11_LATENT16` gives best mean-mse profile.
- `S16_REG_LATENT16` gives best median-mse profile.

### 3) `dyn_p14` variants underperform here
- `S10/S12/S13` consistently sit around `mean_mse ~6.28-6.39` with mostly negative mean skill.
- They are below the top `S04/S11/S16` set.

### 4) Longer epoch caps are not universally better
- `300/300` often degrades results for several scenarios.
- Better patterns are scenario-dependent:
  - `S11` best at `E05 (200/300)`
  - `S16` best median at `E02 (200/200)`, strongest overall tradeoff at `E05`

## Per-Scenario Best Epoch Profile (by mean MSE)
- `S01` -> `E02`
- `S02` -> `E01`
- `S03` -> `E01`
- `S04` -> `E01`
- `S05` -> `E01`
- `S06` -> `E01`
- `S07` -> `E05`
- `S08` -> `E04`
- `S09` -> `E01`
- `S10` -> `E03`
- `S11` -> `E05`
- `S12` -> `E04`
- `S13` -> `E04`
- `S14` -> `E03`
- `S15` -> `E03`
- `S16` -> `E05`
- `S17` -> `E01`

## Fold-Level Risk Summary
- Number of high-error folds across all runs:
  - `mse >= 50`: **48**
  - `mse >= 100`: **24**
  - `mse >= 200`: **18**
  - `mse >= 500`: **12**
- Highest-severity folds are dominated by `S01/S02/S03` and align with extreme negative skill values.

## Stage1 Reuse Note
- `stage1_reuse_rate` is non-zero for many run groups, typically around `0.08-0.22` depending on scenario/epoch profile.
- This indicates checkpoint reuse worked, but signature fragmentation still limits reuse in some groups.

## Recommended Next Step Candidates
Primary candidates for follow-up (seed sweep + robustness checks):
1. `S11_LATENT16 + E05_CAP_200_300`
2. `S04_GRU_H32_L1 + E02_CAP_200_200`
3. `S16_REG_LATENT16 + E05_CAP_200_300`
4. `S16_REG_LATENT16 + E02_CAP_200_200`

De-prioritize for production shortlist:
- `S01/S02/S03` (catastrophic instability)
- `S10/S12/S13` (consistently weaker on this dataset)

## Referenced Artifacts
- `leaderboard.csv`
- `progress_summary.csv`
- `progress_fold_metrics.csv`
- `scenario_epoch_group_summary.csv`
- `worst_folds_by_mse.csv`
- `worst_folds_by_skill.csv`
- `run_status.json`
