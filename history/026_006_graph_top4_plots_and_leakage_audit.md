## Objective

Implement `notebooks/006_graph.ipynb` to visualize top-4 Hana model behavior from completed 005 runs, including true-vs-prediction trajectories, fold-wise performance comparisons, and explicit leakage diagnostics.

## Scope

- Create a complete notebook at `notebooks/006_graph.ipynb`.
- Read existing experiment artifacts from `notebooks/report/005_model/hana/` and checkpoints from `notebooks/checkpoints/hana/`.
- Reconstruct fold predictions for selected top runs using Stage2 checkpoints plus Stage1 signatures.
- Generate comparison plots and concise narrative outputs.
- Add validation cells for temporal split integrity and leakage checks.

## File List

- `notebooks/006_graph.ipynb` (create/populate)

## Implementation Steps

1. Add notebook setup/config cells for paths, plotting style, and selected top-4 run IDs.
2. Load leaderboard and fold metrics, verify run availability, and summarize baseline metrics.
3. Implement checkpoint utilities:
   - discover Stage2 fold checkpoints for each run,
   - map each evaluation fold to latest available refit checkpoint,
   - load Stage1 state via `stage1_signature`,
   - reconstruct forecaster internals (scalers, lavar backbone, bucket heads, indices, guardrails).
4. Recompute fold predictions for the top-4 runs on Hana `X/y` with strict historical windows.
5. Build long-form prediction dataframe and aggregate series for plotting.
6. Create visualizations:
   - true vs predicted trajectories,
   - fold-wise MSE/MAE/skill time series,
   - run-level distribution comparison.
7. Add leakage audit cells with assertions:
   - fold step equals horizon (14) for all runs,
   - no overlap in forecast windows,
   - train context ends before forecast start,
   - prediction context uses only pre-`t_end` slices.
8. Add compact interpretation/next-step markdown cells.

## Validation Criteria

- Notebook loads without missing-path errors when 005 artifacts exist.
- Top-4 run reconstruction executes and returns non-empty prediction rows.
- Fold-level metrics trend plots render for all selected runs.
- Leakage audit assertions pass and print explicit PASS summaries.
- Notebook remains data-source aligned to unified Hana `X/y` only.
