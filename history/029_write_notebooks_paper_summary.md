## Objective

Create `notebooks/PAPER.md` documenting the end-to-end Hana experiment narrative across `005_model`, `006_graph`, and `007_ensemble` artifacts, including model narrowing rationale and figure references.

## Scope

- Add a new markdown report at `notebooks/PAPER.md`.
- Summarize progression from 85-run sweep to top-4, then top-2, then blended ensemble.
- Include key metrics grounded in existing CSV artifacts.
- Include links to generated figures under `notebooks/report/006_graph/hana/` and `notebooks/report/007_ensemble/hana/`.

## File List

- `notebooks/PAPER.md`

## Implementation Steps

1. Draft a structured report with objective, protocol, findings, model-selection flow, ensemble outcomes, and conclusion.
2. Use values from current artifacts (`leaderboard.csv`, `worst_folds_by_mse.csv`, `top4_fold_metrics_reconstructed.csv`, `blend_summary.csv`, `blend_eval_temporal_split.csv`).
3. Add markdown image links for the relevant plots.
4. Save file at `notebooks/PAPER.md`.

## Validation Criteria

- `notebooks/PAPER.md` exists and is readable markdown.
- Report explicitly explains 85 -> 4 -> 2 -> blend progression.
- Report includes at least one figure link to existing image files.
