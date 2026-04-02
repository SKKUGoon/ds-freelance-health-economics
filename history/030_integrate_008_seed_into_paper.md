## Objective

Integrate seed-stability findings from `notebooks/report/008_seed/hana/` into `notebooks/PAPER.md`, including placement in the narrative flow, updated conclusions, and figure references.

## Scope

- Update only `notebooks/PAPER.md`.
- Add a new Stage E section for the seed gate.
- Renumber downstream sections.
- Remove outdated "pending seed" language.
- Add 008-seed figures and artifacts.

## File List

- `notebooks/PAPER.md`

## Implementation Steps

1. Insert `Stage E - Seed Stability Gate (008_seed)` after ensemble section.
2. Add summary table and interpretation grounded in `seed_stability_summary.csv` and `seed_per_run_metrics.csv`.
3. Renumber subsequent sections.
4. Update limitations and conclusion to reflect seed gate completion.
5. Add 008-seed figures and artifact links.

## Validation Criteria

- `PAPER.md` contains explicit 008 seed findings.
- No remaining "pending seed gate" statements.
- Figure list includes seed plots from `report/008_seed/hana/`.
- Section numbering is consistent.
