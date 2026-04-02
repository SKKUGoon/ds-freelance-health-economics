## Objective

Refresh `lavar/LAVAR_README.md` so it matches the current `lavar` package implementation, and add Korean and Japanese README variants.

## Scope

- Rewrite the English README around the current package API and architecture.
- Document the implemented Stage 1 dynamics options (`var`, `gru`).
- Document the implemented Stage 2 modes and supported heads (`delta_mse` via MLP or GRU, `nb`, `zinb`).
- Add Korean and Japanese README files with the same structure and up-to-date content.
- Add language navigation links across the README files.

## File List

- Update: `lavar/LAVAR_README.md`
- Add: `lavar/LAVAR_README_KOR.md`
- Add: `lavar/LAVAR_README_JPN.md`

## Implementation Steps

1. Review the current `lavar` package modules to confirm the public API, architecture options, stage modes, and head implementations.
2. Rewrite the English README to reflect the current package layout, training pipeline, configuration surface, and checkpoint outputs.
3. Add Korean and Japanese README variants with the same documentation structure and accurate terminology.
4. Verify that all links, file names, and architecture descriptions match the current codebase.

## Validation Criteria

- The English README matches the current `lavar` implementation and public API.
- The documentation explicitly covers VAR and GRU latent dynamics, baseline and supply-history Stage 2 modes, and MLP/GRU/NB/ZINB head support.
- The Korean and Japanese README files exist and remain structurally aligned with the English README.
- All intra-README links point to existing files.
