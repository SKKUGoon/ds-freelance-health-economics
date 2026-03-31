# 019 — Experiment 012: GRU Decoder Diagnostics (S1 MLP vs S2 GRU)

## Objective

Diagnose **where** and **how** the S2 GRU decoder (from experiment 011) improves over S1 MLP baseline:
- Per-fold catastrophic overlap and win rate
- Per-target density-bucket breakdown (dense/sparse/ultra)
- S1+S2 ensemble blend grid
- Horizon decomposition h=1..14 (does GRU fix h=1 and h=14 weaknesses from 010?)

## Scope

- Create `notebooks/012_hecon.ipynb` diagnostic notebook
- Output 7 CSV/NPZ files to `notebooks/report/012/`

## Files

| File | Action |
|------|--------|
| `history/019_012_gru_diagnostics.md` | Create (this file) |
| `notebooks/012_hecon.ipynb` | Create — 15-cell diagnostic notebook |

## Implementation Steps

- [x] Create history record
- [x] Cell 0-2: Header, imports, load 011 eval data
- [x] Cell 3-4: Per-fold diagnostics (catastrophic overlap, win count, oracle)
- [x] Cell 5: Holiday proximity analysis
- [x] Cell 6-7: Build/load S1 + S2 rolling prediction cache (`012_s1_s2_manual_cache.npz`)
- [x] Cell 8-9: Per-target density-bucket breakdown
- [x] Cell 10-11: Ensemble alpha grid (blend S1+S2)
- [x] Cell 12-13: Horizon decomposition h=1..14
- [x] Cell 14: Conclusion markdown

## Validation

- Run all cells in `012_hecon.ipynb`
- Confirm 7 output files in `report/012/`
- Check horizon decomposition shows S2 skill at h=1 and h=14
- Check bucket summary for dense target improvements (H02AB02, N02AX02)
