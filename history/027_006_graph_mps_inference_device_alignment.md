## Objective

Fix `notebooks/006_graph.ipynb` checkpoint reconstruction so inference runs on MPS and outputs are returned to CPU/NumPy for metricing and plotting.

## Scope

- Update only `notebooks/006_graph.ipynb` reconstruction helper.
- No training behavior changes.
- Ensure all loaded modules/tensors are on the same MPS device at inference time.

## File List

- `notebooks/006_graph.ipynb`

## Implementation Steps

1. In `_rebuild_forecaster_from_payload`, force reconstruction device to MPS (with availability check).
2. Move `forecaster._lavar` and all available stage2 heads to MPS and set eval mode.
3. Move guardrail tensor (`_upper_bounds`) to MPS.
4. Keep prediction outputs as CPU NumPy (existing `predict()` behavior) for downstream Pandas/plot usage.

## Validation Criteria

- No `Input and parameter tensors are not at the same device` error during `forecaster.predict`.
- Top-4 fold reconstruction cell executes with MPS inference.
- Output CSV/plots are generated as before.
