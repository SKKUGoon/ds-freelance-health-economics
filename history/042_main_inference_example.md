## Objective

Add a top-level `main.py` that demonstrates the production-style, single-step SOTA inference workflow for the final ensemble model using pre-trained saved weights.

## Scope

- Create an illustrative CLI entrypoint for loading saved `S11` and `S04` model artifacts.
- Show how `patient.parquet` and `drugs.parquet` would be loaded as feature and target history inputs.
- Run a single 14-day forecast using only the most recent context window.
- Blend member forecasts with the repo's final ensemble weight.
- Keep the example intentionally schema-agnostic and clearly documented as illustrative.

## Files

- `main.py`

## Implementation Steps

1. Add a small CLI using `argparse` for parquet paths, model paths, and output path.
2. Define helper functions for loading the two saved forecasters.
3. Define helper functions that read parquet inputs and convert them to numeric matrices.
4. Extract the last `dyn_p` rows of `X` and `y` as production inference context.
5. Predict with `S11` and `S04`, then blend with `0.55 / 0.45` weights.
6. Return predictions as a dataframe and optionally write them to parquet.
7. Include concise comments explaining that this is a production inference example, not rolling backtest logic.

## Validation Criteria

- `main.py` is syntactically valid Python.
- The script imports the public `LAVARForecaster` API from the package.
- The workflow reflects single-step inference only, with no rolling evaluation or retraining logic.
- The ensemble blend uses the final SOTA recipe: `0.55 * S11 + 0.45 * S04`.
