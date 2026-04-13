from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from lavar import LAVARForecaster


S11_WEIGHT = 0.55
S04_WEIGHT = 0.45


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Illustrative production inference entrypoint for the final "
            "S11/S04 LAVAR ensemble."
        )
    )
    parser.add_argument(
        "--patient-parquet",
        default="patient.parquet",
        help="Feature-history parquet used as X input.",
    )
    parser.add_argument(
        "--drugs-parquet",
        default="drugs.parquet",
        help="Target-history parquet used as y input.",
    )
    parser.add_argument(
        "--s11-weights",
        default="s11_model.pth",
        help="Saved S11_LATENT16 model artifact.",
    )
    parser.add_argument(
        "--s04-weights",
        default="s04_model.pth",
        help="Saved S04_GRU_H32_L1 model artifact.",
    )
    parser.add_argument(
        "--output",
        default="ensemble_predictions.parquet",
        help="Optional parquet path for the blended 14-day forecast.",
    )
    return parser.parse_args()


def load_forecasters(
    s11_path: str, s04_path: str
) -> tuple[LAVARForecaster, LAVARForecaster]:
    s11 = LAVARForecaster.load(s11_path)
    s04 = LAVARForecaster.load(s04_path)
    return s11, s04


def load_matrix(parquet_path: str) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    if df.empty:
        raise ValueError(f"Parquet file is empty: {parquet_path}")

    numeric_df = df.select_dtypes(include=[np.number]).copy()
    if numeric_df.empty:
        raise ValueError(
            f"No numeric columns found in parquet file: {parquet_path}. "
            "This example assumes model inputs are numeric matrices."
        )
    return numeric_df


def build_recent_context(
    x_df: pd.DataFrame,
    y_df: pd.DataFrame,
    dyn_p: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(x_df) < dyn_p or len(y_df) < dyn_p:
        raise ValueError(
            f"Need at least {dyn_p} rows in both inputs. "
            f"Got len(X)={len(x_df)}, len(y)={len(y_df)}."
        )

    x_recent = x_df.tail(dyn_p).to_numpy(dtype=np.float32)
    y_recent = y_df.tail(dyn_p).to_numpy(dtype=np.float32)
    return x_recent, y_recent


def predict_ensemble(
    x_recent: np.ndarray,
    y_recent: np.ndarray,
    s11: LAVARForecaster,
    s04: LAVARForecaster,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    s11_pred = s11.predict(x_recent, y_recent=y_recent).astype(np.float32)
    s04_pred = s04.predict(x_recent, y_recent=y_recent).astype(np.float32)
    blend_pred = (S11_WEIGHT * s11_pred) + (S04_WEIGHT * s04_pred)
    return s11_pred, s04_pred, blend_pred


def make_prediction_frame(
    blend_pred: np.ndarray,
    s11_pred: np.ndarray,
    s04_pred: np.ndarray,
) -> pd.DataFrame:
    horizon, target_dim = blend_pred.shape
    rows: list[dict[str, float | int]] = []

    for h in range(horizon):
        row: dict[str, float | int] = {"horizon_day": h + 1}
        for j in range(target_dim):
            row[f"target_{j}_ensemble"] = float(blend_pred[h, j])
            row[f"target_{j}_s11"] = float(s11_pred[h, j])
            row[f"target_{j}_s04"] = float(s04_pred[h, j])
        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()

    # This is a single production-style inference step.
    # It intentionally avoids any rolling/backtest or retraining workflow.
    s11, s04 = load_forecasters(args.s11_weights, args.s04_weights)

    x_df = load_matrix(args.patient_parquet)
    y_df = load_matrix(args.drugs_parquet)

    dyn_p = int(s11.cfg.dyn_p)
    if int(s04.cfg.dyn_p) != dyn_p:
        raise ValueError(
            f"Model context mismatch: S11 dyn_p={s11.cfg.dyn_p}, S04 dyn_p={s04.cfg.dyn_p}"
        )

    x_recent, y_recent = build_recent_context(x_df, y_df, dyn_p=dyn_p)
    s11_pred, s04_pred, blend_pred = predict_ensemble(x_recent, y_recent, s11, s04)
    prediction_df = make_prediction_frame(blend_pred, s11_pred, s04_pred)

    output_path = Path(args.output)
    prediction_df.to_parquet(output_path, index=False)

    print("Saved next-14-day ensemble forecast")
    print(f"  patient parquet: {args.patient_parquet}")
    print(f"  drugs parquet:   {args.drugs_parquet}")
    print(f"  s11 weights:     {args.s11_weights}")
    print(f"  s04 weights:     {args.s04_weights}")
    print(f"  output:          {output_path}")
    print(f"  blend:           {S11_WEIGHT:.2f} * S11 + {S04_WEIGHT:.2f} * S04")


if __name__ == "__main__":
    main()
