from __future__ import annotations

import numpy as np
import pandas as pd


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true_flat = np.asarray(y_true, dtype="float64").reshape(-1)
    y_pred_flat = np.asarray(y_pred, dtype="float64").reshape(-1)
    mask = np.isfinite(y_true_flat) & np.isfinite(y_pred_flat)
    if not mask.any():
        return {"MAE": np.nan, "RMSE": np.nan, "NMAE": np.nan, "CVRMSE": np.nan, "n": 0}

    error = y_pred_flat[mask] - y_true_flat[mask]
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(np.square(error))))
    mean_load = float(np.mean(y_true_flat[mask]))
    denom = mean_load if abs(mean_load) > 1e-12 else np.nan
    return {
        "MAE": mae,
        "RMSE": rmse,
        "NMAE": float(mae / denom) if np.isfinite(denom) else np.nan,
        "CVRMSE": float(rmse / denom) if np.isfinite(denom) else np.nan,
        "n": int(mask.sum()),
    }


def horizon_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    rows = []
    for horizon_idx in range(y_true.shape[1]):
        row = regression_metrics(y_true[:, horizon_idx], y_pred[:, horizon_idx])
        row["horizon"] = horizon_idx + 1
        rows.append(row)
    return pd.DataFrame(rows)
