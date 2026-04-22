from __future__ import annotations

import numpy as np
import pandas as pd

from building_aware.features import make_origins


def split_origin_positions(
    index: pd.DatetimeIndex,
    *,
    target_start: str,
    target_end: str,
    lookback_hours: int,
    forecast_horizon: int,
) -> np.ndarray:
    origins = make_origins(
        index,
        target_start=target_start,
        target_end=target_end,
        lookback_hours=lookback_hours,
        forecast_horizon=forecast_horizon,
    )
    positions = index.get_indexer(origins)
    if (positions < 0).any():
        raise ValueError("Some origin timestamps are missing from the time index.")
    return positions.astype("int64")


def inclusive_time_positions(index: pd.DatetimeIndex, *, start: str, end: str) -> np.ndarray:
    mask = (index >= pd.Timestamp(start)) & (index <= pd.Timestamp(end))
    return np.flatnonzero(mask).astype("int64")
