from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_LAGS = (1, 2, 3, 24, 48, 72, 168)
DEFAULT_ROLLING_WINDOWS = (24, 168)


def add_weather_features(
    features: pd.DataFrame,
    *,
    weather: pd.DataFrame,
    timestamps: pd.DatetimeIndex,
    prefix: str,
    weather_features: tuple[str, ...],
) -> pd.DataFrame:
    """Join weather values at specific timestamps using a stable prefix."""
    aligned = weather.reindex(timestamps)
    for column in weather_features:
        features[f"{prefix}_{column}"] = aligned[column].to_numpy()
    return features


def make_origins(
    index: pd.DatetimeIndex,
    *,
    target_start: str,
    target_end: str,
    lookback_hours: int,
    forecast_horizon: int,
) -> pd.DatetimeIndex:
    """Create forecast origin timestamps whose whole horizon stays in one split."""
    target_start_ts = pd.Timestamp(target_start)
    target_end_ts = pd.Timestamp(target_end)

    min_origin = max(index.min() + pd.Timedelta(hours=lookback_hours), target_start_ts - pd.Timedelta(hours=1))
    max_origin = target_end_ts - pd.Timedelta(hours=forecast_horizon)

    if max_origin < min_origin:
        return pd.DatetimeIndex([], name="origin_timestamp")

    origins = pd.date_range(min_origin, max_origin, freq="h", name="origin_timestamp")
    return origins.intersection(index)


def make_base_features(
    series: pd.Series,
    origins: pd.DatetimeIndex,
    *,
    lags: tuple[int, ...] = DEFAULT_LAGS,
    rolling_windows: tuple[int, ...] = DEFAULT_ROLLING_WINDOWS,
) -> pd.DataFrame:
    """Build origin-time lag and rolling features from historical load only."""
    features = pd.DataFrame(index=origins.copy())
    features.index.name = "origin_timestamp"

    features["load_t"] = series.reindex(origins).to_numpy()
    for lag in lags:
        lag_index = origins - pd.Timedelta(hours=lag)
        features[f"lag_{lag}"] = series.reindex(lag_index).to_numpy()

    for window in rolling_windows:
        rolling = series.rolling(window=window, min_periods=window)
        features[f"rolling_mean_{window}"] = rolling.mean().reindex(origins).to_numpy()
        features[f"rolling_std_{window}"] = rolling.std().reindex(origins).fillna(0.0).to_numpy()

    features["origin_hour"] = origins.hour
    features["origin_dayofweek"] = origins.dayofweek
    features["origin_month"] = origins.month
    features["origin_is_weekend"] = (origins.dayofweek >= 5).astype("int8")
    return features


def add_target_time_features(
    base_features: pd.DataFrame,
    horizon: int,
    *,
    weather: pd.DataFrame | None = None,
    weather_features: tuple[str, ...] = (),
    use_origin_weather: bool = False,
    use_target_weather: bool = False,
) -> pd.DataFrame:
    """Add target-time features for one direct forecasting horizon."""
    target_times = base_features.index + pd.Timedelta(hours=horizon)
    features = base_features.copy()
    features["horizon"] = horizon
    features["target_hour"] = target_times.hour
    features["target_dayofweek"] = target_times.dayofweek
    features["target_month"] = target_times.month
    features["target_is_weekend"] = (target_times.dayofweek >= 5).astype("int8")

    if weather is not None and weather_features:
        if use_origin_weather:
            features = add_weather_features(
                features,
                weather=weather,
                timestamps=base_features.index,
                prefix="origin_weather",
                weather_features=weather_features,
            )
        if use_target_weather:
            features = add_weather_features(
                features,
                weather=weather,
                timestamps=target_times,
                prefix="target_weather",
                weather_features=weather_features,
            )
    return features


def add_global_target_features(
    base_features: pd.DataFrame,
    horizon: int,
    *,
    weather: pd.DataFrame | None = None,
    weather_features: tuple[str, ...] = (),
    use_origin_weather: bool = False,
    use_target_weather: bool = False,
) -> pd.DataFrame:
    """Add target-time and optional multi-site weather features for global models."""
    if "origin_timestamp" not in base_features.columns:
        raise ValueError("Global features require an 'origin_timestamp' column.")

    origin_times = pd.DatetimeIndex(pd.to_datetime(base_features["origin_timestamp"]))
    target_times = origin_times + pd.Timedelta(hours=horizon)

    features = base_features.copy()
    features["horizon"] = horizon
    features["target_hour"] = target_times.hour
    features["target_dayofweek"] = target_times.dayofweek
    features["target_month"] = target_times.month
    features["target_is_weekend"] = (target_times.dayofweek >= 5).astype("int8")

    if weather is not None and weather_features:
        if "_site_id" not in features.columns:
            raise ValueError("Global weather features require a '_site_id' helper column.")
        site_ids = features["_site_id"].astype(str)

        if use_origin_weather:
            origin_weather = weather.reindex(
                pd.MultiIndex.from_arrays([site_ids, origin_times], names=["site_id", "timestamp"])
            )
            for column in weather_features:
                features[f"origin_weather_{column}"] = origin_weather[column].to_numpy()

        if use_target_weather:
            target_weather = weather.reindex(
                pd.MultiIndex.from_arrays([site_ids, target_times], names=["site_id", "timestamp"])
            )
            for column in weather_features:
                features[f"target_weather_{column}"] = target_weather[column].to_numpy()

    helper_columns = [column for column in ("origin_timestamp", "_site_id") if column in features.columns]
    if helper_columns:
        features = features.drop(columns=helper_columns)
    return features


def make_target_matrix(
    series: pd.Series,
    origins: pd.DatetimeIndex,
    *,
    forecast_horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return target values and target timestamps shaped [n_origins, horizon]."""
    values = []
    timestamps = []
    for horizon in range(1, forecast_horizon + 1):
        target_times = origins + pd.Timedelta(hours=horizon)
        values.append(series.reindex(target_times).to_numpy(dtype="float32"))
        timestamps.append(target_times.to_numpy())

    y = np.stack(values, axis=1)
    target_timestamp_matrix = np.stack(timestamps, axis=1)
    return y, target_timestamp_matrix


def make_supervised_split(
    series: pd.Series,
    *,
    target_start: str,
    target_end: str,
    lookback_hours: int,
    forecast_horizon: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    origins = make_origins(
        series.index,
        target_start=target_start,
        target_end=target_end,
        lookback_hours=lookback_hours,
        forecast_horizon=forecast_horizon,
    )
    x = make_base_features(series, origins)
    y, target_timestamps = make_target_matrix(series, origins, forecast_horizon=forecast_horizon)
    return x, y, target_timestamps
