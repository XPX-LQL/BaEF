from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation

from building_aware.features import add_target_time_features


@dataclass
class HistoricalAverage:
    """Historical mean by day-of-week and hour-of-day."""

    global_mean: float | None = None
    hour_mean: pd.Series | None = None
    dow_hour_mean: pd.Series | None = None

    def fit(self, series: pd.Series, *, start: str, end: str) -> "HistoricalAverage":
        history = series.loc[pd.Timestamp(start) : pd.Timestamp(end)].dropna()
        if history.empty:
            raise ValueError("HistoricalAverage received an empty training series.")

        frame = history.to_frame("load")
        frame["dayofweek"] = frame.index.dayofweek
        frame["hour"] = frame.index.hour

        self.global_mean = float(frame["load"].mean())
        self.hour_mean = frame.groupby("hour")["load"].mean()
        self.dow_hour_mean = frame.groupby(["dayofweek", "hour"])["load"].mean()
        return self

    def predict(self, target_timestamps: np.ndarray) -> np.ndarray:
        if self.global_mean is None or self.hour_mean is None or self.dow_hour_mean is None:
            raise RuntimeError("HistoricalAverage must be fitted before prediction.")

        flat = pd.DatetimeIndex(pd.to_datetime(target_timestamps.reshape(-1)))
        keys = pd.MultiIndex.from_arrays([flat.dayofweek, flat.hour], names=["dayofweek", "hour"])
        pred = self.dow_hour_mean.reindex(keys).to_numpy(dtype="float64")

        fallback_hour = self.hour_mean.reindex(flat.hour).to_numpy(dtype="float64")
        pred = np.where(np.isfinite(pred), pred, fallback_hour)
        pred = np.where(np.isfinite(pred), pred, self.global_mean)
        return pred.reshape(target_timestamps.shape).astype("float32")


def fit_lightgbm_direct(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_val: pd.DataFrame,
    y_val: np.ndarray,
    *,
    forecast_horizon: int,
    params: dict[str, Any],
    weather: pd.DataFrame | None = None,
    weather_features: tuple[str, ...] = (),
    use_origin_weather: bool = False,
    use_target_weather: bool = False,
) -> list[LGBMRegressor]:
    """Fit one LightGBM model per forecast horizon."""
    models: list[LGBMRegressor] = []
    model_params = dict(params)
    stopping_rounds = int(model_params.pop("early_stopping_rounds", 50))

    for horizon in range(1, forecast_horizon + 1):
        x_train_h = add_target_time_features(
            x_train,
            horizon,
            weather=weather,
            weather_features=weather_features,
            use_origin_weather=use_origin_weather,
            use_target_weather=use_target_weather,
        )
        x_val_h = add_target_time_features(
            x_val,
            horizon,
            weather=weather,
            weather_features=weather_features,
            use_origin_weather=use_origin_weather,
            use_target_weather=use_target_weather,
        )
        y_train_h = y_train[:, horizon - 1]
        y_val_h = y_val[:, horizon - 1]

        train_mask = np.isfinite(y_train_h)
        val_mask = np.isfinite(y_val_h)

        model = LGBMRegressor(**model_params)
        callbacks = [log_evaluation(period=0)]
        fit_kwargs: dict[str, Any] = {}
        if val_mask.any():
            callbacks.append(early_stopping(stopping_rounds=stopping_rounds, verbose=False))
            fit_kwargs["eval_set"] = [(x_val_h.loc[val_mask], y_val_h[val_mask])]
            fit_kwargs["eval_metric"] = "l1"

        model.fit(
            x_train_h.loc[train_mask],
            y_train_h[train_mask],
            callbacks=callbacks,
            **fit_kwargs,
        )
        models.append(model)

    return models


def predict_lightgbm_direct(
    models: list[LGBMRegressor],
    x_base: pd.DataFrame,
    *,
    forecast_horizon: int,
    weather: pd.DataFrame | None = None,
    weather_features: tuple[str, ...] = (),
    use_origin_weather: bool = False,
    use_target_weather: bool = False,
) -> np.ndarray:
    preds = []
    for horizon, model in enumerate(models, start=1):
        x_h = add_target_time_features(
            x_base,
            horizon,
            weather=weather,
            weather_features=weather_features,
            use_origin_weather=use_origin_weather,
            use_target_weather=use_target_weather,
        )
        preds.append(model.predict(x_h).astype("float32"))
    return np.stack(preds, axis=1)
