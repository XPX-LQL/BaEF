from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation


FeatureBuilder = Callable[[pd.DataFrame, int], pd.DataFrame]


def fit_lightgbm_direct_with_builder(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_val: pd.DataFrame,
    y_val: np.ndarray,
    *,
    forecast_horizon: int,
    params: dict[str, Any],
    feature_builder: FeatureBuilder,
) -> list[LGBMRegressor]:
    """Fit one LightGBM model per horizon using a custom feature builder."""
    models: list[LGBMRegressor] = []
    model_params = dict(params)
    stopping_rounds = int(model_params.pop("early_stopping_rounds", 50))

    for horizon in range(1, forecast_horizon + 1):
        x_train_h = feature_builder(x_train, horizon)
        x_val_h = feature_builder(x_val, horizon)
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


def predict_lightgbm_direct_with_builder(
    models: list[LGBMRegressor],
    x_base: pd.DataFrame,
    *,
    forecast_horizon: int,
    feature_builder: FeatureBuilder,
) -> np.ndarray:
    preds = []
    for horizon, model in enumerate(models, start=1):
        x_h = feature_builder(x_base, horizon)
        preds.append(model.predict(x_h).astype("float32"))
    return np.stack(preds, axis=1)
