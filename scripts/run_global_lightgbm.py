from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from building_aware.baselines import HistoricalAverage
from building_aware.data_utils import (
    load_electricity_wide,
    load_metadata,
    load_weather_for_sites,
    select_diverse_electricity_meters,
)
from building_aware.features import add_global_target_features, make_supervised_split
from building_aware.global_lightgbm import (
    fit_lightgbm_direct_with_builder,
    predict_lightgbm_direct_with_builder,
)
from building_aware.metrics import horizon_metrics, regression_metrics


CATEGORICAL_METADATA_COLUMNS = [
    "building_id",
    "site_id",
    "primaryspaceusage",
    "sub_primaryspaceusage",
    "timezone",
]

NUMERIC_METADATA_COLUMNS = [
    "sqm",
    "log_sqm",
    "sqm_missing",
    "yearbuilt",
    "yearbuilt_missing",
    "numberoffloors",
    "numberoffloors_missing",
]

METADATA_MODEL_COLUMNS = CATEGORICAL_METADATA_COLUMNS + NUMERIC_METADATA_COLUMNS
HELPER_COLUMNS = ["origin_timestamp", "_site_id"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run global multi-building LightGBM baselines.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "global_lightgbm_250.json",
        help="Path to the JSON experiment config.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def get_run_name(config: dict[str, Any], default: str) -> str:
    value = str(config.get("run_name", default)).strip()
    return value or default


def as_float(value: Any) -> float:
    numeric = pd.to_numeric(value, errors="coerce")
    return float(numeric) if pd.notna(numeric) else float("nan")


def make_metadata_features(row: pd.Series) -> dict[str, Any]:
    sqm = as_float(row.get("sqm"))
    yearbuilt = as_float(row.get("yearbuilt"))
    floors = as_float(row.get("numberoffloors"))

    return {
        "building_id": str(row.get("building_id", "Unknown")),
        "site_id": str(row.get("site_id", "Unknown")),
        "_site_id": str(row.get("site_id", "Unknown")),
        "primaryspaceusage": str(row.get("primaryspaceusage", "Unknown")) if pd.notna(row.get("primaryspaceusage")) else "Unknown",
        "sub_primaryspaceusage": str(row.get("sub_primaryspaceusage", "Unknown")) if pd.notna(row.get("sub_primaryspaceusage")) else "Unknown",
        "timezone": str(row.get("timezone", "Unknown")) if pd.notna(row.get("timezone")) else "Unknown",
        "sqm": 0.0 if not np.isfinite(sqm) else sqm,
        "log_sqm": 0.0 if not np.isfinite(sqm) else float(np.log1p(max(sqm, 0.0))),
        "sqm_missing": int(not np.isfinite(sqm)),
        "yearbuilt": 0.0 if not np.isfinite(yearbuilt) else yearbuilt,
        "yearbuilt_missing": int(not np.isfinite(yearbuilt)),
        "numberoffloors": 0.0 if not np.isfinite(floors) else floors,
        "numberoffloors_missing": int(not np.isfinite(floors)),
    }


def cast_categoricals(datasets: dict[str, dict[str, Any]]) -> None:
    for column in CATEGORICAL_METADATA_COLUMNS:
        values: list[str] = []
        for split_data in datasets.values():
            if column in split_data["x"].columns:
                values.extend(split_data["x"][column].fillna("Unknown").astype(str).tolist())
        categories = sorted(set(values))
        dtype = pd.CategoricalDtype(categories=categories)
        for split_data in datasets.values():
            if column in split_data["x"].columns:
                split_data["x"][column] = split_data["x"][column].fillna("Unknown").astype(str).astype(dtype)


def build_global_datasets(
    electricity: pd.DataFrame,
    selected_meters: pd.DataFrame,
    *,
    splits: dict[str, dict[str, str]],
    lookback_hours: int,
    forecast_horizon: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, np.ndarray]]:
    datasets = {
        split_name: {"x_parts": [], "y_parts": [], "target_parts": []}
        for split_name in splits
    }
    historical_predictions = {"val": [], "test": []}

    selected_by_id = selected_meters.set_index("building_id")
    for meter_id, meter_meta in selected_by_id.iterrows():
        meter_meta = meter_meta.copy()
        meter_meta["building_id"] = meter_id
        series = electricity[meter_id].astype("float32")
        missing = int(series.isna().sum())
        if missing:
            series = series.interpolate(method="time", limit_direction="both").ffill().bfill()

        metadata_features = make_metadata_features(meter_meta)
        historical = HistoricalAverage().fit(
            series,
            start=splits["train"]["target_start"],
            end=splits["train"]["target_end"],
        )

        for split_name, split_cfg in splits.items():
            x, y, target_timestamps = make_supervised_split(
                series,
                target_start=split_cfg["target_start"],
                target_end=split_cfg["target_end"],
                lookback_hours=lookback_hours,
                forecast_horizon=forecast_horizon,
            )
            x = x.copy()
            x["origin_timestamp"] = x.index
            for key, value in metadata_features.items():
                x[key] = value

            datasets[split_name]["x_parts"].append(x.reset_index(drop=True))
            datasets[split_name]["y_parts"].append(y)
            datasets[split_name]["target_parts"].append(target_timestamps)

            if split_name in historical_predictions:
                historical_predictions[split_name].append(historical.predict(target_timestamps))

    final_datasets: dict[str, dict[str, Any]] = {}
    for split_name, split_data in datasets.items():
        final_datasets[split_name] = {
            "x": pd.concat(split_data["x_parts"], ignore_index=True),
            "y": np.concatenate(split_data["y_parts"], axis=0),
            "target_timestamps": np.concatenate(split_data["target_parts"], axis=0),
        }

    cast_categoricals(final_datasets)
    historical_concat = {
        split_name: np.concatenate(parts, axis=0)
        for split_name, parts in historical_predictions.items()
    }
    return final_datasets, historical_concat


def feature_subset(frame: pd.DataFrame, *, include_metadata: bool) -> pd.DataFrame:
    keep = [column for column in frame.columns if column not in METADATA_MODEL_COLUMNS]
    if include_metadata:
        keep = keep + [column for column in METADATA_MODEL_COLUMNS if column in frame.columns]
    return frame[keep].copy()


def make_feature_builder(
    *,
    weather: pd.DataFrame | None,
    weather_features: tuple[str, ...],
    use_origin_weather: bool,
    use_target_weather: bool,
):
    def build(frame: pd.DataFrame, horizon: int) -> pd.DataFrame:
        return add_global_target_features(
            frame,
            horizon,
            weather=weather,
            weather_features=weather_features,
            use_origin_weather=use_origin_weather,
            use_target_weather=use_target_weather,
        )

    return build


def add_metric_rows(
    rows: list[dict[str, Any]],
    *,
    model: str,
    split: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_meters: int,
) -> None:
    row = regression_metrics(y_true, y_pred)
    row.update({"split": split, "model": model, "num_meters": num_meters})
    rows.append(row)


def add_horizon_rows(
    rows: list[pd.DataFrame],
    *,
    model: str,
    split: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    frame = horizon_metrics(y_true, y_pred)
    frame["split"] = split
    frame["model"] = model
    rows.append(frame)


def per_building_metrics(
    *,
    x_split: pd.DataFrame,
    y_true: np.ndarray,
    prediction_sets: dict[str, np.ndarray],
    selected_meters: pd.DataFrame,
    split: str,
) -> pd.DataFrame:
    rows = []
    metadata_lookup = selected_meters.set_index("building_id")
    building_ids = x_split["building_id"].astype(str).to_numpy()

    for building_id in sorted(set(building_ids)):
        mask = building_ids == building_id
        meta = metadata_lookup.loc[building_id]
        for model_name, y_pred in prediction_sets.items():
            row = regression_metrics(y_true[mask], y_pred[mask])
            row.update(
                {
                    "split": split,
                    "model": model_name,
                    "building_id": building_id,
                    "site_id": meta.get("site_id", ""),
                    "primaryspaceusage": meta.get("primaryspaceusage", ""),
                    "completeness": meta.get("completeness", np.nan),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def save_sample_predictions(
    *,
    output_path: Path,
    datasets: dict[str, dict[str, Any]],
    prediction_sets: dict[str, dict[str, np.ndarray]],
    selected_meters: pd.DataFrame,
) -> None:
    sample_meter = str(selected_meters.iloc[0]["building_id"])
    split_name = "test"
    horizon_idx = 23
    x_split = datasets[split_name]["x"]
    mask = x_split["building_id"].astype(str).to_numpy() == sample_meter
    target_timestamps = pd.to_datetime(datasets[split_name]["target_timestamps"][mask, horizon_idx])

    frame = pd.DataFrame(
        {
            "building_id": sample_meter,
            "target_timestamp": target_timestamps,
            "horizon": horizon_idx + 1,
            "y_true": datasets[split_name]["y"][mask, horizon_idx],
        }
    )
    for model_name, y_pred in prediction_sets[split_name].items():
        frame[model_name] = y_pred[mask, horizon_idx]

    start = frame["target_timestamp"].min()
    frame = frame.loc[frame["target_timestamp"] < start + pd.Timedelta(days=7)].copy()
    frame.to_csv(output_path, index=False)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    run_name = get_run_name(config, "global_lightgbm")

    data_dir = PROJECT_ROOT / config["data_dir"]
    splits = config["splits"]
    lookback_hours = int(config["lookback_hours"])
    forecast_horizon = int(config["forecast_horizon"])
    selection = config["selection"]
    weather_config = config.get("weather", {})

    output_results = PROJECT_ROOT / "outputs" / "results"
    output_predictions = PROJECT_ROOT / "outputs" / "predictions"
    output_results.mkdir(parents=True, exist_ok=True)
    output_predictions.mkdir(parents=True, exist_ok=True)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Data dir: {data_dir}")
    print(f"Task: global multi-building, past {lookback_hours} hours -> next {forecast_horizon} hours")

    metadata = load_metadata(data_dir)
    electricity = load_electricity_wide(data_dir)
    selected_meters = select_diverse_electricity_meters(
        electricity,
        metadata,
        max_meters=int(selection["max_meters"]),
        meters_per_primary_use=int(selection["meters_per_primary_use"]),
        primary_uses=selection.get("primary_uses"),
        min_completeness=float(selection.get("min_completeness", 0.95)),
        site_balanced=bool(selection.get("site_balanced", False)),
        max_meters_per_site=selection.get("max_meters_per_site"),
    )
    selected_path = output_results / f"{run_name}_selected_meters.csv"
    selected_meters.to_csv(selected_path, index=False)

    meter_ids = selected_meters["building_id"].tolist()
    electricity = electricity[meter_ids]
    print("Selected meters by primary use:")
    print(selected_meters["primaryspaceusage"].value_counts().to_string())

    datasets, historical_predictions = build_global_datasets(
        electricity,
        selected_meters,
        splits=splits,
        lookback_hours=lookback_hours,
        forecast_horizon=forecast_horizon,
    )
    for split_name, split_data in datasets.items():
        print(f"{split_name}: rows={len(split_data['x'])}, targets={split_data['y'].size}")

    weather = None
    weather_features: tuple[str, ...] = ()
    use_origin_weather = False
    use_target_weather = False
    if weather_config.get("enabled", False):
        weather_features = tuple(weather_config.get("features", []))
        use_origin_weather = bool(weather_config.get("use_origin_weather", False))
        use_target_weather = bool(weather_config.get("use_target_weather", True))
        weather, weather_info = load_weather_for_sites(
            data_dir,
            selected_meters["site_id"].astype(str).tolist(),
            weather_features=weather_features,
            reference_index=electricity.index,
        )
        print("Weather info:")
        print(f"  sites: {weather_info['site_ids']}")
        print(f"  rows: {weather_info['rows']}")
        print(f"  features: {weather_info['features']}")

    load_only_builder = make_feature_builder(
        weather=None,
        weather_features=(),
        use_origin_weather=False,
        use_target_weather=False,
    )
    weather_builder = make_feature_builder(
        weather=weather,
        weather_features=weather_features,
        use_origin_weather=use_origin_weather,
        use_target_weather=use_target_weather,
    )

    x_load_train = feature_subset(datasets["train"]["x"], include_metadata=False)
    x_load_val = feature_subset(datasets["val"]["x"], include_metadata=False)
    x_load_test = feature_subset(datasets["test"]["x"], include_metadata=False)
    x_building_train = feature_subset(datasets["train"]["x"], include_metadata=True)
    x_building_val = feature_subset(datasets["val"]["x"], include_metadata=True)
    x_building_test = feature_subset(datasets["test"]["x"], include_metadata=True)

    prediction_sets: dict[str, dict[str, np.ndarray]] = {
        "val": {"HistoricalAverage": historical_predictions["val"]},
        "test": {"HistoricalAverage": historical_predictions["test"]},
    }

    experiments = [
        ("GlobalLightGBM-load-only", x_load_train, x_load_val, x_load_test, load_only_builder),
        ("GlobalLightGBM-load-weather", x_load_train, x_load_val, x_load_test, weather_builder),
        ("GlobalLightGBM-building-aware", x_building_train, x_building_val, x_building_test, weather_builder),
    ]

    for model_name, x_train, x_val, x_test, builder in experiments:
        print(f"Training {model_name}...")
        models = fit_lightgbm_direct_with_builder(
            x_train,
            datasets["train"]["y"],
            x_val,
            datasets["val"]["y"],
            forecast_horizon=forecast_horizon,
            params=dict(config["lightgbm"]),
            feature_builder=builder,
        )
        prediction_sets["val"][model_name] = predict_lightgbm_direct_with_builder(
            models,
            x_val,
            forecast_horizon=forecast_horizon,
            feature_builder=builder,
        )
        prediction_sets["test"][model_name] = predict_lightgbm_direct_with_builder(
            models,
            x_test,
            forecast_horizon=forecast_horizon,
            feature_builder=builder,
        )

    metric_rows: list[dict[str, Any]] = []
    horizon_rows: list[pd.DataFrame] = []
    per_building_frames: list[pd.DataFrame] = []
    for split_name, split_predictions in prediction_sets.items():
        y_true = datasets[split_name]["y"]
        for model_name, y_pred in split_predictions.items():
            add_metric_rows(
                metric_rows,
                model=model_name,
                split=split_name,
                y_true=y_true,
                y_pred=y_pred,
                num_meters=len(selected_meters),
            )
            add_horizon_rows(horizon_rows, model=model_name, split=split_name, y_true=y_true, y_pred=y_pred)

        per_building_frames.append(
            per_building_metrics(
                x_split=datasets[split_name]["x"],
                y_true=y_true,
                prediction_sets=split_predictions,
                selected_meters=selected_meters,
                split=split_name,
            )
        )

    metrics = pd.DataFrame(metric_rows)
    metrics = metrics[["split", "model", "num_meters", "MAE", "RMSE", "NMAE", "CVRMSE", "n"]]
    metrics_path = output_results / f"{run_name}_metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    horizon_frame = pd.concat(horizon_rows, ignore_index=True)
    horizon_frame = horizon_frame[["split", "model", "horizon", "MAE", "RMSE", "NMAE", "CVRMSE", "n"]]
    horizon_path = output_results / f"{run_name}_horizon_metrics.csv"
    horizon_frame.to_csv(horizon_path, index=False)

    per_building = pd.concat(per_building_frames, ignore_index=True)
    per_building_path = output_results / f"{run_name}_per_building_metrics.csv"
    per_building.to_csv(per_building_path, index=False)

    sample_path = output_predictions / f"{run_name}_test_h24_sample_predictions.csv"
    save_sample_predictions(
        output_path=sample_path,
        datasets=datasets,
        prediction_sets=prediction_sets,
        selected_meters=selected_meters,
    )

    print("\nMetrics:")
    print(metrics.to_string(index=False))
    print(f"\nSaved selected meters: {selected_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved horizon metrics: {horizon_path}")
    print(f"Saved per-building metrics: {per_building_path}")
    print(f"Saved sample predictions: {sample_path}")


if __name__ == "__main__":
    main()
