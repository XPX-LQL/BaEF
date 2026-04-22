from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from building_aware.data_utils import (
    load_electricity_wide,
    load_metadata,
    load_weather_for_sites,
    select_diverse_electricity_meters,
)
from building_aware.itransformer import BuildingAwareITransformer, MultiMeterWindowContextDataset
from building_aware.lstm import fit_meter_scaler
from building_aware.metrics import horizon_metrics, regression_metrics
from building_aware.time_splits import inclusive_time_positions, split_origin_positions


CONTINUOUS_METADATA_COLUMNS = {"sqm", "log_sqm", "yearbuilt", "numberoffloors"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run building-aware iTransformer experiments.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "building_aware_itransformer_250_noweather.json",
        help="Path to the JSON experiment config.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


def fill_meter_values(frame: pd.DataFrame) -> pd.DataFrame:
    filled = frame.apply(pd.to_numeric, errors="coerce")
    filled = filled.interpolate(method="time", limit_direction="both").ffill().bfill()
    return filled.astype("float32")


def prepare_metadata_tensors(
    selected_meters: pd.DataFrame,
    *,
    categorical_columns: list[str],
    numeric_columns: list[str],
) -> tuple[torch.Tensor, list[int], torch.Tensor, dict[str, Any]]:
    frame = selected_meters.copy()
    for column in categorical_columns:
        frame[column] = frame[column].fillna("Unknown").astype(str)

    categorical_indices = []
    categorical_cardinalities = []
    categorical_maps = {}
    for column in categorical_columns:
        values = frame[column]
        categories = sorted(values.unique().tolist())
        mapping = {category: idx for idx, category in enumerate(categories)}
        categorical_indices.append(values.map(mapping).to_numpy(dtype="int64"))
        categorical_cardinalities.append(len(categories))
        categorical_maps[column] = mapping

    if categorical_indices:
        categorical_tensor = torch.tensor(np.stack(categorical_indices, axis=1), dtype=torch.long)
    else:
        categorical_tensor = torch.empty(len(frame), 0, dtype=torch.long)

    raw_numeric = pd.DataFrame(index=frame.index)
    sqm = pd.to_numeric(frame.get("sqm"), errors="coerce")
    yearbuilt = pd.to_numeric(frame.get("yearbuilt"), errors="coerce")
    floors = pd.to_numeric(frame.get("numberoffloors"), errors="coerce")

    raw_numeric["sqm"] = sqm
    raw_numeric["log_sqm"] = np.log1p(sqm.clip(lower=0))
    raw_numeric["sqm_missing"] = sqm.isna().astype("float32")
    raw_numeric["yearbuilt"] = yearbuilt
    raw_numeric["yearbuilt_missing"] = yearbuilt.isna().astype("float32")
    raw_numeric["numberoffloors"] = floors
    raw_numeric["numberoffloors_missing"] = floors.isna().astype("float32")

    numeric_frame = pd.DataFrame(index=frame.index)
    numeric_stats = {}
    for column in numeric_columns:
        values = raw_numeric[column].astype("float32")
        if column in CONTINUOUS_METADATA_COLUMNS:
            median = float(values.median()) if values.notna().any() else 0.0
            values = values.fillna(median)
            mean = float(values.mean())
            std = float(values.std(ddof=0))
            if std < 1e-6:
                std = 1.0
            values = (values - mean) / std
            numeric_stats[column] = {"median": median, "mean": mean, "std": std}
        else:
            values = values.fillna(0.0)
            numeric_stats[column] = {"mean": float(values.mean()), "std": float(values.std(ddof=0))}
        numeric_frame[column] = values.astype("float32")

    numeric_tensor = torch.tensor(numeric_frame.to_numpy(dtype="float32"), dtype=torch.float32)
    info = {
        "categorical_maps": categorical_maps,
        "categorical_cardinalities": categorical_cardinalities,
        "categorical_columns": categorical_columns,
        "numeric_columns": numeric_columns,
        "numeric_stats": numeric_stats,
    }
    return categorical_tensor, categorical_cardinalities, numeric_tensor, info


def build_weather_values(
    weather: pd.DataFrame,
    selected_meters: pd.DataFrame,
    *,
    weather_features: list[str],
    reference_index: pd.DatetimeIndex,
) -> np.ndarray:
    values = np.zeros((len(reference_index), len(selected_meters), len(weather_features)), dtype="float32")
    for meter_idx, site_id in enumerate(selected_meters["site_id"].astype(str)):
        site_weather = weather.xs(site_id, level="site_id")[weather_features]
        site_weather = site_weather.reindex(reference_index).interpolate(method="time", limit_direction="both").ffill().bfill()
        values[:, meter_idx, :] = np.nan_to_num(site_weather.to_numpy(dtype="float32"), nan=0.0)
    return values


def normalize_weather_values(weather_values: np.ndarray, train_positions: np.ndarray) -> tuple[np.ndarray, dict[str, list[float]]]:
    train_values = weather_values[train_positions].reshape(-1, weather_values.shape[-1])
    mean = np.nanmean(train_values, axis=0).astype("float32")
    std = np.nanstd(train_values, axis=0).astype("float32")
    mean = np.where(np.isfinite(mean), mean, 0.0).astype("float32")
    std = np.where(np.isfinite(std), std, 1.0).astype("float32")
    std = np.where(std < 1e-6, 1.0, std).astype("float32")
    normalized = (weather_values - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
    return normalized.astype("float32"), {"mean": mean.tolist(), "std": std.tolist()}


def make_weather_context(
    weather_values_norm: np.ndarray,
    *,
    origin_positions: np.ndarray,
    forecast_horizon: int,
    use_origin_weather: bool,
    use_future_weather_mean: bool,
) -> np.ndarray:
    parts = []
    if use_origin_weather:
        parts.append(np.nan_to_num(weather_values_norm[origin_positions], nan=0.0))
    if use_future_weather_mean:
        future_parts = [
            np.nan_to_num(weather_values_norm[origin + 1 : origin + forecast_horizon + 1], nan=0.0).mean(axis=0)
            for origin in origin_positions
        ]
        parts.append(np.stack(future_parts, axis=0).astype("float32"))
    if not parts:
        raise ValueError("At least one weather context component must be enabled.")
    return np.concatenate(parts, axis=-1).astype("float32")


def make_empty_weather_context(
    *,
    origin_positions: np.ndarray,
    num_variables: int,
) -> np.ndarray:
    return np.zeros((len(origin_positions), num_variables, 0), dtype="float32")


def make_criterion(training_config: dict[str, Any], scaler) -> tuple[nn.Module, str]:
    loss_name = str(training_config.get("loss", "normalized_mse")).strip().lower()
    if loss_name not in {"normalized_mse", "mse"}:
        raise ValueError(
            "Unsupported loss. Building-aware iTransformer currently supports only 'normalized_mse' / 'mse'."
        )
    return nn.MSELoss(), "normalized_mse"


def make_loader(
    values_norm: np.ndarray,
    weather_context: np.ndarray,
    *,
    origin_positions: np.ndarray,
    lookback_hours: int,
    forecast_horizon: int,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> tuple[MultiMeterWindowContextDataset, DataLoader]:
    dataset = MultiMeterWindowContextDataset(
        values_norm,
        weather_context,
        origin_positions=origin_positions,
        lookback_hours=lookback_hours,
        forecast_horizon=forecast_horizon,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return dataset, loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    clip_grad_norm: float,
) -> tuple[float, float, float]:
    model.train()
    total_loss = 0.0
    total_prediction_loss = 0.0
    total_regularization_loss = 0.0
    total_count = 0
    for x, y, weather_context in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        weather_context = weather_context.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        pred = model(x, weather_context)
        prediction_loss = criterion(pred, y)
        regularization_loss = (
            model.regularization_loss()
            if hasattr(model, "regularization_loss")
            else torch.zeros((), device=device)
        )
        loss = prediction_loss + regularization_loss
        loss.backward()
        if clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        optimizer.step()

        batch_size = x.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_prediction_loss += float(prediction_loss.item()) * batch_size
        total_regularization_loss += float(regularization_loss.item()) * batch_size
        total_count += batch_size
    return (
        total_loss / max(total_count, 1),
        total_prediction_loss / max(total_count, 1),
        total_regularization_loss / max(total_count, 1),
    )


def evaluate_loss_and_original_rmse(
    model: nn.Module,
    loader: DataLoader,
    *,
    criterion: nn.Module,
    scaler,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    squared_error_sum = 0.0
    total_values = 0
    std = torch.tensor(scaler.std.reshape(1, 1, -1), dtype=torch.float32, device=device)
    with torch.no_grad():
        for x, y, weather_context in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            weather_context = weather_context.to(device, non_blocking=True)
            pred = model(x, weather_context)
            loss = criterion(pred, y)

            batch_size = x.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_batches += batch_size

            original_error = (pred - y) * std
            squared_error_sum += float((original_error**2).sum().item())
            total_values += int(original_error.numel())

    mean_loss = total_loss / max(total_batches, 1)
    original_rmse = float(np.sqrt(squared_error_sum / max(total_values, 1)))
    return mean_loss, original_rmse


def inverse_transform(values_norm: np.ndarray, *, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return values_norm * std.reshape(1, 1, -1) + mean.reshape(1, 1, -1)


def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    *,
    scaler,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    pred_parts = []
    true_parts = []
    with torch.no_grad():
        for x, y, weather_context in loader:
            x = x.to(device, non_blocking=True)
            weather_context = weather_context.to(device, non_blocking=True)
            pred = model(x, weather_context).cpu().numpy()
            pred_parts.append(pred)
            true_parts.append(y.numpy())

    pred_norm = np.concatenate(pred_parts, axis=0)
    true_norm = np.concatenate(true_parts, axis=0)
    pred = inverse_transform(pred_norm, mean=scaler.mean, std=scaler.std)
    true = inverse_transform(true_norm, mean=scaler.mean, std=scaler.std)
    return true, pred


def per_building_metrics(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    meter_ids: list[str],
    selected_meters: pd.DataFrame,
    split: str,
    model_name: str,
) -> pd.DataFrame:
    rows = []
    metadata_lookup = selected_meters.set_index("building_id")
    for meter_idx, building_id in enumerate(meter_ids):
        meta = metadata_lookup.loc[building_id]
        row = regression_metrics(y_true[:, :, meter_idx], y_pred[:, :, meter_idx])
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
    y_true: np.ndarray,
    prediction_map: dict[str, np.ndarray],
    origin_positions: np.ndarray,
    time_index: pd.DatetimeIndex,
    meter_ids: list[str],
    forecast_horizon: int,
) -> None:
    sample_meter_idx = 0
    horizon_idx = forecast_horizon - 1
    target_times = time_index[origin_positions + forecast_horizon]
    frame = pd.DataFrame(
        {
            "building_id": meter_ids[sample_meter_idx],
            "target_timestamp": target_times,
            "horizon": forecast_horizon,
            "y_true": y_true[:, horizon_idx, sample_meter_idx],
        }
    )
    for model_name, y_pred in prediction_map.items():
        frame[model_name] = y_pred[:, horizon_idx, sample_meter_idx]
    start = frame["target_timestamp"].min()
    frame = frame.loc[frame["target_timestamp"] < start + pd.Timedelta(days=7)].copy()
    frame.to_csv(output_path, index=False)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    run_name = str(config.get("run_name", "building_aware_itransformer")).strip() or "building_aware_itransformer"

    training_config = config["training"]
    set_seed(int(training_config["seed"]))

    data_dir = PROJECT_ROOT / config["data_dir"]
    splits = config["splits"]
    lookback_hours = int(config["lookback_hours"])
    forecast_horizon = int(config["forecast_horizon"])
    selection = config["selection"]
    weather_config = config["weather"]
    metadata_config = config["metadata"]

    output_results = PROJECT_ROOT / "outputs" / "results"
    output_predictions = PROJECT_ROOT / "outputs" / "predictions"
    output_checkpoints = PROJECT_ROOT / "outputs" / "checkpoints"
    output_results.mkdir(parents=True, exist_ok=True)
    output_predictions.mkdir(parents=True, exist_ok=True)
    output_checkpoints.mkdir(parents=True, exist_ok=True)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Data dir: {data_dir}")
    print(f"Task: building-aware iTransformer, past {lookback_hours} hours -> next {forecast_horizon} hours")

    metadata = load_metadata(data_dir)
    electricity_all = load_electricity_wide(data_dir)
    selected_meters = select_diverse_electricity_meters(
        electricity_all,
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
    electricity = fill_meter_values(electricity_all[meter_ids])
    values = electricity.to_numpy(dtype="float32")

    print("Selected meters by primary use:")
    print(selected_meters["primaryspaceusage"].value_counts().to_string())

    train_positions = inclusive_time_positions(
        electricity.index,
        start=splits["train"]["target_start"],
        end=splits["train"]["target_end"],
    )
    scaler = fit_meter_scaler(values, train_positions)
    values_norm = scaler.transform(values).astype("float32")

    origin_positions = {
        split_name: split_origin_positions(
            electricity.index,
            target_start=split_cfg["target_start"],
            target_end=split_cfg["target_end"],
            lookback_hours=lookback_hours,
            forecast_horizon=forecast_horizon,
        )
        for split_name, split_cfg in splits.items()
    }

    categorical_tensor, categorical_cardinalities, numeric_tensor, metadata_info = prepare_metadata_tensors(
        selected_meters,
        categorical_columns=list(metadata_config["categorical_columns"]),
        numeric_columns=list(metadata_config["numeric_columns"]),
    )
    print("Metadata context:")
    print(f"  categorical_cardinalities: {categorical_cardinalities}")
    print(f"  numeric_dim: {numeric_tensor.shape[1]}")

    weather_features = list(weather_config.get("features", []))
    use_origin_weather = bool(weather_config.get("use_origin_weather", True))
    use_future_weather_mean = bool(weather_config.get("use_future_weather_mean", True))
    weather_enabled = bool(weather_config.get("enabled", True)) and bool(weather_features) and (
        use_origin_weather or use_future_weather_mean
    )

    if weather_enabled:
        weather, weather_info = load_weather_for_sites(
            data_dir,
            selected_meters["site_id"].astype(str).tolist(),
            weather_features=weather_features,
            reference_index=electricity.index,
        )
        weather_values = build_weather_values(
            weather,
            selected_meters,
            weather_features=weather_features,
            reference_index=electricity.index,
        )
        weather_values_norm, weather_stats = normalize_weather_values(weather_values, train_positions)
        weather_context = {
            split_name: make_weather_context(
                weather_values_norm,
                origin_positions=positions,
                forecast_horizon=forecast_horizon,
                use_origin_weather=use_origin_weather,
                use_future_weather_mean=use_future_weather_mean,
            )
            for split_name, positions in origin_positions.items()
        }
    else:
        weather_info = {
            "enabled": False,
            "reason": "Weather context disabled by config.",
            "features": weather_features,
        }
        weather_stats = {"mean": [], "std": []}
        weather_context = {
            split_name: make_empty_weather_context(
                origin_positions=positions,
                num_variables=len(meter_ids),
            )
            for split_name, positions in origin_positions.items()
        }
    weather_dim = int(weather_context["train"].shape[-1])
    print("Weather context:")
    if weather_enabled:
        print(f"  sites: {sorted(set(selected_meters['site_id'].astype(str).tolist()))}")
    else:
        print("  disabled: True")
    print(f"  features: {weather_features}")
    print(f"  weather_dim: {weather_dim}")

    batch_size = int(training_config["batch_size"])
    num_workers = int(training_config.get("num_workers", 0))
    train_dataset, train_loader = make_loader(
        values_norm,
        weather_context["train"],
        origin_positions=origin_positions["train"],
        lookback_hours=lookback_hours,
        forecast_horizon=forecast_horizon,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_dataset, val_loader = make_loader(
        values_norm,
        weather_context["val"],
        origin_positions=origin_positions["val"],
        lookback_hours=lookback_hours,
        forecast_horizon=forecast_horizon,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_dataset, test_loader = make_loader(
        values_norm,
        weather_context["test"],
        origin_positions=origin_positions["test"],
        lookback_hours=lookback_hours,
        forecast_horizon=forecast_horizon,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    print(f"train: samples={len(train_dataset)}, targets={len(train_dataset) * forecast_horizon * len(meter_ids)}")
    print(f"val: samples={len(val_dataset)}, targets={len(val_dataset) * forecast_horizon * len(meter_ids)}")
    print(f"test: samples={len(test_dataset)}, targets={len(test_dataset) * forecast_horizon * len(meter_ids)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    baseline_model_name = (
        str(config["model"].get("baseline_model_name")).strip()
        if config["model"].get("baseline_model_name")
        else ("GlobalITransformer-load-weather" if weather_enabled else "GlobalITransformer-load-only")
    )
    model_specs = [
        (baseline_model_name, False),
        (str(config["model"].get("metadata_model_name", "GlobalITransformer-building-aware")), True),
    ]

    metrics_parts = []
    horizon_parts = []
    per_building_parts = []
    prediction_map: dict[str, np.ndarray] = {}
    context_rows = {}
    final_test_true: np.ndarray | None = None

    for model_name, use_metadata in model_specs:
        print(f"\nTraining {model_name}...")
        set_seed(int(training_config["seed"]))
        model = BuildingAwareITransformer(
            seq_len=lookback_hours,
            num_variables=len(meter_ids),
            forecast_horizon=forecast_horizon,
            d_model=int(config["model"]["d_model"]),
            n_heads=int(config["model"]["n_heads"]),
            num_layers=int(config["model"]["num_layers"]),
            d_ff=int(config["model"]["d_ff"]),
            dropout=float(config["model"]["dropout"]),
            weather_dim=weather_dim,
            categorical_cardinalities=categorical_cardinalities if use_metadata else (),
            categorical_indices=categorical_tensor if use_metadata else torch.empty(len(meter_ids), 0, dtype=torch.long),
            numeric_static=numeric_tensor if use_metadata else torch.empty(len(meter_ids), 0, dtype=torch.float32),
            metadata_dropout=float(config["model"].get("metadata_dropout", 0.0)) if use_metadata else 0.0,
            gate_l1_penalty=float(config["model"].get("gate_l1_penalty", 0.0)) if use_metadata else 0.0,
            use_metadata=use_metadata,
        ).to(device)

        criterion, loss_name = make_criterion(training_config, scaler)
        criterion = criterion.to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(training_config["learning_rate"]),
            weight_decay=float(training_config["weight_decay"]),
        )

        epochs = int(training_config["epochs"])
        patience = int(training_config["patience"])
        clip_grad_norm = float(training_config["clip_grad_norm"])
        best_val = float("inf")
        best_state = copy.deepcopy(model.state_dict())
        wait = 0
        history_rows = []

        for epoch in range(1, epochs + 1):
            train_loss, train_prediction_loss, train_regularization_loss = train_one_epoch(
                model,
                train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                clip_grad_norm=clip_grad_norm,
            )
            val_loss, val_original_rmse = evaluate_loss_and_original_rmse(
                model,
                val_loader,
                criterion=criterion,
                scaler=scaler,
                device=device,
            )
            history_rows.append(
                {
                    "model": model_name,
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_prediction_loss": train_prediction_loss,
                    "train_regularization_loss": train_regularization_loss,
                    "val_loss": val_loss,
                    "val_original_rmse": val_original_rmse,
                }
            )
            print(
                f"{model_name} epoch={epoch:02d} train_loss={train_loss:.6f} train_pred={train_prediction_loss:.6f} "
                f"train_reg={train_regularization_loss:.6f} val_loss={val_loss:.6f} "
                f"val_original_rmse={val_original_rmse:.6f}"
            )

            if val_loss < best_val:
                best_val = val_loss
                best_state = copy.deepcopy(model.state_dict())
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print(f"{model_name} early stopping at epoch {epoch}.")
                    break

        model.load_state_dict(best_state)
        checkpoint_path = output_checkpoints / f"{run_name}_{model_name.lower().replace('-', '_')}.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": config,
                "best_val_mse_norm": best_val,
                "early_stopping_metric": "val_loss",
                "loss": loss_name,
                "use_metadata": use_metadata,
            },
            checkpoint_path,
        )
        history_path = output_results / f"{run_name}_{model_name.lower().replace('-', '_')}_training_log.csv"
        pd.DataFrame(history_rows).to_csv(history_path, index=False)

        split_outputs = {}
        for split_name, loader in [("val", val_loader), ("test", test_loader)]:
            y_true, y_pred = collect_predictions(model, loader, scaler=scaler, device=device)
            split_outputs[split_name] = {"y_true": y_true, "y_pred": y_pred}

        for split_name, split_output in split_outputs.items():
            y_true = split_output["y_true"]
            y_pred = split_output["y_pred"]

            row = regression_metrics(y_true, y_pred)
            row.update(
                {
                    "split": split_name,
                    "model": model_name,
                    "num_meters": len(meter_ids),
                    "best_val_mse_norm": best_val,
                    "early_stopping_metric": "val_loss",
                    "loss": loss_name,
                }
            )
            metrics_parts.append(pd.DataFrame([row]))

            horizon_frame = horizon_metrics(y_true, y_pred)
            horizon_frame["split"] = split_name
            horizon_frame["model"] = model_name
            horizon_parts.append(horizon_frame)

            per_building_parts.append(
                per_building_metrics(
                    y_true=y_true,
                    y_pred=y_pred,
                    meter_ids=meter_ids,
                    selected_meters=selected_meters,
                    split=split_name,
                    model_name=model_name,
                )
            )

        prediction_map[model_name] = split_outputs["test"]["y_pred"]
        final_test_true = split_outputs["test"]["y_true"]
        context_rows[model_name] = {
            "history_path": str(history_path),
            "checkpoint_path": str(checkpoint_path),
            "best_val_mse_norm": best_val,
            "use_metadata": use_metadata,
            "use_weather": weather_enabled,
        }

    metrics = pd.concat(metrics_parts, ignore_index=True)
    metrics = metrics[["split", "model", "num_meters", "MAE", "RMSE", "NMAE", "CVRMSE", "n", "best_val_mse_norm", "early_stopping_metric", "loss"]]
    metrics_path = output_results / f"{run_name}_metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    horizon_frame = pd.concat(horizon_parts, ignore_index=True)
    horizon_frame = horizon_frame[["split", "model", "horizon", "MAE", "RMSE", "NMAE", "CVRMSE", "n"]]
    horizon_path = output_results / f"{run_name}_horizon_metrics.csv"
    horizon_frame.to_csv(horizon_path, index=False)

    per_building = pd.concat(per_building_parts, ignore_index=True)
    per_building_path = output_results / f"{run_name}_per_building_metrics.csv"
    per_building.to_csv(per_building_path, index=False)

    context = {
        "run_name": run_name,
        "metadata_info": metadata_info,
        "weather_info": weather_info,
        "weather_stats": weather_stats,
        "models": context_rows,
    }
    context_path = output_results / f"{run_name}_context.json"
    context_path.write_text(json.dumps(context, ensure_ascii=False, indent=2), encoding="utf-8")

    sample_path = output_predictions / f"{run_name}_test_h24_sample_predictions.csv"
    if final_test_true is not None:
        save_sample_predictions(
            output_path=sample_path,
            y_true=final_test_true,
            prediction_map=prediction_map,
            origin_positions=origin_positions["test"],
            time_index=electricity.index,
            meter_ids=meter_ids,
            forecast_horizon=forecast_horizon,
        )

    print("\nMetrics:")
    print(metrics.to_string(index=False))
    print(f"\nSaved selected meters: {selected_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved horizon metrics: {horizon_path}")
    print(f"Saved per-building metrics: {per_building_path}")
    print(f"Saved context: {context_path}")
    print(f"Saved sample predictions: {sample_path}")


if __name__ == "__main__":
    main()
