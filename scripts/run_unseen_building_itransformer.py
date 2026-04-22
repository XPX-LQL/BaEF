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

from building_aware.data_utils import (  # noqa: E402
    load_electricity_wide,
    load_metadata,
    load_weather_for_sites,
    select_diverse_electricity_meters,
)
from building_aware.itransformer import BuildingAwareITransformer  # noqa: E402
from building_aware.lstm import fit_meter_scaler  # noqa: E402
from building_aware.metrics import horizon_metrics, regression_metrics  # noqa: E402
from building_aware.time_splits import inclusive_time_positions, split_origin_positions  # noqa: E402
from run_building_aware_itransformer import (  # noqa: E402
    CONTINUOUS_METADATA_COLUMNS,
    build_weather_values,
    collect_predictions,
    fill_meter_values,
    make_empty_weather_context,
    make_criterion,
    make_loader,
    per_building_metrics,
    save_sample_predictions,
    train_one_epoch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run unseen-building iTransformer evaluation.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "building_aware_unseen_itransformer_250_noweather.json",
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


def split_unseen_buildings(
    selected_meters: pd.DataFrame,
    *,
    holdout_per_primary_use: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    heldout_indices: list[int] = []
    frame = selected_meters.reset_index(drop=True).copy()
    frame["primaryspaceusage"] = frame["primaryspaceusage"].fillna("Unknown")

    for _, group in frame.groupby("primaryspaceusage", sort=True):
        indices = group.index.to_numpy()
        rng.shuffle(indices)
        num_holdout = min(int(holdout_per_primary_use), max(len(indices) - 1, 0))
        heldout_indices.extend(indices[:num_holdout].tolist())

    heldout_set = set(heldout_indices)
    support = frame.loc[~frame.index.isin(heldout_set)].reset_index(drop=True)
    heldout = frame.loc[frame.index.isin(heldout_set)].reset_index(drop=True)
    selected_with_role = frame.copy()
    selected_with_role["evaluation_role"] = np.where(frame.index.isin(heldout_set), "heldout", "support")
    return support, heldout, selected_with_role


def raw_numeric_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    raw = pd.DataFrame(index=frame.index)
    sqm = pd.to_numeric(frame.get("sqm"), errors="coerce")
    yearbuilt = pd.to_numeric(frame.get("yearbuilt"), errors="coerce")
    floors = pd.to_numeric(frame.get("numberoffloors"), errors="coerce")

    raw["sqm"] = sqm
    raw["log_sqm"] = np.log1p(sqm.clip(lower=0))
    raw["sqm_missing"] = sqm.isna().astype("float32")
    raw["yearbuilt"] = yearbuilt
    raw["yearbuilt_missing"] = yearbuilt.isna().astype("float32")
    raw["numberoffloors"] = floors
    raw["numberoffloors_missing"] = floors.isna().astype("float32")
    return raw


def fit_transfer_metadata(
    support: pd.DataFrame,
    heldout: pd.DataFrame,
    *,
    categorical_columns: list[str],
    numeric_columns: list[str],
) -> tuple[torch.Tensor, torch.Tensor, list[int], torch.Tensor, torch.Tensor, dict[str, Any]]:
    support_frame = support.copy()
    heldout_frame = heldout.copy()
    for frame in (support_frame, heldout_frame):
        frame["primaryspaceusage"] = frame["primaryspaceusage"].fillna("Unknown")
        frame["sub_primaryspaceusage"] = frame["sub_primaryspaceusage"].fillna("Unknown")
        frame["timezone"] = frame["timezone"].fillna("Unknown")

    categorical_maps: dict[str, dict[str, int]] = {}
    support_categorical = []
    heldout_categorical = []
    cardinalities = []
    for column in categorical_columns:
        support_values = support_frame[column].fillna("Unknown").astype(str)
        categories = sorted(set(support_values.tolist()) | {"Unknown"})
        mapping = {category: idx for idx, category in enumerate(categories)}
        unknown_idx = mapping["Unknown"]

        heldout_values = heldout_frame[column].fillna("Unknown").astype(str)
        support_categorical.append(support_values.map(mapping).to_numpy(dtype="int64"))
        heldout_categorical.append(
            heldout_values.map(lambda value: mapping.get(value, unknown_idx)).to_numpy(dtype="int64")
        )
        categorical_maps[column] = mapping
        cardinalities.append(len(categories))

    if support_categorical:
        support_categorical_tensor = torch.tensor(np.stack(support_categorical, axis=1), dtype=torch.long)
        heldout_categorical_tensor = torch.tensor(np.stack(heldout_categorical, axis=1), dtype=torch.long)
    else:
        support_categorical_tensor = torch.empty(len(support_frame), 0, dtype=torch.long)
        heldout_categorical_tensor = torch.empty(len(heldout_frame), 0, dtype=torch.long)

    support_raw = raw_numeric_metadata(support_frame)
    heldout_raw = raw_numeric_metadata(heldout_frame)
    support_numeric = pd.DataFrame(index=support_frame.index)
    heldout_numeric = pd.DataFrame(index=heldout_frame.index)
    numeric_stats: dict[str, dict[str, float]] = {}
    for column in numeric_columns:
        support_values = support_raw[column].astype("float32")
        heldout_values = heldout_raw[column].astype("float32")
        if column in CONTINUOUS_METADATA_COLUMNS:
            median = float(support_values.median()) if support_values.notna().any() else 0.0
            support_values = support_values.fillna(median)
            heldout_values = heldout_values.fillna(median)
            mean = float(support_values.mean())
            std = float(support_values.std(ddof=0))
            if not np.isfinite(std) or std < 1e-6:
                std = 1.0
            support_values = (support_values - mean) / std
            heldout_values = (heldout_values - mean) / std
            numeric_stats[column] = {"median": median, "mean": mean, "std": std}
        else:
            support_values = support_values.fillna(0.0)
            heldout_values = heldout_values.fillna(0.0)
            numeric_stats[column] = {
                "mean": float(support_values.mean()),
                "std": float(support_values.std(ddof=0)),
            }
        support_numeric[column] = support_values.astype("float32")
        heldout_numeric[column] = heldout_values.astype("float32")

    support_numeric_tensor = torch.tensor(support_numeric.to_numpy(dtype="float32"), dtype=torch.float32)
    heldout_numeric_tensor = torch.tensor(heldout_numeric.to_numpy(dtype="float32"), dtype=torch.float32)
    info = {
        "categorical_maps": categorical_maps,
        "categorical_cardinalities": cardinalities,
        "numeric_stats": numeric_stats,
        "numeric_columns": numeric_columns,
        "note": "Categorical maps and numeric statistics are fitted on support buildings only.",
    }
    return (
        support_categorical_tensor,
        heldout_categorical_tensor,
        cardinalities,
        support_numeric_tensor,
        heldout_numeric_tensor,
        info,
    )


def fit_transfer_type_indices(
    support: pd.DataFrame,
    heldout: pd.DataFrame,
    *,
    type_column: str,
) -> tuple[torch.Tensor, torch.Tensor, int, dict[str, int]]:
    support_values = support[type_column].fillna("Unknown").astype(str)
    categories = sorted(set(support_values.tolist()) | {"Unknown"})
    mapping = {category: idx for idx, category in enumerate(categories)}
    unknown_idx = mapping["Unknown"]
    heldout_values = heldout[type_column].fillna("Unknown").astype(str)
    support_indices = torch.tensor(support_values.map(mapping).to_numpy(dtype="int64"), dtype=torch.long)
    heldout_indices = torch.tensor(
        heldout_values.map(lambda value: mapping.get(value, unknown_idx)).to_numpy(dtype="int64"),
        dtype=torch.long,
    )
    return support_indices, heldout_indices, len(categories), mapping


def fit_weather_stats(weather_values: np.ndarray, train_positions: np.ndarray) -> dict[str, np.ndarray]:
    train_values = weather_values[train_positions].reshape(-1, weather_values.shape[-1])
    mean = np.nanmean(train_values, axis=0).astype("float32")
    std = np.nanstd(train_values, axis=0).astype("float32")
    mean = np.where(np.isfinite(mean), mean, 0.0).astype("float32")
    std = np.where(np.isfinite(std), std, 1.0).astype("float32")
    std = np.where(std < 1e-6, 1.0, std).astype("float32")
    return {"mean": mean, "std": std}


def apply_weather_stats(weather_values: np.ndarray, weather_stats: dict[str, np.ndarray]) -> np.ndarray:
    mean = weather_stats["mean"].reshape(1, 1, -1)
    std = weather_stats["std"].reshape(1, 1, -1)
    normalized = (weather_values - mean) / std
    return np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0).astype("float32")


def make_weather_contexts(
    weather_values_norm: np.ndarray,
    *,
    origin_positions: dict[str, np.ndarray],
    forecast_horizon: int,
    use_origin_weather: bool,
    use_future_weather_mean: bool,
) -> dict[str, np.ndarray]:
    contexts = {}
    for split_name, positions in origin_positions.items():
        parts = []
        if use_origin_weather:
            parts.append(np.nan_to_num(weather_values_norm[positions], nan=0.0))
        if use_future_weather_mean:
            future_parts = [
                np.nan_to_num(weather_values_norm[origin + 1 : origin + forecast_horizon + 1], nan=0.0).mean(axis=0)
                for origin in positions
            ]
            parts.append(np.stack(future_parts, axis=0).astype("float32"))
        if not parts:
            raise ValueError("At least one weather context component must be enabled.")
        contexts[split_name] = np.concatenate(parts, axis=-1).astype("float32")
    return contexts


def build_model(
    *,
    config: dict[str, Any],
    use_metadata: bool,
    num_variables: int,
    weather_dim: int,
    categorical_tensor: torch.Tensor,
    categorical_cardinalities: list[int],
    numeric_tensor: torch.Tensor,
    type_indices: torch.Tensor,
    num_building_types: int,
    device: torch.device,
) -> BuildingAwareITransformer:
    model_config = config["model"]
    variable_embedding_mode = str(model_config.get("variable_embedding_mode", "none"))
    if variable_embedding_mode != "none":
        raise ValueError("Unseen-building evaluation requires variable_embedding_mode='none'.")

    metadata_adapter_type = str(model_config.get("metadata_adapter_type", "metadata"))
    use_metadata_adapter = bool(model_config.get("use_metadata_adapter", True)) and use_metadata
    use_weather_response_adapter = bool(model_config.get("use_weather_response_adapter", False)) and use_metadata
    use_weather_token_gating_adapter = bool(model_config.get("use_weather_token_gating_adapter", False)) and use_metadata
    use_residual_output_gate = bool(model_config.get("use_residual_output_gate", False))

    model = BuildingAwareITransformer(
        seq_len=int(config["lookback_hours"]),
        num_variables=num_variables,
        forecast_horizon=int(config["forecast_horizon"]),
        d_model=int(model_config["d_model"]),
        n_heads=int(model_config["n_heads"]),
        num_layers=int(model_config["num_layers"]),
        d_ff=int(model_config["d_ff"]),
        dropout=float(model_config["dropout"]),
        weather_dim=weather_dim,
        categorical_cardinalities=categorical_cardinalities if use_metadata else (),
        categorical_indices=categorical_tensor if use_metadata else torch.empty(num_variables, 0, dtype=torch.long),
        numeric_static=numeric_tensor if use_metadata else torch.empty(num_variables, 0, dtype=torch.float32),
        type_indices=type_indices if use_metadata else None,
        num_building_types=num_building_types if use_metadata else None,
        metadata_adapter_type=metadata_adapter_type if use_metadata else "metadata",
        use_metadata_adapter=use_metadata_adapter,
        metadata_dropout=float(model_config.get("metadata_dropout", 0.0)) if use_metadata else 0.0,
        gate_l1_penalty=float(model_config.get("gate_l1_penalty", 0.0)) if use_metadata else 0.0,
        use_weather_response_adapter=use_weather_response_adapter,
        weather_response_dropout=(
            float(model_config.get("weather_response_dropout", 0.0)) if use_weather_response_adapter else 0.0
        ),
        weather_response_gate_l1_penalty=(
            float(model_config.get("weather_response_gate_l1_penalty", 0.0))
            if use_weather_response_adapter
            else 0.0
        ),
        use_weather_token_gating_adapter=use_weather_token_gating_adapter,
        weather_token_dropout=(
            float(model_config.get("weather_token_dropout", 0.0)) if use_weather_token_gating_adapter else 0.0
        ),
        weather_token_gate_l1_penalty=(
            float(model_config.get("weather_token_gate_l1_penalty", 0.0))
            if use_weather_token_gating_adapter
            else 0.0
        ),
        use_residual_output_gate=use_residual_output_gate,
        residual_output_gate_l1_penalty=(
            float(model_config.get("residual_output_gate_l1_penalty", 0.0))
            if use_residual_output_gate
            else 0.0
        ),
        residual_output_gate_use_time=bool(model_config.get("residual_output_gate_use_time", True)),
        use_relation_bias=False,
        variable_embedding_mode=variable_embedding_mode,
        use_metadata=use_metadata,
    )
    return model.to(device)


def transfer_compatible_state(source: nn.Module, target: nn.Module) -> list[str]:
    source_state = source.state_dict()
    target_state = target.state_dict()
    compatible = {}
    skipped = []
    for key, value in source_state.items():
        if key in target_state and target_state[key].shape == value.shape:
            compatible[key] = value
        else:
            skipped.append(key)
    target_state.update(compatible)
    target.load_state_dict(target_state)
    return skipped


def train_transfer_model(
    *,
    model_name: str,
    use_metadata: bool,
    config: dict[str, Any],
    support_loaders: dict[str, DataLoader],
    heldout_loaders: dict[str, DataLoader],
    support_scaler,
    heldout_scaler,
    support_tensors: dict[str, Any],
    heldout_tensors: dict[str, Any],
    categorical_cardinalities: list[int],
    num_building_types: int,
    weather_dim: int,
    support_meters: pd.DataFrame,
    heldout_meters: pd.DataFrame,
    device: torch.device,
    output_checkpoints: Path,
    run_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    training_config = config["training"]
    set_seed(int(training_config["seed"]))

    model = build_model(
        config=config,
        use_metadata=use_metadata,
        num_variables=len(support_meters),
        weather_dim=weather_dim,
        categorical_tensor=support_tensors["categorical"],
        categorical_cardinalities=categorical_cardinalities,
        numeric_tensor=support_tensors["numeric"],
        type_indices=support_tensors["type_indices"],
        num_building_types=num_building_types,
        device=device,
    )
    criterion, loss_name = make_criterion(training_config, support_scaler)
    criterion = criterion.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )

    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    wait = 0
    history_rows = []
    epochs = int(training_config["epochs"])
    patience = int(training_config["patience"])
    clip_grad_norm = float(training_config["clip_grad_norm"])

    for epoch in range(1, epochs + 1):
        train_loss, train_prediction_loss, train_regularization_loss = train_one_epoch(
            model,
            support_loaders["train"],
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            clip_grad_norm=clip_grad_norm,
        )
        val_true, val_pred = collect_predictions(
            model,
            support_loaders["val"],
            scaler=support_scaler,
            device=device,
        )
        val_metrics = regression_metrics(val_true, val_pred)
        val_norm_loss = 0.0
        total_count = 0
        model.eval()
        with torch.no_grad():
            for x, y, weather_context in support_loaders["val"]:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                weather_context = weather_context.to(device, non_blocking=True)
                pred = model(x, weather_context)
                loss = criterion(pred, y)
                batch_size = x.shape[0]
                val_norm_loss += float(loss.item()) * batch_size
                total_count += batch_size
        val_norm_loss /= max(total_count, 1)

        history_rows.append(
            {
                "model": model_name,
                "epoch": epoch,
                "train_loss": train_loss,
                "train_prediction_loss": train_prediction_loss,
                "train_regularization_loss": train_regularization_loss,
                "support_val_loss": val_norm_loss,
                "support_val_RMSE": val_metrics["RMSE"],
                "loss": loss_name,
                "variable_embedding_mode": str(config["model"].get("variable_embedding_mode", "none")),
            }
        )
        print(
            f"{model_name} epoch={epoch:02d} train_loss={train_loss:.6f} "
            f"train_pred={train_prediction_loss:.6f} train_reg={train_regularization_loss:.6f} "
            f"support_val_loss={val_norm_loss:.6f} support_val_RMSE={val_metrics['RMSE']:.6f}"
        )

        if val_norm_loss < best_val:
            best_val = val_norm_loss
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
            "best_support_val_mse_norm": best_val,
            "use_metadata": use_metadata,
            "loss": loss_name,
            "variable_embedding_mode": str(config["model"].get("variable_embedding_mode", "none")),
        },
        checkpoint_path,
    )

    heldout_model = build_model(
        config=config,
        use_metadata=use_metadata,
        num_variables=len(heldout_meters),
        weather_dim=weather_dim,
        categorical_tensor=heldout_tensors["categorical"],
        categorical_cardinalities=categorical_cardinalities,
        numeric_tensor=heldout_tensors["numeric"],
        type_indices=heldout_tensors["type_indices"],
        num_building_types=num_building_types,
        device=device,
    )
    skipped_state_keys = transfer_compatible_state(model, heldout_model)

    metric_rows = []
    horizon_rows = []
    per_building_rows = []
    prediction_outputs: dict[str, dict[str, np.ndarray]] = {}
    evaluation_plan = [
        ("support_val", model, support_loaders["val"], support_scaler, support_meters),
        ("support_test", model, support_loaders["test"], support_scaler, support_meters),
        ("heldout_val", heldout_model, heldout_loaders["val"], heldout_scaler, heldout_meters),
        ("heldout_test", heldout_model, heldout_loaders["test"], heldout_scaler, heldout_meters),
    ]
    for split_name, eval_model, loader, scaler, meter_frame in evaluation_plan:
        y_true, y_pred = collect_predictions(eval_model, loader, scaler=scaler, device=device)
        prediction_outputs[split_name] = {"y_true": y_true, "y_pred": y_pred}

        row = regression_metrics(y_true, y_pred)
        row.update(
            {
                "split": split_name,
                "model": model_name,
                "num_meters": len(meter_frame),
                "support_meters": len(support_meters),
                "heldout_meters": len(heldout_meters),
                "best_support_val_mse_norm": best_val,
                "loss": loss_name,
            }
        )
        metric_rows.append(row)

        horizon_frame = horizon_metrics(y_true, y_pred)
        horizon_frame["split"] = split_name
        horizon_frame["model"] = model_name
        horizon_rows.append(horizon_frame)

        per_building_rows.append(
            per_building_metrics(
                y_true=y_true,
                y_pred=y_pred,
                meter_ids=meter_frame["building_id"].tolist(),
                selected_meters=meter_frame,
                split=split_name,
                model_name=model_name,
            )
        )

    info = {
        "history": pd.DataFrame(history_rows),
        "checkpoint_path": str(checkpoint_path),
        "best_support_val_mse_norm": best_val,
        "skipped_state_keys": skipped_state_keys,
        "loss": loss_name,
    }
    return (
        pd.DataFrame(metric_rows),
        pd.concat(horizon_rows, ignore_index=True),
        pd.concat(per_building_rows, ignore_index=True),
        pd.DataFrame(history_rows),
        prediction_outputs,
        info,
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    training_config = config["training"]
    seed = int(training_config["seed"])
    set_seed(seed)

    data_dir = PROJECT_ROOT / config["data_dir"]
    splits = config["splits"]
    lookback_hours = int(config["lookback_hours"])
    forecast_horizon = int(config["forecast_horizon"])
    selection = config["selection"]
    weather_config = config["weather"]
    metadata_config = config["metadata"]
    unseen_config = config["unseen_split"]
    run_name = str(config.get("run_name", "building_aware_unseen_itransformer_250_noweather"))

    output_results = PROJECT_ROOT / "outputs" / "results"
    output_predictions = PROJECT_ROOT / "outputs" / "predictions"
    output_checkpoints = PROJECT_ROOT / "outputs" / "checkpoints"
    output_results.mkdir(parents=True, exist_ok=True)
    output_predictions.mkdir(parents=True, exist_ok=True)
    output_checkpoints.mkdir(parents=True, exist_ok=True)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Data dir: {data_dir}")
    print("Task: unseen-building iTransformer evaluation")

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
    support_meters, heldout_meters, selected_with_role = split_unseen_buildings(
        selected_meters,
        holdout_per_primary_use=int(unseen_config["holdout_per_primary_use"]),
        seed=int(unseen_config.get("seed", seed)),
    )
    selected_path = output_results / f"{run_name}_selected_meters.csv"
    selected_with_role.to_csv(selected_path, index=False)

    print("Selected meters by role and primary use:")
    print(selected_with_role.groupby(["evaluation_role", "primaryspaceusage"]).size().to_string())

    support_ids = support_meters["building_id"].tolist()
    heldout_ids = heldout_meters["building_id"].tolist()
    electricity_support = fill_meter_values(electricity_all[support_ids])
    electricity_heldout = fill_meter_values(electricity_all[heldout_ids])
    support_values = electricity_support.to_numpy(dtype="float32")
    heldout_values = electricity_heldout.to_numpy(dtype="float32")

    train_positions = inclusive_time_positions(
        electricity_support.index,
        start=splits["train"]["target_start"],
        end=splits["train"]["target_end"],
    )
    support_scaler = fit_meter_scaler(support_values, train_positions)
    heldout_scaler = fit_meter_scaler(heldout_values, train_positions)
    support_values_norm = support_scaler.transform(support_values).astype("float32")
    heldout_values_norm = heldout_scaler.transform(heldout_values).astype("float32")

    origin_positions = {
        split_name: split_origin_positions(
            electricity_support.index,
            target_start=split_cfg["target_start"],
            target_end=split_cfg["target_end"],
            lookback_hours=lookback_hours,
            forecast_horizon=forecast_horizon,
        )
        for split_name, split_cfg in splits.items()
    }

    weather_features = list(weather_config.get("features", []))
    weather_enabled = bool(weather_config.get("enabled", True)) and bool(weather_features) and (
        bool(weather_config.get("use_origin_weather", True))
        or bool(weather_config.get("use_future_weather_mean", True))
    )
    if weather_enabled:
        all_site_ids = selected_meters["site_id"].astype(str).tolist()
        weather, weather_info = load_weather_for_sites(
            data_dir,
            all_site_ids,
            weather_features=weather_features,
            reference_index=electricity_support.index,
        )
        support_weather_values = build_weather_values(
            weather,
            support_meters,
            weather_features=weather_features,
            reference_index=electricity_support.index,
        )
        heldout_weather_values = build_weather_values(
            weather,
            heldout_meters,
            weather_features=weather_features,
            reference_index=electricity_support.index,
        )
        weather_stats = fit_weather_stats(support_weather_values, train_positions)
        support_weather_norm = apply_weather_stats(support_weather_values, weather_stats)
        heldout_weather_norm = apply_weather_stats(heldout_weather_values, weather_stats)
        support_weather_context = make_weather_contexts(
            support_weather_norm,
            origin_positions=origin_positions,
            forecast_horizon=forecast_horizon,
            use_origin_weather=bool(weather_config.get("use_origin_weather", True)),
            use_future_weather_mean=bool(weather_config.get("use_future_weather_mean", True)),
        )
        heldout_weather_context = make_weather_contexts(
            heldout_weather_norm,
            origin_positions=origin_positions,
            forecast_horizon=forecast_horizon,
            use_origin_weather=bool(weather_config.get("use_origin_weather", True)),
            use_future_weather_mean=bool(weather_config.get("use_future_weather_mean", True)),
        )
        weather_dim = int(support_weather_context["train"].shape[-1])
        print("Weather context:")
        print(f"  sites: {weather_info['site_ids']}")
        print(f"  features: {weather_features}")
        print(f"  weather_dim: {weather_dim}")
    else:
        support_weather_context = {
            split_name: make_empty_weather_context(
                origin_positions=origin_positions[split_name],
                num_variables=len(support_meters),
            )
            for split_name in origin_positions
        }
        heldout_weather_context = {
            split_name: make_empty_weather_context(
                origin_positions=origin_positions[split_name],
                num_variables=len(heldout_meters),
            )
            for split_name in origin_positions
        }
        weather_dim = 0
        weather_info = {
            "enabled": False,
            "features": [],
            "note": "Weather branch disabled by config; unseen-building transfer is evaluated with load-only inputs.",
        }
        weather_stats = None
        print("Weather context: disabled")

    (
        support_categorical,
        heldout_categorical,
        categorical_cardinalities,
        support_numeric,
        heldout_numeric,
        metadata_info,
    ) = fit_transfer_metadata(
        support_meters,
        heldout_meters,
        categorical_columns=list(metadata_config["categorical_columns"]),
        numeric_columns=list(metadata_config["numeric_columns"]),
    )
    type_column = str(metadata_config.get("type_column", "primaryspaceusage"))
    support_type_indices, heldout_type_indices, num_building_types, type_map = fit_transfer_type_indices(
        support_meters,
        heldout_meters,
        type_column=type_column,
    )
    print("Metadata context:")
    print(f"  categorical_columns: {metadata_config['categorical_columns']}")
    print(f"  categorical_cardinalities: {categorical_cardinalities}")
    print(f"  numeric_dim: {support_numeric.shape[1]}")
    print(f"  type_column: {type_column}")
    print(f"  num_building_types: {num_building_types}")

    batch_size = int(training_config["batch_size"])
    num_workers = int(training_config.get("num_workers", 0))
    support_loaders = {}
    heldout_loaders = {}
    for split_name in ["train", "val", "test"]:
        _, loader = make_loader(
            support_values_norm,
            support_weather_context[split_name],
            origin_positions=origin_positions[split_name],
            lookback_hours=lookback_hours,
            forecast_horizon=forecast_horizon,
            batch_size=batch_size,
            shuffle=(split_name == "train"),
            num_workers=num_workers,
        )
        support_loaders[split_name] = loader
    for split_name in ["val", "test"]:
        _, loader = make_loader(
            heldout_values_norm,
            heldout_weather_context[split_name],
            origin_positions=origin_positions[split_name],
            lookback_hours=lookback_hours,
            forecast_horizon=forecast_horizon,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        heldout_loaders[split_name] = loader

    print(
        f"support_meters={len(support_meters)} heldout_meters={len(heldout_meters)} "
        f"train_samples={len(support_loaders['train'].dataset)} "
        f"heldout_test_samples={len(heldout_loaders['test'].dataset)}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    support_tensors = {
        "categorical": support_categorical,
        "numeric": support_numeric,
        "type_indices": support_type_indices,
    }
    heldout_tensors = {
        "categorical": heldout_categorical,
        "numeric": heldout_numeric,
        "type_indices": heldout_type_indices,
    }
    baseline_model_name = (
        str(config["model"].get("baseline_model_name")).strip()
        if config["model"].get("baseline_model_name")
        else ("GlobalITransformer-unseen-load-weather" if weather_enabled else "GlobalITransformer-unseen-load-only")
    )
    experiments = [
        (baseline_model_name, False),
        (
            str(config["model"].get("metadata_model_name", "GlobalITransformer-unseen-building-aware")),
            True,
        ),
    ]

    metric_frames = []
    horizon_frames = []
    per_building_frames = []
    history_frames = []
    heldout_prediction_map: dict[str, np.ndarray] = {}
    heldout_y_true = None
    transfer_info: dict[str, Any] = {}

    for model_name, use_metadata in experiments:
        print(f"\nTraining {model_name} on support buildings...")
        metrics, horizons, per_building, history, split_outputs, info = train_transfer_model(
            model_name=model_name,
            use_metadata=use_metadata,
            config=config,
            support_loaders=support_loaders,
            heldout_loaders=heldout_loaders,
            support_scaler=support_scaler,
            heldout_scaler=heldout_scaler,
            support_tensors=support_tensors,
            heldout_tensors=heldout_tensors,
            categorical_cardinalities=categorical_cardinalities,
            num_building_types=num_building_types,
            weather_dim=weather_dim,
            support_meters=support_meters,
            heldout_meters=heldout_meters,
            device=device,
            output_checkpoints=output_checkpoints,
            run_name=run_name,
        )
        metric_frames.append(metrics)
        horizon_frames.append(horizons)
        per_building_frames.append(per_building)
        history_frames.append(history)
        heldout_prediction_map[model_name] = split_outputs["heldout_test"]["y_pred"]
        if heldout_y_true is None:
            heldout_y_true = split_outputs["heldout_test"]["y_true"]
        transfer_info[model_name] = info

    metrics = pd.concat(metric_frames, ignore_index=True)
    metrics = metrics[
        [
            "split",
            "model",
            "num_meters",
            "support_meters",
            "heldout_meters",
            "MAE",
            "RMSE",
            "NMAE",
            "CVRMSE",
            "n",
            "best_support_val_mse_norm",
            "loss",
        ]
    ]
    metrics_path = output_results / f"{run_name}_metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    horizon_frame = pd.concat(horizon_frames, ignore_index=True)
    horizon_frame = horizon_frame[["split", "model", "horizon", "MAE", "RMSE", "NMAE", "CVRMSE", "n"]]
    horizon_path = output_results / f"{run_name}_horizon_metrics.csv"
    horizon_frame.to_csv(horizon_path, index=False)

    history = pd.concat(history_frames, ignore_index=True)
    history_path = output_results / f"{run_name}_training_log.csv"
    history.to_csv(history_path, index=False)

    per_building = pd.concat(per_building_frames, ignore_index=True)
    per_building_path = output_results / f"{run_name}_per_building_metrics.csv"
    per_building.to_csv(per_building_path, index=False)

    context_path = output_results / f"{run_name}_context.json"
    context_payload = {
        "weather_stats": {
            "mean": weather_stats["mean"].tolist() if weather_stats is not None else [],
            "std": weather_stats["std"].tolist() if weather_stats is not None else [],
            "note": (
                "Weather normalization is fitted on support-building train-period weather only."
                if weather_stats is not None
                else "Weather branch disabled."
            ),
        },
        "weather_info": weather_info,
        "metadata_info": metadata_info,
        "type_info": {"type_column": type_column, "type_map": type_map},
        "support_meter_ids": support_ids,
        "heldout_meter_ids": heldout_ids,
        "transfer_info": {
            model_name: {
                "checkpoint_path": info["checkpoint_path"],
                "best_support_val_mse_norm": info["best_support_val_mse_norm"],
                "skipped_state_keys": info["skipped_state_keys"],
            }
            for model_name, info in transfer_info.items()
        },
    }
    with context_path.open("w", encoding="utf-8") as handle:
        json.dump(context_payload, handle, indent=2)

    sample_path = output_predictions / f"{run_name}_heldout_test_h24_sample_predictions.csv"
    assert heldout_y_true is not None
    save_sample_predictions(
        output_path=sample_path,
        y_true=heldout_y_true,
        prediction_map=heldout_prediction_map,
        origin_positions=origin_positions["test"],
        time_index=electricity_support.index,
        meter_ids=heldout_ids,
        forecast_horizon=forecast_horizon,
    )

    print("\nMetrics:")
    print(metrics.to_string(index=False))
    print(f"\nSaved selected meters: {selected_path}")
    print(f"Saved training log: {history_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved horizon metrics: {horizon_path}")
    print(f"Saved per-building metrics: {per_building_path}")
    print(f"Saved context: {context_path}")
    print(f"Saved sample predictions: {sample_path}")


if __name__ == "__main__":
    main()
