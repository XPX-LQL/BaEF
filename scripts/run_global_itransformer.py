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

from building_aware.data_utils import load_electricity_wide, load_metadata, select_diverse_electricity_meters
from building_aware.itransformer import ITransformerLoadOnly, MultiMeterWindowDataset
from building_aware.lstm import fit_meter_scaler
from building_aware.metrics import horizon_metrics, regression_metrics
from building_aware.time_splits import inclusive_time_positions, split_origin_positions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run global multi-building iTransformer baseline.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "global_itransformer_250.json",
        help="Path to the JSON experiment config.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def get_run_name(config: dict[str, Any], default: str) -> str:
    value = str(config.get("run_name", default)).strip()
    return value or default


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


def make_loader(
    values_norm: np.ndarray,
    *,
    origin_positions: np.ndarray,
    lookback_hours: int,
    forecast_horizon: int,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> tuple[MultiMeterWindowDataset, DataLoader]:
    dataset = MultiMeterWindowDataset(
        values_norm,
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
) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        if clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        optimizer.step()

        batch_size = x.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size
    return total_loss / max(total_count, 1)


def evaluate_norm_loss(
    model: nn.Module,
    loader: DataLoader,
    *,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            pred = model(x)
            loss = criterion(pred, y)
            batch_size = x.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_count += batch_size
    return total_loss / max(total_count, 1)


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
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            pred = model(x).cpu().numpy()
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
    y_pred: np.ndarray,
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
            "GlobalITransformer-load-only": y_pred[:, horizon_idx, sample_meter_idx],
        }
    )
    start = frame["target_timestamp"].min()
    frame = frame.loc[frame["target_timestamp"] < start + pd.Timedelta(days=7)].copy()
    frame.to_csv(output_path, index=False)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    run_name = get_run_name(config, "global_itransformer")

    training_config = config["training"]
    seed = int(training_config["seed"])
    set_seed(seed)

    data_dir = PROJECT_ROOT / config["data_dir"]
    splits = config["splits"]
    lookback_hours = int(config["lookback_hours"])
    forecast_horizon = int(config["forecast_horizon"])
    selection = config["selection"]

    output_results = PROJECT_ROOT / "outputs" / "results"
    output_predictions = PROJECT_ROOT / "outputs" / "predictions"
    output_checkpoints = PROJECT_ROOT / "outputs" / "checkpoints"
    output_results.mkdir(parents=True, exist_ok=True)
    output_predictions.mkdir(parents=True, exist_ok=True)
    output_checkpoints.mkdir(parents=True, exist_ok=True)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Data dir: {data_dir}")
    print(f"Task: global iTransformer load-only, past {lookback_hours} hours -> next {forecast_horizon} hours")

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
    num_variables = len(meter_ids)

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

    batch_size = int(training_config["batch_size"])
    num_workers = int(training_config.get("num_workers", 0))
    train_dataset, train_loader = make_loader(
        values_norm,
        origin_positions=origin_positions["train"],
        lookback_hours=lookback_hours,
        forecast_horizon=forecast_horizon,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_dataset, val_loader = make_loader(
        values_norm,
        origin_positions=origin_positions["val"],
        lookback_hours=lookback_hours,
        forecast_horizon=forecast_horizon,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_dataset, test_loader = make_loader(
        values_norm,
        origin_positions=origin_positions["test"],
        lookback_hours=lookback_hours,
        forecast_horizon=forecast_horizon,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    print(f"train: samples={len(train_dataset)}, targets={len(train_dataset) * forecast_horizon * num_variables}")
    print(f"val: samples={len(val_dataset)}, targets={len(val_dataset) * forecast_horizon * num_variables}")
    print(f"test: samples={len(test_dataset)}, targets={len(test_dataset) * forecast_horizon * num_variables}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model_config = config["model"]
    model = ITransformerLoadOnly(
        seq_len=lookback_hours,
        num_variables=num_variables,
        forecast_horizon=forecast_horizon,
        d_model=int(model_config["d_model"]),
        n_heads=int(model_config["n_heads"]),
        num_layers=int(model_config["num_layers"]),
        d_ff=int(model_config["d_ff"]),
        dropout=float(model_config["dropout"]),
    ).to(device)

    criterion = nn.MSELoss()
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
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            clip_grad_norm=clip_grad_norm,
        )
        val_loss = evaluate_norm_loss(model, val_loader, criterion=criterion, device=device)
        history_rows.append({"epoch": epoch, "train_mse_norm": train_loss, "val_mse_norm": val_loss})
        print(f"epoch={epoch:02d} train_mse_norm={train_loss:.6f} val_mse_norm={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    model.load_state_dict(best_state)
    checkpoint_path = output_checkpoints / f"{run_name}_load_only.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "meter_ids": meter_ids,
            "scaler_mean": scaler.mean,
            "scaler_std": scaler.std,
            "best_val_mse_norm": best_val,
        },
        checkpoint_path,
    )

    history = pd.DataFrame(history_rows)
    history_path = output_results / f"{run_name}_training_log.csv"
    history.to_csv(history_path, index=False)

    model_name = "GlobalITransformer-load-only"
    split_outputs = {}
    for split_name, loader in [("val", val_loader), ("test", test_loader)]:
        y_true, y_pred = collect_predictions(model, loader, scaler=scaler, device=device)
        split_outputs[split_name] = {"y_true": y_true, "y_pred": y_pred}

    metric_rows = []
    horizon_rows = []
    per_building_rows = []
    for split_name, split_output in split_outputs.items():
        y_true = split_output["y_true"]
        y_pred = split_output["y_pred"]

        row = regression_metrics(y_true, y_pred)
        row.update({"split": split_name, "model": model_name, "num_meters": len(meter_ids), "best_val_mse_norm": best_val})
        metric_rows.append(row)

        horizon_frame = horizon_metrics(y_true, y_pred)
        horizon_frame["split"] = split_name
        horizon_frame["model"] = model_name
        horizon_rows.append(horizon_frame)

        per_building_rows.append(
            per_building_metrics(
                y_true=y_true,
                y_pred=y_pred,
                meter_ids=meter_ids,
                selected_meters=selected_meters,
                split=split_name,
                model_name=model_name,
            )
        )

    metrics = pd.DataFrame(metric_rows)
    metrics = metrics[["split", "model", "num_meters", "MAE", "RMSE", "NMAE", "CVRMSE", "n", "best_val_mse_norm"]]
    metrics_path = output_results / f"{run_name}_metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    horizon_frame = pd.concat(horizon_rows, ignore_index=True)
    horizon_frame = horizon_frame[["split", "model", "horizon", "MAE", "RMSE", "NMAE", "CVRMSE", "n"]]
    horizon_path = output_results / f"{run_name}_horizon_metrics.csv"
    horizon_frame.to_csv(horizon_path, index=False)

    per_building = pd.concat(per_building_rows, ignore_index=True)
    per_building_path = output_results / f"{run_name}_per_building_metrics.csv"
    per_building.to_csv(per_building_path, index=False)

    sample_path = output_predictions / f"{run_name}_test_h24_sample_predictions.csv"
    save_sample_predictions(
        output_path=sample_path,
        y_true=split_outputs["test"]["y_true"],
        y_pred=split_outputs["test"]["y_pred"],
        origin_positions=origin_positions["test"],
        time_index=electricity.index,
        meter_ids=meter_ids,
        forecast_horizon=forecast_horizon,
    )

    print("\nMetrics:")
    print(metrics.to_string(index=False))
    print(f"\nSaved selected meters: {selected_path}")
    print(f"Saved training log: {history_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved horizon metrics: {horizon_path}")
    print(f"Saved per-building metrics: {per_building_path}")
    print(f"Saved sample predictions: {sample_path}")
    print(f"Saved checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
