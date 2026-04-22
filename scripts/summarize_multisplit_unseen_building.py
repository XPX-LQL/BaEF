from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize multi-split unseen-building transfer runs."
    )
    parser.add_argument(
        "--base-run-name",
        type=str,
        default="building_aware_unseen_itransformer_250_noweather",
        help="Run name of the reference unseen-building split.",
    )
    parser.add_argument(
        "--base-split-seed",
        type=int,
        default=42,
        help="Held-out split seed used by the reference run name without the _split suffix.",
    )
    parser.add_argument(
        "--baseline-model",
        type=str,
        default="GlobalITransformer-unseen-load-only",
        help="Model name used as the unseen-building transfer baseline.",
    )
    parser.add_argument(
        "--building-aware-model",
        type=str,
        default="GlobalITransformer-unseen-building-aware",
        help="Model name used as the unseen-building building-aware variant.",
    )
    return parser.parse_args()


def read_metrics(path: Path, *, split_seed: int, run_name: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame.insert(0, "run_name", run_name)
    frame.insert(1, "split_seed", split_seed)
    return frame


def collect_metric_frames(base_run_name: str, base_split_seed: int) -> pd.DataFrame:
    frames = []
    base_path = RESULTS_DIR / f"{base_run_name}_metrics.csv"
    if base_path.exists():
        frames.append(read_metrics(base_path, split_seed=base_split_seed, run_name=base_run_name))

    pattern = re.compile(rf"^{re.escape(base_run_name)}_split(\d+)_metrics\.csv$")
    for path in sorted(RESULTS_DIR.glob(f"{base_run_name}_split*_metrics.csv")):
        match = pattern.match(path.name)
        if not match:
            continue
        split_seed = int(match.group(1))
        run_name = path.name.removesuffix("_metrics.csv")
        frames.append(read_metrics(path, split_seed=split_seed, run_name=run_name))

    if not frames:
        raise FileNotFoundError(f"No unseen-building metric files were found for base run '{base_run_name}'.")
    return pd.concat(frames, ignore_index=True, sort=False)


def collect_per_building_frames(base_run_name: str, base_split_seed: int) -> pd.DataFrame:
    frames = []
    base_path = RESULTS_DIR / f"{base_run_name}_per_building_metrics.csv"
    if base_path.exists():
        frame = pd.read_csv(base_path)
        frame.insert(0, "run_name", base_run_name)
        frame.insert(1, "split_seed", base_split_seed)
        frames.append(frame)

    pattern = re.compile(rf"^{re.escape(base_run_name)}_split(\d+)_per_building_metrics\.csv$")
    for path in sorted(RESULTS_DIR.glob(f"{base_run_name}_split*_per_building_metrics.csv")):
        match = pattern.match(path.name)
        if not match:
            continue
        split_seed = int(match.group(1))
        run_name = path.name.removesuffix("_per_building_metrics.csv")
        frame = pd.read_csv(path)
        frame.insert(0, "run_name", run_name)
        frame.insert(1, "split_seed", split_seed)
        frames.append(frame)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics.groupby(["split", "model"], dropna=False)
        .agg(
            num_split_seeds=("split_seed", "nunique"),
            MAE_mean=("MAE", "mean"),
            MAE_std=("MAE", "std"),
            RMSE_mean=("RMSE", "mean"),
            RMSE_std=("RMSE", "std"),
            RMSE_min=("RMSE", "min"),
            RMSE_max=("RMSE", "max"),
            NMAE_mean=("NMAE", "mean"),
            NMAE_std=("NMAE", "std"),
            CVRMSE_mean=("CVRMSE", "mean"),
            CVRMSE_std=("CVRMSE", "std"),
        )
        .reset_index()
        .sort_values(["split", "RMSE_mean", "model"])
    )


def summarize_deltas(metrics: pd.DataFrame, *, baseline_model: str, building_aware_model: str) -> pd.DataFrame:
    pairs = []
    for (split_seed, split), group in metrics.groupby(["split_seed", "split"]):
        by_model = group.set_index("model")
        if {baseline_model, building_aware_model}.issubset(by_model.index):
            baseline = by_model.loc[baseline_model]
            building_aware = by_model.loc[building_aware_model]
            pairs.append(
                {
                    "split_seed": split_seed,
                    "split": split,
                    "baseline_model": baseline_model,
                    "building_aware_model": building_aware_model,
                    "RMSE_delta_building_minus_baseline": float(building_aware["RMSE"] - baseline["RMSE"]),
                    "CVRMSE_delta_building_minus_baseline": float(
                        building_aware["CVRMSE"] - baseline["CVRMSE"]
                    ),
                    "MAE_delta_building_minus_baseline": float(building_aware["MAE"] - baseline["MAE"]),
                }
            )
    if not pairs:
        return pd.DataFrame(
            columns=[
                "split_seed",
                "split",
                "baseline_model",
                "building_aware_model",
                "RMSE_delta_building_minus_baseline",
                "CVRMSE_delta_building_minus_baseline",
                "MAE_delta_building_minus_baseline",
            ]
        )
    return pd.DataFrame(pairs).sort_values(["split", "split_seed"])


def summarize_type_deltas(
    per_building: pd.DataFrame,
    *,
    baseline_model: str,
    building_aware_model: str,
) -> pd.DataFrame:
    if per_building.empty:
        return pd.DataFrame()
    rows = []
    for group_key, group in per_building.groupby(["split_seed", "split", "primaryspaceusage"], dropna=False):
        split_seed, split, primary_use = group_key
        summary = (
            group.groupby("model")
            .agg(
                RMSE=("RMSE", "mean"),
                CVRMSE=("CVRMSE", "mean"),
                MAE=("MAE", "mean"),
                num_buildings=("building_id", "nunique"),
            )
        )
        if {baseline_model, building_aware_model}.issubset(summary.index):
            baseline = summary.loc[baseline_model]
            building_aware = summary.loc[building_aware_model]
            rows.append(
                {
                    "split_seed": split_seed,
                    "split": split,
                    "primaryspaceusage": primary_use,
                    "num_buildings": int(building_aware["num_buildings"]),
                    "RMSE_delta_building_minus_baseline": float(building_aware["RMSE"] - baseline["RMSE"]),
                    "CVRMSE_delta_building_minus_baseline": float(
                        building_aware["CVRMSE"] - baseline["CVRMSE"]
                    ),
                    "MAE_delta_building_minus_baseline": float(building_aware["MAE"] - baseline["MAE"]),
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["split", "primaryspaceusage", "split_seed"])


def main() -> None:
    args = parse_args()
    metrics = collect_metric_frames(args.base_run_name, args.base_split_seed)

    detail_path = RESULTS_DIR / f"{args.base_run_name}_multisplit_detail.csv"
    metrics.to_csv(detail_path, index=False)

    summary = summarize_metrics(metrics)
    summary_path = RESULTS_DIR / f"{args.base_run_name}_multisplit_summary.csv"
    summary.to_csv(summary_path, index=False)

    deltas = summarize_deltas(
        metrics,
        baseline_model=args.baseline_model,
        building_aware_model=args.building_aware_model,
    )
    delta_path = RESULTS_DIR / f"{args.base_run_name}_multisplit_deltas.csv"
    deltas.to_csv(delta_path, index=False)

    per_building = collect_per_building_frames(args.base_run_name, args.base_split_seed)
    type_deltas = summarize_type_deltas(
        per_building,
        baseline_model=args.baseline_model,
        building_aware_model=args.building_aware_model,
    )
    type_delta_path = RESULTS_DIR / f"{args.base_run_name}_multisplit_type_deltas.csv"
    type_deltas.to_csv(type_delta_path, index=False)

    print("Metric summary:")
    print(summary.to_string(index=False))
    print("\nPer-split deltas: building-aware minus baseline")
    print(deltas.to_string(index=False))
    if not deltas.empty:
        delta_summary = (
            deltas.groupby("split")
            .agg(
                num_split_seeds=("split_seed", "nunique"),
                RMSE_delta_mean=("RMSE_delta_building_minus_baseline", "mean"),
                RMSE_delta_std=("RMSE_delta_building_minus_baseline", "std"),
                win_rate=("RMSE_delta_building_minus_baseline", lambda values: float((values < 0).mean())),
            )
            .reset_index()
        )
        print("\nDelta summary:")
        print(delta_summary.to_string(index=False))

    if not type_deltas.empty:
        type_summary = (
            type_deltas.groupby(["split", "primaryspaceusage"], dropna=False)
            .agg(
                num_split_seeds=("split_seed", "nunique"),
                RMSE_delta_mean=("RMSE_delta_building_minus_baseline", "mean"),
                RMSE_delta_std=("RMSE_delta_building_minus_baseline", "std"),
                win_rate=("RMSE_delta_building_minus_baseline", lambda values: float((values < 0).mean())),
            )
            .reset_index()
            .sort_values(["split", "primaryspaceusage"])
        )
        type_summary_path = RESULTS_DIR / f"{args.base_run_name}_multisplit_type_summary.csv"
        type_summary.to_csv(type_summary_path, index=False)
        print("\nBuilding-type delta summary:")
        print(type_summary.to_string(index=False))
        print(f"Saved type summary: {type_summary_path}")

    print(f"\nSaved detail: {detail_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved deltas: {delta_path}")
    print(f"Saved type deltas: {type_delta_path}")


if __name__ == "__main__":
    main()
