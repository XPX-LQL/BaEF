from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize multi-seed seen-building building-aware iTransformer runs."
    )
    parser.add_argument(
        "--base-run-name",
        type=str,
        default="building_aware_itransformer_250_noweather",
        help="Run name of the reference seed run.",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=42,
        help="Seed used by the reference run name without the _seed suffix.",
    )
    parser.add_argument(
        "--baseline-model",
        type=str,
        default="GlobalITransformer-load-only",
        help="Model name used as the seen-building baseline.",
    )
    parser.add_argument(
        "--building-aware-model",
        type=str,
        default="GlobalITransformer-building-aware",
        help="Model name used as the building-aware variant.",
    )
    return parser.parse_args()


def read_metrics(path: Path, *, seed: int, run_name: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame.insert(0, "run_name", run_name)
    frame.insert(1, "seed", seed)
    return frame


def collect_frames(base_run_name: str, base_seed: int) -> pd.DataFrame:
    frames = []
    base_path = RESULTS_DIR / f"{base_run_name}_metrics.csv"
    if base_path.exists():
        frames.append(read_metrics(base_path, seed=base_seed, run_name=base_run_name))

    pattern = re.compile(rf"^{re.escape(base_run_name)}_seed(\d+)_metrics\.csv$")
    for path in sorted(RESULTS_DIR.glob(f"{base_run_name}_seed*_metrics.csv")):
        match = pattern.match(path.name)
        if not match:
            continue
        seed = int(match.group(1))
        run_name = path.name.removesuffix("_metrics.csv")
        frames.append(read_metrics(path, seed=seed, run_name=run_name))

    if not frames:
        raise FileNotFoundError(f"No multiseed metric files were found for base run '{base_run_name}'.")
    return pd.concat(frames, ignore_index=True, sort=False)


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics.groupby(["split", "model"], dropna=False)
        .agg(
            num_seeds=("seed", "nunique"),
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
    for (seed, split), group in metrics.groupby(["seed", "split"]):
        by_model = group.set_index("model")
        if {baseline_model, building_aware_model}.issubset(by_model.index):
            baseline = by_model.loc[baseline_model]
            building_aware = by_model.loc[building_aware_model]
            pairs.append(
                {
                    "seed": seed,
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
                "seed",
                "split",
                "baseline_model",
                "building_aware_model",
                "RMSE_delta_building_minus_baseline",
                "CVRMSE_delta_building_minus_baseline",
                "MAE_delta_building_minus_baseline",
            ]
        )
    return pd.DataFrame(pairs).sort_values(["split", "seed"])


def main() -> None:
    args = parse_args()
    metrics = collect_frames(args.base_run_name, args.base_seed)

    detail_path = RESULTS_DIR / f"{args.base_run_name}_multiseed_detail.csv"
    metrics.to_csv(detail_path, index=False)

    summary = summarize_metrics(metrics)
    summary_path = RESULTS_DIR / f"{args.base_run_name}_multiseed_summary.csv"
    summary.to_csv(summary_path, index=False)

    deltas = summarize_deltas(
        metrics,
        baseline_model=args.baseline_model,
        building_aware_model=args.building_aware_model,
    )
    delta_path = RESULTS_DIR / f"{args.base_run_name}_multiseed_deltas.csv"
    deltas.to_csv(delta_path, index=False)

    print("Metric summary:")
    print(summary.to_string(index=False))
    print("\nPer-seed deltas: building-aware minus baseline")
    print(deltas.to_string(index=False))
    if not deltas.empty:
        delta_summary = (
            deltas.groupby("split")
            .agg(
                num_seeds=("seed", "nunique"),
                RMSE_delta_mean=("RMSE_delta_building_minus_baseline", "mean"),
                RMSE_delta_std=("RMSE_delta_building_minus_baseline", "std"),
                win_rate=("RMSE_delta_building_minus_baseline", lambda values: float((values < 0).mean())),
            )
            .reset_index()
        )
        print("\nDelta summary:")
        print(delta_summary.to_string(index=False))

    print(f"\nSaved detail: {detail_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved deltas: {delta_path}")


if __name__ == "__main__":
    main()
