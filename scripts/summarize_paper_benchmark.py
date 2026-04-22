from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize paper benchmark completion and available metrics.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "configs" / "paper_benchmark_250_manifest.json",
        help="Path to the benchmark manifest JSON.",
    )
    return parser.parse_args()


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    run_name = str(manifest.get("run_name", args.manifest.stem))

    rows: list[dict[str, Any]] = []
    for index, item in enumerate(manifest.get("experiments", []), start=1):
        metrics_rel = Path(item["metrics_file"])
        metrics_path = PROJECT_ROOT / metrics_rel
        row = {
            "order": index,
            "id": item["id"],
            "paper_role": item.get("paper_role", ""),
            "split": item.get("split", ""),
            "model": item.get("model", ""),
            "metrics_file": str(metrics_rel).replace("\\", "/"),
            "note": item.get("note", ""),
            "status": "missing_file",
            "num_rows": 0,
        }

        if metrics_path.exists():
            metrics = pd.read_csv(metrics_path)
            row["num_rows"] = int(len(metrics))
            row["status"] = "file_present"

            if item.get("model") and "model" in metrics.columns:
                metrics = metrics.loc[metrics["model"] == item["model"]].copy()
            if item.get("split") and "split" in metrics.columns:
                metrics = metrics.loc[metrics["split"] == item["split"]].copy()

            if len(metrics) == 1:
                row["status"] = "available"
                metric_row = metrics.iloc[0].to_dict()
                for key in ["num_meters", "MAE", "RMSE", "NMAE", "CVRMSE", "n"]:
                    if key in metric_row:
                        row[key] = metric_row[key]
            elif len(metrics) > 1:
                row["status"] = "multiple_rows"
            else:
                row["status"] = "row_missing"

        rows.append(row)

    summary = pd.DataFrame(rows)
    output_dir = PROJECT_ROOT / "outputs" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{run_name}_status.csv"
    summary.to_csv(output_path, index=False)

    available = int((summary["status"] == "available").sum())
    total = int(len(summary))
    print(f"Manifest: {args.manifest}")
    print(f"Available rows: {available}/{total}")
    print()
    display_columns = ["order", "paper_role", "model", "split", "status", "RMSE", "metrics_file"]
    existing_display_columns = [column for column in display_columns if column in summary.columns]
    print(summary[existing_display_columns].fillna("").to_string(index=False))
    print()
    print(f"Saved summary: {output_path}")


if __name__ == "__main__":
    main()
