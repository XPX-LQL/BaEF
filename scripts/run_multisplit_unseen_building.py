from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_seed_list(value: str) -> list[int]:
    seeds = []
    for part in value.split(","):
        part = part.strip()
        if part:
            seeds.append(int(part))
    if not seeds:
        raise ValueError("At least one split seed is required.")
    return seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run unseen-building iTransformer experiments across multiple held-out split seeds."
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "building_aware_unseen_itransformer_250_noweather.json",
        help="Base unseen-building JSON config to copy and override.",
    )
    parser.add_argument(
        "--split-seeds",
        type=str,
        default="1,7,123,2024",
        help="Comma-separated held-out split seeds. The base config usually keeps split seed 42 as the reference run.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a split seed if its metrics file already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_config_path = args.base_config.resolve()
    with base_config_path.open("r", encoding="utf-8") as handle:
        base_config = json.load(handle)

    base_run_name = str(base_config["run_name"])
    generated_config_dir = PROJECT_ROOT / "outputs" / "generated_configs"
    generated_config_dir.mkdir(parents=True, exist_ok=True)

    for split_seed in parse_seed_list(args.split_seeds):
        run_name = f"{base_run_name}_split{split_seed}"
        metrics_path = PROJECT_ROOT / "outputs" / "results" / f"{run_name}_metrics.csv"
        if args.skip_existing and metrics_path.exists():
            print(f"Skipping split seed {split_seed}: {metrics_path} already exists.")
            continue

        config = json.loads(json.dumps(base_config))
        config["run_name"] = run_name
        config["unseen_split"]["seed"] = split_seed
        config_path = generated_config_dir / f"{run_name}.json"
        with config_path.open("w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2)

        print(f"\nRunning held-out split seed {split_seed} with config {config_path}...")
        subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "run_unseen_building_itransformer.py"),
                "--config",
                str(config_path),
            ],
            cwd=PROJECT_ROOT,
            check=True,
        )


if __name__ == "__main__":
    main()
