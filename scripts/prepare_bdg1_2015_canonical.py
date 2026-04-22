from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_START = pd.Timestamp("2015-01-01 00:00:00")
CANONICAL_END = pd.Timestamp("2015-12-31 23:00:00")
CANONICAL_INDEX = pd.date_range(CANONICAL_START, CANONICAL_END, freq="h", name="timestamp")

WEATHER_COLUMN_MAP = {
    "TemperatureC": "airTemperature",
    "Dew PointC": "dewTemperature",
    "Precipitationmm": "precipDepth1HR",
    "Sea Level PressurehPa": "seaLvlPressure",
    "WindDirDegrees": "windDirection",
    "Wind SpeedKm/h": "windSpeed",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare BDG1 into BDG2-like canonical files for 2015 full-year experiments.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "Building Data Genome Project 1",
        help="Path to the original BDG1 folder.",
    )
    return parser.parse_args()


def load_metadata(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "meta_open.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def load_building_frame(data_dir: Path, building_id: str) -> pd.DataFrame:
    path = data_dir / f"{building_id}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    frame = pd.read_csv(path, parse_dates=["timestamp"])
    frame = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
    return frame


def get_modal_year(timestamps: pd.DatetimeIndex) -> int:
    return int(pd.Series(timestamps.year).mode().iloc[0])


def is_eligible_2015_full_year(frame: pd.DataFrame, building_id: str) -> bool:
    if len(frame) != len(CANONICAL_INDEX):
        return False
    expected_columns = ["timestamp", building_id]
    if list(frame.columns) != expected_columns:
        return False
    timestamps = pd.DatetimeIndex(frame["timestamp"])
    return get_modal_year(timestamps) == 2015


def build_canonical_electricity(data_dir: Path, metadata: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    series_map: dict[str, np.ndarray] = {}
    eligible_rows = []

    for row in metadata.itertuples(index=False):
        building_id = str(row.uid)
        frame = load_building_frame(data_dir, building_id)
        if not is_eligible_2015_full_year(frame, building_id):
            continue
        values = pd.to_numeric(frame[building_id], errors="coerce").astype("float32").to_numpy(dtype="float32")
        if values.shape[0] != len(CANONICAL_INDEX):
            continue
        series_map[building_id] = values

        row_dict = row._asdict()
        timestamps = pd.DatetimeIndex(frame["timestamp"])
        row_dict["window_start"] = str(timestamps.min())
        row_dict["window_end"] = str(timestamps.max())
        eligible_rows.append(row_dict)

    if not series_map:
        raise ValueError("No eligible 2015 full-year BDG1 buildings were found.")

    electricity = pd.DataFrame(series_map, index=CANONICAL_INDEX)
    electricity.index.name = "timestamp"
    electricity = electricity.interpolate(method="time", limit_direction="both").ffill().bfill().astype("float32")
    eligible_metadata = pd.DataFrame(eligible_rows)
    return electricity, eligible_metadata


def make_canonical_metadata(eligible_metadata: pd.DataFrame) -> pd.DataFrame:
    frame = eligible_metadata.copy()
    frame["building_id"] = frame["uid"].astype(str)
    frame["site_id"] = frame["newweatherfilename"].astype(str).str.replace(".csv", "", regex=False)
    frame["primaryspaceusage"] = frame["primaryspaceusage"].fillna("Unknown")
    frame["sub_primaryspaceusage"] = (
        frame["subindustry"].fillna(frame["primaryspaceuse_abbrev"]).fillna(frame["primaryspaceusage"]).astype(str)
    )
    frame["timezone"] = frame["timezone"].fillna("Unknown")
    frame["sqm"] = pd.to_numeric(frame["sqm"], errors="coerce")
    frame["yearbuilt"] = pd.to_numeric(frame["yearbuilt"], errors="coerce")
    frame["numberoffloors"] = pd.to_numeric(frame["numberoffloors"], errors="coerce")

    columns = [
        "building_id",
        "site_id",
        "primaryspaceusage",
        "sub_primaryspaceusage",
        "sqm",
        "timezone",
        "yearbuilt",
        "numberoffloors",
        "industry",
        "subindustry",
        "nickname",
        "primaryspaceuse_abbrev",
        "newweatherfilename",
        "annualschedule",
        "window_start",
        "window_end",
    ]
    return frame[columns].sort_values("building_id").reset_index(drop=True)


def build_canonical_weather(data_dir: Path, metadata: pd.DataFrame) -> pd.DataFrame:
    site_to_weather = metadata[["site_id", "newweatherfilename", "window_start", "window_end"]].copy()
    site_to_weather = (
        site_to_weather.groupby("site_id", as_index=False)
        .agg(
            newweatherfilename=("newweatherfilename", "first"),
            window_start=("window_start", lambda values: pd.Series(values).mode().iloc[0]),
            window_end=("window_end", lambda values: pd.Series(values).mode().iloc[0]),
        )
        .sort_values("site_id")
        .reset_index(drop=True)
    )
    frames = []

    for row in site_to_weather.itertuples(index=False):
        weather_path = data_dir / str(row.newweatherfilename)
        if not weather_path.exists():
            raise FileNotFoundError(f"Missing file: {weather_path}")

        weather = pd.read_csv(weather_path, parse_dates=["timestamp"])
        weather = weather.sort_values("timestamp").drop_duplicates(subset="timestamp")
        weather = weather.set_index("timestamp")
        site_index = pd.date_range(pd.Timestamp(row.window_start), pd.Timestamp(row.window_end), freq="h", name="timestamp")

        numeric = pd.DataFrame(index=weather.index)
        for source_column, target_column in WEATHER_COLUMN_MAP.items():
            if source_column not in weather.columns:
                raise ValueError(f"Missing weather column '{source_column}' in {weather_path}")
            numeric[target_column] = pd.to_numeric(weather[source_column], errors="coerce")

        hourly = numeric.resample("h").mean()
        hourly = hourly.reindex(site_index)
        hourly = hourly.interpolate(method="time", limit_direction="both").ffill().bfill()
        if len(hourly) != len(CANONICAL_INDEX):
            raise ValueError(f"Site '{row.site_id}' produced {len(hourly)} hours instead of {len(CANONICAL_INDEX)}.")
        hourly.index = CANONICAL_INDEX
        hourly = hourly.reset_index()
        hourly["site_id"] = str(row.site_id)
        frames.append(hourly[["timestamp", "site_id", *WEATHER_COLUMN_MAP.values()]])

    canonical_weather = pd.concat(frames, ignore_index=True)
    canonical_weather = canonical_weather.sort_values(["site_id", "timestamp"]).reset_index(drop=True)
    return canonical_weather.astype(
        {
            "airTemperature": "float32",
            "dewTemperature": "float32",
            "precipDepth1HR": "float32",
            "seaLvlPressure": "float32",
            "windDirection": "float32",
            "windSpeed": "float32",
        }
    )


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir

    metadata_raw = load_metadata(data_dir)
    electricity, eligible_metadata = build_canonical_electricity(data_dir, metadata_raw)
    metadata = make_canonical_metadata(eligible_metadata)
    weather = build_canonical_weather(data_dir, metadata)

    electricity_path = data_dir / "electricity_cleaned.csv"
    metadata_path = data_dir / "metadata.csv"
    weather_path = data_dir / "weather.csv"

    electricity.reset_index().to_csv(electricity_path, index=False)
    metadata.to_csv(metadata_path, index=False)
    weather.to_csv(weather_path, index=False)

    print(f"Prepared BDG1 canonical files in: {data_dir}")
    print(f"Eligible full-year 2015 buildings: {len(metadata)}")
    print("Primary-use counts:")
    print(metadata["primaryspaceusage"].value_counts().to_string())
    print("Site counts:")
    print(metadata["site_id"].value_counts().to_string())
    print(f"Saved electricity: {electricity_path}")
    print(f"Saved metadata: {metadata_path}")
    print(f"Saved weather: {weather_path}")


if __name__ == "__main__":
    main()
