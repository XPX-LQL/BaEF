from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def load_single_meter(
    data_dir: Path,
    meter_id: str,
    *,
    fill_missing: bool = True,
) -> tuple[pd.Series, dict[str, Any]]:
    """Load one electricity meter from the wide BDG2 cleaned CSV."""
    electricity_path = data_dir / "electricity_cleaned.csv"
    if not electricity_path.exists():
        raise FileNotFoundError(f"Missing file: {electricity_path}")

    try:
        frame = pd.read_csv(
            electricity_path,
            usecols=["timestamp", meter_id],
            parse_dates=["timestamp"],
        )
    except ValueError as exc:
        header = pd.read_csv(electricity_path, nrows=0)
        suggestions = [col for col in header.columns if meter_id.lower() in col.lower()]
        hint = f" Similar columns: {suggestions[:10]}" if suggestions else ""
        raise ValueError(f"Meter '{meter_id}' was not found in {electricity_path}.{hint}") from exc

    frame = frame.rename(columns={meter_id: "load"})
    frame = frame.sort_values("timestamp").drop_duplicates(subset="timestamp")
    frame = frame.set_index("timestamp")

    full_index = pd.date_range(frame.index.min(), frame.index.max(), freq="h")
    frame = frame.reindex(full_index)
    frame.index.name = "timestamp"

    missing_before_fill = int(frame["load"].isna().sum())
    if fill_missing and missing_before_fill:
        frame["load"] = frame["load"].interpolate(method="time").ffill().bfill()

    series = frame["load"].astype("float32")
    info = {
        "meter_id": meter_id,
        "start": str(series.index.min()),
        "end": str(series.index.max()),
        "hours": int(series.shape[0]),
        "missing_before_fill": missing_before_fill,
        "missing_after_fill": int(series.isna().sum()),
    }
    return series, info


def load_electricity_wide(data_dir: Path, meter_ids: list[str] | None = None) -> pd.DataFrame:
    """Load BDG2 cleaned electricity data in wide format."""
    electricity_path = data_dir / "electricity_cleaned.csv"
    if not electricity_path.exists():
        raise FileNotFoundError(f"Missing file: {electricity_path}")

    usecols = None if meter_ids is None else ["timestamp", *meter_ids]
    frame = pd.read_csv(electricity_path, usecols=usecols, parse_dates=["timestamp"])
    frame = frame.sort_values("timestamp").drop_duplicates(subset="timestamp")
    frame = frame.set_index("timestamp")

    full_index = pd.date_range(frame.index.min(), frame.index.max(), freq="h")
    frame = frame.reindex(full_index)
    frame.index.name = "timestamp"
    return frame


def select_diverse_electricity_meters(
    electricity: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    max_meters: int,
    meters_per_primary_use: int,
    primary_uses: list[str] | None = None,
    min_completeness: float = 0.95,
    site_balanced: bool = False,
    max_meters_per_site: int | None = None,
) -> pd.DataFrame:
    """Select complete meters across several primary-use categories."""
    meter_ids = [column for column in electricity.columns if column != "timestamp"]
    completeness = electricity[meter_ids].notna().mean(axis=0).rename("completeness")

    candidates = metadata.loc[metadata["building_id"].isin(meter_ids)].copy()
    candidates = candidates.merge(completeness, left_on="building_id", right_index=True, how="left")
    candidates = candidates.loc[candidates["completeness"] >= min_completeness].copy()
    candidates["primaryspaceusage"] = candidates["primaryspaceusage"].fillna("Unknown")
    candidates = candidates.sort_values(["completeness", "building_id"], ascending=[False, True])

    selected_parts = []
    selected_ids: set[str] = set()
    if primary_uses is None:
        primary_uses = candidates["primaryspaceusage"].value_counts().index.tolist()

    for primary_use in primary_uses:
        primary_candidates = candidates.loc[
            (candidates["primaryspaceusage"] == primary_use) & (~candidates["building_id"].isin(selected_ids))
        ].copy()
        if site_balanced:
            part = select_site_balanced_rows(
                primary_candidates,
                max_rows=meters_per_primary_use,
                selected_ids=selected_ids,
                max_meters_per_site=max_meters_per_site,
            )
        else:
            part = primary_candidates.head(meters_per_primary_use)
        selected_parts.append(part)
        selected_ids.update(part["building_id"].tolist())

    selected = pd.concat(selected_parts, ignore_index=True) if selected_parts else pd.DataFrame()
    if len(selected) < max_meters:
        fill_candidates = candidates.loc[~candidates["building_id"].isin(selected_ids)].copy()
        if site_balanced:
            fill = select_site_balanced_rows(
                fill_candidates,
                max_rows=max_meters - len(selected),
                selected_ids=selected_ids,
                max_meters_per_site=max_meters_per_site,
            )
        else:
            fill = fill_candidates.head(max_meters - len(selected))
        selected = pd.concat([selected, fill], ignore_index=True)

    selected = selected.head(max_meters).copy()
    return selected.reset_index(drop=True)


def select_site_balanced_rows(
    candidates: pd.DataFrame,
    *,
    max_rows: int,
    selected_ids: set[str],
    max_meters_per_site: int | None,
) -> pd.DataFrame:
    """Round-robin candidate rows across sites while preserving completeness order."""
    if candidates.empty or max_rows <= 0:
        return candidates.head(0)

    selected = []
    site_counts: dict[str, int] = {}
    grouped = {
        str(site_id): group.reset_index(drop=True)
        for site_id, group in candidates.groupby("site_id", dropna=False, sort=True)
    }
    offsets = {site_id: 0 for site_id in grouped}

    while len(selected) < max_rows:
        added = False
        for site_id in sorted(grouped):
            if len(selected) >= max_rows:
                break
            if max_meters_per_site is not None and site_counts.get(site_id, 0) >= max_meters_per_site:
                continue

            group = grouped[site_id]
            offset = offsets[site_id]
            while offset < len(group):
                row = group.iloc[offset]
                offset += 1
                building_id = str(row["building_id"])
                if building_id in selected_ids:
                    continue
                selected.append(row)
                selected_ids.add(building_id)
                site_counts[site_id] = site_counts.get(site_id, 0) + 1
                added = True
                break
            offsets[site_id] = offset

        if not added:
            break

    return pd.DataFrame(selected)


def load_metadata(data_dir: Path) -> pd.DataFrame:
    metadata_path = data_dir / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing file: {metadata_path}")
    return pd.read_csv(metadata_path)


def get_meter_metadata(metadata: pd.DataFrame, meter_id: str) -> dict[str, Any]:
    rows = metadata.loc[metadata["building_id"] == meter_id]
    if rows.empty:
        return {}
    row = rows.iloc[0]
    fields = [
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
    ]
    return {field: row[field] for field in fields if field in row.index}


def load_site_weather(
    data_dir: Path,
    site_id: str,
    *,
    weather_features: list[str] | tuple[str, ...],
    reference_index: pd.DatetimeIndex | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load and impute hourly BDG2 weather for one site."""
    weather_path = data_dir / "weather.csv"
    if not weather_path.exists():
        raise FileNotFoundError(f"Missing file: {weather_path}")

    usecols = ["timestamp", "site_id", *weather_features]
    frame = pd.read_csv(weather_path, usecols=usecols, parse_dates=["timestamp"])
    frame = frame.loc[frame["site_id"] == site_id].copy()
    if frame.empty:
        raise ValueError(f"No weather rows found for site_id='{site_id}' in {weather_path}")

    frame = frame.sort_values("timestamp").drop_duplicates(subset="timestamp")
    frame = frame.set_index("timestamp")
    frame = frame.drop(columns=["site_id"])

    if reference_index is None:
        full_index = pd.date_range(frame.index.min(), frame.index.max(), freq="h")
    else:
        full_index = reference_index
    frame = frame.reindex(full_index)
    frame.index.name = "timestamp"

    missing_before_fill = {column: int(frame[column].isna().sum()) for column in frame.columns}
    frame = frame.apply(pd.to_numeric, errors="coerce")
    frame = frame.interpolate(method="time", limit_direction="both").ffill().bfill()
    missing_after_fill = {column: int(frame[column].isna().sum()) for column in frame.columns}

    info = {
        "site_id": site_id,
        "start": str(frame.index.min()),
        "end": str(frame.index.max()),
        "hours": int(frame.shape[0]),
        "features": list(frame.columns),
        "missing_before_fill": missing_before_fill,
        "missing_after_fill": missing_after_fill,
    }
    return frame.astype("float32"), info


def load_weather_for_sites(
    data_dir: Path,
    site_ids: list[str],
    *,
    weather_features: list[str] | tuple[str, ...],
    reference_index: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load and impute hourly weather for several sites as a MultiIndex frame."""
    weather_path = data_dir / "weather.csv"
    if not weather_path.exists():
        raise FileNotFoundError(f"Missing file: {weather_path}")

    site_ids = sorted({str(site_id) for site_id in site_ids})
    usecols = ["timestamp", "site_id", *weather_features]
    raw = pd.read_csv(weather_path, usecols=usecols, parse_dates=["timestamp"])
    raw["site_id"] = raw["site_id"].astype(str)
    raw = raw.loc[raw["site_id"].isin(site_ids)].copy()

    frames = []
    missing_before: dict[str, dict[str, int]] = {}
    missing_after: dict[str, dict[str, int]] = {}
    for site_id in site_ids:
        site_frame = raw.loc[raw["site_id"] == site_id].copy()
        if site_frame.empty:
            continue

        site_frame = site_frame.sort_values("timestamp").drop_duplicates(subset="timestamp")
        site_frame = site_frame.set_index("timestamp").drop(columns=["site_id"])
        site_frame = site_frame.reindex(reference_index)
        site_frame.index.name = "timestamp"
        site_frame = site_frame.apply(pd.to_numeric, errors="coerce")

        missing_before[site_id] = {column: int(site_frame[column].isna().sum()) for column in site_frame.columns}
        site_frame = site_frame.interpolate(method="time", limit_direction="both").ffill().bfill()
        missing_after[site_id] = {column: int(site_frame[column].isna().sum()) for column in site_frame.columns}
        site_frame["site_id"] = site_id
        frames.append(site_frame.reset_index())

    if not frames:
        raise ValueError(f"No weather rows found for sites: {site_ids}")

    weather = pd.concat(frames, ignore_index=True)
    weather = weather.set_index(["site_id", "timestamp"]).sort_index()

    info = {
        "site_ids": site_ids,
        "features": list(weather_features),
        "rows": int(weather.shape[0]),
        "missing_before_fill": missing_before,
        "missing_after_fill": missing_after,
    }
    return weather.astype("float32"), info
