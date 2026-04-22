from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset


@dataclass
class MeterScaler:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean.reshape(1, -1)) / self.std.reshape(1, -1)

    def inverse_transform_targets(self, values: np.ndarray, meter_indices: np.ndarray) -> np.ndarray:
        means = self.mean[meter_indices].reshape(-1, 1)
        stds = self.std[meter_indices].reshape(-1, 1)
        return values * stds + means


def fit_meter_scaler(values: np.ndarray, train_positions: np.ndarray) -> MeterScaler:
    train_values = values[train_positions]
    mean = np.nanmean(train_values, axis=0).astype("float32")
    std = np.nanstd(train_values, axis=0).astype("float32")
    std = np.where(std < 1e-6, 1.0, std).astype("float32")
    return MeterScaler(mean=mean, std=std)


def make_sample_index(num_meters: int, origin_positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    meter_indices = np.tile(np.arange(num_meters, dtype="int64"), len(origin_positions))
    sample_origins = np.repeat(origin_positions.astype("int64"), num_meters)
    return meter_indices, sample_origins


class MultiMeterSequenceDataset(Dataset):
    def __init__(
        self,
        values_norm: np.ndarray,
        *,
        origin_positions: np.ndarray,
        lookback_hours: int,
        forecast_horizon: int,
    ) -> None:
        self.values_norm = values_norm.astype("float32", copy=False)
        self.lookback_hours = lookback_hours
        self.forecast_horizon = forecast_horizon
        self.num_meters = values_norm.shape[1]
        self.meter_indices, self.sample_origins = make_sample_index(self.num_meters, origin_positions)

    def __len__(self) -> int:
        return int(self.sample_origins.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        meter_idx = int(self.meter_indices[idx])
        origin = int(self.sample_origins[idx])

        start = origin - self.lookback_hours + 1
        end = origin + 1
        target_start = origin + 1
        target_end = origin + self.forecast_horizon + 1

        x = self.values_norm[start:end, meter_idx].reshape(self.lookback_hours, 1)
        y = self.values_norm[target_start:target_end, meter_idx]
        return (
            torch.from_numpy(x),
            torch.from_numpy(y.astype("float32", copy=False)),
            torch.tensor(meter_idx, dtype=torch.long),
        )


class LoadOnlyLSTM(nn.Module):
    def __init__(
        self,
        *,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        forecast_horizon: int,
    ) -> None:
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, forecast_horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.encoder(x)
        return self.head(hidden[-1])
