from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset


class MultiMeterWindowDataset(Dataset):
    """Multivariate windows shaped for iTransformer: [time, buildings]."""

    def __init__(
        self,
        values_norm: np.ndarray,
        *,
        origin_positions: np.ndarray,
        lookback_hours: int,
        forecast_horizon: int,
    ) -> None:
        self.values_norm = values_norm.astype("float32", copy=False)
        self.origin_positions = origin_positions.astype("int64", copy=False)
        self.lookback_hours = lookback_hours
        self.forecast_horizon = forecast_horizon

    def __len__(self) -> int:
        return int(self.origin_positions.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        origin = int(self.origin_positions[idx])
        start = origin - self.lookback_hours + 1
        end = origin + 1
        target_start = origin + 1
        target_end = origin + self.forecast_horizon + 1

        x = self.values_norm[start:end, :]
        y = self.values_norm[target_start:target_end, :]
        return torch.from_numpy(x), torch.from_numpy(y)


class MultiMeterWindowContextDataset(Dataset):
    """Multivariate windows plus per-building dynamic context."""

    def __init__(
        self,
        values_norm: np.ndarray,
        weather_context: np.ndarray,
        *,
        origin_positions: np.ndarray,
        lookback_hours: int,
        forecast_horizon: int,
    ) -> None:
        self.values_norm = values_norm.astype("float32", copy=False)
        self.weather_context = weather_context.astype("float32", copy=False)
        self.origin_positions = origin_positions.astype("int64", copy=False)
        self.lookback_hours = lookback_hours
        self.forecast_horizon = forecast_horizon

    def __len__(self) -> int:
        return int(self.origin_positions.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        origin = int(self.origin_positions[idx])
        start = origin - self.lookback_hours + 1
        end = origin + 1
        target_start = origin + 1
        target_end = origin + self.forecast_horizon + 1

        x = self.values_norm[start:end, :]
        y = self.values_norm[target_start:target_end, :]
        context = self.weather_context[idx]
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(context)


class ITransformerLoadOnly(nn.Module):
    """Compact load-only iTransformer baseline."""

    def __init__(
        self,
        *,
        seq_len: int,
        num_variables: int,
        forecast_horizon: int,
        d_model: int,
        n_heads: int,
        num_layers: int,
        d_ff: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.num_variables = num_variables
        self.forecast_horizon = forecast_horizon

        self.value_embedding = nn.Linear(seq_len, d_model)
        self.variable_embedding = nn.Parameter(torch.zeros(1, num_variables, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, forecast_horizon),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.variable_embedding, std=0.02)
        nn.init.xavier_uniform_(self.value_embedding.weight)
        nn.init.zeros_(self.value_embedding.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = x.transpose(1, 2)
        tokens = self.value_embedding(tokens) + self.variable_embedding
        encoded = self.encoder(tokens)
        prediction = self.head(encoded)
        return prediction.transpose(1, 2)


class BuildingMetadataAdapter(nn.Module):
    """Building-aware token conditioning with static metadata."""

    def __init__(
        self,
        *,
        num_variables: int,
        d_model: int,
        categorical_cardinalities: list[int] | tuple[int, ...],
        categorical_indices: torch.Tensor,
        numeric_static: torch.Tensor,
        dropout: float,
        metadata_dropout: float = 0.0,
        gate_l1_penalty: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_variables = num_variables
        self.d_model = d_model
        self.metadata_dropout = float(metadata_dropout)
        self.gate_l1_penalty = float(gate_l1_penalty)
        self._last_gate_mean: torch.Tensor | None = None

        self.categorical_embeddings = nn.ModuleList(
            [nn.Embedding(cardinality, d_model) for cardinality in categorical_cardinalities]
        )
        self.register_buffer("categorical_indices", categorical_indices.long())
        self.register_buffer("numeric_static", numeric_static.float())

        numeric_dim = int(self.numeric_static.shape[1])
        self.numeric_embedding = nn.Linear(numeric_dim, d_model) if numeric_dim else None
        self.metadata_encoder = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gamma = nn.Linear(d_model, d_model)
        self.beta = nn.Linear(d_model, d_model)
        self.gate = nn.Linear(d_model, d_model)
        self.token_norm = nn.LayerNorm(d_model)
        self._init_weights()

    def _init_weights(self) -> None:
        for embedding in self.categorical_embeddings:
            nn.init.normal_(embedding.weight, mean=0.0, std=0.02)
        if self.numeric_embedding is not None:
            nn.init.xavier_uniform_(self.numeric_embedding.weight)
            nn.init.zeros_(self.numeric_embedding.bias)
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.0)

    def metadata_embedding(self) -> torch.Tensor:
        metadata = torch.zeros(
            self.num_variables,
            self.d_model,
            device=self.categorical_indices.device,
        )
        for column_idx, embedding in enumerate(self.categorical_embeddings):
            metadata = metadata + embedding(self.categorical_indices[:, column_idx])
        if self.numeric_embedding is not None:
            metadata = metadata + self.numeric_embedding(self.numeric_static)
        return self.metadata_encoder(metadata)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        metadata = self.metadata_embedding()
        gamma = self.gamma(metadata).unsqueeze(0)
        beta = self.beta(metadata).unsqueeze(0)
        gate = torch.sigmoid(self.gate(metadata)).unsqueeze(0)
        self._last_gate_mean = gate.mean()
        conditioned_delta = gate * (gamma * self.token_norm(tokens) + beta)
        if self.training and self.metadata_dropout > 0:
            keep_prob = max(1.0 - self.metadata_dropout, 1e-6)
            mask = torch.empty(
                tokens.shape[0],
                tokens.shape[1],
                1,
                device=tokens.device,
                dtype=tokens.dtype,
            ).bernoulli_(keep_prob)
            conditioned_delta = conditioned_delta * mask / keep_prob
        return tokens + conditioned_delta

    def regularization_loss(self) -> torch.Tensor:
        if self.gate_l1_penalty <= 0 or self._last_gate_mean is None:
            return self.numeric_static.sum() * 0.0
        return self.gate_l1_penalty * self._last_gate_mean


class BuildingAwareITransformer(nn.Module):
    """iTransformer with optional weather context and a metadata adapter."""

    def __init__(
        self,
        *,
        seq_len: int,
        num_variables: int,
        forecast_horizon: int,
        d_model: int,
        n_heads: int,
        num_layers: int,
        d_ff: int,
        dropout: float,
        weather_dim: int,
        categorical_cardinalities: list[int] | tuple[int, ...] = (),
        categorical_indices: torch.Tensor | None = None,
        numeric_static: torch.Tensor | None = None,
        metadata_dropout: float = 0.0,
        gate_l1_penalty: float = 0.0,
        use_metadata: bool = True,
        **_: object,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.num_variables = num_variables
        self.forecast_horizon = forecast_horizon
        self.weather_dim = weather_dim
        self.use_metadata = bool(use_metadata)

        self.value_embedding = nn.Linear(seq_len, d_model)
        self.variable_embedding = nn.Parameter(torch.zeros(1, num_variables, d_model))
        self.weather_embedding = (
            nn.Sequential(
                nn.Linear(weather_dim, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model),
            )
            if weather_dim > 0
            else None
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, forecast_horizon),
        )
        self._init_weights()

        if categorical_indices is None:
            categorical_indices = torch.empty(num_variables, 0, dtype=torch.long)
        if numeric_static is None:
            numeric_static = torch.empty(num_variables, 0, dtype=torch.float32)
        self.metadata_adapter = (
            BuildingMetadataAdapter(
                num_variables=num_variables,
                d_model=d_model,
                categorical_cardinalities=categorical_cardinalities,
                categorical_indices=categorical_indices,
                numeric_static=numeric_static,
                dropout=dropout,
                metadata_dropout=metadata_dropout,
                gate_l1_penalty=gate_l1_penalty,
            )
            if self.use_metadata
            else None
        )

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.variable_embedding, std=0.02)
        nn.init.xavier_uniform_(self.value_embedding.weight)
        nn.init.zeros_(self.value_embedding.bias)

    def forward(self, x: torch.Tensor, weather_context: torch.Tensor) -> torch.Tensor:
        tokens = x.transpose(1, 2)
        tokens = self.value_embedding(tokens) + self.variable_embedding
        if self.weather_embedding is not None:
            weather_tokens = self.weather_embedding(weather_context)
            tokens = tokens + weather_tokens
        if self.metadata_adapter is not None:
            tokens = self.metadata_adapter(tokens)
        encoded = self.encoder(tokens)
        prediction = self.head(encoded)
        return prediction.transpose(1, 2)

    def regularization_loss(self) -> torch.Tensor:
        if self.metadata_adapter is None:
            return self.variable_embedding.sum() * 0.0
        return self.metadata_adapter.regularization_loss()
