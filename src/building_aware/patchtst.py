from __future__ import annotations

import torch
from torch import nn


class PatchTSTLoadOnly(nn.Module):
    """A compact univariate PatchTST-style baseline for load-only forecasting."""

    def __init__(
        self,
        *,
        seq_len: int,
        patch_len: int,
        stride: int,
        d_model: int,
        n_heads: int,
        num_layers: int,
        d_ff: int,
        dropout: float,
        forecast_horizon: int,
    ) -> None:
        super().__init__()
        if seq_len < patch_len:
            raise ValueError("seq_len must be >= patch_len.")

        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.num_patches = ((seq_len - patch_len) // stride) + 1

        self.patch_embedding = nn.Linear(patch_len, d_model)
        self.position_embedding = nn.Parameter(torch.zeros(1, self.num_patches, d_model))
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
            nn.Flatten(start_dim=1),
            nn.Dropout(dropout),
            nn.Linear(self.num_patches * d_model, forecast_horizon),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        nn.init.xavier_uniform_(self.patch_embedding.weight)
        nn.init.zeros_(self.patch_embedding.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, 1]
        x = x.squeeze(-1)
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        tokens = self.patch_embedding(patches)
        tokens = tokens + self.position_embedding
        encoded = self.encoder(tokens)
        return self.head(encoded)
