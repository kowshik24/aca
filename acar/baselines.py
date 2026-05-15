from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn

from .models import InterventionSpec


class StaticIITBaseline(nn.Module):
    """Per-step static intervention parameters (non-amortized baseline)."""

    def __init__(self, max_prog_len: int, d_model: int, num_layers: int, max_positions: int):
        super().__init__()
        self.layer_read = nn.Parameter(torch.randint(0, num_layers, (max_prog_len,), dtype=torch.long), requires_grad=False)
        self.layer_write = nn.Parameter(torch.randint(0, num_layers, (max_prog_len,), dtype=torch.long), requires_grad=False)
        self.position_write = nn.Parameter(torch.randint(0, max_positions, (max_prog_len,), dtype=torch.long), requires_grad=False)
        self.gain = nn.Parameter(torch.randn(max_prog_len, d_model) * 0.01)
        self.bias = nn.Parameter(torch.zeros(max_prog_len, d_model))

    def forward(self, prog_ids: torch.Tensor) -> InterventionSpec:
        bsz, seq = prog_ids.shape
        layer_read = self.layer_read[:seq].repeat(bsz)
        layer_write = self.layer_write[:seq].repeat(bsz)
        position_write = self.position_write[:seq].repeat(bsz)
        gain = self.gain[:seq].repeat(bsz, 1)
        bias = self.bias[:seq].repeat(bsz, 1)
        return InterventionSpec(layer_read=layer_read, layer_write=layer_write, position_write=position_write, gain=gain, bias=bias)
