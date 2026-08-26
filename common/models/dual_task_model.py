# -*- coding: utf-8 -*-
"""SPRTL wrapper: train main_block; aux_block is a frozen source-weight reference."""
from __future__ import annotations

import torch

from .gcn_property_block import GcnPropertyBlock


class DualTaskModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.main_block = GcnPropertyBlock()
        self.aux_block = GcnPropertyBlock()

    def forward(self, data):
        return self.main_block(data)

    def load_aux_source_state(self, source_state_dict) -> None:
        """Load source weights into aux_block and freeze them."""
        missing, unexpected = self.aux_block.load_state_dict(source_state_dict, strict=True)
        if missing or unexpected:
            raise RuntimeError(
                f"Source state_dict incompatible: missing={missing}, unexpected={unexpected}"
            )
        for p in self.aux_block.parameters():
            p.requires_grad = False
