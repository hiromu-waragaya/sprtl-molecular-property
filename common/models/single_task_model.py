# -*- coding: utf-8 -*-
from __future__ import annotations

import torch

from .gcn_property_block import GcnPropertyBlock


class SingleTaskModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = GcnPropertyBlock()

    def forward(self, data):
        return self.block(data)

    def load_source_state(self, source_state_dict) -> None:
        missing, unexpected = self.block.load_state_dict(source_state_dict, strict=True)
        if missing or unexpected:
            raise RuntimeError(
                f"Source state_dict incompatible: missing={missing}, unexpected={unexpected}"
            )
