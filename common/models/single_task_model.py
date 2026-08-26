# -*- coding: utf-8 -*-
"""SingleTaskModel: General GCN と Finetuning が共有する単一タスクラッパー。

両モデルの差分は「重みの初期状態のみ」に閉じ込めるため、
学習対象は常に `block` の全パラメータ (requires_grad=True)。
"""
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
        """1_Training_SourceModel が保存した state_dict をブロックに読み込む。

        Source の MolecularGCN と GcnPropertyBlock は層構成・キーが完全一致するため
        `strict=True` でロード可能。失敗した場合は明示的に例外を投げる。
        """
        missing, unexpected = self.block.load_state_dict(source_state_dict, strict=True)
        # load_state_dict は strict=True なら IncompatibleKeys を返すが
        # キー不一致時には RuntimeError を投げる。念のため空チェックも残す。
        if missing or unexpected:
            raise RuntimeError(
                f"Source state_dict incompatible: missing={missing}, unexpected={unexpected}"
            )
