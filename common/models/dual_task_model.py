# -*- coding: utf-8 -*-
"""DualTaskModel: Soft Transfer 用ラッパー。

設計上の最重要ポイント:
1. forward は **main_block の出力のみ** を返す (B, 1)。aux_block は
   forward 経路から完全に切り離されている。
2. aux_block は Source の重みを保持する「参照テーブル」であり、
   soft_sharing loss の計算時にのみ参照される。
3. aux_block の全パラメータは `requires_grad=False` に設定する。

→ λ=0 のときに soft_sharing 項が 0 となり、loss が
   SingleTaskModel と完全に同一 (= MSE(main_block(data), y)) となる。
→ また aux_block のパラメータは forward に出ないため、auxi の種類
   (alpha / zpve / gap / r2 / cv 等) が変わっても λ=0 では main の挙動に
   一切影響しない (source 不変性)。

初期化乱数の整合性:
- SingleTaskModel.__init__ では `GcnPropertyBlock()` を 1 回呼ぶ。
- DualTaskModel.__init__ では `GcnPropertyBlock()` を main, aux の順に呼ぶ。
- いずれも `set_seeds(seed)` 直後に構築すれば、最初に作られる Block
  (= SingleTaskModel.block / DualTaskModel.main_block) の初期重みは
  完全に一致する。
"""
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
        """aux_block に Source の state_dict を読み込み、全パラメータを凍結する。"""
        missing, unexpected = self.aux_block.load_state_dict(source_state_dict, strict=True)
        if missing or unexpected:
            raise RuntimeError(
                f"Source state_dict incompatible: missing={missing}, unexpected={unexpected}"
            )
        for p in self.aux_block.parameters():
            p.requires_grad = False
