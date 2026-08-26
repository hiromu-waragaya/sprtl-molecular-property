# -*- coding: utf-8 -*-
"""モデル定義モジュール。

GcnPropertyBlock: 唯一の GCN 本体（BN なし）。三モデルで共有する。
SingleTaskModel: General GCN / Finetuning 用ラッパー。
DualTaskModel:   SoftTransfer 用ラッパー（main + aux）。
"""
from .gcn_property_block import GcnPropertyBlock
from .single_task_model import SingleTaskModel
from .dual_task_model import DualTaskModel

__all__ = ["GcnPropertyBlock", "SingleTaskModel", "DualTaskModel"]
