# -*- coding: utf-8 -*-
"""乱数シード管理。

参考実装 (99_Sample_Implemention) と同じ呼び出しパターンを集約することで、
三モデル (General / Finetuning / SoftTransfer) で同一 seed のときの
数値挙動を一致させることを目的とする。
"""
from __future__ import annotations

import random
from typing import List

import numpy as np
import torch


def set_seeds(seed: int) -> None:
    """torch / numpy / random / cuda の seed を一括設定する。

    cudnn は deterministic モードに固定し、benchmark を無効化する。
    """
    seed_value = int(seed)
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def make_loader_seed(seed: int, num_epochs: int) -> List[int]:
    """各 epoch ごとの DataLoader 系 seed を作る。

    参考実装と同様、`set_seeds(seed)` 直後に `random.randint` を
    `num_epochs + 1` 回呼ぶ。返り値は長さ `num_epochs + 1` の int リスト。
    エポック番号は 1-origin で使うため、[0] はダミーとして無視するのが慣例。
    """
    set_seeds(seed)
    return [random.randint(1, 100000) for _ in range(int(num_epochs) + 1)]
