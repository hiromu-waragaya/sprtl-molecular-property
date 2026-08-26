# -*- coding: utf-8 -*-
from __future__ import annotations

import random
from typing import List

import numpy as np
import torch


def set_seeds(seed: int) -> None:
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
    set_seeds(seed)
    return [random.randint(1, 100000) for _ in range(int(num_epochs) + 1)]
