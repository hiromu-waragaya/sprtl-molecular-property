# -*- coding: utf-8 -*-
"""MAE (physical units), transfer degree, and SPRTL sharing loss."""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch


SUPPORTED_SHARING_SCOPES = {
    "weight_all",
    "weight_no_output",
    "all_params_all",
    "all_params_no_output",
}


def should_share_parameter(name: str, sharing_scope: str = "weight_all") -> bool:
    if sharing_scope not in SUPPORTED_SHARING_SCOPES:
        raise ValueError(f"Unsupported sharing_scope: {sharing_scope}")
    is_output = name.startswith("mlp_out.")
    is_weight = "weight" in name
    if sharing_scope == "weight_all":
        return is_weight
    if sharing_scope == "weight_no_output":
        return is_weight and not is_output
    if sharing_scope == "all_params_all":
        return True
    if sharing_scope == "all_params_no_output":
        return not is_output
    raise AssertionError(f"Unhandled sharing_scope: {sharing_scope}")


def compute_transfer_degree(main_block, aux_block, sharing_scope: str = "weight_all") -> float:
    prm_main = []
    prm_aux = []
    for (m_name, m_param), (a_name, a_param) in zip(
        main_block.named_parameters(),
        aux_block.named_parameters(),
    ):
        if should_share_parameter(m_name, sharing_scope):
            prm_main.append(m_param.view(-1).detach().cpu().numpy())
            prm_aux.append(a_param.view(-1).detach().cpu().numpy())
    if not prm_main:
        return float("nan")
    main_flat = np.concatenate(prm_main)
    aux_flat = np.concatenate(prm_aux)
    if main_flat.std() == 0 or aux_flat.std() == 0:
        return float("nan")
    return float(np.corrcoef(main_flat, aux_flat)[0, 1])


def compute_soft_sharing_loss(
    main_block: torch.nn.Module,
    aux_block: torch.nn.Module,
    device: Optional[torch.device] = None,
    sharing_scope: str = "weight_all",
) -> torch.Tensor:
    if device is None:
        device = next(main_block.parameters()).device
    soft_loss = torch.tensor(0.0, device=device)
    for (m_name, m_param), (a_name, a_param) in zip(
        main_block.named_parameters(),
        aux_block.named_parameters(),
    ):
        if should_share_parameter(m_name, sharing_scope):
            diff = m_param - a_param
            if diff.ndim >= 2:
                soft_loss = soft_loss + torch.norm(diff, p="fro")
            else:
                soft_loss = soft_loss + torch.norm(diff, p=2)
    return soft_loss
