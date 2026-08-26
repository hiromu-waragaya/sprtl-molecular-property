# -*- coding: utf-8 -*-
"""GCN 本体ブロック。

三モデル (General / Finetuning / SoftTransfer) の **唯一の GCN 実装**。
BatchNorm1d は完全に削除しており、Soft / Single どちらの参考実装とも
ここで初めてアーキテクチャが厳密に同一となる。

ハイパラ:
- n_features = 75 (mol2vec の出力次元)
- n_conv_hidden = 1 (graphconv_hidden の段数)
- n_mlp_hidden = 1 (mlp_hidden の段数)
- dim = 32 (隠れ次元)

state_dict のキーは 1_Training_SourceModel/source_gcn_train.py 内で
保存される Source Model (MolecularGCN) と一致するため、
`block.load_state_dict(source_state)` で互換読み込みが可能。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.nn import Linear, ModuleList
from torch_geometric.nn import GCNConv, global_add_pool


class GcnPropertyBlock(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.n_features = 75
        self.n_conv_hidden = 1
        self.n_mlp_hidden = 1
        self.dim = 32
        self.graphconv1 = GCNConv(self.n_features, self.dim)
        self.graphconv_hidden = ModuleList(
            [GCNConv(self.dim, self.dim, cached=False) for _ in range(self.n_conv_hidden)]
        )
        self.mlp_hidden = ModuleList(
            [Linear(self.dim, self.dim) for _ in range(self.n_mlp_hidden)]
        )
        self.mlp_out = Linear(self.dim, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.graphconv1(x, edge_index))
        for graphconv in self.graphconv_hidden:
            x = graphconv(x, edge_index)
        x = global_add_pool(x, data.batch)
        for fc_mlp in self.mlp_hidden:
            x = F.relu(fc_mlp(x))
        x = self.mlp_out(x)
        return x
