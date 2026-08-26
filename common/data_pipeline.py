# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from torch_geometric.loader import DataLoader

from .graph_featurization import mol2vec
from .seeding import set_seeds


def load_qm9(csv_path: Path) -> Tuple[pd.DataFrame, list]:
    """Load QM9 CSV (ms932) and featurize valid SMILES."""
    df = pd.read_csv(str(csv_path), encoding="ms932", sep=",")
    if "smiles" not in df.columns:
        raise ValueError(f"CSV must contain 'smiles' column: {csv_path}")

    valid_rows = []
    mols = []
    for _, row in df.iterrows():
        mol = Chem.MolFromSmiles(str(row["smiles"]))
        if mol is None:
            continue
        valid_rows.append(row)
        mols.append(mol)

    n_invalid = len(df) - len(mols)
    if n_invalid:
        print(f"[data_pipeline] Excluded invalid SMILES: {n_invalid}")
    valid_df = pd.DataFrame(valid_rows).reset_index(drop=True)
    print(f"[data_pipeline] Valid molecules: {len(valid_df)} (from {len(df)} rows)")

    print("[data_pipeline] Graph featurization...")
    all_x = [mol2vec(m) for m in mols]
    print("[data_pipeline] Graph featurization done.")
    return valid_df, all_x


def build_split(n_all: int, train_datasize: int, seed: int) -> Tuple[List[int], List[int]]:
    if train_datasize < 1 or train_datasize >= n_all:
        raise ValueError(
            f"Invalid split: n_all={n_all}, train_datasize={train_datasize}"
        )
    indices = list(range(n_all))
    gen = torch.Generator().manual_seed(int(seed))
    train_subset, test_subset = torch.utils.data.random_split(
        indices,
        lengths=[train_datasize, n_all - train_datasize],
        generator=gen,
    )
    return list(train_subset.indices), list(test_subset.indices)


def load_split_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_split_json(
    path: Path,
    *,
    seed: int,
    n_all: int,
    train_datasize: int,
    train_idx: List[int],
    test_idx: List[int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "seed": int(seed),
                "n_all": int(n_all),
                "train_datasize": int(train_datasize),
                "n_test": len(test_idx),
                "train_idx": list(train_idx),
                "test_idx": list(test_idx),
            },
            f,
            indent=2,
        )


def get_or_build_split(
    splits_root: Path,
    n_all: int,
    train_datasize: int,
    seed: int,
) -> Tuple[List[int], List[int], Path]:
    """Load a fixed split JSON, or create one if missing."""
    splits_root.mkdir(parents=True, exist_ok=True)
    path = splits_root / f"target{train_datasize}_seed{seed}.json"
    if path.is_file():
        d = load_split_json(path)
        if d.get("n_all") != n_all or d.get("train_datasize") != train_datasize:
            raise RuntimeError(
                f"Existing split JSON inconsistent: {path} "
                f"(n_all={d.get('n_all')} vs {n_all}, "
                f"train_datasize={d.get('train_datasize')} vs {train_datasize})"
            )
        return list(d["train_idx"]), list(d["test_idx"]), path

    train_idx, test_idx = build_split(n_all, train_datasize, seed)
    save_split_json(
        path,
        seed=seed,
        n_all=n_all,
        train_datasize=train_datasize,
        train_idx=train_idx,
        test_idx=test_idx,
    )
    return train_idx, test_idx, path


def compute_norm_params_from_train(y_raw_train: np.ndarray) -> Tuple[float, float]:
    """Train-only z-score mean and std (ddof=0)."""
    arr = np.asarray(y_raw_train, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    if std == 0.0:
        raise ValueError("std is zero on training split.")
    return mean, std


def attach_y_and_build_loaders(
    all_x: list,
    y_norm: np.ndarray,
    train_idx: List[int],
    test_idx: List[int],
    seed: int,
    train_batch_size: int = 32,
    test_batch_size: int = 4096,
) -> Tuple[DataLoader, DataLoader]:
    set_seeds(seed)

    train_x = []
    for i in train_idx:
        g = all_x[i]
        g.y = torch.FloatTensor([float(y_norm[i])])
        train_x.append(g)

    test_x = []
    for i in test_idx:
        g = all_x[i]
        g.y = torch.FloatTensor([float(y_norm[i])])
        test_x.append(g)

    set_seeds(seed)
    train_loader = DataLoader(
        train_x, batch_size=train_batch_size, shuffle=True, drop_last=False
    )
    set_seeds(seed)
    test_loader = DataLoader(
        test_x, batch_size=test_batch_size, shuffle=True, drop_last=False
    )
    return train_loader, test_loader
