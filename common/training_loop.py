# -*- coding: utf-8 -*-
"""学習ループ共通実装。

General GCN / Finetuning / SoftTransfer のいずれもここで定義された
`train_one_epoch` / `evaluate` / `fit_model` を経由するため、
モデル間の数値挙動を「初期重み」「optimizer のパラメータ集合」「lamb の値」
の 3 点だけに集約できる。

特に Soft Transfer (lamb=0) のとき、`train_one_epoch` の loss 計算は
General と完全に同一 (= MSE(model(data), y.unsqueeze(1))) になる。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import r2_score

from .metrics import compute_soft_sharing_loss
from .seeding import make_loader_seed, set_seeds


def train_one_epoch(
    model: torch.nn.Module,
    train_loader,
    optimizer: torch.optim.Optimizer,
    seed_num: int,
    device: torch.device,
    lamb: float = 0.0,
    sharing_scope: str = "weight_all",
) -> float:
    """1 epoch の学習。返り値は loss_all / num_batches。

    set_seeds の密な呼び出しは参考実装の数値再現性のためにそのまま踏襲。
    """
    model.train()
    set_seeds(seed_num)
    loss_all = 0.0
    n_batches = 0
    for data in train_loader:
        set_seeds(seed_num)
        data = data.to(device)
        set_seeds(seed_num)
        optimizer.zero_grad()
        set_seeds(seed_num)
        pred = model(data)  # SingleTask: block(data), Dual: main_block(data)
        set_seeds(seed_num)
        loss = F.mse_loss(pred, data.y.unsqueeze(1))
        if lamb > 0:
            # Soft sharing 項は DualTaskModel 限定。
            if not (hasattr(model, "main_block") and hasattr(model, "aux_block")):
                raise RuntimeError(
                    "lamb > 0 requires a DualTaskModel with main_block / aux_block."
                )
            set_seeds(seed_num)
            sharing = compute_soft_sharing_loss(
                model.main_block,
                model.aux_block,
                device,
                sharing_scope=sharing_scope,
            )
            loss = loss + lamb * sharing
        set_seeds(seed_num)
        loss.backward()
        set_seeds(seed_num)
        loss_all += loss.item()
        set_seeds(seed_num)
        optimizer.step()
        n_batches += 1
    return loss_all / max(n_batches, 1)


def evaluate(
    model: torch.nn.Module,
    loader,
    seed_num: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """評価。targets, preds, R2 (バッチ平均) を返す。

    参考実装の `test(loader, seed_num)` と同等。
    """
    model.eval()
    set_seeds(seed_num)
    preds = []
    targets = []
    r2_list = []
    with torch.no_grad():
        for data in loader:
            set_seeds(seed_num)
            data = data.to(device)
            set_seeds(seed_num)
            y_pred = model(data)
            set_seeds(seed_num)
            preds.append(y_pred.detach().cpu().numpy())
            set_seeds(seed_num)
            targets.append(data.y.detach().cpu().numpy())
            set_seeds(seed_num)
            r2_list.append(
                r2_score(
                    y_pred.detach().cpu().numpy(),
                    data.y.detach().cpu().numpy(),
                )
            )
    return (
        np.concatenate(targets),
        np.concatenate(preds),
        float(np.mean(r2_list)),
    )


def fit_model(
    model: torch.nn.Module,
    train_loader,
    test_loader,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: torch.device,
    norm_std: float,
    lamb: float = 0.0,
    sharing_scope: str = "weight_all",
    transfer_degree_fn: Optional[Callable[[torch.nn.Module], float]] = None,
    extra_metric_fns: Optional[Dict[str, Callable[[torch.nn.Module], float]]] = None,
    state_save_dir: Optional[Path] = None,
    state_filename_prefix: str = "model",
    metrics_jsonl_path: Optional[Path] = None,
    log_prefix: str = "",
    log_every: int = 10,
) -> Tuple[dict, dict]:
    """学習ループ全体の高レベルラッパー。

    Returns:
        history: dict of per-epoch metrics (list 形式)
        summary: dict (best_train_mae / best_test_mae / best_test_epoch)

    Note:
        `log_every` は **stdout への進捗 print の間引き間隔** のみを制御する
        (人間向け表示)。epoch ごとの train/test 性能を保持する
        `metrics.jsonl` への書き込み・best 選択・state 保存は、間引きとは
        無関係に **毎 epoch** 実行される (永続データは欠落しない)。
        epoch==1 / epoch==epochs / epoch%log_every==0 のときに print する。
        log_every<=1 のときは全 epoch を print (従来挙動)。
    """
    train_seed_list = make_loader_seed(42, epochs)
    test_seed_list = make_loader_seed(43, epochs)

    history = {
        "train_mse": [],
        "train_r2": [],
        "test_r2": [],
        "train_mae": [],
        "test_mae": [],
    }
    if transfer_degree_fn is not None:
        history["transfer_degree"] = []
    if extra_metric_fns:
        for name in extra_metric_fns:
            history[name] = []

    mae_fn = torch.nn.L1Loss()

    best_train_mae: Optional[float] = None
    best_test_mae: Optional[float] = None
    best_test_epoch: Optional[int] = None

    if state_save_dir is not None:
        state_save_dir = Path(state_save_dir)
        state_save_dir.mkdir(parents=True, exist_ok=True)

    if metrics_jsonl_path is not None:
        metrics_jsonl_path = Path(metrics_jsonl_path)
        metrics_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_jsonl_path.write_text("", encoding="utf-8")

    for epoch in range(1, epochs + 1):
        train_seed = train_seed_list[epoch]
        train_mse = train_one_epoch(
            model,
            train_loader,
            optimizer,
            train_seed,
            device,
            lamb=lamb,
            sharing_scope=sharing_scope,
        )
        test_seed = test_seed_list[epoch]
        y_train, y_train_pred, train_r2 = evaluate(
            model, train_loader, test_seed, device
        )
        train_mae_t = mae_fn(
            torch.tensor(y_train_pred).to(device),
            torch.tensor(y_train).unsqueeze(1).to(device),
        ) * norm_std
        train_mae = float(train_mae_t.detach().cpu().item())

        # 参考実装にあわせ、test 評価前にも seed をリセット (実害はないが
        # 三モデル間の数値挙動を完全に揃えるための保険)
        set_seeds(test_seed)
        y_test, y_test_pred, test_r2 = evaluate(
            model, test_loader, test_seed, device
        )
        test_mae_t = mae_fn(
            torch.tensor(y_test_pred).to(device),
            torch.tensor(y_test).unsqueeze(1).to(device),
        ) * norm_std
        test_mae = float(test_mae_t.detach().cpu().item())

        if best_train_mae is None or train_mae < best_train_mae:
            best_train_mae = train_mae
            if state_save_dir is not None:
                torch.save(
                    model.state_dict(),
                    state_save_dir / f"{state_filename_prefix}_train_best.pth",
                )
        if best_test_mae is None or test_mae < best_test_mae:
            best_test_mae = test_mae
            best_test_epoch = epoch
            if state_save_dir is not None:
                torch.save(
                    model.state_dict(),
                    state_save_dir / f"{state_filename_prefix}_test_best.pth",
                )

        history["train_mse"].append(float(train_mse))
        history["train_r2"].append(float(train_r2))
        history["test_r2"].append(float(test_r2))
        history["train_mae"].append(train_mae)
        history["test_mae"].append(test_mae)

        td_val: Optional[float] = None
        if transfer_degree_fn is not None:
            td_val = float(transfer_degree_fn(model))
            history["transfer_degree"].append(td_val)
        extra_metric_vals = {}
        if extra_metric_fns:
            for name, fn in extra_metric_fns.items():
                extra_metric_vals[name] = float(fn(model))
                history[name].append(extra_metric_vals[name])

        if metrics_jsonl_path is not None:
            row = {
                "epoch": epoch,
                "train_mse": float(train_mse),
                "train_r2": float(train_r2),
                "test_r2": float(test_r2),
                "train_mae": train_mae,
                "test_mae": test_mae,
            }
            if td_val is not None:
                row["transfer_degree"] = td_val
            row.update(extra_metric_vals)
            with open(metrics_jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        if (log_every <= 1) or (epoch == 1) or (epoch == epochs) or (epoch % log_every == 0):
            print(
                f"{log_prefix}[ep {epoch:3d}] "
                f"train_mse={train_mse:.6f} train_mae={train_mae:.6f} "
                f"test_mae={test_mae:.6f} test_r2={test_r2:.4f}"
                + (f" td={td_val:.4f}" if td_val is not None else "")
            )

    summary = {
        "best_train_mae": best_train_mae,
        "best_test_mae": best_test_mae,
        "best_test_epoch": best_test_epoch,
    }
    return history, summary
