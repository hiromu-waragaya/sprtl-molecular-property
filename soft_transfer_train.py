# -*- coding: utf-8 -*-
"""Soft Transfer 学習スクリプト。

設計上の最重要ポイント:
1. DualTaskModel.forward は main_block の出力のみを返す (B, 1)。
   aux_block は forward 経路から完全に切り離されており、Source 重みの
   「参照テーブル」としてのみ存在する。
2. 損失は MSE(main_pred, y) + lamb * soft_sharing_loss。
   aux 値は y に含めない、aux_loss は計算しない。
3. λ=0 のとき: soft_sharing 項が無効化され、損失は SingleTaskModel と
   完全に同一。main_block の初期重みも SingleTaskModel.block と
   (同一 seed 直後構築なら) 一致するため、Soft(λ=0) と General GCN は
   epoch 単位で完全一致する (検証 A)。
4. λ=0 では aux_block の重みが forward と loss どちらにも影響しないため、
   auxi を alpha→zpve→... のように変えても main 側の挙動は不変
   (検証 B)。

各 λ について「モデル再構築 → aux 重み読込・凍結 → fit」を行い、
λ 単位の学習結果を保存する。

環境変数:
  QM9_CSV_PATH, SHARED_SPLITS_ROOT, TRANSFER_PARAMS_ROOT
  SMOKE_TEST=1            : epochs=3、lambdas=[0.0, 0.01] (CLI 未指定時)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from common.paths import (  # noqa: E402
    bootstrap_conda_libstdcxx,
    resolve_qm9_csv,
    resolve_shared_splits_root,
    resolve_transfer_params_root,
)

bootstrap_conda_libstdcxx()

if not os.environ.get("DISPLAY") and not os.environ.get("MPLBACKEND"):
    import matplotlib  # noqa: E402

    matplotlib.use("Agg")

import torch  # noqa: E402

from common.data_pipeline import (  # noqa: E402
    attach_y_and_build_loaders,
    compute_norm_params_from_train,
    get_or_build_split,
    load_qm9,
)
from common.metrics import (  # noqa: E402
    SUPPORTED_SHARING_SCOPES,
    compute_transfer_degree,
)
from common.models.dual_task_model import DualTaskModel  # noqa: E402
from common.seeding import set_seeds  # noqa: E402
from common.training_loop import fit_model  # noqa: E402


RESULTS_ROOT = Path(__file__).resolve().parent / "results"
DEFAULT_LR = 0.01
DEFAULT_SHARING_SCOPE = "weight_no_output"


# 参考実装 99_Sample_Implemention/SoftTransfer/soft_transfer_train.py の
# DEFAULT_LAMBDA_LIST に λ=0 を先頭に追加したフルグリッド。
DEFAULT_LAMBDA_LIST_FULL: List[float] = [
    0.0,
    0.000000001, 0.00000001, 0.0000001, 0.0000002, 0.0000003, 0.0000004, 0.0000005,
    0.0000006, 0.0000007, 0.0000008, 0.0000009, 0.000001, 0.000005, 0.00001, 0.00005,
    0.0001, 0.001, 0.00125, 0.0015, 0.00175, 0.002, 0.00225, 0.0025, 0.00275, 0.003,
    0.00325, 0.0035, 0.00375, 0.004, 0.00425, 0.0045, 0.00475, 0.005, 0.007, 0.008,
    0.009, 0.0095, 0.01, 0.011, 0.012, 0.013, 0.014, 0.015, 0.016, 0.017, 0.018,
    0.019, 0.02, 0.021, 0.022, 0.023, 0.024, 0.025, 0.026, 0.027, 0.028, 0.029, 0.03,
    0.0325, 0.035, 0.0375, 0.04, 0.0425, 0.045, 0.0475, 0.05, 0.07, 0.08, 0.09,
    0.091, 0.092, 0.093, 0.094, 0.095, 0.096, 0.097, 0.098, 0.099, 0.1,
]


# Round 2 診断を踏まえた本番用 λ グリッド。
# 低 λ と高 λ の transfer degree 飽和域を粗くし、有望な 0.016-0.03 周辺は密に残す。
DEFAULT_LAMBDA_LIST_REDUCED_FULL: List[float] = [
    0.0,
    0.00000001, 0.0000001, 0.0000005, 0.000001, 0.000005, 0.00001,
    0.00005, 0.0001, 0.0005,
    0.001, 0.00125, 0.0015, 0.00175, 0.002,
    0.0025, 0.003, 0.0035, 0.004, 0.0045, 0.005,
    0.006, 0.007, 0.008, 0.009,
    0.01, 0.011, 0.012, 0.013, 0.014, 0.015,
    0.016, 0.017, 0.018, 0.019,
    0.02, 0.021, 0.022, 0.023, 0.024, 0.025,
    0.026, 0.027, 0.028, 0.029, 0.03,
    0.0325, 0.035, 0.0375, 0.04, 0.0425, 0.045, 0.0475, 0.05,
    0.06, 0.07, 0.08, 0.1,
]


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Soft Transfer training (full-grid lambdas, frozen aux Source)."
    )
    p.add_argument(
        "--seed", dest="seeds", type=int, action="append", required=True,
        help="実験乱数シード (複数指定可)",
    )
    p.add_argument(
        "--main-prop", dest="main_props", type=str, action="append", required=True,
        help="ターゲット物性 (複数指定可)",
    )
    p.add_argument(
        "--auxi", dest="auxis", type=str, action="append", required=True,
        help="Source 物性 (複数指定可)",
    )
    p.add_argument(
        "--lambda", dest="lambdas", type=float, action="append", default=None,
        help="λ 値 (複数指定可。未指定時はフルグリッド、SMOKE_TEST=1 では [0, 0.01])",
    )
    p.add_argument("--train-datasize", type=int, default=200)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LR,
        help="Adam learning rate",
    )
    p.add_argument(
        "--sharing-scope",
        default=DEFAULT_SHARING_SCOPE,
        choices=sorted(SUPPORTED_SHARING_SCOPES),
        help="Soft sharing 対象 parameter の範囲",
    )
    p.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="stdout 進捗 print の間引き間隔 (metrics.jsonl は毎 epoch 記録)。1 で全 epoch。",
    )
    p.add_argument(
        "--lambdas-full", action="store_true",
        help="--lambda 未指定でもフルグリッドを明示的に使う (SMOKE_TEST=1 を上書き)",
    )
    p.add_argument(
        "--lambdas-reduced-full", action="store_true",
        help="Round 2 診断後の縮約フルグリッドを使う",
    )
    p.add_argument(
        "--run-label",
        default=None,
        help="Optional result namespace under Results/<target>_soft_<auxi>/",
    )
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing runs.")
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a lambda run if its summary.json already exists.",
    )
    return p.parse_args(argv)


def _source_path(transfer_params_root: Path, auxi: str) -> Path:
    return transfer_params_root / f"{auxi}_source_model_state.pth"


def _resolve_lambdas(args, smoke: bool) -> List[float]:
    """λ リストを決定。

    - --lambda が明示指定されていればそれを使う。
    - --lambdas-full フラグがあればフルグリッド。
    - --lambdas-reduced-full フラグがあれば縮約フルグリッド。
    - 上記いずれも無く SMOKE_TEST=1 なら [0.0, 0.01]。
    - そうでなければフルグリッド。
    """
    if args.lambdas is not None:
        lambdas = [float(x) for x in args.lambdas]
    elif args.lambdas_full:
        lambdas = list(DEFAULT_LAMBDA_LIST_FULL)
    elif args.lambdas_reduced_full:
        lambdas = list(DEFAULT_LAMBDA_LIST_REDUCED_FULL)
    elif smoke:
        lambdas = [0.0, 0.01]
    else:
        lambdas = list(DEFAULT_LAMBDA_LIST_FULL)

    # 重複を除き安定にソート (λ=0 は必ず含める)
    seen = set()
    uniq = []
    for x in lambdas:
        key = float(x)
        if key not in seen:
            seen.add(key)
            uniq.append(key)
    if 0.0 not in seen:
        uniq = [0.0] + uniq
    return uniq


def _lambda_dir_name(lamb: float) -> str:
    """ファイルシステムフレンドリな λ ディレクトリ名。"""
    if lamb == 0.0:
        return "lambda_0.0"
    # 小さい λ は指数表記、大きいものは小数表記
    return f"lambda_{lamb:.12g}"


def _result_base_dir(target: str, auxi: str, run_label: str | None) -> Path:
    base = RESULTS_ROOT / f"{target}_soft_{auxi}"
    if run_label:
        return base / run_label
    return base


def run_one_seed_scenario(
    target: str,
    auxi: str,
    seed: int,
    lambdas: List[float],
    train_datasize: int,
    epochs: int,
    lr: float,
    sharing_scope: str,
    valid_df,
    all_x,
    splits_root: Path,
    transfer_params_root: Path,
    device: torch.device,
    run_label: str | None,
    overwrite: bool,
    skip_existing: bool,
    log_every: int = 10,
) -> List[dict]:
    if target == auxi:
        raise ValueError(f"target and auxi must differ: {target} vs {auxi}")

    source_path = _source_path(transfer_params_root, auxi)
    if not source_path.is_file():
        raise FileNotFoundError(f"Source weights not found: {source_path}")

    print(
        f"==================== soft | target={target} auxi={auxi} seed={seed} "
        f"lr={lr:g} sharing_scope={sharing_scope} (|lambdas|={len(lambdas)}) ===================="
    )

    set_seeds(seed)
    n_all = len(valid_df)
    train_idx, test_idx, split_path = get_or_build_split(
        splits_root, n_all, train_datasize, seed
    )
    print(f"split: {split_path}")
    print(f"train: {len(train_idx)}, test: {len(test_idx)}")

    y_raw = valid_df[target].astype(float).values
    mean, std = compute_norm_params_from_train(y_raw[train_idx])
    y_norm = (y_raw - mean) / std

    train_loader, test_loader = attach_y_and_build_loaders(
        all_x, y_norm, train_idx, test_idx, seed
    )

    scenario_label = f"{target}_soft_{auxi}"
    seed_dir = _result_base_dir(target, auxi, run_label) / f"seed{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    with open(seed_dir / "norm_params.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "target": target,
                "auxi": auxi,
                "lr": float(lr),
                "sharing_scope": sharing_scope,
                "run_label": run_label,
                "mean": float(mean),
                "std": float(std),
                "n_train": len(train_idx),
                "scope": "train_only",
                "source_path": str(source_path),
            },
            f,
            indent=2,
        )

    records: List[dict] = []
    for lamb in lambdas:
        lamb_dir = seed_dir / _lambda_dir_name(lamb)
        summary_path = lamb_dir / "summary.json"
        if summary_path.is_file() and skip_existing and not overwrite:
            print(f"[soft/{target}<-{auxi}/s{seed}/lamb={lamb}] skip existing summary")
            with open(summary_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            records.append(
                {
                    "lambda": float(existing.get("lambda", lamb)),
                    "lr": existing.get("lr"),
                    "sharing_scope": existing.get("sharing_scope"),
                    "best_train_mae": existing.get("best_train_mae"),
                    "best_test_mae": existing.get("best_test_mae"),
                    "best_test_epoch": existing.get("best_test_epoch"),
                    "transfer_degree_at_best": existing.get("transfer_degree_at_best"),
                    "transfer_degree_shared_scope_at_best": existing.get(
                        "transfer_degree_shared_scope_at_best"
                    ),
                }
            )
            continue

        state_dir = lamb_dir / "state"
        lamb_dir.mkdir(parents=True, exist_ok=True)
        state_dir.mkdir(parents=True, exist_ok=True)

        # モデルを再構築。set_seeds(seed) 直後に DualTaskModel() を構築すれば
        # main_block の初期重みは SingleTaskModel.block と完全一致する (検証 A の前提)。
        set_seeds(seed)
        model = DualTaskModel().to(device)
        source_state = torch.load(str(source_path), map_location=device)
        model.load_aux_source_state(source_state)
        set_seeds(seed)
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=float(lr),
        )

        history, summary = fit_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            epochs=epochs,
            device=device,
            norm_std=std,
            lamb=lamb,
            sharing_scope=sharing_scope,
            transfer_degree_fn=lambda m: compute_transfer_degree(
                m.main_block, m.aux_block
            ),
            extra_metric_fns={
                "transfer_degree_shared_scope": lambda m: compute_transfer_degree(
                    m.main_block, m.aux_block, sharing_scope=sharing_scope
                )
            },
            state_save_dir=state_dir,
            state_filename_prefix=f"lamb{lamb}",
            metrics_jsonl_path=lamb_dir / "metrics.jsonl",
            log_prefix=f"[soft/{target}<-{auxi}/s{seed}/lamb={lamb}] ",
            log_every=log_every,
        )

        best_epoch = summary["best_test_epoch"]
        td_at_best: Optional[float] = None
        if best_epoch is not None and history.get("transfer_degree"):
            td_at_best = float(history["transfer_degree"][best_epoch - 1])
        td_shared_at_best: Optional[float] = None
        if best_epoch is not None and history.get("transfer_degree_shared_scope"):
            td_shared_at_best = float(
                history["transfer_degree_shared_scope"][best_epoch - 1]
            )

        rec = {
            "lambda": float(lamb),
            "lr": float(lr),
            "sharing_scope": sharing_scope,
            "best_train_mae": summary["best_train_mae"],
            "best_test_mae": summary["best_test_mae"],
            "best_test_epoch": best_epoch,
            "transfer_degree_at_best": td_at_best,
            "transfer_degree_shared_scope_at_best": td_shared_at_best,
        }
        records.append(rec)

        with open(lamb_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model": "soft_transfer",
                    "target": target,
                    "auxi": auxi,
                    "seed": int(seed),
                    "lambda": float(lamb),
                    "lr": float(lr),
                    "sharing_scope": sharing_scope,
                    "run_label": run_label,
                    "epochs": int(epochs),
                    "train_datasize": int(train_datasize),
                    **summary,
                    "transfer_degree_at_best": td_at_best,
                    "transfer_degree_shared_scope_at_best": td_shared_at_best,
                },
                f,
                indent=2,
            )
        print(
            f"[soft/{target}<-{auxi}/s{seed}/lamb={lamb}] BEST "
            f"train_mae={summary['best_train_mae']:.6f} "
            f"test_mae={summary['best_test_mae']:.6f} "
            f"epoch={best_epoch} td={td_at_best} td_scope={td_shared_at_best}"
        )

    # seed 横断: λ 一覧の summary
    with open(seed_dir / "lambda_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": "soft_transfer",
                "target": target,
                "auxi": auxi,
                "seed": int(seed),
                "lr": float(lr),
                "sharing_scope": sharing_scope,
                "run_label": run_label,
                "epochs": int(epochs),
                "train_datasize": int(train_datasize),
                "records": records,
            },
            f,
            indent=2,
        )

    return records


def main(argv=None):
    args = parse_args(argv)
    smoke = os.environ.get("SMOKE_TEST") == "1"
    if smoke:
        epochs = 3
        print("[soft] SMOKE_TEST=1 -> epochs=3")
    else:
        epochs = args.epochs

    lambdas = _resolve_lambdas(args, smoke)
    print(f"[soft] |lambdas|={len(lambdas)}; first={lambdas[:3]} last={lambdas[-3:]}")
    print(f"[soft] lr={args.lr:g}; sharing_scope={args.sharing_scope}")

    csv_path = resolve_qm9_csv()
    splits_root = resolve_shared_splits_root()
    transfer_params_root = resolve_transfer_params_root()

    if not csv_path.is_file():
        raise FileNotFoundError(f"QM9 CSV not found: {csv_path}")
    if not transfer_params_root.is_dir():
        raise FileNotFoundError(
            f"Transfer params directory not found: {transfer_params_root}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[soft] device={device}")

    valid_df, all_x = load_qm9(csv_path)

    for target in args.main_props:
        for auxi in args.auxis:
            if target == auxi:
                print(f"[soft] skip target==auxi: {target}")
                continue
            for seed in args.seeds:
                run_one_seed_scenario(
                    target=target,
                    auxi=auxi,
                    seed=seed,
                    lambdas=lambdas,
                    train_datasize=args.train_datasize,
                    epochs=epochs,
                    lr=float(args.lr),
                    sharing_scope=args.sharing_scope,
                    valid_df=valid_df,
                    all_x=all_x,
                    splits_root=splits_root,
                    transfer_params_root=transfer_params_root,
                    device=device,
                    run_label=args.run_label,
                    overwrite=bool(args.overwrite),
                    skip_existing=bool(args.skip_existing),
                    log_every=int(args.log_every),
                )

    print("[soft] Complete!")


if __name__ == "__main__":
    main(sys.argv[1:])
