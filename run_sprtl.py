# -*- coding: utf-8 -*-
"""Public entry point for SPRTL (Soft Transfer) target-task training.

Loads a pretrained source GCN and trains the target task with starting-point
regularization. Unlike ``soft_transfer_train.py``'s CLI (which prepends λ=0.0),
this wrapper calls ``run_one_seed_scenario`` with a single λ.

Environment variables (setdefault to this repository if unset):
  QM9_CSV_PATH, SHARED_SPLITS_ROOT, TRANSFER_PARAMS_ROOT
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

os.environ.setdefault("QM9_CSV_PATH", str(REPO_ROOT / "data" / "qm9_dataset.csv"))
os.environ.setdefault("SHARED_SPLITS_ROOT", str(REPO_ROOT / "splits"))
os.environ.setdefault("TRANSFER_PARAMS_ROOT", str(REPO_ROOT / "pretrained"))

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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

from common.data_pipeline import load_qm9  # noqa: E402


def _load_flat_yaml(path: Path) -> dict:
    """Minimal YAML mapping loader (no nested keys). Avoids a PyYAML dependency."""
    out = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, val = line.split(":", 1)
        out[key.strip()] = val.strip()
    return out


def _import_soft_module():
    spec = importlib.util.spec_from_file_location(
        "soft_transfer_train",
        str(REPO_ROOT / "soft_transfer_train.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["soft_transfer_train"] = mod
    spec.loader.exec_module(mod)
    return mod


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="SPRTL / Soft Transfer: train a target property from a frozen source GCN."
    )
    p.add_argument("--config", type=str, default=None, help="Flat YAML config (see configs/).")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--target", default=None)
    p.add_argument("--auxi", default=None, help="Source property name (e.g. gap).")
    p.add_argument("--lambda-val", type=float, default=None)
    p.add_argument("--train-datasize", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--sharing-scope", default=None)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--run-label", default=None, help="Optional result namespace under results/.")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip if summary.json already exists for this lambda.",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cfg = {}
    if args.config:
        cfg_path = Path(args.config)
        if not cfg_path.is_file():
            cfg_path = REPO_ROOT / args.config
        if not cfg_path.is_file():
            raise FileNotFoundError(f"config not found: {args.config}")
        cfg = _load_flat_yaml(cfg_path)

    # YAML uses `source`; CLI uses `--auxi`.
    source_from_cfg = cfg.get("source", cfg.get("auxi"))

    seed = args.seed if args.seed is not None else (int(cfg["seed"]) if "seed" in cfg else None)
    target = args.target if args.target is not None else cfg.get("target")
    auxi = args.auxi if args.auxi is not None else source_from_cfg
    lambda_val = (
        args.lambda_val
        if args.lambda_val is not None
        else (float(cfg["lambda"]) if "lambda" in cfg else None)
    )
    train_datasize = (
        args.train_datasize
        if args.train_datasize is not None
        else (int(cfg["train_datasize"]) if "train_datasize" in cfg else 200)
    )
    epochs = (
        args.epochs if args.epochs is not None else (int(cfg["epochs"]) if "epochs" in cfg else 1000)
    )
    lr = args.lr if args.lr is not None else (float(cfg["lr"]) if "lr" in cfg else 0.01)
    sharing_scope = (
        args.sharing_scope
        if args.sharing_scope is not None
        else cfg.get("sharing_scope", "weight_no_output")
    )

    missing = [
        name
        for name, val in (
            ("seed", seed),
            ("target", target),
            ("auxi/source", auxi),
            ("lambda", lambda_val),
        )
        if val is None
    ]
    if missing:
        raise SystemExit(
            "missing required settings: "
            + ", ".join(missing)
            + ". Pass CLI flags or --config configs/alpha_from_gap.yaml"
        )

    csv_path = resolve_qm9_csv()
    splits_root = resolve_shared_splits_root()
    transfer_params_root = resolve_transfer_params_root()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[sprtl] repo={REPO_ROOT}")
    print(f"[sprtl] csv={csv_path}")
    print(f"[sprtl] splits={splits_root}")
    print(f"[sprtl] pretrained={transfer_params_root}")
    print(f"[sprtl] device={device}")
    print(
        f"[sprtl] target={target} auxi={auxi} seed={seed} "
        f"lambda={lambda_val} epochs={epochs} train_datasize={train_datasize}"
    )

    if not csv_path.is_file():
        raise FileNotFoundError(
            f"QM9 CSV not found: {csv_path}\nSee data/README.md for placement."
        )

    valid_df, all_x = load_qm9(csv_path)

    soft_mod = _import_soft_module()
    records = soft_mod.run_one_seed_scenario(
        target=str(target),
        auxi=str(auxi),
        seed=int(seed),
        lambdas=[float(lambda_val)],
        train_datasize=int(train_datasize),
        epochs=int(epochs),
        lr=float(lr),
        sharing_scope=str(sharing_scope),
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
    rec = records[0] if records else {}
    print(json.dumps({"status": "ok", **rec}, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1:])
