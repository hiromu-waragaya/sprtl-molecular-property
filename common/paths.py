# -*- coding: utf-8 -*-
"""Path helpers. Override with QM9_CSV_PATH, TRANSFER_PARAMS_ROOT, SHARED_SPLITS_ROOT."""
from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_ROOT = PROJECT_ROOT / "0_Datasets"
QM9_CSV_DEFAULT = DATASETS_ROOT / "qm9_dataset.csv"
TRANSFER_PARAMS_ROOT_DEFAULT = PROJECT_ROOT / "Transfer_Params"
SHARED_SPLITS_ROOT_DEFAULT = PROJECT_ROOT / "Shared_Splits"


def resolve_qm9_csv() -> Path:
    env = os.environ.get("QM9_CSV_PATH")
    if env:
        return Path(env)
    return QM9_CSV_DEFAULT


def resolve_transfer_params_root() -> Path:
    env = os.environ.get("TRANSFER_PARAMS_ROOT")
    if env:
        return Path(env)
    return TRANSFER_PARAMS_ROOT_DEFAULT


def resolve_shared_splits_root() -> Path:
    env = os.environ.get("SHARED_SPLITS_ROOT")
    if env:
        return Path(env)
    return SHARED_SPLITS_ROOT_DEFAULT


def _resolve_conda_prefix() -> str | None:
    pfx = os.environ.get("CONDA_PREFIX")
    if pfx:
        return pfx
    exe = Path(sys.executable).resolve()
    if exe.parent.name != "bin":
        return None
    candidate = exe.parent.parent
    if (candidate / "lib" / "libstdc++.so.6").is_file():
        return str(candidate)
    return None


def bootstrap_conda_libstdcxx() -> None:
    """Prepend conda libstdc++ so GLIBCXX matches the env on older hosts."""
    pfx = _resolve_conda_prefix()
    if not pfx:
        return
    libdir = Path(pfx) / "lib"
    lstd = libdir / "libstdc++.so.6"
    if lstd.is_file():
        try:
            import ctypes

            ctypes.CDLL(str(lstd), mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass
    d = str(libdir)
    ld0 = os.environ.get("LD_LIBRARY_PATH", "")
    if d not in ld0.split(":"):
        os.environ["LD_LIBRARY_PATH"] = f"{d}:{ld0}" if ld0 else d
