# Starting-Point Regularized Transfer Learning (SPRTL)

Author implementation for the paper:

**Starting-Point Regularized Transfer Learning for Molecular Property Prediction: Mitigating Negative Transfer in Low-Data Regimes**

This repository contains the minimum needed to run SPRTL on a target property using a pretrained source model, plus the representative numerical result.

## Paper

Coming soon.

## What this repository contains

- Pretrained source GCN weights (`gap`)
- SPRTL training code for a target property (`alpha`)
- The representative run (`seed=42`, `lambda=0.032`) and its metrics

## Requirements

See `environment.yml`. The paper numbers were produced with conda env `sprtl` (Python 3.9, PyTorch 2.0.1+cu118, PyG 2.3.1, RDKit 2023.03.2).

## Installation

```bash
conda env create -f environment.yml
conda activate sprtl
```

PyTorch Geometric CUDA wheels may need a separate install matching your CUDA version.

## Data

Place the QM9 CSV as described in [`data/README.md`](data/README.md).

## Usage

```bash
python run_sprtl.py --config configs/alpha_from_gap.yaml
```

Or equivalently:

```bash
python run_sprtl.py \
  --seed 42 --target alpha --auxi gap \
  --lambda-val 0.032 --epochs 1000 \
  --train-datasize 200 --lr 0.01 \
  --sharing-scope weight_no_output
```

## Repository structure

```text
.
├── run_sprtl.py
├── sprtl_train.py
├── common/
├── configs/alpha_from_gap.yaml
├── pretrained/gap_source_model_state.pth
├── splits/target200_seed42.json
├── data/README.md
└── results/alpha_from_gap/seed42/
```

## Contact

Hiromu Waragaya  
[chlorine017@stu.kanazawa-u.ac.jp](mailto:chlorine017@stu.kanazawa-u.ac.jp)
