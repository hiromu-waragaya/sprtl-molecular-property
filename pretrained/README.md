# Pretrained source model

`gap_source_model_state.pth` is the QM9 **gap** source GCN used for the representative SPRTL run (`alpha <- gap`).

- Format: raw `state_dict` of `GcnPropertyBlock` (no wrapper)
- Size: about 22 KB
- Training: single-task GCN on QM9 gap (see the research workspace `1_Training_SourceModel/`)

`run_sprtl.py` loads `{auxi}_source_model_state.pth` from this directory (or `TRANSFER_PARAMS_ROOT`).
