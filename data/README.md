# Data

This repository does not ship the QM9 CSV. Place a local copy at:

```text
data/qm9_dataset.csv
```

## File used in the paper

- Path in the research workspace: `0_Datasets/qm9_dataset.csv`
- Encoding: `ms932`
- Separator: comma
- Required columns: `smiles` and the target property (`alpha`, …)
- Valid molecules after SMILES parsing: **133885**

SHA256 of the exact file used for the paper numbers:

```
e3e723a250f8fb26cf3089f2dc2c83a50d786cee585276c7f4e66c60cd909cb3
```

`run_sprtl.py` reads this path via `QM9_CSV_PATH` (set automatically). Do not commit the CSV.
