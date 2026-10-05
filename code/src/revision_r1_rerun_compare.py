"""Compare a full pipeline rerun (run_framework.py into a separate folder) with the saved tables (supervisor review M5).
For every table the rerun writes, the saved file of the same name is compared value by value.
Writes NEW tables_revision_r1/r1_rerun_comparison.csv.
usage: python revision_r1_rerun_compare.py <package_root> <rerun_folder>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT, RUN = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
OUT = ROOT / "tables_revision_r1" / "r1_rerun_comparison.csv"
if OUT.exists():
    raise SystemExit(f"Refusing to overwrite {OUT}")
rows = []
for new in sorted((RUN / "tables").glob("*.csv")):
    old = ROOT / "tables" / new.name
    r = {"table": new.name, "saved_exists": old.exists()}
    if old.exists():
        a, b = pd.read_csv(old), pd.read_csv(new)
        r.update(rows_saved=len(a), rows_rerun=len(b), same_columns=list(a.columns) == list(b.columns))
        if r["same_columns"] and len(a) == len(b):
            num = [c for c in a.columns if pd.api.types.is_numeric_dtype(a[c]) and pd.api.types.is_numeric_dtype(b[c])
                   and not pd.api.types.is_bool_dtype(a[c]) and not pd.api.types.is_bool_dtype(b[c])]
            txt = [c for c in a.columns if c not in num]
            d = (a[num] - b[num]).abs().to_numpy(float) if num else np.zeros((0, 0))
            both_nan = (a[num].isna() & b[num].isna()).to_numpy() if num else np.zeros((0, 0), bool)
            d = np.where(both_nan, 0.0, d)
            scale = np.maximum(a[num].abs().to_numpy(float), 1e-12) if num else d
            r.update(max_abs_diff=float(np.nanmax(d)) if d.size else 0.0,
                     max_rel_diff=float(np.nanmax(np.where(both_nan, 0, d / scale))) if d.size else 0.0,
                     nan_mismatch=int(((a[num].isna() != b[num].isna()).to_numpy()).sum()) if num else 0,
                     text_identical=bool(a[txt].astype(str).equals(b[txt].astype(str))) if txt else True)
    rows.append(r)
out = pd.DataFrame(rows)
with open(OUT, "x", encoding="utf-8", newline="") as fh:
    out.to_csv(fh, index=False)
print(out.to_string(index=False))
