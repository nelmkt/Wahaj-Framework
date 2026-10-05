"""Scenes per summer month (supervisor review m5) from code/gee/panel_meta.json. Writes NEW tables_revision_r1/r1_scene_counts.csv."""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(sys.argv[1]).resolve()
OUT = ROOT / "tables_revision_r1" / "r1_scene_counts.csv"
if OUT.exists():
    raise SystemExit(f"Refusing to overwrite {OUT}")
meta = json.loads((ROOT / "code" / "gee" / "panel_meta.json").read_text(encoding="utf-8"))
rows = [{"year": int(k[:4]), "month": int(k[5:]), "scenes_cloud_cover_below_20": v} for k, v in meta["scenes_per_year_month"].items()]
d = pd.DataFrame(rows).pivot(index="year", columns="month", values="scenes_cloud_cover_below_20")
d.columns = [f"month_{m:02d}" for m in d.columns]
d["total"] = d.sum(axis=1)
d = d.reset_index()
with open(OUT, "x", encoding="utf-8", newline="") as fh:
    d.to_csv(fh, index=False)
print(d.to_string(index=False))
