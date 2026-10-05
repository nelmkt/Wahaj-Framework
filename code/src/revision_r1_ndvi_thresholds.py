"""NDVI-threshold sensitivity (supervisor review, item on the 0.15 / 0.30 thresholds): the primary match repeated on
panels exported with gee_r1_export_panel_thresholds.py. Writes NEW tables_revision_r1/r1_ndvi_thresholds.csv.
usage: python revision_r1_ndvi_thresholds.py <package_root>
"""
import dataclasses
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "code" / "src"))

import pandas as pd  # noqa: E402

import fw_matching as matching  # noqa: E402
import fw_panel as panel  # noqa: E402
from fw_config import SETTINGS, Config  # noqa: E402

OUT = ROOT / "tables_revision_r1" / "r1_ndvi_thresholds.csv"
if OUT.exists():
    raise SystemExit(f"Refusing to overwrite {OUT}")
G = ROOT / "code" / "gee"
RUNS = [("0.15", "0.30", G / "panel.csv"), ("0.10", "0.30", G / "panel_thr_nv010_g030_r1.csv"),
        ("0.20", "0.30", G / "panel_thr_nv020_g030_r1.csv"), ("0.15", "0.25", G / "panel_thr_nv015_g025_r1.csv"),
        ("0.15", "0.35", G / "panel_thr_nv015_g035_r1.csv")]
rows = []
for nv, gr, path in RUNS:
    cfg = dataclasses.replace(Config(), panel_path=path, out_dir=ROOT)
    df, meta = panel.load(cfg)
    key = panel.strata(df, cfg, baseline_only=True)
    boot = matching.Bootstrap(df.block, cfg.n_boot, cfg.seed)
    for s in SETTINGS:
        t = df.group.eq("greened") & df.setting.eq(s)
        c = df.group.eq("control") & df.setting.eq(s) & df.d_own_built.abs().lt(cfg.stable_surface_max)
        px = matching.att(df, t, c, "d_lst", key, cfg, boot, dose="n_greened_px")
        full = matching.att(df, t & df.n_greened_px.eq(9), c, "d_lst", key, cfg, boot)
        rows.append({"ndvi_nonvegetated_below": float(nv), "ndvi_greened_at_least": float(gr), "setting": s,
                     "greened_cells": int(t.sum()), "matched_cells": px["n_treated_matched"],
                     "per_pixel_C": px["estimate_C"], "lo_C": px["lo_C"], "hi_C": px["hi_C"],
                     "fully_greened_cells": full["n_treated_matched"], "fully_greened_cell_C": full["estimate_C"],
                     "fully_greened_lo_C": full["lo_C"], "fully_greened_hi_C": full["hi_C"],
                     "panel_cells": len(df), "panel": path.name})
out = pd.DataFrame(rows)
with open(OUT, "x", encoding="utf-8", newline="") as fh:
    out.to_csv(fh, index=False)
print(out.drop(columns=["panel"]).to_string(index=False))
