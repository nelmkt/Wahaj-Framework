"""Matched contrasts with re-retrieved LST (supervisor review M1). Runs only after gee_r1_lst_reretrieval.py.

Replaces lst_<year> in the panel by lst_rr_<year> (joined on cell centre) and repeats the primary match.
Writes NEW tables_revision_r1/r1_reretrieved_contrasts.csv.
usage: python revision_r1_reretrieved_contrasts.py <package_root>
"""
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "code" / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import fw_matching as matching  # noqa: E402
import fw_panel as panel  # noqa: E402
from fw_config import SETTINGS, Config  # noqa: E402

RR = ROOT / "code" / "gee" / "panel_lst_reretrieved_r1.csv"
OUT = ROOT / "tables_revision_r1" / "r1_reretrieved_contrasts.csv"
if not RR.exists():
    raise SystemExit(f"{RR} not found: run manuscript/gee_r1_lst_reretrieval.py first")
if OUT.exists():
    raise SystemExit(f"Refusing to overwrite {OUT}")

cfg = Config(panel_path=ROOT / "code" / "gee" / "panel.csv", out_dir=ROOT)
raw = pd.read_csv(cfg.panel_path)
rr = pd.read_csv(RR)
key = ["lon", "lat"]
raw[key] = raw[key].round(6)
rr[key] = rr[key].round(6)
merged = raw.merge(rr, on=key, how="left", validate="one_to_one")
rows = []
for version in ("product LST", "re-retrieved LST"):
    d = merged.copy()
    if version != "product LST":
        for y in cfg.pre_years + cfg.mid_years + cfg.post_years:
            d[f"lst_{y}"] = d[f"lst_rr_{y}"]
    df = panel.derive(d.dropna(subset=[f"lst_{y}" for y in cfg.pre_years + cfg.post_years]), cfg)
    k = panel.strata(df, cfg, baseline_only=True)
    boot = matching.Bootstrap(df.block, cfg.n_boot, cfg.seed)
    for s in SETTINGS:
        t = df.group.eq("greened") & df.setting.eq(s)
        c = df.group.eq("control") & df.setting.eq(s) & df.d_own_built.abs().lt(cfg.stable_surface_max)
        for label, lo, hi in (("all", 1, 9), ("1–2", 1, 2), ("3–4", 3, 4), ("5–8", 5, 8), ("9", 9, 9)):
            sel = t & df.n_greened_px.between(lo, hi)
            f = matching.att(df, sel, c, "d_lst", k, cfg, boot, dose="n_greened_px")
            rows.append({"lst": version, "setting": s, "pixels": label, "n_matched": f["n_treated_matched"],
                         "per_pixel_C": f["estimate_C"], "lo_C": f["lo_C"], "hi_C": f["hi_C"]})
out = pd.DataFrame(rows)
with open(OUT, "x", encoding="utf-8", newline="") as fh:
    out.to_csv(fh, index=False)
print(out.to_string(index=False))
