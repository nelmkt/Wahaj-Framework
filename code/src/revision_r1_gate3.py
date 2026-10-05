"""Gate 3 and gate 2 on the PRIMARY match (supervisor review M5): the framework's own comparison
(fw_validate.compare) and support screen, applied with the pre-treatment-only strata instead of the
concurrent-change strata of the pipeline's main run.

Read-only on the package; writes one NEW table tables_revision_r1/r1_gate3_primary.csv.
usage: python revision_r1_gate3.py <package_root>      (PYTHONDONTWRITEBYTECODE=1; ~10-20 minutes)
"""
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "code" / "src"))

import pandas as pd  # noqa: E402

import fw_matching as matching  # noqa: E402
import fw_model as model  # noqa: E402
import fw_panel as panel  # noqa: E402
import fw_validate as validate  # noqa: E402
from fw_config import SETTINGS, Config  # noqa: E402

OUT = ROOT / "tables_revision_r1" / "r1_gate3_primary.csv"
if OUT.exists():
    raise SystemExit(f"Refusing to overwrite {OUT}")

cfg = Config(panel_path=ROOT / "code" / "gee" / "panel.csv", out_dir=ROOT)
df, _ = panel.load(cfg)
key = panel.strata(df, cfg, baseline_only=True)
R = {}
for s in SETTINGS:
    t = df.group.eq("greened") & df.setting.eq(s)
    c = df.group.eq("control") & df.setting.eq(s) & df.d_own_built.abs().lt(cfg.stable_surface_max)
    R[s] = {"treated": t, "controls": c, "key": key, "weights": matching.control_weights(df, t, c, key, cfg)}

rows = []
for label, excl in (("A", None), ("B", cfg.strict_exclusion_m)):
    print(f"fitting Model {label} with {cfg.n_refits} refits", flush=True)
    M = model.Model(df, cfg, exclude_near_m=excl)
    for s in SETTINGS:
        rs = R[s]
        matched = rs["treated"] & (rs["weights"] > 0)
        cells = df[matched]
        sup = pd.Series(False, index=df.index)
        sup.loc[cells.index] = M.supported(cells) & M.supported(cells, ndvi=cells.ndvi_pre, ndbi=cells.ndbi_pre)
        for lo, hi in cfg.dose_bins:
            mask = matched & df.n_greened_px.between(lo, hi)
            if mask.sum() < cfg.min_cells_reported:
                continue
            r = validate.compare(df, M, rs, mask, cfg)
            rows.append({"model": label, "match": "primary (pre-treatment-only strata)", "setting": s,
                         "pixels": f"{lo}–{hi}" if lo != hi else str(lo), "n_supported": int((mask & sup).sum()),
                         **{k: r[k] for k in ("n_cells", "n_blocks", "measured_C", "measured_lo_C", "measured_hi_C",
                                              "model_C", "model_lo_C", "model_hi_C", "difference_C",
                                              "difference_lo_C", "difference_hi_C", "n_joint_valid", "status")}})
out = pd.DataFrame(rows)
with open(OUT, "x", encoding="utf-8", newline="") as fh:
    out.to_csv(fh, index=False)
print(out.to_string(index=False))
