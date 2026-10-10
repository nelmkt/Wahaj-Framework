"""Compare saved May–September and newly exported June–September LST panels.

The first table repeats the primary pre-treatment-only match within each
panel, so its populations may differ. The paired table fixes original class,
dose, strata and control eligibility on cells exported in both panels and
changes only the LST outcome. Neither comparison verifies treatment identity.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from fw_config import Config, SETTINGS
import fw_panel as panel
import fw_matching as matching

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OLD = ROOT / "code" / "gee" / "panel.csv"
DEFAULT_NEW = ROOT / "code" / "gee" / "panel_jun_sep.csv"
DOSES = (("all", None, None), ("1–2", 1, 2), ("3–4", 3, 4),
         ("5–8", 5, 8), ("9", 9, 9))


def checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def matched_rows(df: pd.DataFrame, cfg: Config, panel_name: str) -> list[dict]:
    key = panel.strata(df, cfg, baseline_only=True)
    boot = matching.Bootstrap(df.block, cfg.n_boot, cfg.seed)
    rows = []
    for setting in SETTINGS:
        t0 = df.group.eq("greened") & df.setting.eq(setting)
        c = df.group.eq("control") & df.setting.eq(setting) & df.d_own_built.abs().lt(cfg.stable_surface_max)
        for label, lo, hi in DOSES:
            t = t0 if lo is None else t0 & df.n_greened_px.between(lo, hi)
            r = matching.att(df, t, c, "d_lst", key, cfg, boot, dose="n_greened_px")
            matched = t & matching.control_weights(df, t, c, key, cfg).gt(0)
            n_blocks = int(df.loc[matched, "block"].nunique())
            reportable = int(matched.sum()) >= 10 and n_blocks >= 5
            rows.append(dict(panel=panel_name, setting=setting, dose_class=label,
                             n_classified=int(t.sum()), n_matched=r["n_treated_matched"],
                             n_blocks=n_blocks, reportable=reportable,
                             per_pixel_C=r["estimate_C"] if reportable else np.nan,
                             lo_C=r["lo_C"] if reportable else np.nan,
                             hi_C=r["hi_C"] if reportable else np.nan,
                             n_boot_valid=r["n_boot_valid"]))
    return rows


def paired_rows(old: pd.DataFrame, new: pd.DataFrame, cfg: Config) -> tuple[list[dict], pd.DataFrame]:
    keep = ["lon", "lat", "group", "d_lst"] + [f"lst_{y}" for y in (2014, 2015, 2018, 2019, 2024, 2025)]
    linked = old.merge(new[keep], on=["lon", "lat"], how="inner", validate="one_to_one",
                       suffixes=("_old", "_new"))
    overlap = (old[["lon", "lat", "group"]].rename(columns={"group": "group_old"})
               .merge(new[["lon", "lat", "group"]].rename(columns={"group": "group_new"}),
                      on=["lon", "lat"], how="outer", validate="one_to_one"))
    overlap["group_old"] = overlap.group_old.fillna("absent")
    overlap["group_new"] = overlap.group_new.fillna("absent")
    overlap_counts = (overlap.groupby(["group_old", "group_new"], dropna=False).size()
                      .reset_index(name="n_cells"))
    key = panel.strata(linked, cfg, baseline_only=True)
    boot = matching.Bootstrap(linked.block, cfg.n_boot, cfg.seed)
    rows = []
    for setting in SETTINGS:
        t0 = linked.group_old.eq("greened") & linked.setting.eq(setting)
        c = (linked.group_old.eq("control") & linked.setting.eq(setting)
             & linked.d_own_built.abs().lt(cfg.stable_surface_max))
        for label, lo, hi in DOSES:
            t = t0 if lo is None else t0 & linked.n_greened_px.between(lo, hi)
            before = matching.att(linked, t, c, "d_lst_old", key, cfg, boot,
                                  dose="n_greened_px", keep_reps=True)
            after = matching.att(linked, t, c, "d_lst_new", key, cfg, boot,
                                 dose="n_greened_px", keep_reps=True)
            diffs = after["_reps"] - before["_reps"]
            diffs = diffs[np.isfinite(diffs)]
            matched = t & matching.control_weights(linked, t, c, key, cfg).gt(0)
            n_blocks = int(linked.loc[matched, "block"].nunique())
            reportable = int(matched.sum()) >= 10 and n_blocks >= 5
            rows.append(dict(setting=setting, dose_class=label, n_original_classified_in_overlap=int(t.sum()),
                             n_matched_same_cells=int(matched.sum()),
                             n_blocks_same_cells=n_blocks, reportable=reportable,
                             old_per_pixel_C=before["estimate_C"] if reportable else np.nan,
                             new_per_pixel_C=after["estimate_C"] if reportable else np.nan,
                             new_minus_old_C=(after["estimate_C"] - before["estimate_C"]) if reportable else np.nan,
                             paired_lo_C=float(np.percentile(diffs, 2.5)) if reportable and len(diffs) else np.nan,
                             paired_hi_C=float(np.percentile(diffs, 97.5)) if reportable and len(diffs) else np.nan,
                             n_paired_boot_valid=len(diffs)))
    return rows, overlap_counts


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--old", type=Path, default=DEFAULT_OLD)
    ap.add_argument("--new", type=Path, default=DEFAULT_NEW)
    ap.add_argument("--tag", default="v2", help="Output version tag; use a new tag to preserve earlier tables")
    args = ap.parse_args()
    if not args.tag.isalnum():
        raise SystemExit("Output tag must contain only letters and digits.")
    old_cfg = Config(panel_path=args.old)
    new_cfg = Config(panel_path=args.new)
    old, _ = panel.load(old_cfg)
    new, new_meta = panel.load(new_cfg)
    if tuple(new_meta.get("months", [])) != (6, 7, 8, 9):
        raise SystemExit("The new panel metadata does not identify a June–September export.")
    rows = matched_rows(old, old_cfg, "saved May–September") + matched_rows(new, new_cfg, "new June–September")
    paired, overlap = paired_rows(old, new, old_cfg)
    tables = ROOT / "tables"
    for name, data in ((f"gee_common_month_primary_{args.tag}.csv", pd.DataFrame(rows)),
                       (f"gee_common_month_paired_{args.tag}.csv", pd.DataFrame(paired)),
                       (f"gee_common_month_overlap_{args.tag}.csv", overlap)):
        out = tables / name
        if out.exists():
            raise SystemExit(f"Refusing to overwrite {out}")
        data.to_csv(out, index=False)
        print(f"Wrote {out}: {len(data)} rows")
    print("Panel SHA-256", checksum(args.old), checksum(args.new))
    print(pd.DataFrame(rows).query("setting == 'outside the built-up area' and dose_class == 'all'").to_string(index=False))
    print(pd.DataFrame(paired).query("setting == 'outside the built-up area'").to_string(index=False))


if __name__ == "__main__":
    main()
