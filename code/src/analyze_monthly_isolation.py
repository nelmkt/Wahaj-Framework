"""Analyze a new GEE monthly-NDVI and complete 30 m neighborhood export.

Run only after code/gee/gee_monthly_isolation.py has produced its raw CSV.
Monthly persistence is descriptive evidence about the vegetation path, not
proof of irrigation, and isolation is a site-selection sensitivity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fw_config import Config, SETTINGS
import fw_panel as panel
import fw_matching as matching

ROOT = Path(__file__).resolve().parents[2]
MONTH_KEYS = [f"{year}{month:02d}" for year in (2024, 2025) for month in (5, 6, 7, 8, 9)]
REQUIRED_RECIPE_REVISION = "native-30m-neighborhood-v4"


def link_export(df: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """Require near-complete one-to-one alignment with the unchanged panel grid."""
    key = ["lon_key", "lat_key"]
    d = df.assign(lon_key=df.lon.round(6), lat_key=df.lat.round(6))
    e = raw.assign(lon_key=raw.lon.round(6), lat_key=raw.lat.round(6))
    assert not e.duplicated(key).any(), "The new export has duplicate cell centres."
    e = e.drop(columns=["lon", "lat"])
    linked = d.merge(e, on=key, how="left", validate="one_to_one", sort=False)
    classified = linked.group.eq("greened")
    coverage = linked.loc[classified, "green_count_cell"].notna().mean()
    if coverage < 0.99:
        raise SystemExit(f"Only {coverage:.1%} of classified greened panel cells joined; check GEE grid/recipe drift.")
    matched = linked.loc[classified & linked.green_count_cell.notna()]
    disagreement = np.abs(matched.green_count_cell - matched.n_greened_px)
    if (disagreement > 0.25).any():
        raise SystemExit(f"{int((disagreement > 0.25).sum())} dose counts disagree; do not interpret this export.")
    for month in MONTH_KEYS:
        valid = matched[f"ndvi_valid_green_count_{month}"]
        high = matched[f"ndvi_ge30_green_count_{month}"]
        bad = valid.lt(-.25) | high.lt(-.25) | valid.gt(matched.n_greened_px + .25) | high.gt(valid + .25)
        if bad.any():
            raise SystemExit(f"{int(bad.sum())} impossible pixel counts in {month}; do not interpret this export.")
    return linked


def persistence_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for setting in SETTINGS:
        for label, lo, hi in (("1–2", 1, 2), ("3–4", 3, 4), ("5–8", 5, 8), ("9/9", 9, 9)):
            selected = df[df.group.eq("greened") & df.setting.eq(setting) &
                          df.n_greened_px.between(lo, hi)].copy()
            if selected.empty:
                continue
            valid = np.column_stack([selected[f"ndvi_valid_green_count_{k}"].to_numpy(float)
                                     for k in MONTH_KEYS])
            high = np.column_stack([selected[f"ndvi_ge30_green_count_{k}"].to_numpy(float)
                                    for k in MONTH_KEYS])
            means = np.column_stack([selected[f"ndvi_mean_{k}"].to_numpy(float)
                                     for k in MONTH_KEYS])
            means[means < -1000] = np.nan
            required = np.ceil(selected.n_greened_px.to_numpy(float)[:, None] / 2)
            adequate = valid + 0.25 >= required
            n_adequate = adequate.sum(axis=1)
            high_share = np.divide(high, valid, out=np.full_like(high, np.nan), where=valid > 0)
            high_share[~adequate] = np.nan
            cell_mean_share = np.nanmean(high_share, axis=1)
            sufficiently_seen = n_adequate >= 8
            reportable = len(selected) >= 10
            rows.append({
                "setting": setting, "dose_class": label, "n_cells": len(selected),
                "reportable": reportable,
                "n_at_least_8_of_10_months_adequately_observed": int(sufficiently_seen.sum()),
                "median_adequately_observed_months": float(np.median(n_adequate)) if reportable else np.nan,
                "median_fraction_green_pixel_months_ndvi_ge_0_30": (
                    float(np.nanmedian(cell_mean_share)) if reportable else np.nan),
                "share_cells_ge80pct_green_pixel_months_among_adequately_observed": (
                    float(np.mean(cell_mean_share[sufficiently_seen] >= .8))
                    if reportable and sufficiently_seen.any() else np.nan),
                "median_monthly_ndvi_range": (float(np.nanmedian(
                    np.nanmax(means, axis=1) - np.nanmin(means, axis=1))) if reportable else np.nan),
                "interpretation": "endpoint vegetation persistence only; not an irrigation classifier",
            })
    return pd.DataFrame(rows)


def isolation_rows(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    boot = matching.Bootstrap(df.block, cfg.n_boot, cfg.seed)
    rows = []
    for setting in SETTINGS:
        treated = (df.group.eq("greened") & df.setting.eq(setting) &
                   df.n_greened_px.between(1, 2))
        controls = (df.group.eq("control") & df.setting.eq(setting) &
                    df.d_own_built.abs().lt(cfg.stable_surface_max))
        for spec, baseline_only in (("pre-treatment-only strata", True),
                                    ("concurrent-change strata", False)):
            key = panel.strata(df, cfg, baseline_only=baseline_only)
            full = matching.att(df, treated, controls, "d_lst", key, cfg, boot,
                                dose="n_greened_px", keep_reps=True)
            screens = [
                ("all 1–2-pixel cells", treated),
                ("no greened pixel in eight neighboring 90 m cells",
                 treated & df.green_count_8_neighbor_cells.lt(0.5)),
                ("no other greened 30 m pixel within 300 m of cell centre",
                 treated & df.green_count_other_within_300m_center.lt(0.5)),
                ("both 30 m screens",
                 treated & df.green_count_8_neighbor_cells.lt(0.5) &
                 df.green_count_other_within_300m_center.lt(0.5)),
            ]
            for name, selected in screens:
                result = matching.att(df, selected, controls, "d_lst", key, cfg, boot,
                                      dose="n_greened_px", keep_reps=True)
                paired = result["_reps"] - full["_reps"]
                paired = paired[np.isfinite(paired)]
                matched = selected & matching.control_weights(df, selected, controls, key, cfg).gt(0)
                rows.append({
                    "setting": setting, "specification": spec, "screen": name,
                    "n_classified": int(selected.sum()),
                    "n_matched": int(matched.sum()),
                    "n_matched_blocks": int(df.loc[matched, "block"].nunique()),
                    "per_pixel_C": result["estimate_C"], "lo_C": result["lo_C"],
                    "hi_C": result["hi_C"],
                    "difference_from_full_C": result["estimate_C"] - full["estimate_C"],
                    "difference_lo_C": float(np.percentile(paired, 2.5)) if len(paired) else np.nan,
                    "difference_hi_C": float(np.percentile(paired, 97.5)) if len(paired) else np.nan,
                    "baseline_lst_C": float(df.loc[matched, "lst_pre"].mean()) if matched.any() else np.nan,
                    "coast_km": float(df.loc[matched, "coast_km"].mean()) if matched.any() else np.nan,
                    "n_boot_valid": result["n_boot_valid"],
                })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, default=ROOT / "tables" / "gee_monthly_isolation_raw.csv")
    args = parser.parse_args()
    if not args.raw.is_file():
        raise SystemExit(f"Missing {args.raw}; first run code/gee/gee_monthly_isolation.py with Earth Engine.")
    metadata = json.loads(args.raw.with_suffix(".json").read_text(encoding="utf-8"))
    print(f"NEGI monthly analyzer: isolation requires {REQUIRED_RECIPE_REVISION} "
          "and zero neighborhood-order violations", flush=True)
    if metadata.get("source_recipe") != "code/gee/export_panel.py":
        raise SystemExit("Unexpected source recipe; inspect the raw export before analysis.")
    current_sha = hashlib.sha256((ROOT / "code" / "gee" / "panel.csv").read_bytes()).hexdigest()
    if metadata.get("source_panel_sha256") != current_sha:
        raise SystemExit("The raw export refers to a different panel.csv; do not merge these runs.")
    cfg = Config(panel_path=ROOT / "code" / "gee" / "panel.csv")
    df, _ = panel.load(cfg)
    linked = link_export(df, pd.read_csv(args.raw))
    persistence = persistence_rows(linked)
    suffix = args.raw.stem.replace("monthly_isolation_", "")
    persistence_out = ROOT / "tables" / f"gee_monthly_persistence_from_{suffix}.csv"
    isolation_out = ROOT / "tables" / f"gee_exact_isolation_effects_from_{suffix}.csv"
    if persistence_out.exists() or isolation_out.exists():
        raise SystemExit("Refusing to overwrite a prior monthly or isolation analysis.")
    persistence.to_csv(persistence_out, index=False)
    print(f"Analyzed {int(linked.group.eq('greened').sum())} classified greened cells.")
    print(persistence.to_string(index=False))
    greened = linked.loc[linked.group.eq("greened")]
    bad = (greened.green_count_8_neighbor_cells
           - greened.green_count_other_within_300m_center).gt(.25)
    if bad.any():
        raise SystemExit(f"{int(bad.sum())} neighborhood-order violations: the 300 m count "
                         "is smaller than the eight-neighbor count; persistence was saved, "
                         "isolation was withheld.")
    if (metadata.get("recipe_revision") != REQUIRED_RECIPE_REVISION
            or metadata.get("neighbor_order_violations") != 0):
        raise SystemExit("Isolation withheld: export metadata lacks the verified v4 "
                         "30 m neighborhood recipe with zero violations.")
    isolation = isolation_rows(linked, cfg)
    isolation.to_csv(isolation_out, index=False)
    print(isolation.to_string(index=False))


if __name__ == "__main__":
    main()
