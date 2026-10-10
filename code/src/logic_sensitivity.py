"""Recompute design and proximity diagnostics from the bundled annual cell panel.

The proximity screen sees all exported, eligible greened *cells*, not every 30 m
greened pixel. It cannot certify isolation from unexported mixed cells or provide
monthly vegetation persistence. It is reported only as a panel-available proxy.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import t as student_t

from fw_config import Config, SETTINGS
import fw_panel as panel
import fw_matching as matching


ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "tables"


def nearest_other_exported_greened_m(df: pd.DataFrame) -> pd.Series:
    """Centre-to-centre distance to the nearest other exported eligible greened cell."""
    mask = df.group.eq("greened")
    lat0 = np.radians(float(df.lat.mean()))
    xy = np.column_stack((df.loc[mask, "lon"].to_numpy() * 111320 * np.cos(lat0),
                          df.loc[mask, "lat"].to_numpy() * 110570))
    distances = cKDTree(xy).query(xy, k=2)[0][:, 1]
    out = pd.Series(np.nan, index=df.index)
    out.loc[mask] = distances
    return out


def estimate_rows(df, cfg, boot, nearest):
    rows, deletions, summaries, proxy = [], [], [], []
    for setting in SETTINGS:
        T = df.group.eq("greened") & df.setting.eq(setting)
        C = (df.group.eq("control") & df.setting.eq(setting)
             & df.d_own_built.abs().lt(cfg.stable_surface_max))
        for spec, key in (("pre-treatment-only strata", panel.strata(df, cfg, baseline_only=True)),
                          ("concurrent-change strata", panel.strata(df, cfg))):
            matched = T & matching.control_weights(df, T, C, key, cfg).gt(0)
            dose2 = df.loc[matched].groupby("block").n_greened_px.apply(lambda v: float(np.square(v).sum()))
            shares = dose2 / dose2.sum()
            leverage_equivalent = 1 / float(np.square(shares).sum())
            for label, treatment in [("all", T)] + [
                (f"{lo}–{hi}" if lo != hi else str(lo), T & df.n_greened_px.between(lo, hi))
                for lo, hi in cfg.dose_bins
            ]:
                matched_class = treatment & matching.control_weights(df, treatment, C, key, cfg).gt(0)
                if matched_class.sum() < cfg.min_cells_reported:
                    continue
                slope = matching.att(df, treatment, C, "d_lst", key, cfg, boot, dose="n_greened_px")
                cell = matching.att(df, treatment, C, "d_lst", key, cfg, boot)
                rows.append(dict(setting=setting, specification=spec, dose_class=label,
                                 n_classified=int(treatment.sum()), n_matched=int(matched_class.sum()),
                                 n_treated_blocks=int(df.loc[matched_class, "block"].nunique()),
                                 mean_dose=float(df.loc[matched_class, "n_greened_px"].mean()),
                                 per_pixel_C=slope["estimate_C"], per_pixel_lo_C=slope["lo_C"],
                                 per_pixel_hi_C=slope["hi_C"], cell_C=cell["estimate_C"],
                                 cell_lo_C=cell["lo_C"], cell_hi_C=cell["hi_C"],
                                 n_boot_valid=slope["n_boot_valid"]))
                block_values = sorted(df.loc[matched_class, "block"].unique())
                delete_values = []
                for block in block_values:
                    drop = df.block.eq(block)
                    r = matching.att(df, treatment & ~drop, C & ~drop, "d_lst", key, cfg, None,
                                     dose="n_greened_px")
                    delete_values.append(r["estimate_C"])
                    deletions.append(dict(setting=setting, specification=spec, dose_class=label,
                                          deleted_block=block, deleted_matched_treated=int((matched_class & drop).sum()),
                                          deleted_dose_squared_share=float(np.square(
                                              df.loc[matched_class & drop, "n_greened_px"]).sum()
                                              / np.square(df.loc[matched_class, "n_greened_px"]).sum()),
                                          per_pixel_C=r["estimate_C"], n_matched_after=r["n_treated_matched"]))
                x = np.asarray(delete_values, float)
                valid = x[np.isfinite(x)]
                if len(valid) == len(x) and len(valid) > 2:
                    jk_se = float(np.sqrt((len(valid) - 1) / len(valid) * np.square(valid - valid.mean()).sum()))
                    critical = student_t.ppf(.975, len(valid) - 1)
                    jk_lo, jk_hi = slope["estimate_C"] - critical * jk_se, slope["estimate_C"] + critical * jk_se
                else:
                    jk_se = jk_lo = jk_hi = np.nan
                summaries.append(dict(setting=setting, specification=spec, dose_class=label,
                                      n_deleted_blocks=len(x), n_valid_deletions=len(valid),
                                      base_per_pixel_C=slope["estimate_C"], deletion_min_C=np.min(valid),
                                      deletion_max_C=np.max(valid), jackknife_se_C=jk_se,
                                      jackknife_95_lo_C=jk_lo, jackknife_95_hi_C=jk_hi,
                                      leverage_equivalent_blocks=leverage_equivalent if label == "all" else np.nan,
                                      largest_block_dose_squared_share=float(shares.max()) if label == "all" else np.nan,
                                      note="exploratory cluster jackknife; unequal blocks and rematching limit interval interpretation"))
            small = T & df.n_greened_px.between(1, 2)
            full = matching.att(df, small, C, "d_lst", key, cfg, boot,
                                dose="n_greened_px", keep_reps=True)
            for radius in (0, 300, 430, 600):
                subset = small if radius == 0 else small & nearest.gt(radius)
                r = matching.att(df, subset, C, "d_lst", key, cfg, boot,
                                 dose="n_greened_px", keep_reps=True)
                cell = matching.att(df, subset, C, "d_lst", key, cfg, boot)
                paired = r["_reps"] - full["_reps"]
                paired = paired[np.isfinite(paired)]
                m = subset & matching.control_weights(df, subset, C, key, cfg).gt(0)
                proxy.append(dict(setting=setting, specification=spec,
                                  nearest_other_exported_greened_cell_centre_gt_m=radius,
                                  n_classified=int(subset.sum()), n_matched=int(m.sum()),
                                  n_treated_blocks=int(df.loc[m, "block"].nunique()),
                                  per_pixel_C=r["estimate_C"], lo_C=r["lo_C"], hi_C=r["hi_C"],
                                  difference_from_full_per_pixel_C=r["estimate_C"] - full["estimate_C"],
                                  difference_lo_C=float(np.percentile(paired, 2.5)) if len(paired) else np.nan,
                                  difference_hi_C=float(np.percentile(paired, 97.5)) if len(paired) else np.nan,
                                  cell_C=cell["estimate_C"], cell_lo_C=cell["lo_C"], cell_hi_C=cell["hi_C"],
                                  baseline_lst_C=float(df.loc[m, "lst_pre"].mean()) if m.any() else np.nan,
                                  coast_km=float(df.loc[m, "coast_km"].mean()) if m.any() else np.nan,
                                  n_boot_valid=r["n_boot_valid"],
                                  screen_scope="other eligible classified greened 90 m cells only; not complete 30 m greening"))
    return map(pd.DataFrame, (rows, deletions, summaries, proxy))


def sio_integrity(cfg):
    raw = pd.read_csv(cfg.sio_path)
    out = raw[["year", "branch", "irrigated_area_ha"]].copy()
    out["source_total_m3"] = raw[["reclaimed_m3", "groundwater_m3", "agri_drainage_m3"]].fillna(0).sum(axis=1)
    out["depth_m"] = out.source_total_m3 / (out.irrigated_area_ha * 10000)
    water = pd.read_csv(TABLES / "water.csv")
    low, high = float(water.depth_m.min()), float(water.depth_m.max())
    out["inside_assumed_depth_band"] = out.depth_m.between(low, high)
    out["duplicate_branch_year_in_aggregated_file"] = out.duplicated(["year", "branch"], keep=False)
    out["source_join_verified"] = False
    return out


def main():
    cfg = Config(panel_path=ROOT / "code" / "gee" / "panel.csv", out_dir=ROOT)
    df, _ = panel.load(cfg)
    boot = matching.Bootstrap(df.block, cfg.n_boot, cfg.seed)
    nearest = nearest_other_exported_greened_m(df)
    outputs = estimate_rows(df, cfg, boot, nearest)
    for name, table in zip(("logic_primary_class_estimates", "logic_block_deletions",
                            "logic_block_summary", "logic_panel_proximity_proxy"), outputs):
        table.to_csv(TABLES / f"{name}.csv", index=False)
        print(f"{name}: {len(table)} rows")
    sio = sio_integrity(cfg)
    sio.to_csv(TABLES / "logic_sio_integrity.csv", index=False)
    print(f"SIO: {len(sio)} aggregated branch-years, {int(sio.inside_assumed_depth_band.sum())} inside scenario band")


if __name__ == "__main__":
    main()
