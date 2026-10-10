"""Measured cooling: matched difference-in-differences with a spatial block bootstrap, by setting and by how much of
a cell greened.

For every greened cell: its summer LST change (2014–15 → 2024–25) minus the mean change of control cells in its
stratum — ground in the same setting, local region, coastal band, building cover, development of the surroundings and
material that stayed unvegetated and unchanged (coarsened exact matching). Averaging within a dose class produces a
matched cell-level contrast; the slope through those contrasts is a zero-intercept per-greened-pixel summary, not an
identified marginal return to planting one additional pixel (0.09 ha).

The pipeline's legacy ``main`` strata condition on neighbourhood development during the treatment period. The
interpretation-leading pre-treatment-only specification omits that variable; the legacy main output is a labelled
sensitivity on a narrower population. Neither comparison proves parallel trends or validates the stable-surface
control filter.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fw_config import SETTINGS, Config
from fw_panel import _grid_id, coast_band, strata


class Bootstrap:
    """One set of block draws shared by every estimate, so their intervals are comparable."""

    def __init__(self, blocks: pd.Series, n_boot: int, seed: int):
        codes, uniq = pd.factorize(blocks)
        self.codes, self.n_blocks = codes, len(uniq)
        self.block_ids = pd.Index(uniq)
        self.replicate_ids = np.arange(n_boot, dtype=int)
        rng = np.random.default_rng(seed)
        self.draws = rng.multinomial(self.n_blocks, np.full(self.n_blocks, 1 / self.n_blocks), size=n_boot)

    def weights(self, r: int) -> np.ndarray:
        return self.draws[r][self.codes].astype(float)


def _point(y, t, c, key, eligible, w, dose=None):
    """Matched difference (mean over treated) or, with dose, its least-squares slope through the origin."""
    wc = w * c
    n_c = np.bincount(key, wc, minlength=len(eligible))
    s_c = np.bincount(key, wc * y, minlength=len(eligible))
    ok = t & eligible[key] & (n_c[key] > 0)
    if not np.any(w[ok] > 0):
        return np.nan
    mu = np.divide(s_c, n_c, out=np.zeros_like(s_c), where=n_c > 0)
    diff = y[ok] - mu[key[ok]]
    if dose is None:
        return float(np.sum(w[ok] * diff) / np.sum(w[ok]))
    d = dose[ok]
    return float(np.sum(w[ok] * d * diff) / np.sum(w[ok] * d * d))


def att(df, treated, controls, outcome, key, cfg: Config, boot: Bootstrap | None, dose: str | None = None,
        keep_reps: bool = False) -> dict:
    y = df[outcome].to_numpy(float)
    kc, _ = pd.factorize(key)
    t, c = treated.to_numpy(bool), controls.to_numpy(bool)
    n_ctrl = np.bincount(kc, c.astype(float), minlength=kc.max() + 1)
    n_trt = np.bincount(kc, t.astype(float), minlength=kc.max() + 1)
    eligible = n_ctrl >= cfg.min_controls
    dz = df[dose].to_numpy(float) if dose else None
    est = _point(y, t, c, kc, eligible, np.ones(len(y)), dz)
    reps = np.array([_point(y, t, c, kc, eligible, boot.weights(r), dz) for r in range(len(boot.draws))]) if boot else np.array([])
    raw = reps
    reps = reps[np.isfinite(reps)]
    used = t & eligible[kc]
    extra = {"_reps": raw} if keep_reps else {}
    return {**extra, "estimate_C": est,
            "lo_C": float(np.percentile(reps, 2.5)) if len(reps) else np.nan,
            "hi_C": float(np.percentile(reps, 97.5)) if len(reps) else np.nan,
            "n_treated": int(t.sum()), "n_treated_matched": int(used.sum()),
            "n_controls_matched": int((c & eligible[kc] & (n_trt[kc] > 0)).sum()),
            "n_strata": int((eligible & (n_trt > 0)).sum()),
            "n_boot_requested": int(len(raw)), "n_boot_valid": int(len(reps)),
            "n_boot_missing": int(len(raw) - len(reps))}


def control_weights(df, treated, controls, key, cfg: Config) -> pd.Series:
    """Weight of each cell in the matched comparison: 1 for a matched treated cell; for a control, the treated cells of
    its stratum divided by the stratum's controls."""
    kc, _ = pd.factorize(key)
    t, c = treated.to_numpy(bool), controls.to_numpy(bool)
    n_ctrl = np.bincount(kc, c.astype(float), minlength=kc.max() + 1)
    n_trt = np.bincount(kc, t.astype(float), minlength=kc.max() + 1)
    eligible = n_ctrl >= cfg.min_controls
    w = np.where(c & eligible[kc], n_trt[kc] / np.maximum(n_ctrl[kc], 1), 0.0)
    return pd.Series(w + np.where(t & eligible[kc], 1.0, 0.0), index=df.index)


BALANCE_VARS = {"lst_pre": "LST 2014–15 (°C)", "ndvi_pre": "NDVI 2014–15", "emis_pre": "emissivity (fixed)",
                "coast_km": "distance to coast (km)", "elev": "elevation (m)", "ghsl_2015": "own building cover 2015",
                "ghsl_nb_2015": "building cover within 500 m, 2015", "d_nb_built": "surroundings built-up change"}


def balance(df, treated, controls, key, cfg: Config) -> pd.DataFrame:
    w = control_weights(df, treated, controls, key, cfg)
    t, c = treated & (w > 0), controls & (w > 0)
    rows = []
    for v, label in BALANCE_VARS.items():
        sd = np.sqrt((df.loc[treated, v].var() + df.loc[controls, v].var()) / 2)
        mt, mc_all = df.loc[t, v].mean(), df.loc[controls, v].mean()
        mc = np.average(df.loc[c, v], weights=w[c]) if c.any() else np.nan
        rows.append({"variable": label, "greened": mt, "all controls": mc_all, "matched controls": mc,
                     "std diff before": (df.loc[treated, v].mean() - mc_all) / sd if sd > 0 else np.nan,
                     "std diff after": (mt - mc) / sd if sd > 0 else np.nan})
    return pd.DataFrame(rows)


def yearly(df, treated, controls, key, cfg: Config, boot: Bootstrap) -> pd.DataFrame:
    """Per-pixel effect in every year relative to 2014–15: the whole path, not just the endpoints."""
    d = df.copy()
    rows = []
    for y in cfg.pre_years + cfg.mid_years + cfg.post_years:
        d["_dy"] = d[f"lst_{y}"] - d["lst_pre"]
        r = att(d, treated, controls, "_dy", key, cfg, boot, dose="n_greened_px")
        rows.append({"year": y, **{k: r[k] for k in ("estimate_C", "lo_C", "hi_C")}})
    return pd.DataFrame(rows)


def southern_belt_mask(df: pd.DataFrame, cfg: Config) -> pd.Series:
    """Exploratory latitude-belt proxy for the southern arc, not a pre-specified geomorphological delineation."""
    lower, upper = cfg.arc_lat_bounds
    return df["lat"].between(lower, upper, inclusive="left")


def cluster_diagnostics(df, treated, controls, key, cfg: Config, estimand: str,
                        variant: str = "main", dose: str | None = "n_greened_px") -> dict:
    """Counts and design concentration, not an effective sample-size estimate.

    Dose-squared shares describe the denominator of the through-origin slope among matched treated cells. They
    do not quantify every source of influence, and neither blocks nor regions are known to be independent.
    """
    w = control_weights(df, treated, controls, key, cfg)
    mt, mc = treated & (w > 0), controls & (w > 0)
    contribution = df.loc[mt, dose].astype(float) ** 2 if dose else pd.Series(1.0, index=df.index[mt])
    by_block = contribution.groupby(df.loc[mt, "block"]).sum()
    by_region = contribution.groupby(_grid_id(df.loc[mt], cfg.region_deg)).sum()

    def share(values):
        return float(values.max() / values.sum()) if len(values) and values.sum() > 0 else np.nan

    return {"variant": variant, "estimand": estimand,
            "n_treated": int(treated.sum()), "n_treated_matched": int(mt.sum()),
            "n_controls_matched": int(mc.sum()),
            "n_treated_blocks_matched": int(df.loc[mt, "block"].nunique()),
            "n_control_blocks_matched": int(df.loc[mc, "block"].nunique()),
            "n_comparison_blocks_matched": int(df.loc[mt | mc, "block"].nunique()),
            "n_treated_regions_matched": int(len(by_region)),
            "largest_block_dose_squared_share": share(by_block),
            "largest_region_dose_squared_share": share(by_region),
            "leverage_basis": "squared greened pixels" if dose else "matched treated cell count"}


def geographic_checks(df, treated, controls, key, cfg: Config, boot: Bootstrap) -> dict:
    """Exploratory geographic exclusions with the same spatial-bootstrap draws as the primary estimate."""
    arc = southern_belt_mask(df, cfg)
    variants = (("main", treated, controls),
                ("without southern latitude belt (exploratory proxy)", treated & ~arc, controls & ~arc))
    dose_rows, diagnostics = [], []
    for variant, t, c in variants:
        diagnostics.append(cluster_diagnostics(df, t, c, key, cfg, "per greened pixel", variant))
        for lo, hi in cfg.dose_bins:
            tb = t & df["n_greened_px"].between(lo, hi)
            label = f"{lo}–{hi}" if lo != hi else str(lo)
            r = att(df, tb, c, "d_lst", key, cfg, boot)
            reportable = r["n_treated_matched"] >= cfg.min_cells_reported
            if not reportable:
                r.update(estimate_C=np.nan, lo_C=np.nan, hi_C=np.nan)
            dose_rows.append({"variant": variant, "pixels": label, "reportable": reportable, **r,
                              "belt_lat_lower": cfg.arc_lat_bounds[0], "belt_lat_upper_exclusive": cfg.arc_lat_bounds[1]})
            diagnostics.append(cluster_diagnostics(df, tb, c, key, cfg, f"dose {label}", variant))

    regions = _grid_id(df, cfg.region_deg)
    matched = treated & (control_weights(df, treated, controls, key, cfg) > 0)
    base = att(df, treated, controls, "d_lst", key, cfg, None, dose="n_greened_px")["estimate_C"]
    total_dose_squared = float((df.loc[matched, "n_greened_px"] ** 2).sum())
    influence = []
    for region in sorted(regions[treated].unique()):
        drop = regions == region
        r = att(df, treated & ~drop, controls & ~drop, "d_lst", key, cfg, boot, dose="n_greened_px")
        dose_squared = float((df.loc[matched & drop, "n_greened_px"] ** 2).sum())
        influence.append({"excluded_region": region, "n_treated_removed": int((treated & drop).sum()),
                          "n_treated_matched_removed": int((matched & drop).sum()),
                          "n_controls_removed": int((controls & drop).sum()),
                          "removed_dose_squared_share": dose_squared / total_dose_squared if total_dose_squared else np.nan,
                          "main_estimate_C": base, "change_from_main_C": r["estimate_C"] - base, **r})
    return {"geographic_dose": pd.DataFrame(dose_rows), "region_influence": pd.DataFrame(influence),
            "cluster_diagnostics": pd.DataFrame(diagnostics)}


def _one_setting(df, s, key, cfg, boot):
    g, S = df["group"], df["setting"] == s
    stable = df["d_own_built"].abs() < cfg.stable_surface_max
    T, C, R = (g == "greened") & S, (g == "control") & S & stable, (g == "ring") & S & stable
    L = T & df["late"]
    out = {"n": {"greened": int(T.sum()), "late": int(L.sum()), "ring": int(((g == "ring") & S).sum()),
                 "ring_stable": int(R.sum()), "control": int(((g == "control") & S).sum()), "control_stable": int(C.sum()),
                 "greened_px": int(df.loc[T, "n_greened_px"].sum())}}
    est = {"per_pixel": att(df, T, C, "d_lst", key, cfg, boot, dose="n_greened_px", keep_reps=True),
           "all_greened": att(df, T, C, "d_lst", key, cfg, boot, keep_reps=True),
           "placebo_per_pixel": att(df, L, C, "d_lst_mid", key, cfg, boot, dose="n_greened_px"),
           "late_per_pixel": att(df, L, C, "d_lst", key, cfg, boot, dose="n_greened_px"),
           "ring": att(df, R, C, "d_lst", key, cfg, boot),
           "d_emis": att(df, T, C, "d_emis", key, cfg, None)}
    est["all_greened"]["mean_px"] = float(df.loc[T, "n_greened_px"].mean()) if T.any() else np.nan
    reps = {"all_greened": est["all_greened"].pop("_reps")}
    out["estimates"] = est
    dose = []
    for lo, hi in cfg.dose_bins:
        Tb = T & df["n_greened_px"].between(lo, hi)
        if Tb.sum() < cfg.min_cells_reported:
            dose.append({"pixels": f"{lo}–{hi}" if lo != hi else f"{lo}", "n_treated": int(Tb.sum()), "estimate_C": np.nan,
                         "lo_C": np.nan, "hi_C": np.nan, "n_treated_matched": 0, "d_ndvi": np.nan})
            continue
        r = att(df, Tb, C, "d_lst", key, cfg, boot, keep_reps=True)
        if r["n_treated_matched"] < cfg.min_cells_reported:
            r.update(estimate_C=np.nan, lo_C=np.nan, hi_C=np.nan)
        reps[f"dose {lo}-{hi}"] = r.pop("_reps")
        dose.append({"pixels": f"{lo}–{hi}" if lo != hi else f"{lo}", **r,
                     "d_ndvi": att(df, Tb, C, "d_ndvi", key, cfg, None)["estimate_C"]})
    out["dose"] = pd.DataFrame(dose)
    band = pd.Series(coast_band(df["coast_km"], cfg).to_numpy(), index=df.index)
    coast = []
    for b in coast_band([0.0], cfg).cat.categories:
        Tb = T & (band == b)
        if Tb.sum() < cfg.min_cells_reported:
            continue
        r = att(df, Tb, C, "d_lst", key, cfg, boot, dose="n_greened_px", keep_reps=True)
        reps[f"coast {b}"] = r.pop("_reps")
        if r["n_treated_matched"] < cfg.min_cells_reported:
            continue
        coast.append({"coast_band": b, **r, "mean_px": float(df.loc[Tb, "n_greened_px"].mean())})
    out["by_coast"] = pd.DataFrame(coast)
    out["reps"] = reps
    out["replicate_ids"] = boot.replicate_ids.copy()
    out["treated"] = T
    out["controls"], out["key"] = C, key
    out["balance"] = balance(df, T, C, key, cfg)
    out["yearly"] = yearly(df, T, C, key, cfg, boot)
    out["weights"] = control_weights(df, T, C, key, cfg)

    sens = []

    def add(name, **kw):
        r = att(df, kw.get("T", T), kw.get("C", C), "d_lst", kw.get("key", key), cfg, kw.get("boot", boot), dose="n_greened_px")
        sens.append({"variant": name, **r})

    add("main (as above)")
    add("controls whatever happened to their own surface", C=(g == "control") & S)
    add(f"controls more than {cfg.far_control_m:.0f} m from any greening", C=C & (df["dist_greened_m"] > cfg.far_control_m))
    add("strata: region and coastal band only", key=strata(df, cfg, coarse=True))
    add("baseline-only strata: omit neighbourhood built-up change", key=strata(df, cfg, baseline_only=True))
    add(f"regions of {cfg.region_deg / 2:.2f}° instead of {cfg.region_deg:.2f}°", key=strata(df, cfg, region_deg=cfg.region_deg / 2))
    top = _grid_id(df[T], cfg.region_deg).value_counts()
    if len(top) > 1:
        drop = _grid_id(df, cfg.region_deg) == top.index[0]
        add(f"without the region holding most greened cells ({top.iloc[0]} of {int(T.sum())})", T=T & ~drop, C=C & ~drop)
    arc = southern_belt_mask(df, cfg)
    add("without southern latitude belt (exploratory proxy)", T=T & ~arc, C=C & ~arc)
    add(f"bootstrap blocks of {cfg.block_deg * 2:.2f}° instead of {cfg.block_deg:.2f}°",
        boot=Bootstrap(_grid_id(df, cfg.block_deg * 2), len(boot.draws), cfg.seed))
    out["sensitivity"] = pd.DataFrame(sens)
    out.update(geographic_checks(df, T, C, key, cfg, boot))
    return out


def run_all(df: pd.DataFrame, cfg: Config) -> dict:
    boot = Bootstrap(df["block"], cfg.n_boot, cfg.seed)
    key = strata(df, cfg)
    res = {s: _one_setting(df, s, key, cfg, boot) for s in SETTINGS}
    T = df["group"] == "greened"
    res["blocks"] = boot.n_blocks
    res["emis_slope_per_001"] = float(df.loc[T, "emis_slope_per_001"].median())
    for s in SETTINGS:
        res[s]["reps"]["per_pixel"] = res[s]["estimates"]["per_pixel"].pop("_reps")
    a, b = (res[s]["reps"]["per_pixel"] for s in SETTINGS)
    d = (b - a)[np.isfinite(a) & np.isfinite(b)]
    res["difference_per_pixel"] = {"estimate_C": res[SETTINGS[1]]["estimates"]["per_pixel"]["estimate_C"]
                                   - res[SETTINGS[0]]["estimates"]["per_pixel"]["estimate_C"],
                                   "lo_C": float(np.percentile(d, 2.5)) if len(d) else np.nan,
                                   "hi_C": float(np.percentile(d, 97.5)) if len(d) else np.nan}
    res["top_region_share"] = {}
    for s in SETTINGS:
        px = df.loc[T & (df["setting"] == s)]
        reg = px.groupby(_grid_id(px, cfg.region_deg))["n_greened_px"].sum() if len(px) else pd.Series(dtype=float)
        res["top_region_share"][s] = float(reg.max() / reg.sum()) if len(reg) else 0.0
    res["pixels_by_setting"] = {s: int(df.loc[T & (df["setting"] == s), "n_greened_px"].sum()) for s in SETTINGS}
    return res
