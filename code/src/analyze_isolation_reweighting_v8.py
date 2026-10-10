"""Saved-panel follow-up for outside 1–2-pixel isolation and class-slope spread.

Uses only the saved cell panel, native-pixel-count export, coastline table,
Figure 1 population check, and combined-exposure table. No Earth Engine call.
The primary match and block draws reproduce the existing pre-treatment-only
analysis. New files are versioned and existing results are never overwritten.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fw_config import Config, SETTINGS

ROOT = Path(__file__).resolve().parents[2]
PANEL = ROOT / "code/gee/panel.csv"
RAW = ROOT / "code/gee/pixel_external_counts_v8_raw.csv"
COAST = ROOT / "code/src/jeddah_coastline_ne10m.csv"
EFFECTS = ROOT / "tables/combined_exposure_effects_v8.csv"
FIG1 = ROOT / "tables/figure1_population_comparison_v8.csv"
OUT_COV = ROOT / "tables/isolation_covariates_v8_followup.csv"
OUT_WEIGHT = ROOT / "tables/isolation_reweighted_slopes_v8_followup.csv"
OUT_SPREAD = ROOT / "tables/combined_class_spread_v8_followup.csv"
OUT_NOTE = ROOT / "manuscript/ISOLATION_REWEIGHTING_V8.md"
RADII = (90, 180, 300)
CLASSES = (("1–2", 1, 2), ("3–4", 3, 4), ("5–8", 5, 8), ("9/9", 9, 9))


def coast_km(df: pd.DataFrame, coast: pd.DataFrame) -> np.ndarray:
    """Chunked nearest-point distance, algebraically identical to fw_panel."""
    k = 111.32 * np.cos(np.radians(float(df.lat.mean())))
    shore = np.c_[coast.lon.to_numpy(float) * k, coast.lat.to_numpy(float) * 110.57]
    cells = np.c_[df.lon.to_numpy(float) * k, df.lat.to_numpy(float) * 110.57]
    return np.concatenate([np.sqrt(((part[:, None, :] - shore[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
                           for part in np.array_split(cells, 30)])


def grid_id(df: pd.DataFrame, degrees: float) -> pd.Series:
    return (np.floor(df.lon / degrees).astype(int).astype(str) + "_" +
            np.floor(df.lat / degrees).astype(int).astype(str))


def load_saved(cfg: Config) -> pd.DataFrame:
    df = pd.read_csv(PANEL)
    years = cfg.pre_years + cfg.mid_years + cfg.post_years
    need = ["lon", "lat", "cls", "greened_frac", "built_pre", "built_post", "nb_built_pre", "nb_built_post",
            "elev", "dist_greened_m", "ghsl_2015", "ghsl_nb_2015", "late_frac", "rnd"] + [
                f"{v}_{y}" for v in ("ndvi", "ndbi", "lst", "emis") for y in years]
    df = df.dropna(subset=need).reset_index(drop=True).copy()
    df["coast_km"] = coast_km(df, pd.read_csv(COAST))
    df["lst_pre"] = df[[f"lst_{y}" for y in cfg.pre_years]].mean(axis=1)
    df["lst_post"] = df[[f"lst_{y}" for y in cfg.post_years]].mean(axis=1)
    df["emis_pre"] = df[[f"emis_{y}" for y in cfg.pre_years]].mean(axis=1)
    df["d_lst"] = df.lst_post - df.lst_pre
    df["n_greened_px"] = np.rint(df.greened_frac * 9).astype(int)
    df["setting"] = np.where(df.ghsl_nb_2015 >= cfg.builtup_min, SETTINGS[1], SETTINGS[0])
    df["d_own_built"] = df.built_post - df.built_pre
    df["block"] = grid_id(df, cfg.block_deg)
    df["southern_belt"] = df.lat.between(*cfg.arc_lat_bounds, inclusive="left")
    key = df.setting + "|" + grid_id(df, cfg.region_deg)
    for part in (np.digitize(df.coast_km, cfg.coast_bins_km),
                 np.digitize(df.ghsl_2015, cfg.own_ghsl_bins),
                 np.digitize(df.emis_pre, cfg.emis_bins)):
        key = key + "|" + pd.Series(part, index=df.index).astype(str)
    df["key"] = key

    raw_meta = json.loads(RAW.with_suffix(".json").read_text(encoding="utf-8"))
    if raw_meta.get("recipe_revision") != "unique-external-native30-v8d" or any(raw_meta.get("checks", {}).values()):
        raise SystemExit("The saved external-pixel export is not the zero-violation v8d version")
    if raw_meta.get("source_panel_sha256") != hashlib.sha256(PANEL.read_bytes()).hexdigest():
        raise SystemExit("External-pixel export does not match the saved panel")
    raw = pd.read_csv(RAW)
    if len(raw) != 1333 or len(set(zip(raw.lon.round(6), raw.lat.round(6)))) != 1333:
        raise SystemExit("Expected 1,333 unique raw classified cells")
    df["lon_key"], df["lat_key"] = df.lon.round(6), df.lat.round(6)
    raw["lon_key"], raw["lat_key"] = raw.lon.round(6), raw.lat.round(6)
    df = df.merge(raw.drop(columns=["lon", "lat"]), on=["lon_key", "lat_key"], how="left", validate="one_to_one", sort=False)
    greened = df.cls.eq(1)
    if df.loc[greened, "own_green_px"].isna().any() or not df.loc[greened, "own_green_px"].eq(df.loc[greened, "n_greened_px"]).all():
        raise SystemExit("Saved panel and native-pixel own doses do not reconcile")
    for radius in RADII:
        df[f"combined_green_px_{radius}m"] = df[f"combined_green_px_{radius}m"].fillna(0)

    controls = df.cls.eq(3) & df.setting.eq(SETTINGS[0]) & df.d_own_built.abs().lt(cfg.stable_surface_max)
    n_c = df.loc[controls, "key"].value_counts()
    matched = df.cls.eq(1) & df.setting.eq(SETTINGS[0]) & df.key.isin(n_c[n_c >= cfg.min_controls].index)
    row = pd.read_csv(FIG1).query("setting == @SETTINGS[0] and group == 'primary matched treated'")
    if len(row) != 1 or int(row.iloc[0].n_cells) != int(matched.sum()):
        raise SystemExit("Primary outside matched count disagrees with Figure 1")
    if not np.isclose(df.loc[matched, "coast_km"].mean(), float(row.iloc[0].coast_km), atol=1e-8):
        raise SystemExit("Coast distance disagrees with the saved Figure 1 population table")
    return df


def moments(x: np.ndarray, centre: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Linear, quadratic and pairwise terms for three fixed pre-treatment covariates."""
    z = (np.asarray(x, float) - centre) / scale
    return np.column_stack((z, z * z, z[:, 0] * z[:, 1], z[:, 0] * z[:, 2], z[:, 1] * z[:, 2]))


def entropy_weights(source: np.ndarray, target: np.ndarray, base_source: np.ndarray,
                    base_target: np.ndarray, tol: float = 1e-8) -> np.ndarray | None:
    """Minimum-KL positive weights matching nine target moments; None if no solution."""
    valid_s = np.asarray(base_source, float) > 0
    valid_t = np.asarray(base_target, float) > 0
    if valid_s.sum() < source.shape[1] + 1 or not valid_t.any():
        return None
    a = source[valid_s]
    bs = np.asarray(base_source, float)[valid_s]
    target_mean = np.average(target[valid_t], axis=0, weights=np.asarray(base_target, float)[valid_t])
    lam = np.zeros(a.shape[1])

    def state(candidate):
        logw = np.log(bs) + a @ candidate
        w = np.exp(logw - logw.max())
        w /= w.sum()
        mean = w @ a
        return w, mean, mean - target_mean

    for _ in range(100):
        w, mean, error = state(lam)
        if np.max(np.abs(error)) < tol:
            result = np.zeros(len(source))
            result[valid_s] = w
            return result
        centred = a - mean
        hessian = (centred * w[:, None]).T @ centred
        step = np.linalg.lstsq(hessian, error, rcond=1e-12)[0]
        current = np.linalg.norm(error)
        for half in range(30):
            candidate = lam - step / (2 ** half)
            if np.linalg.norm(state(candidate)[2]) < current:
                lam = candidate
                break
        else:
            return None
    return None


def slope(dose: np.ndarray, contrast: np.ndarray, weights: np.ndarray) -> float:
    denominator = np.sum(weights * dose * dose)
    return float(np.sum(weights * dose * contrast) / denominator) if denominator > 0 else np.nan


def interval(values: list[float]) -> tuple[float, float, int]:
    arr = np.asarray(values, float)
    arr = arr[np.isfinite(arr)]
    if not len(arr):
        return np.nan, np.nan, 0
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)), len(arr)


def analyze(df: pd.DataFrame, cfg: Config):
    setting = SETTINGS[0]
    controls = (df.cls.eq(3) & df.setting.eq(setting) & df.d_own_built.abs().lt(cfg.stable_surface_max)).to_numpy()
    key, key_labels = pd.factorize(df.key)
    n_ctrl = np.bincount(key, controls.astype(float), minlength=len(key_labels))
    eligible = n_ctrl >= cfg.min_controls
    treated = (df.cls.eq(1) & df.setting.eq(setting)).to_numpy() & eligible[key]
    doses = df.n_greened_px.to_numpy(float)
    y = df.d_lst.to_numpy(float)
    class_masks = {name: treated & (doses >= low) & (doses <= high) for name, low, high in CLASSES}
    if [int(class_masks[name].sum()) for name, _, _ in CLASSES] != [419, 183, 244, 160]:
        raise SystemExit("Primary outside class counts do not reconcile with saved effects")
    small = class_masks["1–2"]
    small_ix = np.flatnonzero(small)
    x = df.loc[small, ["lst_pre", "coast_km", "ghsl_2015"]].to_numpy(float)
    centre, scale = x.mean(axis=0), x.std(axis=0)
    if (scale <= 0).any():
        raise SystemExit("A required baseline covariate has no variation")
    design = moments(x, centre, scale)
    ext = {radius: df[f"external_green_px_{radius}m"].to_numpy(float) for radius in RADII}
    iso_small = {radius: ext[radius][small_ix] == 0 for radius in RADII}
    block_code, blocks = pd.factorize(df.block)
    rng = np.random.default_rng(cfg.seed)
    draws = rng.multinomial(len(blocks), np.full(len(blocks), 1 / len(blocks)), size=cfg.n_boot)

    def contrasts(block_weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        weighted_controls = block_weights * controls
        count = np.bincount(key, weighted_controls, minlength=len(key_labels))
        total = np.bincount(key, weighted_controls * y, minlength=len(key_labels))
        mean = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
        valid = treated & (count[key] > 0) & (block_weights > 0)
        return y - mean[key], valid

    cov_rows = []
    point_rw = {}
    point_weight = {}
    point_iso = {}
    point_non = {}
    full_contrast, _ = contrasts(np.ones(len(df)))
    for radius in RADII:
        isolated = iso_small[radius]
        for label, select in (("isolated", isolated), ("non-isolated", ~isolated)):
            ix = small_ix[select]
            selected = df.iloc[ix]
            cov_rows.append(dict(radius_m=radius, subset=label, n_cells=len(ix), n_blocks=selected.block.nunique(),
                                 status="diagnostic_only_under_20_cells" if len(ix) < 20 else "estimate",
                                 baseline_lst_C=float(selected.lst_pre.mean()), coast_km=float(selected.coast_km.mean()),
                                 ghsl_2015_built_fraction=float(selected.ghsl_2015.mean()),
                                 southern_belt_share=float(selected.southern_belt.mean()),
                                 mean_own_pixels=float(selected.n_greened_px.mean())))
        iix, nix = small_ix[isolated], small_ix[~isolated]
        w = entropy_weights(design[~isolated], design[isolated],
                            np.ones(len(nix)), np.ones(len(iix)))
        if w is None:
            raise SystemExit(f"Exact nine-moment calibration is infeasible at {radius} m")
        point_weight[radius] = w
        point_iso[radius] = slope(doses[iix], full_contrast[iix], np.ones(len(iix)))
        point_non[radius] = slope(doses[nix], full_contrast[nix], np.ones(len(nix)))
        point_rw[radius] = slope(doses[nix], full_contrast[nix], w)

    class_doses = {"own": doses, **{f"combined_{r}m": df[f"combined_green_px_{r}m"].to_numpy(float) for r in RADII}}
    class_point = {label: {name: slope(field[mask], full_contrast[mask], np.ones(mask.sum()))
                           for name, field in class_doses.items()} for label, mask in class_masks.items()}
    class_reps = {(label, exposure): [] for label in class_masks for exposure in class_doses}
    iso_reps = {radius: [] for radius in RADII}
    non_reps = {radius: [] for radius in RADII}
    rw_reps = {radius: [] for radius in RADII}
    for j, draw in enumerate(draws, 1):
        bw = draw[block_code].astype(float)
        contrast, valid = contrasts(bw)
        for label, mask in class_masks.items():
            select = mask & valid
            for exposure, field in class_doses.items():
                class_reps[label, exposure].append(slope(field[select], contrast[select], bw[select]))
        small_valid = valid[small_ix]
        for radius in RADII:
            isolated = iso_small[radius]
            i_local = isolated & small_valid
            n_local = ~isolated & small_valid
            iix, nix = small_ix[i_local], small_ix[n_local]
            iso_reps[radius].append(slope(doses[iix], contrast[iix], bw[iix]))
            non_reps[radius].append(slope(doses[nix], contrast[nix], bw[nix]))
            fit = entropy_weights(design[n_local], design[i_local], bw[nix], bw[iix])
            rw_reps[radius].append(slope(doses[nix], contrast[nix], fit) if fit is not None else np.nan)
        if j % 250 == 0:
            print(f"  saved-panel block draws {j}/{cfg.n_boot}", flush=True)

    saved = pd.read_csv(EFFECTS)
    rows = []
    for radius in RADII:
        isolated = iso_small[radius]
        iix, nix = small_ix[isolated], small_ix[~isolated]
        w = point_weight[radius]
        source_x = x[~isolated]
        target_x = x[isolated]
        weighted_mean = w @ source_x
        for label, point, reps in (("isolated", point_iso[radius], iso_reps[radius]),
                                   ("non-isolated", point_non[radius], non_reps[radius]),
                                   ("non-isolated, calibrated", point_rw[radius], rw_reps[radius])):
            lo, hi, valid = interval(reps)
            selected = iix if label == "isolated" else nix
            rows.append(dict(radius_m=radius, subset=label, n_cells=len(selected),
                             n_blocks=df.iloc[selected].block.nunique(),
                             status="diagnostic_only_under_20_cells" if len(selected) < 20 else "estimate",
                             per_own_pixel_slope_C=point, lo_C=lo, hi_C=hi, valid_block_draws=valid,
                             calibration_basis="2014–15 LST, coast km, GHSL own built fraction: linear, square, pairwise",
                             weight_ess=1 / np.sum(w * w) if label.endswith("calibrated") else np.nan,
                             largest_weight_share=w.max() if label.endswith("calibrated") else np.nan,
                             max_abs_balanced_mean_error=float(np.max(np.abs(weighted_mean - target_x.mean(axis=0))))
                             if label.endswith("calibrated") else np.nan,
                             weighted_southern_belt_share=float(w @ df.iloc[nix].southern_belt.to_numpy(float))
                             if label.endswith("calibrated") else np.nan))
        diff = np.asarray(rw_reps[radius]) - np.asarray(iso_reps[radius])
        lo, hi, valid = interval(diff)
        rows.append(dict(radius_m=radius, subset="calibrated non-isolated minus isolated", n_cells=len(nix),
                         n_blocks=df.iloc[nix].block.nunique(), status="paired_comparison",
                         per_own_pixel_slope_C=point_rw[radius] - point_iso[radius], lo_C=lo, hi_C=hi,
                         valid_block_draws=valid, calibration_basis="paired spatial-block draws; calibration refit"))

    spread_rows = []
    for exposure in class_doses:
        point_values = np.array([class_point[label][exposure] for label, _, _ in CLASSES])
        rep_matrix = np.array([class_reps[label, exposure] for label, _, _ in CLASSES]).T
        spread_reps = np.full(cfg.n_boot, np.nan)
        valid = np.isfinite(rep_matrix).all(axis=1)
        spread_reps[valid] = np.ptp(rep_matrix[valid], axis=1)
        lo, hi, n = interval(spread_reps)
        spread_rows.append(dict(exposure=exposure, radius_m=0 if exposure == "own" else int(exposure.split("_")[1][:-1]),
                                statistic="range across four class slopes", value_C_per_pixel=float(np.ptp(point_values)),
                                lo_C_per_pixel=lo, hi_C_per_pixel=hi, valid_block_draws=n))
        for (label, _, _), value in zip(CLASSES, point_values):
            clo, chi, cn = interval(class_reps[label, exposure])
            spread_rows.append(dict(exposure=exposure, radius_m=0 if exposure == "own" else int(exposure.split("_")[1][:-1]),
                                    statistic=f"{label} class slope", value_C_per_pixel=float(value),
                                    lo_C_per_pixel=clo, hi_C_per_pixel=chi, valid_block_draws=cn))
            if exposure == "own":
                reference = saved[(saved.setting == setting) & (saved.own_dose_class == label)].iloc[0]
                columns = ("own_slope_C_per_own_px", "own_lo_C", "own_hi_C")
            else:
                radius = int(exposure.split("_")[1][:-1])
                reference = saved[(saved.setting == setting) & (saved.own_dose_class == label) & saved.radius_m.eq(radius)].iloc[0]
                columns = ("combined_slope_C_per_combined_px", "combined_lo_C", "combined_hi_C")
            if not np.allclose([value, clo, chi], [reference[c] for c in columns], atol=1e-7):
                raise SystemExit(f"Recomputed {label}/{exposure} slope or interval disagrees with saved table")
    own_spread = np.array([class_reps[label, "own"] for label, _, _ in CLASSES]).T
    own_range = np.where(np.isfinite(own_spread).all(axis=1), np.ptp(own_spread, axis=1), np.nan)
    own_point_range = float(np.ptp([class_point[label]["own"] for label, _, _ in CLASSES]))
    for radius in RADII:
        exposure = f"combined_{radius}m"
        mat = np.array([class_reps[label, exposure] for label, _, _ in CLASSES]).T
        new_range = np.where(np.isfinite(mat).all(axis=1), np.ptp(mat, axis=1), np.nan)
        lo, hi, n = interval(new_range - own_range)
        new_point_range = float(np.ptp([class_point[label][exposure] for label, _, _ in CLASSES]))
        spread_rows.append(dict(exposure=exposure, radius_m=radius, statistic="combined minus own class-slope range",
                                value_C_per_pixel=new_point_range - own_point_range, lo_C_per_pixel=lo,
                                hi_C_per_pixel=hi, valid_block_draws=n))
    return pd.DataFrame(cov_rows), pd.DataFrame(rows), pd.DataFrame(spread_rows)


def main():
    for output in (OUT_COV, OUT_WEIGHT, OUT_SPREAD, OUT_NOTE):
        if output.exists():
            raise SystemExit(f"Refusing to overwrite existing output: {output}")
    cfg = Config(panel_path=PANEL)
    print("Loading the saved panel, coastline, and native-pixel counts...", flush=True)
    df = load_saved(cfg)
    print("Reconstructing the primary match and 2,000 spatial-block draws...", flush=True)
    cov, weights, spread = analyze(df, cfg)
    cov.to_csv(OUT_COV, index=False)
    weights.to_csv(OUT_WEIGHT, index=False)
    spread.to_csv(OUT_SPREAD, index=False)
    lines = ["# Outside small-patch isolation and reweighting", "",
             "Saved-data follow-up only. The primary pre-treatment-only matching rules and 2,000 spatial-block draws are unchanged. Coast distance uses the saved Jeddah coastline table and the original nearest-point formula. Built fraction is own-cell GHSL 2015 built-surface share; the southern belt is the earlier exploratory latitude proxy, 21.3°N ≤ latitude < 21.4°N.", "",
             "The calibrated non-isolated comparison uses positive minimum-KL weights to match isolated cells on the means, squares, and pairwise products of baseline LST, coast distance, and GHSL built fraction. We refit weights in each spatial-block draw. Effective sample size is the Kish value 1/Σw² for normalized point-estimate weights. This balances observed covariate moments, not unobserved site attributes or exposure mechanisms.", "",
             "The three new CSV tables contain subgroup covariates, isolated and calibrated non-isolated slopes, and class-slope dispersion. The script independently reproduces every saved outside class slope and interval before writing them. No intervention threshold or matching rule was changed.", "",
             "Reweighting cannot adjust for unmeasured land use, irrigation source, vegetation age, surface-material or emissivity differences, spatial spillover, or untreated temperature trends. It cannot create support where covariate combinations are absent; weight ESS and failed bootstrap calibrations expose that limitation. Southern-belt membership is reported but is not among the three balanced covariates. The combined-exposure denominator also counts external pixels for multiple focal cells, so convergence of its class slopes is descriptive and may be mechanical.", ""]
    OUT_NOTE.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT_COV.name}, {OUT_WEIGHT.name}, {OUT_SPREAD.name}, {OUT_NOTE.name}", flush=True)


if __name__ == "__main__":
    main()
