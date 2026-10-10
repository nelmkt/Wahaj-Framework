"""Analyze a verified *new* native-pixel external-count export.

Run after the owner completes the Earth Engine export, from the package root:
    python code/src/analyze_combined_exposure_v8.py

Produces three new CSV tables and a short interpretation note. The primary
effect analysis keeps pre-treatment-only matching; the Figure 5a comparison
reuses its 744/310 concurrent-change sensitivity population. Neighbour pixels
can belong to several focal cells, so combined exposure is a descriptive
neighbourhood attribution, not additive treated area or observed water use.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fw_config import Config, SETTINGS
import fw_matching as matching
import fw_panel as panel

ROOT = Path(__file__).resolve().parents[2]
PANEL = ROOT / "code/gee/panel.csv"
OLD_SCREEN = ROOT / "code/gee/pixel_isolation_raw.csv"
DEFAULT_RAW = ROOT / "code/gee/pixel_external_counts_v8_raw.csv"
REVISION = "unique-external-native30-v8d"
RADII = (90, 180, 300)
DOSES = (("all", 1, 9), ("1–2", 1, 2), ("3–4", 3, 4), ("5–8", 5, 8), ("9/9", 9, 9))
EFFECTS = ROOT / "tables/combined_exposure_effects_v8.csv"
ISOLATION = ROOT / "tables/combined_exposure_isolation_v8.csv"
RATIOS = ROOT / "tables/combined_exposure_fig5a_v8.csv"
NOTE = ROOT / "manuscript/COMBINED_EXPOSURE_RESULTS_V8.md"


def _ids(frame):
    return set(zip(frame.lon.round(6), frame.lat.round(6)))


def validate(raw_path: Path, df: pd.DataFrame) -> pd.DataFrame:
    meta_path = raw_path.with_suffix(".json")
    if not raw_path.is_file() or not meta_path.is_file():
        raise SystemExit("New pixel-count CSV/JSON are missing. Run the exporter first; no analysis output written.")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("recipe_revision") != REVISION:
        raise SystemExit("Unexpected pixel-count recipe revision")
    if meta.get("source_panel_sha256") != hashlib.sha256(PANEL.read_bytes()).hexdigest():
        raise SystemExit("Pixel-count export does not match the saved panel hash")
    if meta.get("source_pixel_isolation_sha256") != hashlib.sha256(OLD_SCREEN.read_bytes()).hexdigest():
        raise SystemExit("Pixel-count export does not match the saved 300 m isolation screen")
    if meta.get("radii_m") != list(RADII) or not isinstance(meta.get("checks"), dict):
        raise SystemExit("Missing radii/check metadata")
    if any(value != 0 for value in meta["checks"].values()):
        raise SystemExit("Exporter recorded one or more validation violations")
    raw = pd.read_csv(raw_path)
    required = {"lon", "lat", "cell_x", "cell_y", "cell_id", "own_green_px"}
    for radius in RADII:
        required.update({f"external_green_px_{radius}m", f"combined_green_px_{radius}m", f"isolated_{radius}m"})
    if not required.issubset(raw.columns) or len(raw) != meta.get("n_exported_cells") or len(raw) != 1333:
        raise SystemExit("Pixel-count columns or 1,333-cell count are invalid")
    treated = df[df.group.eq("greened")]
    if len(treated) != 1333 or len(_ids(raw)) != 1333 or _ids(raw) != _ids(treated):
        raise SystemExit("Pixel-count coordinate IDs do not match all classified treated cells")
    if raw.cell_id.nunique() != 1333:
        raise SystemExit("New projected cell IDs are not unique")
    for field in ["own_green_px", *(f"external_green_px_{r}m" for r in RADII),
                  *(f"combined_green_px_{r}m" for r in RADII)]:
        values = raw[field].to_numpy(float)
        if not (np.isfinite(values).all() and (values >= 0).all() and np.allclose(values, np.rint(values))):
            raise SystemExit(f"Invalid nonnegative pixel counts in {field}")
    if not (raw.external_green_px_90m.le(raw.external_green_px_180m).all() and
            raw.external_green_px_180m.le(raw.external_green_px_300m).all()):
        raise SystemExit("External pixel counts decrease with radius")
    for radius in RADII:
        ext = raw[f"external_green_px_{radius}m"]
        if not (raw[f"combined_green_px_{radius}m"].eq(raw.own_green_px + ext).all() and
                raw[f"isolated_{radius}m"].eq(ext.eq(0).astype(int)).all()):
            raise SystemExit(f"Combined exposure/isolation arithmetic fails at {radius} m")
    old = pd.read_csv(OLD_SCREEN)
    if len(old) != 1333 or _ids(old) != _ids(raw):
        raise SystemExit("Earlier pixel-centred screen has different treated-cell IDs")
    old = old.assign(lon_key=old.lon.round(6), lat_key=old.lat.round(6))
    r = raw.assign(lon_key=raw.lon.round(6), lat_key=raw.lat.round(6))
    check = r.merge(old[["lon_key", "lat_key", "max_external_green_within300_any_own30"]],
                    on=["lon_key", "lat_key"], how="left", validate="one_to_one")
    if not check.max_external_green_within300_any_own30.lt(.5).eq(check.external_green_px_300m.eq(0)).all():
        raise SystemExit("300 m isolation flags disagree with the earlier pixel-centred screen")
    left = df.assign(lon_key=df.lon.round(6), lat_key=df.lat.round(6))
    merged = left.merge(r.drop(columns=["lon", "lat"]), on=["lon_key", "lat_key"],
                        how="left", validate="one_to_one")
    green = merged.group.eq("greened")
    if (merged.loc[green, "own_green_px"].isna().any() or
            not merged.loc[green, "own_green_px"].eq(merged.loc[green, "n_greened_px"]).all()):
        raise SystemExit("Own greened counts disagree with the saved panel")
    for radius in RADII:
        field = f"combined_green_px_{radius}m"
        merged[field] = merged[field].fillna(0).astype(float)
        merged[f"external_green_px_{radius}m"] = merged[f"external_green_px_{radius}m"].fillna(0).astype(float)
    return merged


def _mean_exposure_draws(df, selected, controls, key, boot, cfg, field):
    """Mean exposure on the matched/resampled treated cells used by each ATT draw."""
    code, unique = pd.factorize(key)
    c = controls.to_numpy(bool)
    eligible = np.bincount(code, c.astype(float), minlength=len(unique)) >= cfg.min_controls
    t = selected.to_numpy(bool) & eligible[code]
    values = df[field].to_numpy(float)
    out = np.full(cfg.n_boot, np.nan)
    for i in range(cfg.n_boot):
        w = boot.weights(i)
        nc = np.bincount(code, w*c, minlength=len(unique))
        ok = t & (nc[code] > 0) & (w > 0)
        if ok.any():
            out[i] = np.average(values[ok], weights=w[ok])
    return out


def _interval(values):
    valid = np.asarray(values, float)
    valid = valid[np.isfinite(valid)]
    return (float(np.percentile(valid, 2.5)), float(np.percentile(valid, 97.5)), len(valid)) if len(valid) else (np.nan, np.nan, 0)


def effect_table(df, cfg, boot):
    key = panel.strata(df, cfg, baseline_only=True)
    rows = []
    for setting in SETTINGS:
        controls = df.group.eq("control") & df.setting.eq(setting) & df.d_own_built.abs().lt(cfg.stable_surface_max)
        for label, low, high in DOSES:
            treated = df.group.eq("greened") & df.setting.eq(setting) & df.n_greened_px.between(low, high)
            if not treated.any():
                continue
            print(f"  combined exposure: {setting}, own dose {label}", flush=True)
            matched = treated & matching.control_weights(df, treated, controls, key, cfg).gt(0)
            n, blocks = int(matched.sum()), int(df.loc[matched, "block"].nunique())
            reportable = n >= 20 and blocks >= 5
            own = matching.att(df, treated, controls, "d_lst", key, cfg,
                               boot if reportable else None, dose="n_greened_px", keep_reps=reportable)
            for radius in RADII:
                external = f"external_green_px_{radius}m"
                combined = f"combined_green_px_{radius}m"
                result = matching.att(df, treated, controls, "d_lst", key, cfg,
                                      boot if reportable else None, dose=combined, keep_reps=reportable)
                delta = result["estimate_C"] - own["estimate_C"]
                paired = result["_reps"] - own["_reps"] if reportable else []
                dlo, dhi, valid = _interval(paired)
                isolated = matched & df[external].eq(0)
                rows.append(dict(setting=setting, own_dose_class=label, radius_m=radius,
                                 status="estimate" if reportable else "diagnostic_only_under_20_cells_or_5_blocks",
                                 n_matched=n, n_blocks=blocks,
                                 mean_own_green_px=float(df.loc[matched, "n_greened_px"].mean()) if n else np.nan,
                                 mean_external_green_px=float(df.loc[matched, external].mean()) if n else np.nan,
                                 mean_combined_green_px=float(df.loc[matched, combined].mean()) if n else np.nan,
                                 share_with_external_green=float(df.loc[matched, external].gt(0).mean()) if n else np.nan,
                                 n_isolated_matched=int(isolated.sum()),
                                 n_isolated_blocks=int(df.loc[isolated, "block"].nunique()),
                                 own_slope_C_per_own_px=own["estimate_C"] if reportable else np.nan,
                                 own_lo_C=own["lo_C"] if reportable else np.nan,
                                 own_hi_C=own["hi_C"] if reportable else np.nan,
                                 combined_slope_C_per_combined_px=result["estimate_C"] if reportable else np.nan,
                                 combined_lo_C=result["lo_C"] if reportable else np.nan,
                                 combined_hi_C=result["hi_C"] if reportable else np.nan,
                                 combined_minus_own_slope_C=delta if reportable else np.nan,
                                 difference_lo_C=dlo if reportable else np.nan,
                                 difference_hi_C=dhi if reportable else np.nan,
                                 n_valid_paired_draws=valid))
    return pd.DataFrame(rows)


def isolation_table(df, cfg, boot):
    """Outside small-patch mixing excess by increasingly strict pixel isolation."""
    key = panel.strata(df, cfg, baseline_only=True)
    setting = SETTINGS[0]
    controls = df.group.eq("control") & df.setting.eq(setting) & df.d_own_built.abs().lt(cfg.stable_surface_max)
    small = df.group.eq("greened") & df.setting.eq(setting) & df.n_greened_px.between(1, 2)
    full = df.group.eq("greened") & df.setting.eq(setting) & df.n_greened_px.eq(9)
    ref = matching.att(df, full, controls, "d_lst", key, cfg, boot, keep_reps=True)
    if ref["n_treated_matched"] < 20:
        raise SystemExit("The 9/9 mixing reference lacks 20 matched cells")
    rows = []
    for label, radius in (("all", 0), ("90 m isolated", 90), ("180 m isolated", 180), ("300 m isolated", 300)):
        print(f"  small-patch isolation: {label}", flush=True)
        selected = small if radius == 0 else small & df[f"external_green_px_{radius}m"].eq(0)
        matched = selected & matching.control_weights(df, selected, controls, key, cfg).gt(0)
        n, blocks = int(matched.sum()), int(df.loc[matched, "block"].nunique())
        diagnostic = n < 20 or blocks < 5
        if not n:
            rows.append(dict(setting=setting, subset=label, radius_m=radius, n_matched=0, n_blocks=0,
                             status="not_testable_no_cells"))
            continue
        measured = matching.att(df, selected, controls, "d_lst", key, cfg,
                                None if diagnostic else boot, keep_reps=not diagnostic)
        slope = matching.att(df, selected, controls, "d_lst", key, cfg,
                             None if diagnostic else boot, dose="n_greened_px", keep_reps=not diagnostic)
        mean_own = float(df.loc[matched, "n_greened_px"].mean())
        benchmark = ref["estimate_C"] * mean_own / 9
        excess = measured["estimate_C"] - benchmark
        if diagnostic:
            elo, ehi, valid = np.nan, np.nan, 0
        else:
            own_draws = _mean_exposure_draws(df, matched, controls, key, boot, cfg, "n_greened_px")
            elo, ehi, valid = _interval(measured["_reps"] - ref["_reps"] * own_draws / 9)
        rows.append(dict(setting=setting, subset=label, radius_m=radius, n_matched=n, n_blocks=blocks,
                         status="diagnostic_only_under_20_cells_or_5_blocks" if diagnostic else "selected_site_sensitivity",
                         mean_own_green_px=mean_own, cell_contrast_C=measured["estimate_C"],
                         own_slope_C_per_own_px=slope["estimate_C"],
                         linear_mixing_benchmark_C=benchmark, excess_C=excess,
                         excess_lo_C=elo, excess_hi_C=ehi, n_valid_draws=valid,
                         benchmark_reference="primary outside 9/9 matched cell contrast; inherits any bias there"))
    return pd.DataFrame(rows)


def ratio_table(df, cfg):
    """Re-express old Figure 5a on its *same narrower matched cells*."""
    key = panel.strata(df, cfg, baseline_only=False)
    water = pd.read_csv(ROOT / "tables/water.csv")
    central = float(water.loc[water.level.eq("central"), "m3_per_ha_yr"].item())
    saved = pd.read_csv(ROOT / "tables/decision_summary.csv")
    rows = []
    for setting in SETTINGS:
        controls = df.group.eq("control") & df.setting.eq(setting) & df.d_own_built.abs().lt(cfg.stable_surface_max)
        for label, low, high in DOSES[1:]:
            treated = df.group.eq("greened") & df.setting.eq(setting) & df.n_greened_px.between(low, high)
            matched = treated & matching.control_weights(df, treated, controls, key, cfg).gt(0)
            n = int(matched.sum())
            if n < 10:
                continue
            basis = f"measured: {'9' if label == '9/9' else label} of 9 pixels greened"
            row = saved[(saved.setting == setting) & (saved.basis == basis)]
            if len(row) != 1:
                raise SystemExit(f"Old Figure 5a source row missing: {setting}/{label}")
            row = row.iloc[0]
            fit = matching.att(df, treated, controls, "d_lst", key, cfg, None)
            mean_own = float(df.loc[matched, "n_greened_px"].mean())
            old_calc = mean_own * panel.PIXEL_HA * central / abs(fit["estimate_C"])
            if (int(row.cells) != n or not np.isclose(row.pixels_mean, mean_own, atol=1e-6) or
                    not np.isclose(row.cooling_C, fit["estimate_C"], atol=1e-6) or
                    not np.isclose(row.m3_per_degree, old_calc, atol=1e-6)):
                raise SystemExit(f"Saved Figure 5a row does not reconcile: {setting}/{label}")
            for radius in RADII:
                external = f"external_green_px_{radius}m"
                combined = f"combined_green_px_{radius}m"
                mean_external = float(df.loc[matched, external].mean())
                mean_combined = float(df.loc[matched, combined].mean())
                new_calc = mean_combined * panel.PIXEL_HA * central / abs(fit["estimate_C"])
                rows.append(dict(setting=setting, own_dose_class=label, radius_m=radius,
                                 population="744 outside / 310 built-up concurrent-change sensitivity",
                                 n_matched=n, mean_own_green_px=mean_own,
                                 mean_external_green_px=mean_external, mean_combined_green_px=mean_combined,
                                 matched_cell_cooling_C=fit["estimate_C"],
                                 assumed_depth_m=central / 10000,
                                 old_fig5a_m3_year_per_C=old_calc,
                                 combined_footprint_m3_year_per_C=new_calc,
                                 ratio_multiplier=new_calc / old_calc,
                                 ranking_claim="withdrawn",
                                 scope="hypothetical common-depth footprint; shared external pixels can be counted for multiple focal cells"))
    return pd.DataFrame(rows)


def write_note(effects, isolation, ratios, raw):
    try:
        source_label = raw.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        source_label = str(raw.resolve())
    lines = ["# Combined exposure follow-up (descriptive)", "",
             f"Source: `{source_label}`; verified against saved panel and prior pixel-centred 300 m screen.",
             "", "Unique external greened 30 m pixels are counted once per focal cell/radius. A pixel can nevertheless be included by more than one focal cell; these cell-level footprints cannot be summed into a municipal planted-area or water budget.",
             "", "## Primary matched effects", "",
             "The table `tables/combined_exposure_effects_v8.csv` reports own-cell and own-plus-external zero-intercept LST slopes by setting, own-dose class and radius with paired spatial-block intervals. The combined slope is an attributed neighbourhood-exposure summary, not an identified causal marginal effect of one additional planted pixel.",
             "", "## Small-patch excess by isolation distance", "",
             "Negative excess means more measured cooling than linear own-cell area mixing from the primary outside 9/9 reference. The reference may itself be biased. Isolated subsets differ in site composition; decreasing excess with stricter isolation is consistent with neighbour contamination but does not identify it.", "",
             "| Subset | Matched cells | Blocks | Excess °C [95% interval] | Status |",
             "|---|---:|---:|---|---|"]
    for r in isolation.itertuples():
        point = f"{r.excess_C:+.3f}" if pd.notna(r.excess_C) else "unavailable"
        ci = f" [{r.excess_lo_C:+.3f}, {r.excess_hi_C:+.3f}]" if pd.notna(r.excess_lo_C) else " [interval withheld]"
        lines.append(f"| {r.subset} | {r.n_matched} | {r.n_blocks} | {point}{ci} | {r.status} |")
    lines += ["", "## Higher-dose adjacency and resource ratios", ""]
    for dose in ("5–8", "9/9"):
        for radius in RADII:
            r = effects[(effects.setting == SETTINGS[0]) & effects.own_dose_class.eq(dose) & effects.radius_m.eq(radius)].iloc[0]
            lines.append(f"Outside {dose}, {radius} m: {int(r.n_isolated_matched)} of {int(r.n_matched)} matched cells are isolated; mean unique external count {r.mean_external_green_px:.2f}. The combined slope is in the effects table. Isolation contrasts for these high-dose classes are not testable from these sparse isolated subsets.")
    lines += ["", "`tables/combined_exposure_fig5a_v8.csv` gives old and combined-footprint central ratios on the exact narrower Figure 5a matched cells. Their change is driven by substituting own-plus-external counted area in the hypothetical common-irrigation numerator; the LST contrast is held fixed. The previous small-patch ranking remains withdrawn. These ratios are not observed irrigation, net energy or population benefit.", ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    args = ap.parse_args()
    outputs = (EFFECTS, ISOLATION, RATIOS, NOTE)
    if any(path.exists() for path in outputs):
        raise SystemExit("Refusing to overwrite an existing combined-exposure output; version a rerun")
    cfg = Config(panel_path=PANEL)
    print("Loading the saved panel and verifying the new pixel-count export...", flush=True)
    df, _ = panel.load(cfg)
    df = validate(args.raw, df)
    print(f"Verified 1,333 classified cells; computing {cfg.n_boot} spatial-block bootstrap draws per reported comparison...", flush=True)
    boot = matching.Bootstrap(df.block, cfg.n_boot, cfg.seed)
    effects = effect_table(df, cfg, boot)
    isolation = isolation_table(df, cfg, boot)
    print("Rechecking the saved Figure 5a rows and calculating conditional ratios...", flush=True)
    ratios = ratio_table(df, cfg)
    note = write_note(effects, isolation, ratios, args.raw)
    for path, data in ((EFFECTS, effects), (ISOLATION, isolation), (RATIOS, ratios)):
        data.to_csv(path, index=False)
    NOTE.write_text(note, encoding="utf-8")
    print(f"Wrote {EFFECTS.name}, {ISOLATION.name}, {RATIOS.name} and {NOTE.name}")
    print("Combined exposure remains descriptive; Figure 5a ratios remain hypothetical and the small-patch ranking is withdrawn.")


if __name__ == "__main__":
    main()
