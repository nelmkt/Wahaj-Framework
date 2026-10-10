"""Load the cell panel (remote sensing) and derive everything the later steps use."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from fw_config import SETTINGS, Config

COASTLINE = Path(__file__).resolve().parent / "jeddah_coastline_ne10m.csv"
GROUPS = {1: "greened", 2: "ring", 3: "control", 4: "other"}
PIXEL_HA = 0.09
CELL_HA = 9 * PIXEL_HA


def coast_distance_km(lon, lat, coast: pd.DataFrame) -> np.ndarray:
    k = 111.32 * np.cos(np.radians(float(np.mean(lat))))
    tree = cKDTree(np.c_[coast["lon"].to_numpy() * k, coast["lat"].to_numpy() * 110.57])
    return tree.query(np.c_[np.asarray(lon) * k, np.asarray(lat) * 110.57])[0]


def emissivity_slope_K(lst_C, emis, cfg: Config) -> np.ndarray:
    """dT/dε of a single-channel retrieval, first order (Wien approximation, no atmospheric term): -(λT²/c2)/ε, in K
    per unit emissivity. Leaving out the downwelling atmospheric term makes this an upper bound on its size."""
    T = np.asarray(lst_C, float) + 273.15
    return -(cfg.lambda_b10_um * T ** 2 / cfg.c2_umK) / np.asarray(emis, float)


def _mean(df, var, years):
    return df[[f"{var}_{y}" for y in years]].mean(axis=1)


def load(cfg: Config) -> tuple[pd.DataFrame, dict]:
    path = Path(cfg.panel_path)
    df = pd.read_csv(path)
    if "lon" not in df and ".geo" in df:
        xy = df[".geo"].map(lambda g: json.loads(g)["coordinates"])
        df["lon"], df["lat"] = xy.str[0], xy.str[1]
    meta_path = path.with_name(path.stem + "_meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    need = ["lon", "lat", "cls", "greened_frac", "built_pre", "built_post", "nb_built_pre", "nb_built_post", "elev",
            "dist_greened_m", "ghsl_2015", "ghsl_nb_2015", "late_frac", "rnd"] + [f"{v}_{y}" for v in ("ndvi", "ndbi", "lst", "emis")
                                 for y in cfg.pre_years + cfg.mid_years + cfg.post_years]
    missing = [c for c in need if c not in df]
    if missing:
        raise SystemExit(f"{path}: missing columns {missing} (export it with code/gee/export_panel.py)")
    n_read = len(df)
    df = df.dropna(subset=need).copy()
    info = {"n_read": n_read, "n_dropped_missing": n_read - len(df)}
    return derive(df, cfg), {**meta, **info}


def derive(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = df.copy()
    df["group"] = df["cls"].astype(int).map(GROUPS)
    df["n_greened_px"] = np.rint(df["greened_frac"] * 9).astype(int)
    for per, yrs in (("pre", cfg.pre_years), ("mid", cfg.mid_years), ("post", cfg.post_years)):
        for v in ("ndvi", "ndbi", "lst", "emis"):
            df[f"{v}_{per}"] = _mean(df, v, yrs)
    df["d_lst"] = df["lst_post"] - df["lst_pre"]
    df["d_lst_mid"] = df["lst_mid"] - df["lst_pre"]
    df["d_ndvi"] = df["ndvi_post"] - df["ndvi_pre"]
    df["d_emis"] = df["emis_post"] - df["emis_pre"]
    df["emis_slope_per_001"] = emissivity_slope_K(df["lst_post"], df["emis_post"], cfg) * 0.01
    df["late"] = (df["group"] == "greened") & (np.rint(df["late_frac"] * 9) == df["n_greened_px"])
    df["setting"] = np.where(df["ghsl_nb_2015"] >= cfg.builtup_min, SETTINGS[1], SETTINGS[0])
    coast = pd.read_csv(COASTLINE)
    df["coast_km"] = coast_distance_km(df["lon"], df["lat"], coast)
    df["d_own_built"] = df["built_post"] - df["built_pre"]
    df["d_nb_built"] = df["nb_built_post"] - df["nb_built_pre"]
    df["block"] = _grid_id(df, cfg.block_deg)
    df["in_sample"] = df["rnd"] < cfg.sample_fraction
    return df


def _grid_id(df, deg):
    return (np.floor(df["lon"] / deg).astype(int).astype(str) + "_" + np.floor(df["lat"] / deg).astype(int).astype(str))


def strata(df: pd.DataFrame, cfg: Config, coarse: bool = False, region_deg: float | None = None,
           baseline_only: bool = False) -> pd.Series:
    """Coarsened exact matching key.

    The pipeline's legacy ``main`` key additionally conditions on neighbourhood built-up change measured over the treatment period.
    Excluding greened land from that measure does not guarantee it is unaffected by greening or by a shared project.
    Its causal interpretation therefore requires a post-treatment conditioning assumption. ``baseline_only`` drops
    only that change variable and retains all other key covariates; it leads manuscript interpretation and is not
    the region/coast-only sensitivity.
    """
    parts = [df["setting"].astype(str), _grid_id(df, region_deg or cfg.region_deg),
             np.digitize(df["coast_km"], cfg.coast_bins_km).astype(str)]
    if not coarse:
        parts += [np.digitize(df["ghsl_2015"], cfg.own_ghsl_bins).astype(str)]
        if not baseline_only:
            parts += [np.digitize(df["d_nb_built"], cfg.nb_change_bins).astype(str)]
        parts += [np.digitize(df["emis_pre"], cfg.emis_bins).astype(str)]
    key = parts[0]
    for p in parts[1:]:
        key = key + "|" + p
    return key


def coast_band(km, cfg: Config) -> pd.Series:
    """Distance to the coast in the bands used for matching and for comparing places."""
    e = cfg.coast_bins_km
    labels = [f"< {e[0]:g} km"] + [f"{a:g}–{b:g} km" for a, b in zip(e[:-1], e[1:])] + [f"≥ {e[-1]:g} km"]
    return pd.cut(pd.Series(np.asarray(km, float)), [0, *e, np.inf], right=False, labels=labels)
