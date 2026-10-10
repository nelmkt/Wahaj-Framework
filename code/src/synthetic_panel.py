"""A synthetic panel with the columns of code/gee/panel.csv and a known answer, for tests and dry runs.

Truth: greening all 9 pixels of a cell cools it by TRUE_FULL °C (each greened pixel by TRUE_FULL / 9); ring cells cool by
TRUE_RING °C; no cell changes before its greening. Confounding: greened cells sit more often in surroundings that were
built up over the period, which warms every cell there by DEV_WARMING °C, so a naive comparison is biased. Other
cells (cls 4) carry existing vegetation with the same NDVI–LST relation as greening, so a correct model passes the test.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

YEARS = (2014, 2015, 2018, 2019, 2024, 2025)
TRUE_FULL, TRUE_RING, DEV_WARMING = -3.0, -0.4, 1.5


def make(n_control=8000, n_ring=900, n_greened=900, n_other=3000, seed=0) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    rows = []
    for cls, n in ((1, n_greened), (2, n_ring), (3, n_control), (4, n_other)):
        lon = rng.uniform(39.10, 39.30, n)
        lat = rng.uniform(21.35, 21.75, n)
        p_dev = 0.6 if cls in (1, 2) else 0.25
        dev = rng.random(n) < p_dev
        nb_pre = rng.uniform(0.05, 0.6, n)
        nb_post = np.clip(nb_pre + np.where(dev, rng.uniform(0.2, 0.4, n), rng.uniform(-0.02, 0.03, n)), 0, 1)
        built_pre = rng.uniform(0.0, 0.7, n)
        built_post = np.clip(built_pre + rng.normal(0, 0.05, n) + (rng.random(n) < 0.15) * 0.4 * (cls == 3), 0, 1)
        n_px = rng.integers(1, 10, n) if cls == 1 else np.zeros(n, int)
        urban = rng.random(n) < 0.35
        ghsl_nb = np.where(urban, rng.uniform(0.06, 0.4, n), rng.uniform(0.0, 0.045, n))
        ghsl_own = np.where(urban, rng.uniform(0.0, 0.5, n), rng.uniform(0.0, 0.04, n))
        late = (rng.random(n) < 0.5) & (cls == 1)
        elev = rng.uniform(0, 60, n)
        emis0 = rng.uniform(0.95, 0.975, n)
        base = 45 + 0.05 * elev + 2.0 * (lon - 39.1) * 10 + rng.normal(0, 1.0, n)
        veg0 = rng.uniform(0.0, 0.5, n) * (cls == 4)
        d = {"rnd": rng.uniform(0, 0.1, n) if cls in (3, 4) else rng.uniform(0, 1, n), "lon": lon, "lat": lat, "cls": cls, "greened_frac": n_px / 9, "never_frac": np.full(n, float(cls != 1)),
             "built_pre": built_pre, "built_post": built_post, "nb_built_pre": nb_pre, "nb_built_post": nb_post,
             "elev": elev, "ghsl_2015": ghsl_own, "ghsl_nb_2015": ghsl_nb, "pre_bare_frac": np.ones(n),
             "late_frac": np.where(late, n_px / 9, 0.0), "dist_greened_m": np.where(cls == 1, 0, np.where(cls == 2, 60, rng.uniform(310, 3000, n)))}
        for y in YEARS:
            frac_dev = {2014: 0, 2015: 0, 2018: 0.3, 2019: 0.4, 2024: 1, 2025: 1}[y]
            green_on = np.where(late, y >= 2024, y >= 2018) & (cls == 1)
            ndvi = 0.08 + veg0 + rng.normal(0, 0.02, n) + green_on * 0.5 * n_px / 9
            emis = emis0
            lst = (base + (y - 2014) * 0.05 + dev * DEV_WARMING * frac_dev + green_on * TRUE_FULL * n_px / 9
                   + TRUE_FULL / 0.5 * veg0
                   + (cls == 2) * (y >= 2024) * TRUE_RING + rng.normal(0, 0.6, n))
            ndbi = 0.10 - 0.3 * (ndvi - 0.08) + rng.normal(0, 0.02, n)
            d[f"ndvi_{y}"], d[f"lst_{y}"], d[f"emis_{y}"], d[f"ndbi_{y}"] = ndvi, lst, emis, ndbi
        rows.append(pd.DataFrame(d))
    df = pd.concat(rows, ignore_index=True)
    meta = {"years": YEARS, "reference_et_mm_per_year": {"2024": 2260.0, "2025": None}, "greened_area_km2": 6.0,
            "scenes_per_year_month": {f"{y}-{m:02d}": 2 for y in YEARS for m in range(5, 10)}, "synthetic": True}
    return df, meta


def write(path: Path, **kw) -> Path:
    df, meta = make(**kw)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    path.with_name(path.stem + "_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return path


if __name__ == "__main__":
    import sys
    print(write(Path(sys.argv[1] if len(sys.argv) > 1 else "synthetic/greening_panel.csv")))
