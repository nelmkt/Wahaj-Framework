"""Render all four exported panel classes and primary matched donors."""
from pathlib import Path
import argparse

import numpy as np
import pandas as pd

from fw_config import Config, SETTINGS
import fw_panel as panel
import fw_matching as matching
import fw_figures as figures

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tables/figure1_population_comparison_v8.csv"


def comparison(df, cfg):
    key = panel.strata(df, cfg, baseline_only=True)
    treated = df.group.eq("greened")
    controls = df.cls.eq(3) & df.d_own_built.abs().lt(cfg.stable_surface_max)
    weights = matching.control_weights(df, treated, controls, key, cfg)
    matched_treated = treated & weights.gt(0)
    matched_controls = controls & weights.gt(0)
    representative_other = df.cls.eq(4)
    if not df.loc[matched_controls, "cls"].eq(3).all():
        raise ValueError("Matched donor pool contains a non-control class")
    rows = []
    for setting in ("all", *SETTINGS):
        mask = pd.Series(True, index=df.index) if setting == "all" else df.setting.eq(setting)
        for group, selected, weight in (
            ("primary matched treated", matched_treated & mask, pd.Series(1., index=df.index)),
            ("primary matched controls", matched_controls & mask, weights),
            ("other-land representative 10% sample", representative_other & mask, pd.Series(1., index=df.index)),
        ):
            if not selected.any():
                raise ValueError(f"Empty Figure 1 group: {setting}/{group}")
            w = weight[selected].to_numpy(float)
            rows.append({"setting": setting, "group": group, "n_cells": int(selected.sum()),
                         "weight_sum": float(w.sum()),
                         "baseline_lst_C": float(np.average(df.loc[selected, "lst_pre"], weights=w)),
                         "baseline_own_built_fraction": float(np.average(df.loc[selected, "ghsl_2015"], weights=w)),
                         "coast_km": float(np.average(df.loc[selected, "coast_km"], weights=w))})
    return pd.DataFrame(rows), matched_controls


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plot-only", action="store_true", help="Rerender the figure without replacing the v8 table")
    args = ap.parse_args()
    if OUT.exists() and not args.plot_only:
        raise SystemExit(f"Refusing to overwrite {OUT}")
    cfg = Config(panel_path=ROOT / "code/gee/panel.csv")
    df, _ = panel.load(cfg)
    table, used = comparison(df, cfg)
    if args.plot_only:
        saved = pd.read_csv(OUT)
        if not saved.set_index(["setting", "group"])[["n_cells", "baseline_lst_C", "baseline_own_built_fraction", "coast_km"]].round(10).equals(
                table.set_index(["setting", "group"])[["n_cells", "baseline_lst_C", "baseline_own_built_fraction", "coast_km"]].round(10)):
            raise SystemExit("Saved Figure 1 comparison differs from recomputation")
    else:
        table.to_csv(OUT, index=False)
    figures.fig_sample(df, used, ROOT / "figures/png", cfg.stable_surface_max,
                       match_label="primary")
    print(table.to_string(index=False))
    print("Figure 1 PNG/PDF regenerated from the same primary match as the comparison table")
