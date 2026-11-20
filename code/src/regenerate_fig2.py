"""Re-estimate the interpretation-leading pre-treatment-only Figure 2 and save its rows."""
from pathlib import Path

import numpy as np
import pandas as pd

from fw_config import Config, SETTINGS
import fw_panel as panel
import fw_matching as matching
import fw_figures as figures
import fw_followup as followup


def build(df: pd.DataFrame, cfg: Config) -> dict:
    key = panel.strata(df, cfg, baseline_only=True)
    boot = matching.Bootstrap(df.block, cfg.n_boot, cfg.seed)
    result = {}
    estimates, doses, years = [], [], []
    for setting in SETTINGS:
        S = df.setting.eq(setting)
        T = df.group.eq("greened") & S
        stable = df.d_own_built.abs().lt(cfg.stable_surface_max)
        C = df.group.eq("control") & S & stable
        R = df.group.eq("ring") & S & stable
        L = T & df.late
        E = {
            "per_pixel": matching.att(df, T, C, "d_lst", key, cfg, boot, dose="n_greened_px"),
            "all_greened": matching.att(df, T, C, "d_lst", key, cfg, boot),
            "placebo_per_pixel": matching.att(df, L, C, "d_lst_mid", key, cfg, boot, dose="n_greened_px"),
            "ring": matching.att(df, R, C, "d_lst", key, cfg, boot),
        }
        for name, row in E.items():
            estimates.append(dict(setting=setting, estimate=name, **row))
        D = []
        for lo, hi in cfg.dose_bins:
            label = f"{lo}–{hi}" if lo != hi else str(lo)
            t = T & df.n_greened_px.between(lo, hi)
            row = matching.att(df, t, C, "d_lst", key, cfg, boot)
            if row["n_treated_matched"] < cfg.min_cells_reported:
                row.update(estimate_C=np.nan, lo_C=np.nan, hi_C=np.nan)
            D.append(dict(pixels=label, **row))
            doses.append(dict(setting=setting, pixels=label, **row))
        Y = matching.yearly(df, T, C, key, cfg, boot)
        years.append(Y.assign(setting=setting))
        result[setting] = {"estimates": E, "dose": pd.DataFrame(D), "yearly": Y,
                           "treated": T, "controls": C, "key": key,
                           "weights": matching.control_weights(df, T, C, key, cfg)}
    pd.DataFrame(estimates).to_csv("tables/logic_primary_figure2_estimates.csv", index=False)
    pd.DataFrame(doses).to_csv("tables/logic_primary_figure2_dose.csv", index=False)
    pd.concat(years, ignore_index=True).to_csv("tables/logic_primary_figure2_yearly.csv", index=False)
    mixing, coastal = followup.paired_contrasts(df, result, cfg, boot)
    mixing.to_csv("tables/logic_primary_area_mixing.csv", index=False)
    coastal.to_csv("tables/logic_primary_coastal_difference.csv", index=False)
    return result


if __name__ == "__main__":
    cfg = Config()
    df, _ = panel.load(cfg)
    result = build(df, cfg)
    figures.fig_cooling(result, cfg, Path("figures/png"), match_label="pre-treatment-only")
    print("Figure 2 PNG/PDF regenerated with pre-treatment-only matching and trace tables.")
