"""Exploratory baseline-LST checks on the saved observed panel.

The late-greener placebo is stratified by treated-cell baseline LST. The
caliper interaction is a within-selected-population association, not an
identification test for post-period confounding.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from fw_config import Config, SETTINGS
import fw_followup as followup
import fw_matching as matching
import fw_panel as panel


def interaction(y, dose, lst, ti, graph, weights, center, scale):
    denominator = np.asarray(graph @ weights).ravel()
    valid = (denominator > 0) & (weights[ti] > 0)
    if valid.sum() < 10:
        return np.nan, np.nan
    contrast = y[ti[valid]] - np.asarray(graph @ (weights * y)).ravel()[valid] / denominator[valid]
    z = (lst[ti[valid]] - center) / scale
    x = np.column_stack((dose[ti[valid]], dose[ti[valid]] * z))
    sw = np.sqrt(weights[ti[valid]])
    if np.linalg.matrix_rank(x * sw[:, None]) < 2:
        return np.nan, np.nan
    coefficients = np.linalg.lstsq(x * sw[:, None], contrast * sw, rcond=None)[0]
    return float(coefficients[0]), float(coefficients[1])


def main():
    cfg = Config()
    df, _ = panel.load(cfg)
    rs = followup.setup(df, cfg)[SETTINGS[0]]
    matched = rs["treated"] & (rs["weights"] > 0)
    controls, key = rs["controls"], rs["key"]
    boot = matching.Bootstrap(df.block, cfg.n_boot, cfg.seed)

    late = matched & df.late
    assert int(late.sum()) == 289
    terciles, edges = pd.qcut(df.loc[late, "lst_pre"], 3, labels=["cooler", "middle", "hotter"], retbins=True)
    placebo_rows = []
    for label in ("cooler", "middle", "hotter"):
        selected = late.copy()
        selected.loc[late] = terciles.eq(label).to_numpy()
        estimate = matching.att(df, selected, controls, "d_lst_mid", key, cfg, boot, dose="n_greened_px")
        placebo_rows.append(dict(tercile=label, n_cells=int(selected.sum()),
                                 n_blocks=int(df.loc[selected, "block"].nunique()),
                                 baseline_lst_min_C=float(df.loc[selected, "lst_pre"].min()),
                                 baseline_lst_max_C=float(df.loc[selected, "lst_pre"].max()),
                                 estimate_px_C=estimate["estimate_C"], lo_px_C=estimate["lo_C"],
                                 hi_px_C=estimate["hi_C"], n_boot_valid=estimate["n_boot_valid"],
                                 note="2014–15 to 2018–19; all matched controls retained"))
    pd.DataFrame(placebo_rows).to_csv("tables/reviewer_placebo_lst_terciles.csv", index=False)

    sd = ((df.loc[rs["treated"], followup.BALANCE].var() +
           df.loc[controls, followup.BALANCE].var()) / 2) ** .5
    ti, graph = followup.caliper_matrix(df, matched, controls, key, sd, .5, cfg)
    y = df.d_lst.to_numpy(float)
    dose = df.n_greened_px.to_numpy(float)
    lst = df.lst_pre.to_numpy(float)
    results = []
    for label, keep in [("all 0.5-SD caliper cells", np.ones(len(ti), dtype=bool)),
                        ("0.5-SD caliper, 1–2 pixels", dose[ti] <= 2)]:
        ids, g = ti[keep], graph[keep]
        center = float(np.mean(lst[ids]))
        scale = float(np.std(lst[ids], ddof=1))
        point, slope = interaction(y, dose, lst, ids, g, np.ones(len(df)), center, scale)
        draws = np.array([interaction(y, dose, lst, ids, g, boot.weights(i), center, scale)
                          for i in range(cfg.n_boot)])
        row = dict(subset=label, n_cells=len(ids), n_blocks=int(df.iloc[ids].block.nunique()),
                   baseline_lst_mean_C=center, baseline_lst_sd_C=scale,
                   effect_at_mean_lst_px_C=point,
                   interaction_per_baseline_sd_px_C=slope,
                   interaction_lo_C=float(np.nanpercentile(draws[:, 1], 2.5)),
                   interaction_hi_C=float(np.nanpercentile(draws[:, 1], 97.5)),
                   n_boot_valid=int(np.isfinite(draws[:, 1]).sum()),
                   note="Exploratory contrast association; dose times centered baseline LST; block bootstrap")
        results.append(row)
    pd.DataFrame(results).to_csv("tables/reviewer_caliper_lst_interaction.csv", index=False)
    print(f"Saved baseline-LST placebo terciles ({edges[0]:.2f} to {edges[-1]:.2f} °C) and caliper interactions.")


if __name__ == "__main__":
    main()
