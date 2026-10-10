"""Analyses requested in the supervisor review of 5 October 2026 (Sections 1-2), run offline on the saved v11 panel.

Read-only on the package: imports fw_config, fw_panel, fw_matching and fw_model; never NEGI_Framework.py.
Writes NEW tables in tables_revision_r1/ and refuses to overwrite.

  r1_emissivity_diagnostic.csv   M1  product emissivity (ST_EMIS) before/after at greened cells vs controls
  r1_emissivity_bound.csv        M1  first-order bound: own-only slope if greened pixels' emissivity were raised
  r1_dose_contrasts.csv          M4  mean matched contrast for each dose 1-9 (primary match), block-bootstrap intervals
  r1_slope_intercept.csv         M4  own-only slope with and without an intercept
  r1_model_sensitivity.csv       M8  spatial CV: full model, without coordinates, without emissivity; random folds
  r1_variogram.csv               M8  empirical semivariogram of the full model's held-out residuals + exponential fit
  r1_negi_weights.csv            M7  NEGI on measured contrasts for alpha/beta in {0.5,1,2} and p in {0.5,1}
  r1_populations.csv             M5  Model A composition, cells dropped by the primary match, strata, blocks

usage: python revision_r1_analyses.py <package_root>      (PYTHONDONTWRITEBYTECODE=1)
"""
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "code" / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import curve_fit  # noqa: E402
from sklearn.model_selection import GroupKFold, KFold  # noqa: E402

import fw_matching as matching  # noqa: E402
import fw_model as model  # noqa: E402
import fw_panel as panel  # noqa: E402
from fw_config import SETTINGS, Config  # noqa: E402

OUT = ROOT / "tables_revision_r1"
OUT.mkdir(exist_ok=False)


def save(name, frame):
    path = OUT / name
    with open(path, "x", encoding="utf-8", newline="") as fh:
        frame.to_csv(fh, index=False)
    print(f"\n== {name}\n{frame.to_string(index=False)}")


cfg = Config(panel_path=ROOT / "code" / "gee" / "panel.csv", out_dir=ROOT)
df, meta = panel.load(cfg)
OUTSIDE, BUILT = SETTINGS
key = panel.strata(df, cfg, baseline_only=True)
boot = matching.Bootstrap(df.block, cfg.n_boot, cfg.seed)


def groups(setting):
    t = df.group.eq("greened") & df.setting.eq(setting)
    c = df.group.eq("control") & df.setting.eq(setting) & df.d_own_built.abs().lt(cfg.stable_surface_max)
    return t, c


T_OUT, C_OUT = groups(OUTSIDE)
W_OUT = matching.control_weights(df, T_OUT, C_OUT, key, cfg)
M_OUT = T_OUT & W_OUT.gt(0)

rows = []
classes = [("greened, 1–2 px", 1, 2), ("greened, 3–4 px", 3, 4), ("greened, 5–8 px", 5, 8), ("greened, 9/9 px", 9, 9)]
for label, lo, hi in classes + [("matched controls", None, None)]:
    sel = (M_OUT & df.n_greened_px.between(lo, hi)) if lo else (C_OUT & W_OUT.gt(0))
    d = df[sel]
    rows.append({"group": label, "setting": OUTSIDE, "n_cells": len(d),
                 "ndvi_pre": d.ndvi_pre.mean(), "ndvi_post": d.ndvi_post.mean(),
                 "emis_pre": d.emis_pre.mean(), "emis_post": d.emis_post.mean(), "d_emis_mean": d.d_emis.mean(),
                 "d_emis_abs_max": d.d_emis.abs().max(),
                 "share_emis_unchanged": float((d.d_emis.abs() < 5e-4).mean()),
                 "emis_slope_K_per_001_median": d.emis_slope_per_001.median()})
save("r1_emissivity_diagnostic.csv", pd.DataFrame(rows))

rows = []
slope_K = panel.emissivity_slope_K(df["lst_post"], df["emis_post"], cfg)
for delta in (0.0, 0.01, 0.02, 0.03):
    d = df.copy()
    adj = np.where(T_OUT, slope_K * delta * d.n_greened_px / 9.0, 0.0)
    d["_y"] = d["d_lst"] + adj
    fit = matching.att(d, T_OUT, C_OUT, "_y", key, cfg, boot, dose="n_greened_px")
    rows.append({"full_cover_emissivity_increase": delta, "setting": OUTSIDE, "n_matched": fit["n_treated_matched"],
                 "per_pixel_C": fit["estimate_C"], "lo_C": fit["lo_C"], "hi_C": fit["hi_C"],
                 "mean_adjustment_9of9_C": float(np.mean(adj[(T_OUT & df.n_greened_px.eq(9)).to_numpy()])) if delta else 0.0})
save("r1_emissivity_bound.csv", pd.DataFrame(rows))

rows = []
for setting in SETTINGS:
    t, c = groups(setting)
    for dose in range(1, 10):
        sel = t & df.n_greened_px.eq(dose)
        if not sel.any():
            continue
        fit = matching.att(df, sel, c, "d_lst", key, cfg, boot)
        used = sel & matching.control_weights(df, sel, c, key, cfg).gt(0)
        rows.append({"setting": setting, "dose": dose, "n_classified": int(sel.sum()),
                     "n_matched": fit["n_treated_matched"], "n_blocks": int(df.loc[used, "block"].nunique()),
                     "cell_C": fit["estimate_C"], "lo_C": fit["lo_C"], "hi_C": fit["hi_C"],
                     "n_boot_valid": fit["n_boot_valid"]})
save("r1_dose_contrasts.csv", pd.DataFrame(rows))


def slopes(y, t, c, kc, eligible, w, dose):
    """Per-cell matched contrasts (as fw_matching._point) and their weighted fits with and without intercept."""
    wc = w * c
    n_c = np.bincount(kc, wc, minlength=len(eligible))
    s_c = np.bincount(kc, wc * y, minlength=len(eligible))
    ok = t & eligible[kc] & (n_c[kc] > 0)
    ww = w[ok]
    if ww.sum() <= 0:
        return np.nan, np.nan, np.nan
    tau = y[ok] - (s_c / np.where(n_c > 0, n_c, 1))[kc[ok]]
    d = dose[ok]
    b0 = np.sum(ww * d * tau) / np.sum(ww * d * d)
    dm, tm = np.average(d, weights=ww), np.average(tau, weights=ww)
    b1 = np.sum(ww * (d - dm) * (tau - tm)) / np.sum(ww * (d - dm) ** 2)
    return b0, b1, tm - b1 * dm


rows = []
for setting in SETTINGS:
    t, c = groups(setting)
    y, dose = df["d_lst"].to_numpy(float), df["n_greened_px"].to_numpy(float)
    kc, _ = pd.factorize(key)
    tt, cc = t.to_numpy(bool), c.to_numpy(bool)
    eligible = np.bincount(kc, cc.astype(float), minlength=kc.max() + 1) >= cfg.min_controls
    point = slopes(y, tt, cc, kc, eligible, np.ones(len(y)), dose)
    reps = np.array([slopes(y, tt, cc, kc, eligible, boot.weights(r), dose) for r in range(cfg.n_boot)])
    reps = reps[np.isfinite(reps).all(1)]
    for j, name in enumerate(("slope_no_intercept_C_per_px", "slope_with_intercept_C_per_px", "intercept_C")):
        rows.append({"setting": setting, "quantity": name, "estimate": point[j],
                     "lo": float(np.percentile(reps[:, j], 2.5)), "hi": float(np.percentile(reps[:, j], 97.5)),
                     "n_boot_valid": len(reps)})
save("r1_slope_intercept.csv", pd.DataFrame(rows))

train = model.training_set(df, cfg.strict_exclusion_m)
y = train["lst_post"].to_numpy(float)
blocks = train["block"]
gkf = list(GroupKFold(n_splits=cfg.cv_folds).split(train, y, groups=blocks))
rkf = list(KFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.seed).split(train))
rows, oof_full = [], None
for label, feats, folds in (("all seven predictors, spatial folds", cfg.ml_features, gkf),
                            ("without longitude and latitude, spatial folds",
                             tuple(f for f in cfg.ml_features if f not in ("lon", "lat")), gkf),
                            ("without emissivity, spatial folds", tuple(f for f in cfg.ml_features if f != "emis"), gkf),
                            ("all seven predictors, random folds", cfg.ml_features, rkf)):
    c2 = Config(panel_path=cfg.panel_path, out_dir=ROOT, ml_features=feats)
    X = model.features(train, c2)
    oof = np.full(len(y), np.nan)
    for tr, te in folds:
        oof[te] = model._xgb(c2, cfg.seed).fit(X.iloc[tr], y[tr]).predict(X.iloc[te])
    res = y - oof
    rows.append({"model": label, "n": len(y), "n_blocks": int(blocks.nunique()), "folds": cfg.cv_folds,
                 "r2": float(1 - np.sum(res ** 2) / np.sum((y - y.mean()) ** 2)),
                 "rmse_C": float(np.sqrt(np.mean(res ** 2))), "mae_C": float(np.mean(np.abs(res))),
                 "mean_bias_C": float(np.mean(oof - y))})
    if oof_full is None:
        oof_full = oof
ms = pd.DataFrame(rows)
ref = pd.read_csv(ROOT / "tables" / "model_cv.csv").iloc[0]
assert abs(ms.r2[0] - ref.r2) < 1e-9 and abs(ms.rmse_C[0] - ref.rmse_C) < 1e-9, "full model does not reproduce model_cv.csv"
save("r1_model_sensitivity.csv", ms)

rng = np.random.default_rng(cfg.seed)
pick = rng.choice(len(y), size=min(5000, len(y)), replace=False)
kx = 111.32 * np.cos(np.radians(train["lat"].mean()))
xy = np.c_[train["lon"].to_numpy()[pick] * kx, train["lat"].to_numpy()[pick] * 110.57]
r = (y - oof_full)[pick]
i, j = np.triu_indices(len(pick), 1)
h = np.hypot(*(xy[i] - xy[j]).T)
g = 0.5 * (r[i] - r[j]) ** 2
edges = np.arange(0, 20.5, 0.5)
lag = np.digitize(h, edges) - 1
vg = pd.DataFrame({"lag_km": (edges[:-1] + edges[1:]) / 2,
                   "semivariance_C2": [g[lag == k].mean() for k in range(len(edges) - 1)],
                   "n_pairs": [int((lag == k).sum()) for k in range(len(edges) - 1)]})


def expo(hh, nug, sill, a):
    return nug + sill * (1 - np.exp(-3 * hh / a))


(nug, sill, a), _ = curve_fit(expo, vg.lag_km, vg.semivariance_C2, p0=(0.3, 1.0, 3.0), bounds=(0, [10, 10, 50]))
vg["fit_C2"] = expo(vg.lag_km, nug, sill, a)
vg["practical_range_km"] = a
vg["nugget_C2"], vg["partial_sill_C2"] = nug, sill
save("r1_variogram.csv", vg)

trd = pd.read_csv(ROOT / "tables_revision_v11" / "tradeoff_illustration_v11.csv")
rows = []
for _, rr in trd.iterrows():
    for p in (0.5, 1.0):
        cost = rr.greened_fraction_f ** p / trd.greened_fraction_f.max() ** p
        for ratio in (0.5, 1.0, 2.0):
            rows.append({"dose_class": rr.dose_class, "f": rr.greened_fraction_f, "normalized_benefit": rr.normalized_benefit,
                         "p": p, "alpha_over_beta": ratio, "negi_beta1": ratio * rr.normalized_benefit - cost})
save("r1_negi_weights.csv", pd.DataFrame(rows))

trA = model.training_set(df)
rows = [{"item": f"Model A training cells, group {g_}", "value": int(n)} for g_, n in trA.group.value_counts().items()]
rows += [{"item": "Model A training cells, total", "value": len(trA)},
         {"item": "Model B training cells, total", "value": len(train)},
         {"item": "ring cells in the representative sample (rnd < 0.10)", "value": int((df.group.eq("ring") & df.in_sample).sum())},
         {"item": "0.05-degree blocks in Model B training set", "value": int(train.block.nunique())},
         {"item": "0.05-degree blocks in the panel (bootstrap units)", "value": int(boot.n_blocks)}]
kc, _ = pd.factorize(key)
for setting in SETTINGS:
    t, c = groups(setting)
    n_ctrl = np.bincount(kc, c.to_numpy(float), minlength=kc.max() + 1)
    n_trt = np.bincount(kc, t.to_numpy(float), minlength=kc.max() + 1)
    used = (n_ctrl >= cfg.min_controls) & (n_trt > 0)
    dropped = t.to_numpy() & ~(n_ctrl >= cfg.min_controls)[kc]
    rows += [{"item": f"{setting}: greened cells", "value": int(t.sum())},
             {"item": f"{setting}: greened cells dropped (stratum with < {cfg.min_controls} controls)", "value": int(dropped.sum())},
             {"item": f"{setting}: greened cells dropped, of which in a stratum with 0 controls",
              "value": int((dropped & (n_ctrl[kc] == 0)).sum())},
             {"item": f"{setting}: strata used", "value": int(used.sum())},
             {"item": f"{setting}: median controls per used stratum", "value": float(np.median(n_ctrl[used]))},
             {"item": f"{setting}: blocks with matched greened cells",
              "value": int(df.loc[t & matching.control_weights(df, t, c, key, cfg).gt(0), "block"].nunique())}]
save("r1_populations.csv", pd.DataFrame(rows))
print("\nwrote", sorted(p.name for p in OUT.iterdir()))
