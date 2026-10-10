"""Runs the framework in order; reads like the Methods section.

1 remote sensing   the cell panel from Landsat 8 (code/gee/export_panel.py)
2 measured cooling matched difference-in-differences where land actually greened
3 machine learning XGBoost model of summer LST across the city, spatially cross-validated
4 test             the model's predicted cooling for the greened cells against the measured cooling
5 water and energy irrigation water and the electricity to supply it
6 decision support cooling, water and energy per option, from the model only where it passed the test
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import fw_decision as decision, fw_figures as figures, fw_ledger as ledger, fw_matching as matching, fw_model as model, fw_panel as panel, fw_validate as validate
from fw_config import SETTINGS, Config
from fw_report import write_report


def _jsonable(o):
    if isinstance(o, pd.DataFrame):
        return _jsonable(o.to_dict(orient="records"))
    if isinstance(o, (pd.Series, model.Model)):
        return None
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items() if not isinstance(v, (pd.Series, np.ndarray, model.Model))}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, float) and not np.isfinite(o):
        return None
    if isinstance(o, Path):
        root = Path(__file__).resolve().parents[2]
        return str(o.relative_to(root)) if o.is_absolute() and o.is_relative_to(root) else str(o)
    return o


def run(cfg: Config) -> dict:
    out = Path(cfg.out_dir)
    tables, figs = out / "tables", out / "figures_png"
    tables.mkdir(parents=True, exist_ok=True)
    (out / "manuscript").mkdir(parents=True, exist_ok=True)

    df, meta = panel.load(cfg)
    meta["analysis_provenance"] = "Matched estimates and Models A/B were computed in this framework invocation; see the follow-up record for any later table refreshes."
    print(f"1 panel: {len(df):,} cells ({meta['n_dropped_missing']:,} dropped for a missing year)", flush=True)

    R = matching.run_all(df, cfg)
    for st in SETTINGS:
        e = R[st]["estimates"]["per_pixel"]
        print(f"2 measured, {st}: {e['estimate_C']:+.2f} deg C per greened pixel [{e['lo_C']:+.2f}, {e['hi_C']:+.2f}]", flush=True)

    cv = model.spatial_cv(model.training_set(df, cfg.strict_exclusion_m), cfg)
    print(f"3 model: spatial cross-validation R2 {cv['r2']:.3f}, RMSE {cv['rmse_C']:.2f} deg C ({cv['n']:,} cells); "
          f"fitting {cfg.n_refits} refits", flush=True)
    M = model.Model(df, cfg)

    VA = validate.run(df, M, R, cfg)
    M = model.Model(df, cfg, exclude_near_m=cfg.strict_exclusion_m)
    VB = validate.run(df, M, R, cfg)
    regional = validate.region_holdouts(df, M, R, cfg)
    V = {"by_setting": VA["by_setting"].assign(test="A"), "by_dose": VA["by_dose"].assign(test="A"),
         "strict": VB["by_setting"].assign(test="B"), "strict_by_dose": VB["by_dose"].assign(test="B"),
         "strict_by_coast": VB["by_coast"].assign(test="B"),
         "validated": VB["validated"], "validated_A": VA["validated"], "strict_n_train": int(len(M.train)),
         "n_train_A": int(len(model.training_set(df))),
         "supported": VB["supported"], "supported_by_dose": VB["supported_by_dose"], "region_holdouts": regional,
         "support_counts": VB["support_counts"], "support_coasts": VB["support_coasts"]}
    for _, r in pd.concat([V["by_setting"], V["strict"]]).iterrows():
        print(f"4 test {r['test']}, {r['setting']}: model {r['model_C']:+.2f} vs measured {r['measured_C']:+.2f} deg C -> "
              f"{r['status']} (all-cell diagnostic)", flush=True)

    L = ledger.run(cfg, meta)

    D = decision.run(df, M, R, V, L, cfg)

    def stack(name):
        return pd.concat([R[st][name].assign(setting=st) for st in SETTINGS], ignore_index=True)

    pd.DataFrame([{"setting": st, "estimate": k, **v} for st in SETTINGS for k, v in R[st]["estimates"].items()]
                 ).to_csv(tables / "measured_estimates.csv", index=False)
    for name in ("dose", "sensitivity", "balance", "yearly", "by_coast", "geographic_dose", "region_influence", "cluster_diagnostics"):
        stack(name).to_csv(tables / f"measured_{name}.csv", index=False)
    pd.DataFrame([{"setting": st, **R[st]["n"]} for st in SETTINGS]).to_csv(tables / "cells.csv", index=False)
    pd.DataFrame([{k: v for k, v in cv.items() if k != "oof"}]).to_csv(tables / "model_cv.csv", index=False)
    M.importance().to_csv(tables / "model_importance.csv", index=False)
    pd.concat([V["by_setting"], V["strict"]]).to_csv(tables / "test_by_setting.csv", index=False)
    pd.concat([V["by_dose"], V["strict_by_dose"]]).to_csv(tables / "test_by_dose.csv", index=False)
    V["strict_by_coast"].to_csv(tables / "test_by_coast.csv", index=False)
    for name in ("supported", "supported_by_dose", "region_holdouts", "support_counts", "support_coasts"):
        V[name].to_csv(tables / f"test_{name}.csv", index=False)
    L["water"].to_csv(tables / "water.csv", index=False)
    L["energy"].to_csv(tables / "energy.csv", index=False)
    L["sio"].to_csv(tables / "sio_supply_depths.csv", index=False)
    D["summary"].to_csv(tables / "decision_summary.csv", index=False)
    D["by_coast"].to_csv(tables / "decision_by_coast.csv", index=False)
    D["candidates"].to_csv(tables / "decision_bare_places_model.csv", index=False)
    for name in ("scenarios", "ranking", "pairwise"):
        D[name].to_csv(tables / f"decision_{name}.csv", index=False)
    D["allocation"]["selected"].to_csv(tables / "allocation_selected.csv", index=False)

    used = (df["group"] == "control") & sum((R[st]["weights"] > 0) for st in SETTINGS).astype(bool)
    figures.fig_sample(df, used, figs, cfg.stable_surface_max)
    figures.fig_cooling(R, cfg, figs)
    figures.fig_test(cv, V, figs)
    figures.fig_ledger(L, figs)
    figures.fig_decision(D, V, figs)

    res = {"config": {k: _jsonable(v) for k, v in cfg.__dict__.items()}, "meta": meta,
           "measured": {k: v for k, v in R.items()}, "model": {k: v for k, v in cv.items() if k != "oof"},
           "importance": M.importance(), "test": V, "ledger": L, "decision": D}
    clean = _jsonable(res)
    for st in SETTINGS:
        for k in ("reps", "treated", "weights", "controls", "key"):
            clean["measured"][st].pop(k, None)
    (tables / "results.json").write_text(json.dumps(clean, indent=1, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    write_report(res, cfg, out / "manuscript" / "REPORT.md")
    print(f"wrote {out / 'manuscript' / 'REPORT.md'}")
    return res
