"""Invariant checks for the framework (python -m pytest tests/test_framework.py)."""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT))

import fw_ledger as ledger, fw_matching as matching, fw_model as model, fw_panel as panel, fw_validate as validate# noqa: E402
from fw_config import SETTINGS, Config  # noqa: E402
import synthetic_panel as sp  # noqa: E402

FAST = dict(n_boot=200, n_refits=20, region_holdouts=False, jobs=2)


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    p = sp.write(tmp_path_factory.mktemp("syn") / "panel.csv")
    cfg = Config(panel_path=p, **FAST)
    df, meta = panel.load(cfg)
    return cfg, df, meta, matching.run_all(df, cfg)


def test_measured_recovers_known_effect(data):
    cfg, df, meta, R = data
    for st in SETTINGS:
        e = R[st]["estimates"]["per_pixel"]
        assert abs(e["estimate_C"] - sp.TRUE_FULL / 9) < 0.03, st
    T, C = df.group == "greened", df.group == "control"
    naive = ((df.loc[T, "d_lst"] - df.loc[C, "d_lst"].mean()) * df.loc[T, "n_greened_px"]).sum() / (df.loc[T, "n_greened_px"] ** 2).sum()
    assert abs(naive - sp.TRUE_FULL / 9) > 0.03


def test_placebo_and_spillover(data):
    _, _, _, R = data
    for st in SETTINGS:
        p = R[st]["estimates"]["placebo_per_pixel"]
        assert p["lo_C"] < 0.03 and p["hi_C"] > -0.03
        assert abs(R[st]["estimates"]["ring"]["estimate_C"] - sp.TRUE_RING) < 0.15


def test_strata_ignore_what_greening_changes(data):
    cfg, df, _, _ = data
    k0 = panel.strata(df, cfg)
    d = df.copy()
    g = d.group == "greened"
    for c in ["built_post"] + [f"{v}_{y}" for v in ("ndvi", "ndbi", "lst") for y in cfg.post_years]:
        d.loc[g, c] = d.loc[g, c] + 0.37
    assert (panel.strata(panel.derive(d, cfg), cfg) == k0).all()


def test_model_never_sees_greened_cells(data):
    _, df, _, _ = data
    tr = model.training_set(df)
    assert (tr["group"] != "greened").all() and tr["in_sample"].all() and len(tr) > 1000


def test_no_external_tolerance_does_not_release_candidates(data):
    cfg, df, _, R = data
    V = validate.run(df, model.Model(df, cfg), R, cfg)
    assert not any(V["validated"].values())
    assert set(V["supported"]["status"]) == {"tolerance_not_set"}


def test_wrong_model_fails(tmp_path):
    """If existing vegetation in the city is twice as cool as greening makes a place, the model must fail the test."""
    df0, meta = sp.make()
    veg = df0["cls"] == 4
    for y in sp.YEARS:
        df0.loc[veg, f"lst_{y}"] += sp.TRUE_FULL / 0.5 * (df0.loc[veg, f"ndvi_{y}"] - 0.08)
    p = tmp_path / "panel.csv"
    df0.to_csv(p, index=False)
    (tmp_path / "panel_meta.json").write_text(__import__("json").dumps(meta))
    cfg = Config(panel_path=p, validation_margin_C=0.3, min_validation_refits=20, **FAST)
    df, _ = panel.load(cfg)
    R = matching.run_all(df, cfg)
    V = validate.run(df, model.Model(df, cfg), R, cfg)
    assert not any(V["validated"].values())


def test_bootstrap_is_reproducible(data):
    cfg, df, _, _ = data
    a = matching.Bootstrap(df["block"], 50, cfg.seed).draws
    b = matching.Bootstrap(df["block"], 50, cfg.seed).draws
    assert (a == b).all() and (a.sum(axis=1) == a.shape[1]).all()


def test_ledger_arithmetic():
    cfg = Config()
    W = ledger.water(cfg, 2000.0)
    assert W.loc[1, "m3_per_ha_yr"] == pytest.approx(2000 * 0.6 / 0.75 * 10)
    E = ledger.energy(cfg, W)
    assert E.loc[0, "MWh_per_ha_yr_central"] == pytest.approx(W.loc[1, "m3_per_ha_yr"] * 3.25 / 1000)
    S = ledger.sio_depths(cfg)
    r = S[(S.year == 2020) & (S.branch == "Al-Ahsa")].iloc[0]
    assert r["depth_m"] == pytest.approx((81136190 + 870451 + 13139462) / (2488 * 1e4))


def test_pipeline_end_to_end(tmp_path):
    p = sp.write(tmp_path / "panel.csv", n_control=2500, n_ring=300, n_greened=300, n_other=1500)
    from fw_pipeline import run
    res = run(dataclasses.replace(Config(), panel_path=p, out_dir=tmp_path / "out", n_boot=50, n_refits=5,region_holdouts=False,jobs=2))
    rep = (tmp_path / "out" / "manuscript" / "REPORT.md").read_text(encoding="utf-8")
    assert "Synthetic test panel" in rep
    for f in ("fig1_sample", "fig2_cooling", "fig3_model_test", "fig4_water_energy", "fig5_decision"):
        assert (tmp_path / "out" / "figures_png" / f"{f}.png").exists()
    assert sum(res["measured"][st]["n"]["greened"] for st in SETTINGS) == 300
