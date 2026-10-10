"""Tests for v8 source-workbook, exposure and population claims."""
import csv
import builtins
import json
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import rebuild_sio_v8 as sio
from make_population_map_v11 import population

ROOT = Path(__file__).resolve().parents[2]


def table(name):
    return pd.read_csv(ROOT / "tables" / name)


def test_sio_workbooks_rebuild_without_guessing_a_geographic_join():
    area = sio.parse_area(ROOT / "code/src/6.xlsx")
    supply = sio.parse_supply(ROOT / "code/src/8.xlsx")
    result, audit = sio.build(area, supply, pd.read_csv(sio.OLD))
    saved = pd.read_csv(sio.NEW)
    assert len(area) == 17 and len(supply) == 18 and len(result) == 17
    pd.testing.assert_frame_equal(result.reset_index(drop=True), saved, check_dtype=False)
    assert audit._merge.value_counts().to_dict() == {"both": 17, "right_only": 1, "left_only": 0}
    assert np.nanmax(np.abs(audit.depth_difference_m)) == 0
    assert not audit.in_named_jeddah_makkah_branch.any()
    water = table("water.csv")
    valid = audit.depth_m.dropna()
    assert int((~valid.between(water.depth_m.min(), water.depth_m.max())).sum()) == 10


def test_pixel_export_has_only_a_300m_max_screen_and_matches_v7_effects():
    raw = pd.read_csv(ROOT / "code/gee/pixel_isolation_raw.csv")
    meta = json.loads((ROOT / "code/gee/pixel_isolation_raw.json").read_text(encoding="utf-8"))
    assert len(raw) == meta["n_exported_cells"] == 1333
    assert all(meta[k] == 0 for k in ("dose_count_disagreements", "unjoined_cells",
                                        "negative_external_flags", "missing_pixel_centered_screens"))
    assert set(raw.columns) == {"lon", "lat", "green_count_cell",
                                "max_external_green_within300_any_own30", "negative_external_flag"}
    new = table("pixel_isolation_by_dose_v8.csv")
    assert new.loc[new.radius_m.isin([90, 180]), "status"].eq("not_exported").all()
    old = table("gee_pixel_centered_isolation_effects_v7.csv")
    for dose in ("1–2", "exactly one"):
        row = new[(new.setting == "outside the built-up area") & (new.dose_class == dose) &
                  (new.radius_m == 300)].iloc[0]
        old_full = old[(old.dose == dose) & old.screen.eq("all")].iloc[0]
        old_isolated = old[(old.dose == dose) & old.screen.ne("all")].iloc[0]
        assert int(row.n_matched_isolated) >= 20
        assert int(row.n_matched_full) == int(old_full.n_matched)
        assert int(row.n_matched_isolated) == int(old_isolated.n_matched)
        assert np.isclose(row.full_slope_C_per_own_pixel, old_full.per_pixel_C)
        assert np.isclose(row.isolated_slope_C_per_own_pixel, old_isolated.per_pixel_C)
        assert np.isclose(row.isolated_minus_full_C,
                          row.isolated_slope_C_per_own_pixel - row.full_slope_C_per_own_pixel)
        assert row.difference_lo_C < row.isolated_minus_full_C < row.difference_hi_C
    assert len(old) == 4
    high = new[(new.setting == "outside the built-up area") & (new.radius_m == 300) &
               (new.dose_class.isin(["5–8", "9/9"]))]
    assert dict(zip(high.dose_class, high.n_matched_isolated)) == {"5–8": 1.0, "9/9": 0.0}
    assert high.status.eq("diagnostic_only").all()


def test_headline_leverage_and_figure_one_donors():
    headline = table("headline_block_v8.csv")
    assert len(headline) == 5
    assert np.isclose(headline.dose_squared_leverage_share.iloc[1:].sum(), 1)
    pooled = headline.iloc[0]
    assert int(pooled.n_matched) == 1006
    assert np.isclose(pooled.dominant_pooled_block_leverage_share, .5566428735896846)
    assert pooled.jackknife_lo_C < pooled.point_C_per_pixel < pooled.jackknife_hi_C
    assert all(headline.loc[headline.dose_class.isin(["5–8", "9/9"]),
                            "dominant_block_removed_C"] < 0)
    figure = table("figure1_population_comparison_v8.csv")
    overall = figure[figure.setting == "all"].set_index("group")
    assert int(overall.loc["primary matched treated", "n_cells"]) == 1319
    assert int(overall.loc["primary matched controls", "n_cells"]) == 9267
    assert int(overall.loc["other-land representative 10% sample", "n_cells"]) == 4892
    matching_code = (ROOT / "code/src/fw_matching.py").read_text(encoding="utf-8")
    assert "control" in matching_code


def test_population_inventory_covers_all_tables_without_reading_blinded_key():
    current = "table_population_v8_followup10.csv"
    key = (ROOT / "tables/imagery_sampling_key_v7.csv").resolve()
    real_open, real_path_open = builtins.open, Path.open

    def guard_open(file, *args, **kwargs):
        if isinstance(file, (str, os.PathLike)) and Path(file).resolve() == key:
            raise AssertionError("Blinded imagery key contents must not be read")
        return real_open(file, *args, **kwargs)

    def guard_path_open(path, *args, **kwargs):
        if path.resolve() == key:
            raise AssertionError("Blinded imagery key contents must not be read")
        return real_path_open(path, *args, **kwargs)

    with patch("builtins.open", guard_open), patch.object(Path, "open", guard_path_open):
        rows = table(current)
        names = {p.name for p in (ROOT / "tables").glob("*.csv")}
        assert set(rows.table) == names
        assert len(rows) == len(names)
        assert rows.table.is_unique
        assert rows.loc[rows.table.eq("imagery_sampling_key_v7.csv"), "population"].item() == "unopened blinded key"
        assert rows.loc[rows.table.eq("imagery_sampling_key_v7.csv"), "interpretation"].item() == "Filename inventoried only; contents not inspected"
        assert rows.population.notna().all() and rows.population.astype(str).str.strip().ne("").all()
        assert rows.interpretation.notna().all() and rows.interpretation.astype(str).str.strip().ne("").all()
        baseline = table("table_population_v8.csv")
        listed = rows.set_index("table")
        for row in baseline.itertuples(index=False):
            assert listed.loc[row.table, "population"] == row.population
            assert listed.loc[row.table, "interpretation"] == row.interpretation
        for row in rows.itertuples(index=False):
            if row.table != current:
                assert population(row.table) == (row.population, row.interpretation)
