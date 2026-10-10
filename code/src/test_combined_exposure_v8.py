"""Local analyzer checks with a constructed zero-external case; no GEE output."""
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_combined_exposure_v8 import isolation_table, ratio_table
from fw_config import Config
from fw_matching import Bootstrap
import fw_panel

ROOT = Path(__file__).resolve().parents[2]


def test_zero_external_exposure_reproduces_own_area_ratio_and_mixing_reference():
    cfg = Config(panel_path=ROOT / "code/gee/panel.csv", n_boot=8, n_refits=2)
    df, _ = fw_panel.load(cfg)
    for radius in (90, 180, 300):
        df[f"external_green_px_{radius}m"] = 0.0
        df[f"combined_green_px_{radius}m"] = df.n_greened_px.astype(float)
    ratios = ratio_table(df, cfg)
    assert len(ratios) == 21
    assert np.allclose(ratios.ratio_multiplier, 1)
    assert np.allclose(ratios.old_fig5a_m3_year_per_C,
                       ratios.combined_footprint_m3_year_per_C)
    boot = Bootstrap(df.block, cfg.n_boot, cfg.seed)
    isolation = isolation_table(df, cfg, boot)
    main = isolation.iloc[0]
    saved = pd.read_csv(ROOT / "tables/logic_primary_area_mixing.csv")
    saved_small = saved[(saved.setting == "outside the built-up area") & saved.pixels.eq("1–2")].iloc[0]
    assert main.n_matched == 419
    assert np.isclose(main.excess_C, saved_small.difference_C, atol=1e-8)
