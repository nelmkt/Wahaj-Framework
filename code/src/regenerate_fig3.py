"""Regenerate paired Figure 3 PNG/PDF with the saved A/B effect diagnostics."""
from pathlib import Path
import numpy as np
import pandas as pd
from fw_config import Config
import fw_panel as panel
import fw_model as model
import fw_figures as figures


def main():
    cfg = Config()
    df, _ = panel.load(cfg)
    cv = model.spatial_cv(model.training_set(df, cfg.strict_exclusion_m), cfg)
    saved = pd.read_csv("tables/model_cv.csv").iloc[0]
    assert int(saved.n) == int(cv["n"])
    assert np.isclose(saved.r2, cv["r2"], atol=1e-6, rtol=0)
    assert np.isclose(saved.rmse_C, cv["rmse_C"], atol=1e-6, rtol=0)
    by_setting = pd.read_csv("tables/test_by_setting.csv")
    by_dose = pd.read_csv("tables/test_by_dose.csv")
    validation = dict(by_setting=by_setting[by_setting.test == "A"],
                      by_dose=by_dose[by_dose.test == "A"],
                      strict=by_setting[by_setting.test == "B"],
                      strict_by_dose=by_dose[by_dose.test == "B"])
    figures.fig_test(cv, validation, Path("figures/png"))
    print(f"Figure 3 PNG/PDF regenerated together; held-out R2={cv['r2']:.6f}.")


if __name__ == "__main__":
    main()
