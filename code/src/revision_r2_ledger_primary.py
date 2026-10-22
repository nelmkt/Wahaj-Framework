"""Water-energy ledger on the PRIMARY match (reads saved tables only; writes one NEW table).

The pipeline's ledger (fw_decision.py -> tables/decision_summary.csv, decision_scenarios.csv) was written for the
concurrent-change match. This script applies the same central arithmetic to the primary match, so that the ledger
and the NEGI illustration (tradeoff_illustration_v11.py) use one population:

    water_m3       = PIXEL_HA * mean_dose * m3_per_ha_yr[central]        (assumed irrigation of the greened land)
    m3_per_degree  = water_m3 / measured cooling of the class             (cooling = -cell_C)
    MWh_per_degree = m3_per_degree * kWh m-3 / 1000                       (SWRO low/central/high; TSE central)

The 95% range of m3_per_degree uses the class's spatial-block interval of the cooling with the greened area held at
its mean; the pipeline's joint area-and-cooling draws are not saved for the primary match. A check row recomputes the
concurrent-change central values from tables/decision_summary.csv with the same arithmetic.

Source: tables/logic_primary_class_estimates.csv, outside the built-up area, pre-treatment-only strata.
Every water and energy quantity is an assumption.

usage: python revision_r2_ledger_primary.py <package_root>
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fw_config import Config  # noqa: E402
from fw_panel import PIXEL_HA  # noqa: E402

ROOT = Path(sys.argv[1]).resolve()
OUT = ROOT / "tables_revision_r1" / "r2_ledger_primary.csv"
if OUT.exists():
    raise SystemExit(f"Refusing to overwrite {OUT}")

SETTING = "outside the built-up area"
SWRO, TSE = "desalinated seawater (SWRO)", "treated wastewater"
kwh = Config.__dataclass_fields__["kwh_per_m3"].default_factory()

W = pd.read_csv(ROOT / "tables" / "water.csv").set_index("level")["m3_per_ha_yr"]

# check: the same arithmetic reproduces the saved concurrent-change ledger
ds = pd.read_csv(ROOT / "tables" / "decision_summary.csv")
ds = ds[ds["setting"] == SETTING]
chk = PIXEL_HA * ds["pixels_mean"] * W["central"] / -ds["cooling_C"]
assert (abs(chk - ds["m3_per_degree"]) < 1e-6).all(), "arithmetic does not reproduce decision_summary.csv"

pce = pd.read_csv(ROOT / "tables" / "logic_primary_class_estimates.csv")
prim = pce[(pce["setting"] == SETTING) & (pce["specification"] == "pre-treatment-only strata")]
assert len(prim) == 5, len(prim)

rows = []
for _, r in prim.iterrows():
    cooling, c_lo, c_hi = -r["cell_C"], -r["cell_hi_C"], -r["cell_lo_C"]   # c_lo < cooling < c_hi
    water = PIXEL_HA * r["mean_dose"] * W["central"]
    m3 = water / cooling
    row = dict(dose_class=r["dose_class"], cells=int(r["n_matched"]), mean_dose=r["mean_dose"],
               cooling_C=cooling, cooling_lo_C=c_lo, cooling_hi_C=c_hi, water_m3_central=water,
               m3_per_degree=m3, m3_per_degree_low=water / c_hi, m3_per_degree_high=water / c_lo)
    for tag, (lo, mid, hi) in (("swro", kwh[SWRO]), ("tse", kwh[TSE])):
        row[f"{tag}_MWh_per_degree_central"] = m3 * mid / 1000
        row[f"{tag}_MWh_per_degree_low_case"] = m3 * W["low"] / W["central"] * lo / 1000
        row[f"{tag}_MWh_per_degree_high_case"] = m3 * W["high"] / W["central"] * hi / 1000
    rows.append(row)
out = pd.DataFrame(rows)
out["note"] = ("primary match; central irrigation case; range from the cooling interval with area fixed; "
               "water and energy are assumptions")
out.to_csv(OUT, index=False, mode="x")
print(out.drop(columns=["note"]).round(3).to_string(index=False))
print("wrote", OUT)
