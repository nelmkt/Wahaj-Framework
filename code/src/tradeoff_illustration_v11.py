"""Illustrative trade-off calculation from saved v11 tables (reads tables only; writes one NEW table).

Part 1. NEGI evaluated on MEASURED contrasts instead of model predictions (illustration, not a released index value):
    f        mean greened fraction of the dose class = mean_dose / 9
    cooling  measured matched cell contrast of the class, sign reversed (positive = cooler)
    benefit  cooling / S, with S the largest cooling over the four classes
    cost     f**p / max(f**p), p = 0.5 (default) and 1.0 (alternative)
    NEGI     alpha * benefit - beta * cost, alpha = beta = 1.0
    p_star   exponent at which NEGI = 0: ln(benefit) / ln(f)   (undefined where f = 1)
  Source: tables/logic_primary_class_estimates.csv, outside the built-up area, pre-treatment-only strata (primary match).

Part 2. Break-even energy: the hypothetical irrigation energy that one degree of measured cell cooling carries,
  for seawater reverse osmosis (low, central, high water and energy cases) and treated wastewater (central).
  Source: tables/decision_scenarios.csv, outside the built-up area (concurrent-change match).

The dose classes are different cells, not one cell at different greening levels, and every water and energy
quantity is an assumption.

usage: python tradeoff_illustration_v11.py <package_root>
"""
import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(sys.argv[1]).resolve()
OUT = ROOT / "tables_revision_v11" / "tradeoff_illustration_v11.csv"
if OUT.exists():
    raise SystemExit(f"Refusing to overwrite {OUT}")

ALPHA, BETA, P_DEFAULT, P_ALT = 1.0, 1.0, 0.5, 1.0
SWRO, TSE = "desalinated seawater (SWRO)", "treated wastewater"
SETTING = "outside the built-up area"

pce = pd.read_csv(ROOT / "tables" / "logic_primary_class_estimates.csv")
prim = pce[(pce["setting"] == SETTING) & (pce["specification"] == "pre-treatment-only strata") & (pce["dose_class"] != "all")]
assert len(prim) == 4, len(prim)
sc = pd.read_csv(ROOT / "tables" / "decision_scenarios.csv")
ds = pd.read_csv(ROOT / "tables" / "decision_summary.csv")

S = float((-prim["cell_C"]).max())
fmax = {p: max((d / 9.0) ** p for d in prim["mean_dose"]) for p in (P_DEFAULT, P_ALT)}
rows = []
for _, r in prim.iterrows():
    f = float(r["mean_dose"]) / 9.0
    cooling = -float(r["cell_C"])
    benefit = max(cooling, 0.0) / S
    row = dict(dose_class=r["dose_class"], n_matched_primary=int(r["n_matched"]), mean_dose=float(r["mean_dose"]),
               greened_fraction_f=f, measured_cooling_C=cooling, benefit_scale_S_C=S, normalized_benefit=benefit)
    for tag, p in (("p05", P_DEFAULT), ("p10", P_ALT)):
        cost = f ** p / fmax[p]
        row[f"cost_{tag}"] = cost
        row[f"negi_{tag}"] = ALPHA * benefit - BETA * cost
    row["breakeven_exponent_p_star"] = math.log(benefit) / math.log(f) if f < 1 else float("nan")
    option = f"{SETTING}: {r['dose_class']}"
    summ = ds[ds["option_id"] == option]
    assert len(summ) == 1, option
    row["n_cells_concurrent"] = int(summ["cells"].iloc[0])
    row["measured_cooling_concurrent_C"] = -float(summ["cooling_C"].iloc[0])

    def pick(source, water, energy):
        hit = sc[(sc["option_id"] == option) & (sc["source"] == source) & (sc["water_case"] == water) & (sc["energy_case"] == energy)]
        assert len(hit) == 1, (option, source, water, energy, len(hit))
        return float(hit["MWh_per_degree"].iloc[0])

    row["swro_MWh_per_degree_low"] = pick(SWRO, "low", "low")
    row["swro_MWh_per_degree_central"] = pick(SWRO, "central", "central")
    row["swro_MWh_per_degree_high"] = pick(SWRO, "high", "high")
    row["tse_MWh_per_degree_central"] = pick(TSE, "central", "central")
    rows.append(row)
out = pd.DataFrame(rows)
out["note"] = "illustration on measured contrasts; classes are different cells; water and energy are assumptions"
out.to_csv(OUT, index=False, mode="x")
print(out.drop(columns=["note"]).round(4).to_string(index=False))
print("wrote", OUT)
