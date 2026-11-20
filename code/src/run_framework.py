"""Urban greening in Jeddah: matched surface cooling and conditional resource accounting.

Run from the package root:
    python code/src/run_framework.py --panel code/gee/panel.csv --out .

Writes manuscript/REPORT.md, tables/results.json, tables/*.csv and matching
five-figure sets in figures/png/ and figures/pdf/.
"""
from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

from fw_config import Config
from fw_pipeline import run


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panel", default=str(Config.panel_path), help="cell panel from code/gee/export_panel.py")
    ap.add_argument("--out", default=str(Config.out_dir))
    ap.add_argument("--boot", type=int, default=Config.n_boot, help="bootstrap resamples for measured cooling (default 2000)")
    ap.add_argument("--refits", type=int, default=Config.n_refits, help="joint spatial model refits (default 200)")
    ap.add_argument("--validation-margin", type=float, default=None, help="Externally justified maximum acceptable effect error in degrees C; absent = no candidate gate")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--skip-region-holdouts", action="store_true", help="Diagnostic runs only; explicitly recorded")
    ap.add_argument("--allocation", type=Path, default=None, help="Optional comparable intervention table; see inputs template")
    ap.add_argument("--water-budget", type=float, default=None)
    ap.add_argument("--wastewater-cap", type=float, default=None)
    ap.add_argument("--energy-budget", type=float, default=None)
    ap.add_argument("--financial-budget", type=float, default=None)
    a = ap.parse_args()
    run(dataclasses.replace(Config(), panel_path=Path(a.panel), out_dir=Path(a.out), n_boot=a.boot, n_refits=a.refits,
        validation_margin_C=a.validation_margin,jobs=a.jobs,region_holdouts=not a.skip_region_holdouts,
        allocation_path=a.allocation,water_budget_m3=a.water_budget,wastewater_cap_m3=a.wastewater_cap,
        energy_budget_mwh=a.energy_budget,financial_budget=a.financial_budget))


if __name__ == "__main__":
    main()
