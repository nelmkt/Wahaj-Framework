"""Replot paired Figure 4 PNG/PDF from the saved resource ledger."""
import json
from pathlib import Path
import numpy as np

import fw_figures as figures
import fw_ledger as ledger
from fw_config import Config
from run_followup import restore


if __name__ == "__main__":
    result = restore(json.loads(Path("tables/results.json").read_text(encoding="utf-8")))
    verified = ledger.sio_depths(Config())
    old = result["ledger"]["sio"].sort_values(["year", "branch"]).reset_index(drop=True)
    new = verified.sort_values(["year", "branch"]).reset_index(drop=True)
    if not old[["year", "branch"]].equals(new[["year", "branch"]]) or not np.allclose(old.depth_m, new.depth_m, rtol=0, atol=1e-12):
        raise SystemExit("Verified workbook depths differ from plotted saved values; inspect before rendering")
    result["ledger"]["sio"] = verified
    figures.fig_ledger(result["ledger"], Path("figures/png"))
    print("Figure 4 PNG/PDF regenerated; all 17 workbook depths equal the previously plotted values.")
