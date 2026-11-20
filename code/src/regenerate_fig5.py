"""Regenerate paired Figure 5 PNG/PDF from the saved observed-panel results."""
import json
from pathlib import Path
from run_followup import restore
import fw_figures as figures


if __name__ == "__main__":
    raw = json.loads(Path("tables/results.json").read_text(encoding="utf-8"))
    result = restore(raw)
    figures.fig_decision(result["decision"], result["test"], Path("figures/png"))
    print("Figure 5 PNG/PDF regenerated together from saved observed-panel results.")
