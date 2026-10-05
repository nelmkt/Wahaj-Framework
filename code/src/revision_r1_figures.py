"""Two figures for the revision (supervisor review M4 and M7), drawn from the r1 tables. Writes NEW PNGs in figures_r1/.
usage: python revision_r1_figures.py <package_root>
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(sys.argv[1]).resolve()
T = ROOT / "tables_revision_r1"
OUT = ROOT / "figures_r1"
OUT.mkdir(exist_ok=True)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"     # categorical slots 1-3 of the reference palette, fixed order
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e6e5e1"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    path = OUT / name
    with open(path, "xb") as fh:
        fig.savefig(fh, dpi=300, bbox_inches="tight", facecolor="white")
    print("wrote", path)


# ---- Fig. A: mean matched contrast for each dose (primary match)
dc = pd.read_csv(T / "r1_dose_contrasts.csv")
si = pd.read_csv(T / "r1_slope_intercept.csv")
fig, ax = plt.subplots(figsize=(6.2, 3.6))
ax.axhline(0, color=MUTED, lw=0.8)
ax.grid(axis="y", color=GRID, lw=0.6)
for setting, color, marker, dx in (("outside the built-up area", BLUE, "o", -0.08), ("built-up surroundings", ORANGE, "s", 0.08)):
    d = dc[(dc.setting == setting) & (dc.n_matched >= 10)]
    ax.errorbar(d.dose + dx, d.cell_C, yerr=[d.cell_C - d.lo_C, d.hi_C - d.cell_C], fmt=marker, ms=5, color=color,
                ecolor=color, elinewidth=1.2, capsize=0, mec="white", mew=0.8, label=f"{setting} (n ≥ 10 per dose)")
o = si[si.setting == "outside the built-up area"].set_index("quantity").estimate
x = np.linspace(0, 9, 50)
ax.plot(x, o["slope_no_intercept_C_per_px"] * x, color=INK, lw=1, ls="--", label="outside: zero-intercept slope, Eq. (3)")
ax.plot(x, o["intercept_C"] + o["slope_with_intercept_C_per_px"] * x, color=MUTED, lw=1, ls=":",
        label="outside: slope with intercept")
ax.set_xlabel("Greened 30 m pixels in the 90 m cell (dose)")
ax.set_ylabel("Matched LST contrast, 2014–15 to 2024–25 (°C)")
ax.set_xticks(range(1, 10))
ax.set_xlim(0, 9.5)
ax.legend(frameon=False, fontsize=7.5, loc="lower left")
save(fig, "fig_r1_dose_contrasts.png")

# ---- Fig. B: NEGI on measured contrasts over alpha/beta and p
nw = pd.read_csv(T / "r1_negi_weights.csv")
fig, axes = plt.subplots(1, 2, figsize=(6.2, 3.0), sharey=True)
for ax, p in zip(axes, (0.5, 1.0)):
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.grid(axis="y", color=GRID, lw=0.6)
    for ratio, color, marker in ((0.5, BLUE, "o"), (1.0, ORANGE, "s"), (2.0, AQUA, "^")):
        d = nw[(nw.p == p) & (nw.alpha_over_beta == ratio)]
        ax.plot(d.f, d.negi_beta1, color=color, lw=2, marker=marker, ms=5, mec="white", mew=0.8,
                label=f"α/β = {ratio:g}")
    ax.set_title(f"cost exponent p = {p:g}", fontsize=9, color=INK)
    ax.set_xlim(0, 1.05)
axes[0].set_ylabel("NEGI (β = 1)")
fig.supxlabel("Mean greened fraction f of the dose class", fontsize=9, color=INK)
axes[1].legend(frameon=False, fontsize=7.5, loc="upper left")
save(fig, "fig_r1_negi_weights.png")
