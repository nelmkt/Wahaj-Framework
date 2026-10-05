"""Fig. 1 redrawn for the revision (supervisor review m7): same data and primary matched controls as
code/src/regenerate_fig1_v8.py, with a scale bar, a north arrow, the coastline source and plain class names.
Writes NEW figures_r1/fig_r1_cell_classes.png.   usage: python revision_r1_fig1_map.py <package_root>
"""
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "code" / "src"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

import fw_panel as panel  # noqa: E402
from fw_config import SETTINGS, Config  # noqa: E402
from regenerate_fig1_v8 import comparison  # noqa: E402

OUT = ROOT / "figures_r1" / "fig_r1_cell_classes.png"
if OUT.exists():
    raise SystemExit(f"Refusing to overwrite {OUT}")
cfg = Config(panel_path=ROOT / "code" / "gee" / "panel.csv")
df, _ = panel.load(cfg)
table, used = comparison(df, cfg)
coast = pd.read_csv(panel.COASTLINE)
INK = "#263238"
fig, (a, b) = plt.subplots(1, 2, figsize=(11, 5.8), gridspec_kw={"width_ratios": [1, 1.2], "wspace": 0.3})
for cls, color, label in ((4, "#d3e8f0", "other land, random 10% sample"), (3, "#d5d9dd", "never-vegetated controls, random 10% sample"),
                          (2, "#f6d6bb", "never-vegetated ring cells, ≤150 m from greening"), (1, "#bfe0c2", "greened cells")):
    p = df[df.cls.eq(cls)]
    a.scatter(p.lon, p.lat, s=1.5 if cls != 1 else 2.5, c=color, lw=0, label=f"{label} ({len(p):,})", rasterized=True)
ctrl = df[used & df.cls.eq(3)]
a.scatter(ctrl.lon, ctrl.lat, s=1.8, c="#455a64", alpha=0.65, lw=0, label=f"controls used by the primary match ({len(ctrl):,})",
          rasterized=True)
g = df[df.cls.eq(1)]
a.scatter(g.lon, g.lat, s=3.5, c="#a7d7ad", lw=0, label="_nolegend_", rasterized=True)
a.plot(coast.lon, coast.lat, "-", lw=0.7, c="#53616b", label="coastline (Natural Earth 10 m)")
x0, y0, x1, y1 = 39.0, 21.2, 39.4, 21.8
a.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, lw=1.0, ls="--", ec=INK, label="analysis rectangle"))
a.set_xlim(x0 - 0.02, x1 + 0.02)
a.set_ylim(y0 - 0.02, y1 + 0.02)
a.set_aspect(1 / np.cos(np.radians(df.lat.mean())))
a.set_xlabel("longitude (°E)")
a.set_ylabel("latitude (°N)")
a.set_title("a  Cell classes (90 m cells)", loc="left", fontsize=10)
# scale bar: 10 km along the parallel at 21.24° N
km_per_deg = 111.32 * np.cos(np.radians(21.24))
sx, sy, L = 39.02, 21.235, 10 / km_per_deg
a.plot([sx, sx + L], [sy, sy], c=INK, lw=2.2)
for xx in (sx, sx + L):
    a.plot([xx, xx], [sy - 0.006, sy + 0.006], c=INK, lw=1)
a.text(sx + L / 2, sy + 0.012, "10 km", ha="center", va="bottom", fontsize=8, color=INK)
# north arrow
a.annotate("N", xy=(39.05, 21.79), xytext=(39.05, 21.72), ha="center", va="center", fontsize=9, color=INK,
           arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.2))
a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=1, markerscale=4, fontsize=7, handletextpad=0.3, frameon=False)
k = np.arange(1, 10)
for i, (s, col) in enumerate(zip(SETTINGS, ("#2e7d32", "#6a1b9a"))):
    n = df[(df.group == "greened") & (df.setting == s)]["n_greened_px"].value_counts().reindex(k, fill_value=0)
    b.bar(k + (i - 0.5) * 0.4, n.values, width=0.4, color=col, label=s)
b.set_xticks(k)
b.set_xlabel("greened 30 m pixels in the cell (out of 9; one pixel = 0.09 ha)")
b.set_ylabel("greened cells")
b.set_title("b  Greened cells by dose and setting", loc="left", fontsize=10)
b.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=8, frameon=False)
for ax in (a, b):
    ax.spines[["top", "right"]].set_visible(False)
OUT.parent.mkdir(exist_ok=True)
with open(OUT, "xb") as fh:
    fig.savefig(fh, dpi=300, bbox_inches="tight", facecolor="white")
print("wrote", OUT, "| controls used", len(ctrl))
