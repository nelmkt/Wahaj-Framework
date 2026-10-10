"""Five figures. Every label sits in its own space: legends outside the data, estimate values in a separate
right-hand column, no text drawn on top of data."""
from __future__ import annotations

from pathlib import Path
import ast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
import matplotlib.patheffects as path_effects  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fw_config import SETTINGS  # noqa: E402
from fw_panel import COASTLINE  # noqa: E402

COL = {SETTINGS[0]: "#2e7d32", SETTINGS[1]: "#6a1b9a"}
RING, CTRL, INK, INK2 = "#ef6c00", "#9e9e9e", "#212121", "#616161"
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                     "savefig.dpi": 200, "font.family": "DejaVu Sans"})


def _xpos(years):
    """Evenly spaced positions for the three periods, with a gap between periods so year labels never touch."""
    xs, x, prev = [], 0.0, None
    for y in years:
        if prev is not None:
            x += 1.0 if y - prev == 1 else 1.8
        xs.append(x); prev = y
    return np.array(xs)


def _err(c, lo, hi):
    """Error-bar lengths; a point estimate can sit just outside its bootstrap percentile interval, so clip at 0."""
    return [np.maximum(np.asarray(c) - np.asarray(lo), 0), np.maximum(np.asarray(hi) - np.asarray(c), 0)]


def _save(fig, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    pdf = out.parent / "figures_pdf"
    pdf.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", bbox_inches="tight")
    fig.savefig(pdf / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def _export_rect() -> tuple[float, float, float, float]:
    """Read the actual GEE bounding rectangle without importing Earth Engine."""
    source = Path(__file__).resolve().parents[1] / "gee/export_panel.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "RECT" for t in node.targets):
            value = tuple(float(x) for x in ast.literal_eval(node.value))
            if len(value) == 4 and value[0] < value[2] and value[1] < value[3]:
                return value
    raise ValueError("GEE RECT not found or invalid")


def fig_sample(df: pd.DataFrame, used_controls: pd.Series, out: Path, stable_max: float = 0.2,
               match_label: str = "concurrent-change"):
    """a: where the cells are; b: how many greened cells, by setting and by greened pixels per cell."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 5.6), gridspec_kw={"width_ratios": [1, 1.2], "wspace": 0.3})
    coast = pd.read_csv(COASTLINE)
    palette = ((4, "#d3e8f0", "4 other land: random 10%"),
               (3, "#d5d9dd", "3 never green: random 10%"),
               (2, "#f6d6bb", "2 ring: ≤150 m"),
               (1, "#bfe0c2", "1 greened transition"))
    for cls, color, label in palette:
        p = df[df.cls.eq(cls)]
        a.scatter(p.lon, p.lat, s=1.5 if cls != 1 else 2.5, c=color, lw=0,
                  label=f"{label} ({len(p):,})", rasterized=True)
    ctrl = df[used_controls & df.cls.eq(3)]
    a.scatter(ctrl.lon, ctrl.lat, s=1.8, c="#455a64", alpha=0.65, lw=0,
              label=f"matched controls, {match_label} ({len(ctrl):,})", rasterized=True)
    green = df[df.cls.eq(1)]
    a.scatter(green.lon, green.lat, s=3.5, c="#a7d7ad", lw=0,
              label="_nolegend_", rasterized=True)
    a.plot(coast.lon, coast.lat, "-", lw=0.7, c="#53616b", label="coastline / land edge")
    x0, y0, x1, y1 = _export_rect()
    a.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, lw=1.0,
                          ls="--", ec="#263238", label="GEE RECT boundary"))
    pad = 0.02
    a.set_xlim(x0 - pad, x1 + pad)
    a.set_ylim(y0 - pad, y1 + pad)
    a.set_aspect(1 / np.cos(np.radians(df.lat.mean())))
    a.set_xlabel("longitude (°E)"); a.set_ylabel("latitude (°N)")
    a.set_title("a  Where the cells are (90 m)")
    a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, markerscale=4, fontsize=7, handletextpad=0.3)

    k = np.arange(1, 10)
    wbar = 0.4
    for i, s in enumerate(SETTINGS):
        g = df[(df.group == "greened") & (df.setting == s)]
        n = g["n_greened_px"].value_counts().reindex(k, fill_value=0)
        b.bar(k + (i - 0.5) * wbar, n.values, width=wbar, color=COL[s], label=s)
    b.set_xticks(k)
    b.set_xlabel("pixels of the cell that greened (out of 9; one pixel = 0.09 ha)")
    b.set_ylabel("all classified greened cells")
    b.set_title("b  How much of each cell greened (all classified)")
    b.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=8)
    _save(fig, out, "fig1_sample")


def _ok(r):
    return r is not None and np.isfinite(r.get("estimate_C", np.nan))


def _forest(ax, rows, xlim, show_labels):
    y = np.arange(len(rows))[::-1].astype(float)
    for yi, (lab, r, col) in zip(y, rows):
        if _ok(r):
            ax.errorbar(r["estimate_C"], yi, xerr=np.reshape(_err(r["estimate_C"], r["lo_C"], r["hi_C"]), (2, 1)),
                        fmt="o", c=col, ms=5, capsize=3, lw=1.4)
    ax.axvline(0, c=INK, lw=0.8)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlim(*xlim)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows] if show_labels else [], fontsize=8)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    right = ax.twinx()
    right.set_ylim(ax.get_ylim())
    right.set_yticks(y)
    right.set_yticklabels([f"{r['estimate_C']:+.2f} [{r['lo_C']:+.2f}, {r['hi_C']:+.2f}]" if _ok(r) else "too few cells"
                           for _, r, _ in rows], fontsize=8)
    right.tick_params(axis="y", length=0)
    for sp in ("top", "right", "left"):
        right.spines[sp].set_visible(False)


def forest_rows(R: dict, s: str) -> list:
    E, D = R[s]["estimates"], R[s]["dose"]
    rows = [("Per greened pixel (0.09 ha)", E["per_pixel"], COL[s])]
    for _, r in D.iterrows():
        lab = "Cells with all 9 pixels greened" if r["pixels"] == "9" else f"Cells with {r['pixels']} of 9 pixels greened"
        rows.append((lab, r.to_dict(), COL[s]))
    rows += [("Placebo (late greeners, pre-greening)\nper pixel: 2014–15 → 2018–19", E["placebo_per_pixel"], INK2),
             ("Ring cells 0–150 m (contrast)", E["ring"], RING)]
    return rows


def fig_cooling(R: dict, cfg, out: Path, match_label: str = "concurrent-change"):
    """a: effect per greened pixel, year by year; b, c: every estimate for each setting, values in their own column."""
    fig = plt.figure(figsize=(12, 8.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2], hspace=0.6, wspace=0.62)
    a = fig.add_subplot(gs[0, :])
    years = list(cfg.pre_years + cfg.mid_years + cfg.post_years)
    xs = _xpos(years)
    for i, s in enumerate(SETTINGS):
        d = R[s]["yearly"]
        off = (i - 0.5) * 0.14
        a.errorbar(xs + off, d["estimate_C"], yerr=_err(d["estimate_C"], d["lo_C"], d["hi_C"]), fmt="o",
                   c=COL[s], ms=4, lw=1.4, capsize=2, label=s)
    a.axhline(0, c=INK, lw=0.8)
    a.set_xticks(xs, [str(y) for y in years])
    a.set_xlim(xs[0] - 0.5, xs[-1] + 0.5)
    a.set_xlabel("year (summer composite, May–September)")
    a.set_ylabel("°C per greened pixel,\nrelative to 2014–15")
    a.set_title("a  Matched difference per greened pixel (0.09 ha), year by year")
    a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=8)

    blocks = {s: forest_rows(R, s) for s in SETTINGS}
    vals = [v for rows in blocks.values() for _, r, _ in rows if _ok(r) for v in (r["lo_C"], r["hi_C"])]
    lo, hi = min(vals + [0.0]), max(vals + [0.0])
    padx = 0.06 * (hi - lo)
    for j, s in enumerate(SETTINGS):
        ax = fig.add_subplot(gs[1, j])
        _forest(ax, blocks[s], (lo - padx, hi + padx), show_labels=(j == 0))
        ax.set_title(f"{'bc'[j]}  {s[0].upper() + s[1:]}")
        ax.set_xlabel("change in summer LST vs matched controls (°C)")
    fig.text(.12, -.035, "Slope and placebo: °C per greened optical pixel. Dose and ring rows: °C per 90 m cell.", fontsize=8)
    fig.text(.12, -.06, f"Shown contrasts use {match_label} strata. Figure 1 shows all classified cells; the effect rows use matched cells.", fontsize=8)
    fig.text(.12, -.085, "The ring contrast does not identify spillover.", fontsize=8)
    fig.text(.12, -.11, "Markers show selected years only; intervening years were not exported.", fontsize=8)
    _save(fig, out, "fig2_cooling")


def fig_ledger(L: dict, out: Path):
    """a: irrigation depth, this study's range vs SIO-supplied depths; b: electricity per hectare by water source."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.0), gridspec_kw={"width_ratios": [1, 1], "wspace": 0.8})
    W, S = L["water"], L["sio"]
    lo, mid, hi = W["depth_m"]
    a.hlines(1, lo, hi, color=COL[SETTINGS[0]], lw=6, alpha=0.35)
    a.plot([mid], [1], "o", c=COL[SETTINGS[0]], ms=7)
    jit = np.linspace(-0.12, 0.12, len(S))
    a.plot(S["depth_m"], 0 + jit, "o", c=INK2, ms=4, alpha=0.8, mfc="white")
    a.plot([L["sio_quartiles_m"]["median"]], [0], "|", c=INK, ms=16, mew=2)
    a.set_yticks([0, 1])
    a.set_yticklabels(["SIO supply per irrigated ha,\n2020–22 (circles; bar = median)",
                       "assumed irrigation for greening\n(2024 reference ET; scenarios)"])
    a.set_ylim(-0.5, 1.5)
    a.set_xlim(0, max(hi, S["depth_m"].max()) * 1.08)
    a.set_xlabel("irrigation water (m/yr; 1 m = 10,000 m³/ha)")
    a.set_title("a  Hypothetical irrigation depth")
    a.spines["left"].set_visible(False); a.tick_params(axis="y", length=0)

    E = L["energy"]
    yy = np.arange(len(E))[::-1]
    for yi, (_, r) in zip(yy, E.iterrows()):
        b.hlines(yi, r["MWh_per_ha_yr_low"], r["MWh_per_ha_yr_high"], color=INK2, lw=6, alpha=0.35)
        b.plot([r["MWh_per_ha_yr_central"]], [yi], "o", c=INK, ms=7)
    b.set_yticks(yy)
    b.set_yticklabels([f"{s}\n({k} kWh/m³)" for s, k in zip(E["source"], E["kwh_per_m3"])])
    b.set_ylim(-0.5, len(E) - 0.5)
    b.set_xlim(0, E["MWh_per_ha_yr_high"].max() * 1.08)
    b.set_xlabel("electricity for that water (MWh/ha/yr)")
    b.set_title("b  Hypothetical gross supply energy")
    b.spines["left"].set_visible(False); b.tick_params(axis="y", length=0)
    outside = int((~S["depth_m"].between(lo, hi)).sum())
    fig.text(.08, -.10, f"{outside}/{len(S)} workbook-joined SIO branch-years lie outside the assumed band; none is named Jeddah or Makkah.", fontsize=8)
    fig.text(.08, -.18, "Supplementary hypothetical scenario; no observed greening irrigation or energy use. Assumed depth uses 2024 ET.", fontsize=8)
    _save(fig, out, "fig4_water_energy")


def fig_test(cv: dict, V: dict, out: Path):
    """a: the model on areas it never saw; b, c: its predicted cooling for the greened cells against the measured."""
    fig = plt.figure(figsize=(13.2, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.15, 1.15], wspace=0.5)
    a = fig.add_subplot(gs[0, 0])
    o = cv["oof"]
    slate = LinearSegmentedColormap.from_list("slate_density", ["#e6edf2", "#acc2d3", "#5d7f99", "#233e58"])
    hb = a.hexbin(o["observed"], o["predicted"], gridsize=45, mincnt=1,
                  cmap=slate, bins="log", linewidths=0)
    hb.set_rasterized(True)
    cb = fig.colorbar(hb, ax=a, fraction=0.05, pad=0.02)
    cb.set_label("cells per hexagon", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    lo, hi = np.nanpercentile(np.r_[o["observed"], o["predicted"]], [0.5, 99.5])
    identity, = a.plot([lo, hi], [lo, hi], c="black", lw=1.6)
    identity.set_path_effects([path_effects.Stroke(linewidth=2.8, foreground="white"), path_effects.Normal()])
    a.set_xlim(lo, hi); a.set_ylim(lo, hi)
    a.set_xlabel("observed summer LST (°C)")
    a.set_ylabel("predicted, area held out (°C)")
    a.set_title(f"a  Model skill (R² = {cv['r2']:.2f})")
    BA, DA = V["by_setting"], V["by_dose"]
    BB, DB = V["strict"], V["strict_by_dose"]
    ys, axes = [], []
    marks = (("measured", "measured", "o", None), ("model, away from greening (B)", "model", "s", "white"),
             ("model, all non-greened cells (A)", "model", "D", "#bdbdbd"))
    for j, s in enumerate(SETTINGS):
        ax = fig.add_subplot(gs[0, j + 1])
        axes.append(ax)
        groups = [("all", BB[BB["setting"] == s], BA[BA["setting"] == s])]
        for p in DB[DB["setting"] == s]["pixels"]:
            groups.append((p, DB[(DB["setting"] == s) & (DB["pixels"] == p)], DA[(DA["setting"] == s) & (DA["pixels"] == p)]))
        for i, (lab, b_, a_) in enumerate(groups):
            if not len(b_):
                continue
            rb, ra = b_.iloc[0], (a_.iloc[0] if len(a_) else None)
            for k, (name, key, mk, fc) in enumerate(marks):
                r = rb if k < 2 else ra
                if r is None:
                    continue
                col = COL[s] if k == 0 else INK
                ax.errorbar(i + (k - 1) * 0.2, r[f"{key}_C"], yerr=np.reshape(_err(r[f"{key}_C"], r[f"{key}_lo_C"], r[f"{key}_hi_C"]), (2, 1)),
                            fmt=mk, c=col, mfc=fc if fc else col, ms=4.5, capsize=2, lw=1.2, label=name if i == 0 else None)
                ys += [r[f"{key}_lo_C"], r[f"{key}_hi_C"]]
        ax.axhline(0, c=INK, lw=0.8)
        ax.set_xticks(range(len(groups)), [g[0] for g in groups])
        ax.set_xlim(-0.6, len(groups) - 0.4)
        ax.set_xlabel("greened pixels of the cell's 9")
        ax.set_title("b  Outside (744-cell sensitivity)" if j == 0
                     else "c  Built-up (310-cell sensitivity)")
        if j == 0:
            ax.set_ylabel("cooling of the cell (°C)")
    if ys:
        lo_y, hi_y = min(ys + [0]), max(ys + [0])
        for ax in axes:
            ax.set_ylim(lo_y - 0.05 * (hi_y - lo_y), hi_y + 0.08 * (hi_y - lo_y))
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", c=INK2, label="measured (in the setting's colour)"),
         Line2D([], [], marker="s", ls="", c=INK, mfc="white", label=marks[1][0]),
         Line2D([], [], marker="D", ls="", c=INK, mfc="#bdbdbd", label=marks[2][0])]
    fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.62, -0.1), ncol=3, fontsize=8, frameon=False)
    fig.text(.08, -.16, "Panel a pools both settings. Effect panels use concurrent-change strata; outside B has 14/744 supported cells and 0/75 at 9/9.", fontsize=8)
    fig.text(.08, -.21, "These are diagnostic contrasts, not validation of the primary 1,006-cell effects. Joint-refit tail endpoints are approximate; absolute-LST skill does not validate effects.", fontsize=8)
    _save(fig, out, "fig3_model_test")


def fig_decision(D: dict, V: dict, out: Path):
    """Conditional ratios and model predictions decomposed by before/after support."""
    from matplotlib.lines import Line2D
    S,BC=D["summary"],D["by_coast"]
    SC=D.get("support_coasts",pd.DataFrame())
    fig,(a,b)=plt.subplots(1,2,figsize=(14,7),gridspec_kw={"width_ratios":[1.5,1],"wspace":.65})
    fig.subplots_adjust(bottom=.38,top=.87)
    for i,(_,r) in enumerate(S.iterrows()):
        y=len(S)-1-i
        if np.isfinite(r.scenario_low) and np.isfinite(r.scenario_high):
            a.hlines(y,r.scenario_low,r.scenario_high,color=COL[r.setting],lw=7,alpha=.22)
        if np.isfinite(r.m3_per_degree_low) and np.isfinite(r.m3_per_degree_high):
            a.hlines(y,r.m3_per_degree_low,r.m3_per_degree_high,color=INK,lw=1.5)
        a.plot(r.m3_per_degree,y,"o",color=COL[r.setting],ms=5)
    a.set_yticks(range(len(S)),[(("Outside" if r.setting==SETTINGS[0] else "Built-up")+" | "+r.basis.split(": ")[1]) for _,r in S.iloc[::-1].iterrows()],fontsize=8)
    a.set_xlabel("Assumed annual irrigation / summer cell LST reduction\n(m³/year per °C; conditional accounting)",fontsize=8)
    a.set_title("a  Hypothetical irrigation conversion (744-cell sensitivity)")
    a.set_xlim(left=0)
    a.legend(handles=[Line2D([],[],color=INK,lw=1.5,label="Central irrigation: joint 95% interval"),
        Line2D([],[],color=INK,lw=7,alpha=.22,label="Irrigation + sampling scenario envelope")],
        loc="upper center",bbox_to_anchor=(.5,-.24),fontsize=8)
    bands=[x for x in ["< 2 km","2–5 km","5–10 km","≥ 10 km"] if x in set(BC.coast_band)]
    for j,setting in enumerate(SETTINGS):
        for _,r in BC[BC.setting==setting].iterrows():
            x=bands.index(r.coast_band)+(j-.5)*.4
            b.vlines(x,r.lo_C,r.hi_C,color=COL[setting],lw=1.5)
            b.plot(x,r.cooling_px_C,"o",color=COL[setting],ms=4)
            if np.isfinite(r.model_px_C):b.plot(x-.07,r.model_px_C,"x",color=INK,ms=5)
            if len(SC):
                sub=SC[(SC.setting==setting)&(SC.coast_band==r.coast_band)]
                for k,kind in enumerate(["supported","unsupported"]):
                    sr=sub[sub.subset==kind]
                    if len(sr) and int(sr.iloc[0].n_cells) >= 10:
                        sx=x+.06+k*.075
                        b.plot(sx,sr.iloc[0].model_px_C,"s",mec=INK,
                            mfc=INK if kind=="supported" else "white",ms=4)
                        if np.isfinite(sr.iloc[0].measured_px_C):
                            b.plot(sx+.033,sr.iloc[0].measured_px_C,"o",mec="white",
                                   mfc=COL[setting],ms=4,mew=.75)
    b.axhline(0,color=INK,lw=.8)
    b.set_xticks(range(len(bands)),bands)
    b.set_ylabel("LST change per greened optical pixel (°C)")
    b.set_xlabel("Distance to coast")
    b.set_title("b  Model diagnostic (744-cell sensitivity)",pad=34)
    missing=not ((BC.setting==SETTINGS[0])&(BC.coast_band=="2–5 km")).any()
    counts=V.get("support_counts",pd.DataFrame())
    note="Outside: no reportable 2–5 km estimate." if missing else ""
    if missing and len(SC):
        absent=SC[(SC.setting==SETTINGS[0])&(SC.coast_band=="2–5 km")]
        if len(absent):note=f"Outside 2–5 km: n={int(absent.iloc[0].n_all)}; estimate withheld."
    if len(counts):
        q=counts[counts.setting==SETTINGS[0]]
        note+=f"\nOutside support: {q.n_supported.sum():.0f}/{q.n_matched.sum():.0f} matched cells."
    b.text(0,1.025,note,transform=b.transAxes,fontsize=7,va="bottom")
    b.legend(handles=[Line2D([],[],marker="o",ls="",color=COL[s],label="Measured: "+("outside" if s==SETTINGS[0] else "built-up")) for s in SETTINGS]+[
        Line2D([],[],marker="x",ls="",color=INK,label="Model B: all matched cells"),
        Line2D([],[],marker="s",ls="",color=INK,label="Model B: support passed (subset)"),
        Line2D([],[],marker="s",ls="",color=INK,mfc="white",label="Model B: support failed (subset)"),
        Line2D([],[],marker="o",ls="",color=COL[SETTINGS[0]],mfc=COL[SETTINGS[0]],mec="white",
               label="Measured: same subset as square (setting colour)")],
        loc="upper center",bbox_to_anchor=(.5,-.24),fontsize=7)
    fig.text(.08,.055,"Shared irrigation depth makes panel a an algebraic re-expression of cooling per greened area, not an independent resource ranking.",fontsize=8)
    fig.text(.08,.025,"Concurrent-change contrasts. Outside crosses are mostly unsupported; subset markers with n<10 are omitted; squares are not accuracy passes.",fontsize=8)
    _save(fig,out,"fig5_decision")

