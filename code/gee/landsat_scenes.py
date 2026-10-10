"""Find the Landsat scenes behind your LST CSV, and export the same columns for other dates.

Same one-time setup as fetch_landcover.py (pip install earthengine-api; earthengine authenticate).

Your CSV's LST converts back to whole Landsat counts in 81% of pixels and to exact half-counts in the rest: it is a
per-pixel MEDIAN over several scenes, not one scene. `identify` recovers which scenes: it samples 400 of your pixels
in every Landsat 8/9 scene over Jeddah from October of the year before to March of the year after, then tries every
contiguous date window x cloud-mask variant x sensor set and keeps the median that reproduces your values exactly.

1. Which scenes / date window?  (≈10–15 min)

    python code\\gee\\landsat_scenes.py identify --project YOUR-CLOUD-PROJECT-ID ^
        --data "C:\\Users\\Nelly\\Downloads\\Jeddah_LST_Dataset_2023_raw.csv"

   It prints the window, the scenes in it, the cloud mask, how NDVI/NDBI were built, and the exact
   --scene-dates text and export-composite command to use next.

2. The same composite for another year (the replication for a composite):

    python code\\gee\\landsat_scenes.py export-composite --project YOUR-CLOUD-PROJECT-ID --recipe v5 --year 2024 ^
        --data "C:\\Users\\Nelly\\Downloads\\Jeddah_LST_Dataset_2023_raw.csv"

   --recipe v5 is exactly how Jeddah_LST_Dataset_2023_raw.csv was built (checked: it reproduces the CSV's LST, NDVI,
   NDBI and ST_EMIS to machine precision). For another recipe give --start/--end/--mask/--sensors/--index. A single scene instead:
   `export --scene <scene id>`.

   Each output replaces LST, NDVI, NDBI, ST_EMIS and ST_EMSD and keeps every other column (Elevation, purity flag,
   ...) from your original CSV — say in the methods that the purity flag is carried over. Then run
   tools\\scene_replication.py with the files it wrote.
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

COLLECTIONS = {"LC08": "LANDSAT/LC08/C02/T1_L2", "LC09": "LANDSAT/LC09/C02/T1_L2"}
BANDS = ["ST_B10", "ST_EMIS", "ST_EMSD", "SR_B4", "SR_B5", "SR_B6", "QA_PIXEL"]
SCENE_COLS = ["LST", "NDVI", "NDBI", "ST_EMIS", "ST_EMSD"]
MASKS = {"none": (), "cloud": (3,), "cloud+shadow": (3, 4), "full": (1, 2, 3, 4)}
SENSOR_SETS = {"LC08,LC09": ("LC08", "LC09"), "LC08": ("LC08",), "LC09": ("LC09",)}
INDEX_MODES = ("index-of-median-scaled", "median-of-index-scaled", "index-of-median-dn", "median-of-index-dn")
ST_SCALE, ST_OFFSET_K = 0.00341802, 149.0


def lst_counts(lst_c) -> np.ndarray:
    """LST in °C -> Landsat C2 L2 ST_B10 counts (whole for one scene, half-counts for a median of an even number)."""
    return (np.asarray(lst_c, float) + 273.15 - ST_OFFSET_K) / ST_SCALE


def fingerprint(lst_c) -> dict:
    c = lst_counts(lst_c); c = c[np.isfinite(c)]
    whole = float(np.mean(np.abs(c - np.round(c)) < 1e-3))
    half = float(np.mean(np.abs(2 * c - np.round(2 * c)) < 2e-3)) - whole
    return {"whole": whole, "half": half,
            "verdict": "single scene" if whole > 0.999 else "median composite" if whole + half > 0.999 and half > 0.001
            else "neither (mean composite, resampled or rescaled)"}


def bad_mask(qa, mask: str) -> np.ndarray:
    q = np.nan_to_num(np.asarray(qa, float), nan=0).astype(np.int64)
    bad = np.zeros(q.shape, bool)
    for b in MASKS[mask]:
        bad |= ((q >> b) & 1).astype(bool)
    return bad


def _ratio(a, b):
    with np.errstate(invalid="ignore", divide="ignore"):
        return (a - b) / (a + b)


def indices(r, n, w, scaling: str):
    if scaling == "scaled":
        r, n, w = (x * 0.0000275 - 0.2 for x in (r, n, w))
    return _ratio(n, r), _ratio(w, n)


def derive(s: pd.DataFrame, index_scaling: str = "scaled", mask: str = "full") -> pd.DataFrame:
    """One scene's C2 L2 DNs -> the pipeline's columns; masked pixels become NaN."""
    out = pd.DataFrame(index=s.index)
    out["LST"] = s["ST_B10"] * ST_SCALE + ST_OFFSET_K - 273.15
    out["ST_EMIS"] = s["ST_EMIS"] * 0.0001
    out["ST_EMSD"] = s["ST_EMSD"] * 0.0001
    out["NDVI"], out["NDBI"] = indices(s["SR_B4"].to_numpy(float), s["SR_B5"].to_numpy(float),
                                       s["SR_B6"].to_numpy(float), index_scaling)
    out.loc[bad_mask(s["QA_PIXEL"], mask) | s["ST_B10"].isna().to_numpy(), SCENE_COLS] = np.nan
    return out


def search_windows(target: np.ndarray, st: np.ndarray, qa: np.ndarray, dates, sensors, top: int = 10) -> pd.DataFrame:
    """target: CSV counts (n,); st, qa: per-scene ST_B10 counts and QA_PIXEL (k, n), scenes sorted by date.
    Tries every window of consecutive acquisition dates x mask x sensor set (scenes on the same date — adjacent
    rows/paths — go in together); score = share of pixels whose median equals the CSV value exactly."""
    dates, sensors = np.asarray(dates).astype(str), np.asarray(sensors)
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for sname, sset in SENSOR_SETS.items():
            ks = np.flatnonzero(np.isin(sensors, sset))
            if not len(ks):
                continue
            ud = np.unique(dates[ks])
            for mname in MASKS:
                v = st[ks].astype(float).copy()
                v[bad_mask(qa[ks], mname)] = np.nan
                dk = dates[ks]
                for a in range(len(ud)):
                    for b in range(a, len(ud)):
                        sel = (dk >= ud[a]) & (dk <= ud[b])
                        med = np.nanmedian(v[sel], axis=0)
                        ok = np.isfinite(med) & np.isfinite(target)
                        if ok.sum() < 0.5 * np.isfinite(target).sum():
                            continue
                        dev = np.abs(med[ok] - target[ok])
                        rows.append({"sensors": sname, "mask": mname, "first": ud[a], "last": ud[b],
                                     "n_scenes": int(sel.sum()), "exact_share": float(np.mean(dev < 0.01)),
                                     "coverage": float(ok.mean()), "median_abs_dC": float(np.median(dev)) * ST_SCALE})
    T = pd.DataFrame(rows)
    if T.empty:
        return T
    return T.sort_values(["exact_share", "coverage", "n_scenes"], ascending=[False, False, True]).head(top).reset_index(drop=True)


def index_construction(csv_ndvi, csv_ndbi, r, n, w, qa, mask: str) -> pd.DataFrame:
    """Which way were NDVI/NDBI built for the composite? r, n, w, qa: (k, n) for the scenes in the window."""
    bad = bad_mask(qa, mask)
    R, N, W = (np.where(bad, np.nan, x.astype(float)) for x in (r, n, w))
    out = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for scaling in ("scaled", "dn"):
            vi, bi = indices(np.nanmedian(R, 0), np.nanmedian(N, 0), np.nanmedian(W, 0), scaling)
            out.append((f"index-of-median-{scaling}", vi, bi))
            vi_s, bi_s = indices(R, N, W, scaling)
            out.append((f"median-of-index-{scaling}", np.nanmedian(vi_s, 0), np.nanmedian(bi_s, 0)))
    rows = []
    for name, vi, bi in out:
        ok = np.isfinite(vi) & np.isfinite(csv_ndvi)
        rows.append({"index": name, "ndvi_match": float(np.mean(np.abs(vi[ok] - csv_ndvi[ok]) < 1e-5)) if ok.any() else np.nan,
                     "ndbi_match": float(np.mean(np.abs(bi[ok] - csv_ndbi[ok]) < 1e-5)) if ok.any() else np.nan})
    return pd.DataFrame(rows).sort_values(["ndvi_match", "ndbi_match"], ascending=False).reset_index(drop=True)


def coords(df: pd.DataFrame) -> tuple[str, str]:
    lon = next(c for c in ("lon", "longitude", "Longitude", "x", "X") if c in df.columns)
    lat = next(c for c in ("lat", "latitude", "Latitude", "y", "Y") if c in df.columns)
    return lon, lat


def _init(project):
    try:
        import ee
    except ImportError:
        sys.exit("earthengine-api is not installed: run  pip install earthengine-api")
    try:
        ee.Initialize(project=project)
    except Exception:
        ee.Authenticate(); ee.Initialize(project=project)
    return ee


def _sample(ee, img, pts: pd.DataFrame, lon: str, lat: str, bands=BANDS, chunk: int = 2000) -> pd.DataFrame:
    rows = []
    for i0 in range(0, len(pts), chunk):
        part = pts.iloc[i0:i0 + chunk]
        fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([float(r[lon]), float(r[lat])]), {"i": int(i)})
                                   for i, r in part.iterrows()])
        for attempt in range(3):
            try:
                res = img.reduceRegions(collection=fc, reducer=ee.Reducer.first(), scale=30).getInfo()
                break
            except Exception as e:
                if attempt == 2:
                    raise
                print(f"    retrying ({e})"); time.sleep(5)
        rows += [{"i": f["properties"]["i"], **{b: f["properties"].get(b) for b in bands}} for f in res["features"]]
        if len(pts) > chunk:
            print(f"    {min(i0 + chunk, len(pts)):,} / {len(pts):,} pixels", flush=True)
    return pd.DataFrame(rows).set_index("i").sort_index().reindex(pts.index).astype(float)


def _collection(ee, region, start, end, sensors=("LC08", "LC09")):
    col = None
    for s in sensors:
        ic = ee.ImageCollection(COLLECTIONS[s]).filterBounds(region).filterDate(start, end)
        col = ic if col is None else col.merge(ic)
    return col


def _scene_image(ee, scene_id: str):
    return ee.Image(f"{COLLECTIONS[scene_id[:4]]}/{scene_id}").select(BANDS)


def _region(ee, raw, lon, lat):
    return ee.Geometry.Rectangle([raw[lon].min(), raw[lat].min(), raw[lon].max(), raw[lat].max()])


def identify(a):
    raw = pd.read_csv(a.data)
    lon, lat = coords(raw)
    fp = fingerprint(raw["LST"])
    print(f"LST values: {100 * fp['whole']:.1f}% whole Landsat counts, {100 * fp['half']:.1f}% half-counts -> {fp['verdict']}.")
    ee = _init(a.project)
    pts = raw.dropna(subset=["LST", "NDVI", "NDBI", lon, lat]).sample(a.n, random_state=0)
    region = _region(ee, raw, lon, lat)
    col = _collection(ee, region, f"{a.year - 1}-10-01", f"{a.year + 1}-04-01")
    meta = col.reduceColumns(ee.Reducer.toList(4), ["system:index", "DATE_ACQUIRED", "CLOUD_COVER", "SCENE_CENTER_TIME"]
                             ).get("list").getInfo()
    meta = pd.DataFrame(meta, columns=["id", "date", "cloud_cover", "time_utc"])
    meta["id"] = meta["id"].str.replace(r"^\d+_", "", regex=True)
    meta = meta.sort_values(["date", "id"]).reset_index(drop=True)
    print(f"{len(meta)} Landsat 8/9 scenes over the study area, {a.year - 1}-10 to {a.year + 1}-03; "
          f"sampling {len(pts)} of your pixels in each.")
    S = []
    for k, m in meta.iterrows():
        S.append(_sample(ee, _scene_image(ee, m["id"]), pts, lon, lat))
        print(f"  {k + 1:>3}/{len(meta)} {m['id']}  cloud {m['cloud_cover']:5.1f}%", flush=True)
    st = np.stack([s["ST_B10"].to_numpy() for s in S]); qa = np.stack([s["QA_PIXEL"].to_numpy() for s in S])
    print("Trying every date window x cloud mask x sensor set ...", flush=True)
    T = search_windows(lst_counts(pts["LST"]), st, qa, meta["date"].to_numpy(), meta["id"].str[:4].to_numpy())
    out = Path(a.out); T.to_csv(out, index=False)
    if T.empty:
        sys.exit("No window covers enough of your pixels; check --year.")
    print("\nBest windows (exact_share = share of sampled pixels reproduced exactly):")
    print(T.head(6).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    top = T.iloc[0]
    ids = meta[(meta["date"] >= top["first"]) & (meta["date"] <= top["last"]) & meta["id"].str[:4].isin(SENSOR_SETS[top["sensors"]])]
    if top["exact_share"] < 0.95:
        print("\n==> No window reproduces your CSV exactly (best {:.0%}). It may use Landsat 7, Tier 2 scenes, another "
              "cloud mask or a different sampling scale; the date range is in the Earth Engine script that exported it. "
              "The closest window above is still the best guess.".format(top["exact_share"]))
        return
    prev_ = meta[(meta["date"] < top["first"]) & meta["id"].str[:4].isin(SENSOR_SETS[top["sensors"]])]["date"].max()
    next_ = meta[(meta["date"] > top["last"]) & meta["id"].str[:4].isin(SENSOR_SETS[top["sensors"]])]["date"].min()
    print(f"\n==> Your CSV is the per-pixel median of {len(ids)} scenes ({top['sensors']}), {top['first']} to {top['last']}, "
          f"cloud mask '{top['mask']}' ({100 * top['exact_share']:.1f}% of sampled pixels reproduced exactly).")
    print(f"    The script's date filter started after {prev_} and ended before {next_} (the neighbouring scenes).")
    print("    Scenes: " + ", ".join(ids["id"]))
    ties = T[(T["exact_share"] == top["exact_share"]) & (T["first"] == top["first"]) & (T["last"] == top["last"])
             & (T["sensors"] == top["sensors"])]["mask"].tolist()
    if len(ties) > 1:
        print(f"    Cloud masks {', '.join(ties)} give the same result on these pixels; any of them reproduces the composite.")
    sub = [S[i] for i in ids.index]
    ic = index_construction(pts["NDVI"].to_numpy(), pts["NDBI"].to_numpy(), *(np.stack([s[b].to_numpy() for s in sub])
                            for b in ("SR_B4", "SR_B5", "SR_B6", "QA_PIXEL")), top["mask"])
    best = ic.iloc[0]
    print(f"    NDVI/NDBI: best match '{best['index']}' (NDVI {100 * best['ndvi_match']:.0f}%, NDBI {100 * best['ndbi_match']:.0f}% "
          "of pixels exact)" + ("" if best["ndvi_match"] > 0.95 else " — not exact; NDVI/NDBI may have been built differently"))
    txt = f"median of {len(ids)} Landsat 8/9 scenes, {top['first']} to {top['last']}"
    print(f"\nNext:\n  python tools\\rebuild_report.py results_v13 --figures --data \"{a.data}\" --scene-dates \"{txt}\"")
    y = int(top["first"][:4])
    for yy in (y - 1, y + 1):
        print(f"  python code\\gee\\landsat_scenes.py export-composite --project {a.project} --data \"{a.data}\" "
              f"--start {yy}{top['first'][4:]} --end {yy}{top['last'][4:]} --mask {top['mask']} --sensors {top['sensors']} "
              f"--index {best['index']}")
    print("  (for the other years the end date is inclusive; check that each window still holds a similar number of scenes)")
    print(f"\nFull table: {out}")


def _masked(ee, img, mask: str, radsat: bool = False):
    bits = sum(1 << b for b in MASKS[mask])
    if bits:
        img = img.updateMask(img.select("QA_PIXEL").bitwiseAnd(bits).eq(0))
    return img.updateMask(img.select("QA_RADSAT").eq(0)) if radsat else img


RECIPES = {"v5": dict(sensors="LC08", mask="full", radsat=True, max_cloud=20.0, index="median-of-index-scaled")}


def _server_indices(img, scaling: str):
    b = {k: img.select(k) for k in ("SR_B4", "SR_B5", "SR_B6")}
    if scaling == "scaled":
        b = {k: v.multiply(0.0000275).add(-0.2) for k, v in b.items()}
    ndvi = b["SR_B5"].subtract(b["SR_B4"]).divide(b["SR_B5"].add(b["SR_B4"])).rename("NDVI")
    ndbi = b["SR_B6"].subtract(b["SR_B5"]).divide(b["SR_B6"].add(b["SR_B5"])).rename("NDBI")
    return ndvi.addBands(ndbi)


def _write(raw: pd.DataFrame, d: pd.DataFrame, path: Path):
    out = raw.copy()
    out[SCENE_COLS] = d[SCENE_COLS].to_numpy()
    out.to_csv(path, index=False, float_format="%.10f")
    ok = out["LST"].notna().mean()
    print(f"  wrote {path}: {ok:.0%} of pixels usable" + ("  — under 80%: pick a clearer scene/window" if ok < 0.8 else ""))


def export_composite(a):
    ee = _init(a.project)
    raw = pd.read_csv(a.data)
    lon, lat = coords(raw)
    region = _region(ee, raw, lon, lat)
    if a.recipe:
        for k, v in RECIPES[a.recipe].items():
            setattr(a, k, v)
        region = ee.Geometry.Rectangle([39.0, 21.2, 39.4, 21.8])
    if a.year:
        a.start, a.end = f"{a.year}-01-01", f"{a.year}-12-30"
    if not (a.start and a.end):
        sys.exit("give --year, or --start and --end")
    end_excl = (pd.Timestamp(a.end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    ic = _collection(ee, region, a.start, end_excl, SENSOR_SETS[a.sensors])
    if a.max_cloud is not None:
        ic = ic.filter(ee.Filter.lt("CLOUD_COVER", a.max_cloud))
    ic = ic.map(lambda i: _masked(ee, i, a.mask, a.radsat))
    meta = ic.reduceColumns(ee.Reducer.toList(2), ["system:index", "CLOUD_COVER"]).get("list").getInfo()
    print(f"{a.start} to {a.end}: {len(meta)} scenes ({a.sensors}, mask '{a.mask}'"
          + (", no saturation" if a.radsat else "") + (f", scene cloud < {a.max_cloud:g}%" if a.max_cloud is not None else "") + ")")
    print("  " + ", ".join(str(m[0]).split("_")[-1] for m in meta))
    mode, scaling = a.index.rsplit("-", 1)
    st = ic.select(["ST_B10", "ST_EMIS", "ST_EMSD"]).median()
    if mode == "index-of-median":
        idx = _server_indices(ic.select(["SR_B4", "SR_B5", "SR_B6"]).median(), scaling)
    else:
        idx = ic.map(lambda i: _server_indices(i, scaling)).median()
    s = _sample(ee, st.addBands(idx), raw, lon, lat, bands=["ST_B10", "ST_EMIS", "ST_EMSD", "NDVI", "NDBI"])
    d = pd.DataFrame({"LST": s["ST_B10"] * ST_SCALE + ST_OFFSET_K - 273.15, "ST_EMIS": s["ST_EMIS"] * 0.0001,
                      "ST_EMSD": s["ST_EMSD"] * 0.0001, "NDVI": s["NDVI"], "NDBI": s["NDBI"]})
    dates = sorted(str(m[0]).split("_")[-1] for m in meta)
    label = (f"median of {len(meta)} Landsat {'8' if a.sensors == 'LC08' else '8/9'} scenes, "
             f"{pd.to_datetime(dates[0]).date()} to {pd.to_datetime(dates[-1]).date()}") if dates else f"{a.start} to {a.end}"
    path = Path(a.out_dir or Path(a.data).parent) / f"Jeddah_LST_{a.start[:4] if a.year else a.start + '_' + a.end}_median_raw.csv"
    _write(raw, d, path)
    print(f"\nNext (legacy external tool): python tools\\scene_replication.py --reference results_v13 --landcover code\\gee\\Jeddah_landcover.csv "
          f"--scene \"{label}={path}\"")


def export(a):
    ee = _init(a.project)
    raw = pd.read_csv(a.data)
    lon, lat = coords(raw)
    for sid in a.scene:
        date = pd.to_datetime(sid.split("_")[2]).strftime("%Y-%m-%d")
        print(f"{sid} ({date}): sampling {len(raw):,} pixels")
        s = _sample(ee, _scene_image(ee, sid), raw, lon, lat)
        _write(raw, derive(s, a.index_scaling, a.mask), Path(a.out_dir or Path(a.data).parent) / f"Jeddah_LST_{date}_raw.csv")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("identify", help="find the scenes / date window behind your CSV")
    p.add_argument("--project", required=True); p.add_argument("--data", required=True)
    p.add_argument("--year", type=int, default=2023); p.add_argument("--n", type=int, default=400)
    p.add_argument("--out", default=str(Path(__file__).resolve().parent / "scene_windows.csv"))
    p.set_defaults(fn=identify)
    c = sub.add_parser("export-composite", help="the same median composite for another date window")
    c.add_argument("--project", required=True); c.add_argument("--data", required=True)
    c.add_argument("--recipe", choices=list(RECIPES), help="v5 = exactly how Jeddah_LST_Dataset_2023_raw.csv was built")
    c.add_argument("--year", type=int, help="whole calendar year, as the v5 script's filterDate")
    c.add_argument("--start", help="first day, YYYY-MM-DD"); c.add_argument("--end", help="last day, inclusive")
    c.add_argument("--max-cloud", type=float, help="keep scenes with CLOUD_COVER below this")
    c.add_argument("--radsat", action="store_true", help="also mask radiometrically saturated pixels")
    c.add_argument("--mask", choices=list(MASKS), default="full"); c.add_argument("--sensors", choices=list(SENSOR_SETS), default="LC08,LC09")
    c.add_argument("--index", choices=INDEX_MODES, default="index-of-median-scaled", help="how NDVI/NDBI were built (identify tells you)")
    c.add_argument("--out-dir", help="default: the folder of --data")
    c.set_defaults(fn=export_composite)
    q = sub.add_parser("export", help="your CSV's columns from one scene")
    q.add_argument("--project", required=True); q.add_argument("--data", required=True)
    q.add_argument("--scene", action="append", required=True, help="Landsat scene id, e.g. LC09_170045_20230722")
    q.add_argument("--index-scaling", choices=("scaled", "dn"), default="scaled")
    q.add_argument("--mask", choices=list(MASKS), default="full")
    q.add_argument("--out-dir", help="default: the folder of --data")
    q.set_defaults(fn=export)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
