"""NDVI-threshold sensitivity (supervisor review, revision r1): code/gee/export_panel.py with the non-vegetated
threshold (default 0.15) and the greened threshold (default 0.30) as arguments. Everything else is unchanged;
this file was generated from export_panel.py by text substitution of those two literals only.

Original docstring:
Export the cell panel for the framework using Earth Engine's Python API.

After Earth Engine authentication, run from the package root:
    python code/gee/export_panel.py --project YOUR-CLOUD-PROJECT-ID

Writes code/gee/panel.csv and code/gee/panel_meta.json by default. The companion
code/gee/export_panel.js contains the same recipe for the Earth Engine Code Editor.
NDVI transitions do not by themselves establish applied irrigation.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
YEARS = (2014, 2015, 2018, 2019, 2024, 2025)
MONTHS = (5, 6, 7, 8, 9)
RECT = [39.0, 21.2, 39.4, 21.8]
CELL_M = 90
CONTROL_FRACTION = 0.10
GHSL_2015 = "JRC/GHSL/P2023A/GHS_BUILT_S/2015"   # built-up surface, 100 m, epoch 2015 (before the greening)
TILES = (4, 6)              # lon x lat tiles, so each request stays under Earth Engine's 5,000-feature limit
DW_PRE = ("2015-06-01", "2015-10-01")   # Dynamic World starts June 2015; this is inside the "before" period
DW_POST = ("2024-05-01", "2025-10-01")
NB_RADIUS_M = 500
NONVEG, GREEN = 0.15, 0.30   # set from the command line


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


def build(ee):
    rect = ee.Geometry.Rectangle(RECT)
    land = rect.intersection(ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017")
                             .filter(ee.Filter.eq("country_na", "Saudi Arabia")).geometry(), ee.ErrorMargin(1))
    col = (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").filterBounds(rect).filter(ee.Filter.lt("CLOUD_COVER", 20)))
    proj30 = col.first().select("SR_B4").projection()

    def prep(img):
        qa, sat = img.select("QA_PIXEL"), img.select("QA_RADSAT")
        m = (qa.bitwiseAnd(1 << 1).eq(0).And(qa.bitwiseAnd(1 << 2).eq(0)).And(qa.bitwiseAnd(1 << 3).eq(0))
             .And(qa.bitwiseAnd(1 << 4).eq(0)).And(sat.eq(0)))
        o = img.select("SR_B.*").multiply(0.0000275).add(-0.2)
        ndvi = o.normalizedDifference(["SR_B5", "SR_B4"]).rename("NDVI")
        ndwi = o.normalizedDifference(["SR_B3", "SR_B5"]).rename("NDWI")
        ndbi = o.normalizedDifference(["SR_B6", "SR_B5"]).rename("NDBI")
        lst = img.select("ST_B10").multiply(0.00341802).add(149.0).subtract(273.15).rename("LST")
        emis = img.select("ST_EMIS").multiply(0.0001).rename("EMIS")
        # keep the acquisition time, or the per-year/per-month filters below would find nothing
        return ee.Image(ee.Image.cat([ndvi, ndwi, ndbi, lst, emis]).updateMask(m).copyProperties(img, ["system:time_start"]))

    cp = col.map(prep)
    comps, counts = {}, {}
    for y in YEARS:
        monthly = []
        for mth in MONTHS:
            c = cp.filter(ee.Filter.calendarRange(y, y, "year")).filter(ee.Filter.calendarRange(mth, mth, "month"))
            counts[(y, mth)] = c.size()
            monthly.append(c.median().set("n", c.size()))
        comps[y] = ee.ImageCollection.fromImages(monthly).filter(ee.Filter.gt("n", 0)).mean().reproject(proj30)

    water = None
    for y in YEARS:
        w = comps[y].select("NDWI").gt(0)
        water = w if water is None else water.Or(w)
    ok = water.unmask(0).focal_max(radius=300, units="meters").Not()

    nd = {y: comps[y].select("NDVI") for y in YEARS}
    greened = (nd[2014].lt(NONVEG).And(nd[2015].lt(NONVEG)).And(nd[2024].gte(GREEN)).And(nd[2025].gte(GREEN))).And(ok)
    never = ok
    for y in YEARS:
        never = never.And(nd[y].lt(NONVEG))
    dist = greened.selfMask().fastDistanceTransform(64).sqrt().multiply(30).unmask(99999).rename("dist_greened_m")

    dw = ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1").filterBounds(rect).filter(ee.Filter.calendarRange(5, 9, "month")).select("built")

    def to30(im):
        return im.reproject(proj30.atScale(10)).reduceResolution(ee.Reducer.mean(), maxPixels=64).reproject(proj30)

    built_pre = to30(dw.filterDate(*DW_PRE).mean()).rename("built_pre")
    built_post = to30(dw.filterDate(*DW_POST).mean()).rename("built_post")
    # surroundings: mean built-up probability within NB_RADIUS_M, greened pixels left out so a cell's own greening
    # (or its neighbours') never enters the description of how its surroundings developed
    not_green = greened.unmask(0).Not()
    kern = ee.Kernel.circle(NB_RADIUS_M, "meters")
    nb_pre = built_pre.updateMask(not_green).reduceNeighborhood(reducer=ee.Reducer.mean(), kernel=kern,
                                                                skipMasked=False).reproject(proj30).rename("nb_built_pre")
    nb_post = built_post.updateMask(not_green).reduceNeighborhood(reducer=ee.Reducer.mean(), kernel=kern,
                                                                  skipMasked=False).reproject(proj30).rename("nb_built_post")
    elev = ee.Image("USGS/SRTMGL1_003").rename("elev")
    # setting before the greening: share of the ground covered by buildings in 2015, in the cell and within 500 m
    gh = ee.Image(GHSL_2015).select("built_surface").divide(1e4)
    ghsl_own = gh.reproject(proj30).rename("ghsl_2015")
    ghsl_nb = gh.reduceNeighborhood(reducer=ee.Reducer.mean(), kernel=kern).reproject(proj30).rename("ghsl_nb_2015")
    pre_bare = nd[2014].lt(NONVEG).And(nd[2015].lt(NONVEG)).And(ok).unmask(0).rename("pre_bare_frac")
    late = greened.And(nd[2018].lt(NONVEG)).And(nd[2019].lt(NONVEG)).unmask(0).rename("late_frac")   # greened after 2019

    bands30 = [greened.unmask(0).rename("greened_frac"), never.unmask(0).rename("never_frac"), pre_bare, late,
               ok.unmask(0).rename("dry_frac")]
    for y in YEARS:
        bands30 += [comps[y].select("NDVI").rename(f"ndvi_{y}"), comps[y].select("LST").rename(f"lst_{y}"),
                    comps[y].select("EMIS").rename(f"emis_{y}"), comps[y].select("NDBI").rename(f"ndbi_{y}")]
    stack30 = ee.Image.cat(bands30 + [built_pre, built_post, nb_pre, nb_post, ghsl_own, ghsl_nb, elev]).reproject(proj30)
    cell = proj30.scale(3, 3)          # 90 m cells on the Landsat pixel grid: each holds exactly 3 x 3 pixels
    mean90 = stack30.reduceResolution(ee.Reducer.mean(), maxPixels=1024).reproject(cell)
    dmin = dist.reproject(proj30).reduceResolution(ee.Reducer.min(), maxPixels=64).reproject(cell)
    g, nv = mean90.select("greened_frac"), mean90.select("never_frac")
    rnd = ee.Image.random(42).reproject(cell)
    cls = (ee.Image(0).where(nv.eq(1).And(dmin.gt(300)).And(rnd.lt(CONTROL_FRACTION)), 3)
           .where(nv.eq(1).And(dmin.lte(150)), 2)
           .where(g.gte(1 / 9 - 1e-6).And(mean90.select("pre_bare_frac").gte(1 - 1e-6)), 1)).rename("cls").reproject(cell)
    # a random 10% of every other land cell (existing green, built, mixed), so that the draw rnd < CONTROL_FRACTION
    # over all classes is a representative sample of the whole study area
    cls = cls.where(cls.eq(0).And(rnd.lt(CONTROL_FRACTION)).And(mean90.select("dry_frac").gte(1 - 1e-6)), 4).rename("cls")
    out = mean90.addBands(dmin).addBands(cls).addBands(rnd.rename("rnd")).updateMask(cls.gt(0)).clip(land)
    return out, land, cell, counts, greened


def tiles(nx, ny):
    x0, y0, x1, y1 = RECT
    dx, dy = (x1 - x0) / nx, (y1 - y0) / ny
    return [[x0 + i * dx, y0 + j * dy, x0 + (i + 1) * dx, y0 + (j + 1) * dy] for i in range(nx) for j in range(ny)]


def fetch(ee, img, land, cell, box, retries=3):
    geom = ee.Geometry.Rectangle(box).intersection(land, ee.ErrorMargin(1))
    fc = img.sample(region=geom, projection=cell, geometries=True, tileScale=16)
    for attempt in range(retries):
        try:
            res = fc.getInfo()
            break
        except Exception as e:
            if attempt == retries - 1:
                raise
            print(f"    retrying tile ({e})"); time.sleep(10)
    rows = []
    for f in res["features"]:
        p = f["properties"]; lon, lat = f["geometry"]["coordinates"]
        rows.append({"lon": lon, "lat": lat, **p})
    return rows


def main():
    global MONTHS, NONVEG, GREEN
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", required=True)
    ap.add_argument("--nonveg", type=float, default=0.15)
    ap.add_argument("--green", type=float, default=0.30)
    ap.add_argument("--skip-et", action="store_true")
    ap.add_argument("--out", default=str(HERE / "panel.csv"))
    ap.add_argument("--months", nargs="+", type=int, choices=range(1,13), default=list(MONTHS),
                    help="Use 6 7 8 9 for the common-month sensitivity; export to a new file")
    a = ap.parse_args()
    MONTHS = tuple(sorted(set(a.months)))
    NONVEG, GREEN = a.nonveg, a.green
    if Path(a.out).exists():
        sys.exit(f'Refusing to overwrite {a.out}')
    ee = _init(a.project)
    img, land, cell, counts, greened = build(ee)
    meta = {"years": YEARS, "months": MONTHS, "cell_m": CELL_M, "control_fraction": CONTROL_FRACTION,
            "sensor": "LANDSAT/LC08/C02/T1_L2, CLOUD_COVER < 20", "ndvi_nonveg": NONVEG, "ndvi_green": GREEN,
            "scenes_per_year_month": ee.Dictionary({f"{y}-{m:02d}": n for (y, m), n in counts.items()}).getInfo()}
    # reference evapotranspiration over the greened land (TerraClimate pet, mm/month x 0.1)
    tc = ee.ImageCollection("IDAHO_EPSCOR/TERRACLIMATE").select("pet")
    if a.skip_et:
        tc = None
    area = greened.selfMask().reduceToVectors(geometry=land, scale=300, maxPixels=1e10, geometryType="centroid").geometry()

    def annual(y):
        return tc.filterDate(f"{y}-01-01", f"{y + 1}-01-01").sum().multiply(0.1).reduceRegion(
            ee.Reducer.mean(), area, 4638, maxPixels=1e9).get("pet").getInfo()

    pet = {}
    for y in (() if tc is None else (2024, 2025)):
        n = tc.filterDate(f"{y}-01-01", f"{y + 1}-01-01").size().getInfo()
        pet[str(y)] = annual(y) if n == 12 else None
        if n != 12:
            print(f"  TerraClimate has {n} months for {y}; that year is left out of reference ET")
    if tc is not None and not any(v for v in pet.values()):
        pet = {str(y): annual(y) for y in (2022, 2023)}      # when the most recent years are not yet published
    meta["reference_et_mm_per_year"] = pet
    meta["reference_et_source"] = "IDAHO_EPSCOR/TERRACLIMATE pet (Penman-Monteith reference ET), mean over greened land"
    ga = greened.selfMask().multiply(ee.Image.pixelArea()).reduceRegion(ee.Reducer.sum(), land, crs=greened.projection(),
                                                                        maxPixels=1e10, tileScale=16).values().get(0)
    meta["greened_area_km2"] = ee.Number(ga).divide(1e6).getInfo()
    rows = []
    boxes = tiles(*TILES)
    for k, box in enumerate(boxes, 1):
        r = fetch(ee, img, land, cell, box)
        rows += r
        print(f"  tile {k}/{len(boxes)}: {len(r):,} cells", flush=True)
    df = pd.DataFrame(rows).drop_duplicates(["lon", "lat"])
    df.to_csv(a.out, index=False, float_format="%.6f")
    meta["n_cells"] = {str(int(k)): int(v) for k, v in df["cls"].value_counts().items()}
    Path(a.out).with_name(Path(a.out).stem + "_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote {a.out}: {len(df):,} cells (greened {meta['n_cells'].get('1', 0):,}, ring {meta['n_cells'].get('2', 0):,}, "
          f"control {meta['n_cells'].get('3', 0):,}, other {meta['n_cells'].get('4', 0):,}); reference ET {pet} mm/yr")


if __name__ == "__main__":
    main()
