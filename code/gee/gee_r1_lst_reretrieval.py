"""LST re-retrieval with an emissivity that follows current vegetation (supervisor review M1).

Collection 2 ST overrides the emissivity of pixels that were bare in the ASTER era (2000-2008) with bare-soil
emissivity, so ST_EMIS does not rise where vegetation was established later (USGS Collection 2 known issues). This
script inverts the single-channel radiative transfer equation with the product's own atmospheric layers:

    LT = (Lobs - Lup - tau * (1 - e) * Ldown) / (tau * e),     T = K2 / ln(K1 / LT + 1)

with e = fv * E_VEG + (1 - fv) * ST_EMIS, fv = clip((NDVI - NDVI_S) / (NDVI_V - NDVI_S), 0, 1)^2.
E_VEG, NDVI_S and NDVI_V are those of Ermida et al. (2020, Remote Sensing 12, 1471; open-source code\nhttps://github.com/sofiaermida/Landsat_SMW_LST). Greened and control cells were bare in the ASTER period, so ST_EMIS\nis their bare-soil emissivity; where NDVI < NDVI_S the re-retrieval reduces to the product LST (a built-in check).
Masking and compositing are those of code/gee/export_panel.py; the values are sampled at the panel's cell centres.

usage (needs the author's Earth Engine account):
    python gee_r1_lst_reretrieval.py --project YOUR-CLOUD-PROJECT --panel code/gee/panel.csv \
        --out code/gee/panel_lst_reretrieved_r1.csv
Then: python manuscript/revision_r1_reretrieved_contrasts.py <package_root>
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

YEARS = (2014, 2015, 2018, 2019, 2024, 2025)
MONTHS = (5, 6, 7, 8, 9)
RECT = [39.0, 21.2, 39.4, 21.8]
K1, K2 = 774.8853, 1321.0789          # Landsat 8 band 10 (also in each scene's metadata)
E_VEG, NDVI_S, NDVI_V = 0.99, 0.2, 0.86     # Ermida et al. (2020), Landsat_SMW_LST modules compute_FVC.js and compute_emissivity.js


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--panel", default="code/gee/panel.csv")
    ap.add_argument("--out", default="code/gee/panel_lst_reretrieved_r1.csv")
    a = ap.parse_args()
    if Path(a.out).exists():
        sys.exit(f"Refusing to overwrite {a.out}")
    import ee
    ee.Initialize(project=a.project)
    rect = ee.Geometry.Rectangle(RECT)
    col = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").filterBounds(rect).filter(ee.Filter.lt("CLOUD_COVER", 20))
    proj30 = col.first().select("SR_B4").projection()

    def prep(img):
        qa, sat = img.select("QA_PIXEL"), img.select("QA_RADSAT")
        m = (qa.bitwiseAnd(1 << 1).eq(0).And(qa.bitwiseAnd(1 << 2).eq(0)).And(qa.bitwiseAnd(1 << 3).eq(0))
             .And(qa.bitwiseAnd(1 << 4).eq(0)).And(sat.eq(0)))
        o = img.select("SR_B.*").multiply(0.0000275).add(-0.2)
        ndvi = o.normalizedDifference(["SR_B5", "SR_B4"])
        fv = ndvi.subtract(NDVI_S).divide(NDVI_V - NDVI_S).clamp(0, 1).pow(2)
        e_prod = img.select("ST_EMIS").multiply(0.0001)
        e = fv.multiply(E_VEG).add(ee.Image(1).subtract(fv).multiply(e_prod))
        lobs = img.select("ST_TRAD").multiply(0.001)
        lup = img.select("ST_URAD").multiply(0.001)
        ldn = img.select("ST_DRAD").multiply(0.001)
        tau = img.select("ST_ATRAN").multiply(0.0001)
        lt = lobs.subtract(lup).subtract(tau.multiply(ee.Image(1).subtract(e)).multiply(ldn)).divide(tau.multiply(e))
        t = ee.Image(K2).divide(ee.Image(K1).divide(lt).add(1).log()).subtract(273.15).rename("LST_RR")
        return t.updateMask(m).copyProperties(img, ["system:time_start"])

    cp = col.map(prep)
    bands = []
    for y in YEARS:
        monthly = [cp.filter(ee.Filter.calendarRange(y, y, "year")).filter(ee.Filter.calendarRange(mth, mth, "month"))
                   for mth in MONTHS]
        comp = ee.ImageCollection.fromImages([c.median().set("n", c.size()) for c in monthly]) \
            .filter(ee.Filter.gt("n", 0)).mean().reproject(proj30)
        bands.append(comp.rename(f"lst_rr_{y}"))
    cell = proj30.scale(3, 3)
    img = ee.Image.cat(bands).reduceResolution(ee.Reducer.mean(), maxPixels=1024).reproject(cell)

    pts = pd.read_csv(a.panel, usecols=["lon", "lat"])
    rows = []
    for start in range(0, len(pts), 4000):
        chunk = pts.iloc[start:start + 4000]
        fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([x, y]), {"lon": x, "lat": y})
                                   for x, y in zip(chunk.lon, chunk.lat)])
        res = img.sampleRegions(collection=fc, projection=cell, tileScale=16).getInfo()
        rows += [f["properties"] for f in res["features"]]
        print(f"  {min(start + 4000, len(pts)):,} / {len(pts):,}", flush=True)
    out = pd.DataFrame(rows)
    with open(a.out, "x", encoding="utf-8", newline="") as fh:
        out.to_csv(fh, index=False, float_format="%.6f")
    print("wrote", a.out, len(out))


if __name__ == "__main__":
    main()
