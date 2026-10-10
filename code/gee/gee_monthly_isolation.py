"""Export endpoint monthly NDVI and a complete 30 m greening-neighborhood screen.

Run from the package root after Earth Engine authentication:
    python code/gee/gee_monthly_isolation.py --project YOUR-CLOUD-PROJECT-ID

This uses the same 30 m greening definition, cloud mask, and 90 m grid as
export_panel.py. It does not re-export LST or replace the current panel.
The output alone does not identify irrigation: persistent wadi, groundwater,
or discharge-supported vegetation may also remain green in dry months.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import export_panel as source

ROOT = Path(__file__).resolve().parents[2]
YEARS = (2024, 2025)
MONTHS = (5, 6, 7, 8, 9)
NDVI_THRESHOLD = 0.30
RECIPE_REVISION = "native-30m-neighborhood-v4"


def build_diagnostics(ee, panel_image, cell, greened):
    """Return a 90 m image sampled only at the panel's classified greened cells."""
    landsat = (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
               .filterBounds(ee.Geometry.Rectangle(source.RECT))
               .filter(ee.Filter.lt("CLOUD_COVER", 20)))
    proj30 = landsat.first().select("SR_B4").projection()
    green30 = greened.unmask(0).toFloat().rename("green30").reproject(proj30)

    def sum90(image, name):
        return (image.toFloat().reproject(proj30)
                .reduceResolution(reducer=ee.Reducer.mean(), maxPixels=64)
                .reproject(cell).multiply(9).rename(name))

    green_cell = sum90(green30, "green_count_cell")
    neighbor8 = green_cell.reduceNeighborhood(
        reducer=ee.Reducer.sum(),
        kernel=ee.Kernel.square(radius=1, units="pixels", normalize=False),
        skipMasked=False,
    ).subtract(green_cell).max(0).rename("green_count_8_neighbor_cells")
    around300_30 = green30.reduceNeighborhood(
        reducer=ee.Reducer.sum(),
        kernel=ee.Kernel.circle(radius=300, units="meters", normalize=False),
        skipMasked=False,
    ).reproject(proj30)
    around300 = around300_30.reproject(cell)
    other300 = around300.subtract(green_cell).max(0).rename("green_count_other_within_300m_center")

    def prep(image):
        qa, sat = image.select("QA_PIXEL"), image.select("QA_RADSAT")
        clear = (qa.bitwiseAnd(1 << 1).eq(0)
                 .And(qa.bitwiseAnd(1 << 2).eq(0))
                 .And(qa.bitwiseAnd(1 << 3).eq(0))
                 .And(qa.bitwiseAnd(1 << 4).eq(0))
                 .And(sat.eq(0)))
        reflectance = image.select("SR_B.*").multiply(0.0000275).add(-0.2)
        return (reflectance.normalizedDifference(["SR_B5", "SR_B4"])
                .rename("NDVI").updateMask(clear)
                .copyProperties(image, ["system:time_start"]))

    prepared = landsat.map(prep)
    bands = [green_cell, neighbor8, other300]
    scene_counts = {}
    for year in YEARS:
        for month in MONTHS:
            key = f"{year}{month:02d}"
            monthly_collection = (prepared.filter(ee.Filter.calendarRange(year, year, "year"))
                                  .filter(ee.Filter.calendarRange(month, month, "month")))
            n_scenes = int(monthly_collection.size().getInfo())
            scene_counts[key] = n_scenes
            if n_scenes:
                ndvi = monthly_collection.median().select("NDVI").reproject(proj30)
                valid30 = ndvi.mask().And(green30).unmask(0)
                high30 = ndvi.gte(NDVI_THRESHOLD).And(green30).unmask(0)
                mean90 = (ndvi.updateMask(green30)
                          .reduceResolution(ee.Reducer.mean(), maxPixels=64)
                          .reproject(cell).unmask(-9999)
                          .rename(f"ndvi_mean_{key}"))
                valid90 = sum90(valid30, f"ndvi_valid_green_count_{key}")
                high90 = sum90(high30, f"ndvi_ge30_green_count_{key}")
            else:
                mean90 = ee.Image.constant(-9999).reproject(cell).rename(f"ndvi_mean_{key}")
                valid90 = ee.Image.constant(0).reproject(cell).rename(f"ndvi_valid_green_count_{key}")
                high90 = ee.Image.constant(0).reproject(cell).rename(f"ndvi_ge30_green_count_{key}")
            bands.extend((mean90, valid90, high90))
    classified = panel_image.select("cls").eq(1)
    diagnostics = ee.Image.cat(bands).updateMask(classified).clip(ee.Geometry.Rectangle(source.RECT))
    return diagnostics, scene_counts


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", required=True, help="Earth Engine Google Cloud project ID")
    parser.add_argument("--out", type=Path, default=ROOT / "tables" / "gee_monthly_isolation_raw.csv")
    args = parser.parse_args()
    print(f"NEGI Earth Engine exporter recipe: {RECIPE_REVISION}; "
          "requires zero dose and neighborhood-order violations", flush=True)
    if args.out.exists():
        raise SystemExit(f"Refusing to overwrite {args.out}")
    ee = source._init(args.project)
    panel_image, land, cell, _, greened = source.build(ee)
    diagnostics, scene_counts = build_diagnostics(ee, panel_image, cell, greened)
    rows = []
    for index, box in enumerate(source.tiles(*source.TILES), 1):
        part = source.fetch(ee, diagnostics, land, cell, box)
        rows.extend(part)
        print(f"  diagnostic tile {index}/{source.TILES[0] * source.TILES[1]}: {len(part)} cells", flush=True)
    data = pd.DataFrame(rows).drop_duplicates(["lon", "lat"])
    if data.empty:
        raise SystemExit("No classified greened cells returned; no output written.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.out, index=False, float_format="%.6f")
    source_panel = pd.read_csv(ROOT / "code" / "gee" / "panel.csv")
    source_green = source_panel.loc[source_panel.cls.eq(1), ["lon", "lat", "greened_frac"]].copy()
    source_green["lon_key"] = source_green.lon.round(6)
    source_green["lat_key"] = source_green.lat.round(6)
    check = data.assign(lon_key=data.lon.round(6), lat_key=data.lat.round(6)).merge(
        source_green[["lon_key", "lat_key", "greened_frac"]],
        on=["lon_key", "lat_key"], how="left", validate="one_to_one")
    n_disagree = int((check.green_count_cell - (check.greened_frac * 9).round()).abs().gt(.25).sum())
    n_unjoined = int(check.greened_frac.isna().sum())
    n_neighbor_order_violations = int((
        check.green_count_8_neighbor_cells - check.green_count_other_within_300m_center
    ).gt(.25).sum())
    metadata = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_recipe": "code/gee/export_panel.py",
        "recipe_revision": RECIPE_REVISION,
        "source_panel_sha256": hashlib.sha256((ROOT / "code" / "gee" / "panel.csv").read_bytes()).hexdigest(),
        "landsat_collection": "LANDSAT/LC08/C02/T1_L2",
        "scene_cloud_cover_lt": 20,
        "monthly_composite": "median of QA-cleared NDVI",
        "months": list(MONTHS),
        "years": list(YEARS),
        "greened_pixel_ndvi_threshold": NDVI_THRESHOLD,
        "neighbor_screen": "all 30 m pixels meeting the endpoint transition, including those in mixed 90 m cells",
        "distance_definition": "300 m from 90 m cell centre to 30 m pixel centres, excluding own nine 30 m pixels",
        "n_exported_cells": int(len(data)),
        "dose_count_disagreements_vs_source_panel": n_disagree,
        "unjoined_cells_vs_source_panel": n_unjoined,
        "neighbor_order_violations": n_neighbor_order_violations,
        "count_method": "30 m floating mean times nine aligned source pixels",
        "scenes_per_month": scene_counts,
    }
    args.out.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if n_disagree or n_unjoined or n_neighbor_order_violations:
        raise SystemExit(f"Export saved for diagnosis but rejected: {n_disagree} dose-count disagreements and "
                         f"{n_unjoined} unjoined cells and {n_neighbor_order_violations} neighborhood-order "
                         f"violations; do not interpret isolation from {args.out}.")
    print(f"Wrote {args.out}: {len(data)} classified greened cells")


if __name__ == "__main__":
    main()
