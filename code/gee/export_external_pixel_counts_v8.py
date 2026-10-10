"""Export unique external 30 m greening counts for each classified 90 m cell.

Run from the package root, in the owner's authenticated Earth Engine session:
    python code/gee/export_external_pixel_counts_v8.py --project satttt-500210

Earth Engine supplies the *same* greening mask and grids as export_panel.py.
Pixel IDs come directly from Earth Engine's pixelCoordinates bands on those
grids; distances between native pixel centres use the saved affine transform.
A neighbouring pixel enters a radius once if it is within that radius
of at least one own greened pixel; all nine own-cell pixels are excluded.
No LST, irrigation or causal effect is exported here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import export_panel as source

ROOT = Path(__file__).resolve().parents[2]
REVISION = "unique-external-native30-v8d"
RADII_M = (90, 180, 300)
DEFAULT_OUT = ROOT / "code/gee/pixel_external_counts_v8_raw.csv"
PANEL = ROOT / "code/gee/panel.csv"
OLD_SCREEN = ROOT / "code/gee/pixel_isolation_raw.csv"
PROTECTED = {ROOT / "code/gee" / name for name in (
    "panel.csv", "panel_meta.json", "panel_jun_sep.csv",
    "pixel_isolation_raw.csv", "monthly_isolation_raw.csv")}


def integer(value, label: str) -> int:
    number = float(value)
    if not math.isfinite(number) or abs(number - round(number)) > 0.01:
        raise ValueError(f"{label} is not an integer pixel coordinate/count: {value}")
    return int(round(number))


def center_index(value, label: str) -> int:
    """EE pixelCoordinates reports pixel centres at index + 0.5 on this grid."""
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} is not a finite pixel-centre coordinate: {value}")
    index = math.floor(number)
    if abs(number - (index + 0.5)) > 0.01:
        raise ValueError(f"{label} is not a half-integer pixel-centre coordinate: {value}")
    return index


def parent_index(value, label: str) -> int:
    """A 30 m centre lies at 1/6, 1/2 or 5/6 of its 90 m parent cell."""
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} is not a finite parent-cell coordinate: {value}")
    index = math.floor(number)
    fraction = number - index
    if min(abs(fraction - position) for position in (1/6, 1/2, 5/6)) > 0.01:
        raise ValueError(f"{label} is not a 30 m pixel centre within a 90 m parent: {value}")
    return index


def offsets_for_radii(transform, radii=RADII_M):
    """Enumerate native-grid offsets by metric centre distance, including boundary pixels."""
    a, b, _, d, e, _ = map(float, transform)
    trace = a*a + b*b + d*d + e*e
    determinant_sq = (a*e - b*d)**2
    small_eigen = (trace - math.sqrt(max(0.0, trace*trace - 4*determinant_sq))) / 2
    if small_eigen <= 0:
        raise ValueError("Pixel transform has no invertible metric basis")
    max_step = math.ceil(max(radii) / math.sqrt(small_eigen)) + 1
    result = []
    for dx in range(-max_step, max_step + 1):
        for dy in range(-max_step, max_step + 1):
            distance = math.hypot(a*dx + b*dy, d*dx + e*dy)
            included = tuple(radius for radius in radii if distance <= radius + 1e-7)
            if included:
                result.append((dx, dy, included))
    return result


def unique_external_counts(own_points, own_cell, pixel_index, offsets, radii=RADII_M):
    """Count a union of external greened pixel IDs, never a sum over own pixels."""
    found = {radius: set() for radius in radii}
    for x, y in own_points:
        if pixel_index.get((x, y)) != own_cell:
            raise ValueError("An own greened pixel is missing or assigned to another cell")
        for dx, dy, included in offsets:
            candidate = (x + dx, y + dy)
            parent = pixel_index.get(candidate)
            if parent is None or parent == own_cell:
                continue
            for radius in included:
                found[radius].add(candidate)
    return {radius: len(found[radius]) for radius in radii}


def _native_image(ee, greened, proj30, cell):
    native_ids = ee.Image.pixelCoordinates(proj30).rename(["native_x", "native_y"]).reproject(proj30)
    parent_ids = ee.Image.pixelCoordinates(cell).rename(["cell_x", "cell_y"]).reproject(proj30)
    return (greened.selfMask().toByte().rename("green").reproject(proj30)
            .addBands(native_ids).addBands(parent_ids))


def _classified_image(ee, panel_image, cell, greened, proj30):
    own = (greened.unmask(0).toFloat().reduceResolution(ee.Reducer.mean(), maxPixels=64)
           .reproject(cell).multiply(9).rename("own_count_ee"))
    ids = ee.Image.pixelCoordinates(cell).rename(["cell_x", "cell_y"]).reproject(cell)
    return own.addBands(ids).updateMask(panel_image.select("cls").eq(1))


def _fetch_projected(image, sampling_projection, region):
    return image.sample(region=region, projection=sampling_projection,
                        geometries=True, tileScale=16).getInfo()["features"]


def _classified_tile(ee, image, land, cell, proj30, box):
    region = ee.Geometry.Rectangle(box).intersection(land, ee.ErrorMargin(1))
    features = _fetch_projected(image, cell, region)
    return [{"lon": f["geometry"]["coordinates"][0],
             "lat": f["geometry"]["coordinates"][1], **f["properties"]} for f in features]


def _pixel_tile(ee, image, proj30, box, depth=0):
    region = ee.Geometry.Rectangle(box).buffer(360, ee.ErrorMargin(1))
    try:
        features = _fetch_projected(image, proj30, region)
        if len(features) <= 4500 or depth >= 4:
            return [f["properties"] for f in features]
    except Exception:
        if depth >= 4:
            raise
    x0, y0, x1, y1 = box
    xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
    return sum((_pixel_tile(ee, image, proj30, child, depth + 1) for child in (
        [x0, y0, xm, ym], [xm, y0, x1, ym],
        [x0, ym, xm, y1], [xm, ym, x1, y1])), [])


def _catalogue(rows):
    pixels = {}
    own_by_cell = defaultdict(list)
    for row in rows:
        xy = (center_index(row["native_x"], "native_x"), center_index(row["native_y"], "native_y"))
        parent = (parent_index(row["cell_x"], "cell_x"), parent_index(row["cell_y"], "cell_y"))
        if xy in pixels:
            if pixels[xy] != parent:
                raise ValueError("Duplicate native pixel has conflicting parent-cell IDs")
            continue
        pixels[xy] = parent
        own_by_cell[parent].append(xy)
    return pixels, own_by_cell


def _ids(frame):
    return set(zip(frame.lon.round(6), frame.lat.round(6)))


def _validated_output(classified, pixels, own_by_cell, transform, panel, old):
    if len(classified) != 1333 or len(panel) != 1333 or len(old) != 1333:
        raise ValueError("Expected exactly 1,333 classified greened cells in all three sources")
    if len(_ids(classified)) != len(classified) or len(_ids(panel)) != len(panel) or len(_ids(old)) != len(old):
        raise ValueError("Duplicate classified-cell coordinate ID")
    if _ids(classified) != _ids(panel) or _ids(classified) != _ids(old):
        raise ValueError("Classified-cell IDs differ from the protected panel or old pixel screen")
    panel_by_id = {(round(r.lon, 6), round(r.lat, 6)): r for r in panel.itertuples()}
    old_by_id = {(round(r.lon, 6), round(r.lat, 6)): r for r in old.itertuples()}
    offsets = offsets_for_radii(transform)
    records = []
    checks = dict(cell_id_disagreements=0, own_pixel_count_disagreements=0,
                  nonmonotone_radius_counts=0, old_isolated_300m_not_zero=0,
                  old_300m_isolation_flag_disagreements=0)
    seen_grid_ids = set()
    for row in classified.itertuples():
        cell = (center_index(row.cell_x, "cell_x"), center_index(row.cell_y, "cell_y"))
        if cell in seen_grid_ids:
            checks["cell_id_disagreements"] += 1
        seen_grid_ids.add(cell)
        key = (round(row.lon, 6), round(row.lat, 6))
        expected = integer(round(float(panel_by_id[key].greened_frac) * 9), "panel dose")
        own = own_by_cell.get(cell, [])
        if len(own) != expected or abs(float(row.own_count_ee) - expected) > 0.25:
            checks["own_pixel_count_disagreements"] += 1
        if not own:
            raise ValueError(f"No greened 30 m pixels found in classified cell {cell}")
        ext = unique_external_counts(own, cell, pixels, offsets)
        if not (ext[90] <= ext[180] <= ext[300]):
            checks["nonmonotone_radius_counts"] += 1
        old_isolated = float(getattr(old_by_id[key], "max_external_green_within300_any_own30")) < .5
        if old_isolated and any(ext[radius] for radius in RADII_M):
            checks["old_isolated_300m_not_zero"] += 1
        if old_isolated != (ext[300] == 0):
            checks["old_300m_isolation_flag_disagreements"] += 1
        record = {"lon": row.lon, "lat": row.lat, "cell_x": cell[0], "cell_y": cell[1],
                  "cell_id": f"{cell[0]}_{cell[1]}", "own_green_px": len(own)}
        for radius in RADII_M:
            record[f"external_green_px_{radius}m"] = ext[radius]
            record[f"combined_green_px_{radius}m"] = len(own) + ext[radius]
            record[f"isolated_{radius}m"] = int(ext[radius] == 0)
        records.append(record)
    if len(seen_grid_ids) != 1333:
        checks["cell_id_disagreements"] += 1333 - len(seen_grid_ids)
    return pd.DataFrame(records), checks


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", required=True, help="Authenticated Earth Engine Cloud project ID")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    output = args.out.resolve()
    if output in {path.resolve() for path in PROTECTED}:
        raise SystemExit("Refusing to replace a protected earlier Earth Engine output")
    if output.exists() or output.with_suffix(".json").exists():
        raise SystemExit("Refusing to overwrite the requested output or its JSON metadata")
    print(f"NEGI pixel-union recipe: {REVISION}", flush=True)
    ee = source._init(args.project)
    panel_image, land, cell, _, greened = source.build(ee)
    landsat = (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
               .filterBounds(ee.Geometry.Rectangle(source.RECT))
               .filter(ee.Filter.lt("CLOUD_COVER", 20)))
    proj30 = landsat.first().select("SR_B4").projection()
    projection = proj30.getInfo()
    cell_projection = cell.getInfo()
    if projection.get("crs") != "EPSG:32637" or len(projection.get("transform", [])) != 6:
        raise SystemExit("Expected the Jeddah Landsat 30 m UTM 37N grid; inspect projection before use")
    if cell_projection.get("crs") != projection["crs"] or len(cell_projection.get("transform", [])) != 6:
        raise SystemExit("Native and 90 m cell grids do not share a UTM CRS/affine transform")
    transform = projection["transform"]
    cell_transform = cell_projection["transform"]
    classified_image = _classified_image(ee, panel_image, cell, greened, proj30)
    native_image = _native_image(ee, greened, proj30, cell)
    cells, pixel_rows = [], []
    boxes = source.tiles(*source.TILES)
    for i, box in enumerate(boxes, 1):
        part = _classified_tile(ee, classified_image, land, cell, proj30, box)
        cells.extend(part)
        natives = _pixel_tile(ee, native_image, proj30, box)
        if part:
            center_index(part[0]["cell_x"], "classified cell_x")
            center_index(part[0]["cell_y"], "classified cell_y")
        if natives:
            for field in ("native_x", "native_y"):
                center_index(natives[0][field], field)
            for field in ("cell_x", "cell_y"):
                parent_index(natives[0][field], field)
        pixel_rows.extend(natives)
        print(f"  tile {i}/{len(boxes)}: {len(part)} classified cells; {len(natives)} buffered green-pixel records", flush=True)
    if not cells or not pixel_rows:
        raise SystemExit("Earth Engine returned no classified cells or no native green pixels; no output written")
    classified = pd.DataFrame(cells).drop_duplicates(["lon", "lat"])
    pixels, own_by_cell = _catalogue(pixel_rows)
    panel = pd.read_csv(PANEL)
    panel = panel.loc[panel.cls.eq(1), ["lon", "lat", "greened_frac"]].copy()
    old = pd.read_csv(OLD_SCREEN)
    data, checks = _validated_output(classified, pixels, own_by_cell, transform, panel, old)
    print("Built-in validation counts:", checks, flush=True)
    if any(checks.values()):
        raise SystemExit("Validation failed; no v8 output written")
    metadata = {"generated_utc": datetime.now(timezone.utc).isoformat(),
                "recipe_revision": REVISION, "project": args.project,
                "source_panel_sha256": hashlib.sha256(PANEL.read_bytes()).hexdigest(),
                "source_pixel_isolation_sha256": hashlib.sha256(OLD_SCREEN.read_bytes()).hexdigest(),
                "n_exported_cells": len(data), "n_catalogue_greened_pixels": len(pixels),
                "projection": projection, "cell_projection": cell_projection,
                "radii_m": list(RADII_M), "checks": checks,
                "distance_definition": "Euclidean projected native-30 m pixel-centre distance; union of unique external greened pixel IDs within each radius of any own greened pixel"}
    output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output, index=False, float_format="%.6f")
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote {output}: {len(data)} classified cells; all validation counts zero")


if __name__ == "__main__":
    main()
