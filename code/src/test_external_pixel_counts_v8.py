"""Synthetic native-pixel cases; these do not call Earth Engine."""
from pathlib import Path
import sys

import pytest

GEE = Path(__file__).resolve().parents[1] / "gee"
sys.path.insert(0, str(GEE))
from export_external_pixel_counts_v8 import _catalogue, center_index, parent_index, offsets_for_radii, unique_external_counts  # noqa: E402


def test_union_counts_a_pixel_in_range_of_multiple_own_pixels_once():
    own_cell = (0, 0)
    own = [(0, 0), (1, 0), (2, 0)]
    pixels = {point: own_cell for point in own}
    pixels.update({(3, 0): (1, 0),
                   (8, 0): (2, 0),
                   (12, 0): (3, 0),
                   (13, 0): (4, 0)})
    offsets = offsets_for_radii((30, 0, 0, 0, -30, 0))
    assert unique_external_counts(own, own_cell, pixels, offsets) == {90: 1, 180: 2, 300: 3}
    assert 3 + unique_external_counts(own, own_cell, pixels, offsets)[90] == 4


def test_zero_external_pixels_is_isolated_at_all_radii():
    own_cell = (4, 6)
    own = [(10, 20), (11, 20)]
    pixels = {point: own_cell for point in own}
    pixels[(30, 20)] = (8, 6)
    offsets = offsets_for_radii((30, 0, 0, 0, -30, 0))
    assert unique_external_counts(own, own_cell, pixels, offsets) == {90: 0, 180: 0, 300: 0}


def test_rotated_metric_transform_and_bad_parent_are_handled():
    own_cell = (0, 0)
    own = [(0, 0), (1, 0)]
    pixels = {(0, 0): own_cell, (1, 0): own_cell, (3, 0): (1, 0)}
    rotated = offsets_for_radii((0, -30, 0, 30, 0, 0))
    assert unique_external_counts(own, own_cell, pixels, rotated) == {90: 1, 180: 1, 300: 1}
    with pytest.raises(ValueError, match="missing or assigned"):
        unique_external_counts(own, (9, 9), pixels, rotated)


def test_catalogue_uses_native_and_parent_pixel_ids_without_geometry_rounding():
    rows = [
        {"native_x": 7.5, "native_y": 10.5, "cell_x": 2.5, "cell_y": 3.5},
        {"native_x": 8.5, "native_y": 10.5, "cell_x": 2.8333333333333, "cell_y": 3.5},
        {"native_x": 7.5, "native_y": 10.5, "cell_x": 2.5, "cell_y": 3.5},
    ]
    pixels, own = _catalogue(rows)
    assert pixels == {(7, 10): (2, 3), (8, 10): (2, 3)}
    assert own[(2, 3)] == [(7, 10), (8, 10)]
    assert center_index(8.5, "native_x") == 8
    assert center_index(-17010.5, "native_x") == -17011
    assert parent_index(2.833333333334243, "cell_x") == 2
    assert parent_index(-5670.833333333333, "cell_x") == -5671
    assert parent_index(2.1666666666667, "cell_x") == 2
    with pytest.raises(ValueError, match="not a half-integer"):
        center_index(-17010.05, "native_x")
    with pytest.raises(ValueError, match="not a 30 m pixel centre"):
        parent_index(2.05, "cell_x")
    with pytest.raises(ValueError, match="conflicting parent"):
        _catalogue(rows + [{"native_x": 7.5, "native_y": 10.5,
                            "cell_x": 3.1666666666667, "cell_y": 3.5}])
