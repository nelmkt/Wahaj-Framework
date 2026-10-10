"""Cross-artifact population and provenance checks for the new follow-up."""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def read(name):
    with (ROOT / "tables" / name).open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def test_neighbor_bins_partition_exactly_one_pixel_classified_cells():
    for name in ("gee_neighbor_count_gradient_v7.csv",
                 "gee_neighbor_count_gradient_lst_tercile_v7_verified.csv"):
        rows = read(name)
        assert len(rows) == 8
        for measure in {r["neighbor_measure"] for r in rows}:
            part = [r for r in rows if r["neighbor_measure"] == measure]
            assert {r["neighbor_count_bin"] for r in part} == {"0", "1–2", "3–8", "≥9"}
            assert sum(int(r["n_classified"]) for r in part) == 280
            assert sum(int(r["n_matched"]) for r in part) <= 280


def test_imagery_queue_is_unlabelled_and_stratified_key_is_separate():
    queue = read("imagery_review_queue_v7.csv")
    assert (ROOT / "tables/imagery_sampling_key_v7.csv").is_file()
    assert len(queue) == 200
    assert len({r["sample_id"] for r in queue}) == 200
    assert all(r["review_status"] == "pending" and not r["land_use_label"] for r in queue)


def test_new_pixel_export_has_distinct_recipe_and_validation_gate():
    script = (ROOT / "code/gee/export_pixel_centered_isolation.py").read_text(encoding="utf-8")
    analyzer = (ROOT / "code/src/analyze_pixel_centered_isolation.py").read_text(encoding="utf-8")
    assert 'pixel-centered-native30-v7' in script and 'pixel-centered-native30-v7' in analyzer
    assert 'dose_count_disagreements' in script
    assert 'negative_external_flags' in analyzer
