"""Cross-check reported populations, structural ratio, and flat figure pairs."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "tables"


def read(name):
    with (TABLES / name).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def close(a, b, tol=1e-8):
    assert abs(float(a)-float(b)) <= tol, (a,b)


def main():
    dose = read("measured_dose.csv")
    retention = read("population_retention.csv")
    support = read("test_support_counts.csv")
    coast = read("coastal_block_counts.csv")
    specs = read("followup_specification_comparison.csv")
    for row in retention:
        matching = [d for d in dose if d["setting"] == row["setting"] and d["pixels"] == row["pixels"]]
        assert len(matching) == 1
        close(row["classified"], matching[0]["n_treated"])
        if int(row["matched"]) >= 10:
            close(row["matched"], matching[0]["n_treated_matched"])
        sr = [s for s in support if s["setting"] == row["setting"] and s["pixels"] == row["pixels"]]
        assert len(sr) == 1
        close(row["matched"], sr[0]["n_matched"])
    for setting in {r["setting"] for r in retention}:
        total = sum(int(r["matched"]) for r in retention if r["setting"] == setting)
        close(total, sum(int(r["n_matched"]) for r in coast if r["setting"] == setting))
        for spec, col in [("main", "matched"), ("calipers 0.5 SD plus 0.5 km coast", "caliper_0.5_sd"),
                          ("calipers 0.2 SD plus 0.5 km coast", "caliper_0.2_sd")]:
            r = [x for x in specs if x["setting"] == setting and x["specification"] == spec]
            assert len(r) == 1
            close(sum(int(x[col]) for x in retention if x["setting"] == setting), r[0]["n_retained"])
    outside = "outside the built-up area"
    a = next(r for r in read("decision_summary.csv") if r["setting"] == outside and r["basis"] == "measured: 9 of 9 pixels greened")
    b = next(r for r in read("test_by_dose.csv") if r["setting"] == outside and r["test"] == "B" and r["pixels"] == "9")
    sr = next(r for r in support if r["setting"] == outside and r["pixels"] == "9")
    close(a["cooling_C"], b["measured_C"])
    close(a["water_m3"], float(a["m3_per_degree"])*abs(float(a["cooling_C"])))
    structural = dict(setting=outside, pixels="9", water_m3_year=a["water_m3"],
        matched_cooling_C=a["cooling_C"], matched_conditional_m3_year_per_C=a["m3_per_degree"],
        model_B_cooling_C=b["model_C"], model_B_substitution_m3_year_per_C=float(a["water_m3"])/abs(float(b["model_C"])),
        substitution_over_matched_ratio=(float(a["water_m3"])/abs(float(b["model_C"]))) / float(a["m3_per_degree"]),
        model_B_supported_cells=sr["n_supported"], model_B_matched_cells=sr["n_matched"],
        note="Alternative denominator, not an independently validated causal estimate or irrigation envelope")
    saved_structural = read("structural_ratio_sensitivity.csv")
    assert len(saved_structural) == 1
    for field, expected in structural.items():
        if isinstance(expected, (int, float)):
            close(saved_structural[0][field], expected)
        else:
            assert saved_structural[0][field] == expected
    caliper = read("reviewer_caliper_by_dose.csv")
    caliper_half = [r for r in caliper if r["caliper_sd"] == "0.5"]
    close(sum(int(r["n_cells"]) for r in caliper_half), 95)
    assert next(r for r in caliper_half if r["pixels"] == "1–2")["n_cells"] == "60"
    profiles = read("reviewer_9pixel_profiles.csv")
    assert next(r for r in profiles if r["group"] == "9/9 dropped by main matching")["n_cells"] == "85"
    leverage = read("reviewer_dose_leverage.csv")
    close(sum(float(r["dose_squared_share"]) for r in leverage), 1)
    assert sum(float(r["dose_squared_share"]) for r in leverage if r["pixels"] in ("5–8", "9")) > .8
    coverage = read("reviewer_holdout_coverage.csv")
    close(sum(int(r["n_matched"]) for r in coverage if r["in_regional_holdout"] == "True"), 716)
    close(sum(int(r["n_matched"]) for r in coverage if r["in_regional_holdout"] == "False"), 28)
    assert sum(r["in_regional_holdout"] == "False" for r in coverage) == 7
    b_dose = [r for r in read("test_by_dose.csv") if r["setting"] == outside and r["test"] == "B"]
    assert [r["pixels"] for r in b_dose] == ["1–2", "3–4", "5–8", "9"]
    assert all(float(a["difference_C"]) < float(b["difference_C"]) for a, b in zip(b_dose, b_dose[1:]))
    holdouts = [r for r in read("test_region_holdouts.csv") if r["setting"] == outside]
    assert len(holdouts) == 8 and all(float(r["difference_C"]) > 0 for r in holdouts)
    built_support = next(r for r in read("test_support_coasts.csv") if r["setting"] == "built-up surroundings"
                         and r["coast_band"] == "≥ 10 km" and r["subset"] == "supported")
    close(built_support["n_cells"], 18)
    close(built_support["model_px_C"], -1.1523787180582683)
    close(built_support["measured_px_C"], -1.431383868098691)
    placebo_terciles = read("reviewer_placebo_lst_terciles.csv")
    close(sum(int(r["n_cells"]) for r in placebo_terciles), 289)
    assert {r["tercile"] for r in placebo_terciles} == {"cooler", "middle", "hotter"}
    assert next(r for r in placebo_terciles if r["tercile"] == "hotter")["n_blocks"] == "7"
    interactions = read("reviewer_caliper_lst_interaction.csv")
    assert {int(r["n_cells"]) for r in interactions} == {95, 60}
    central_water = next(r for r in read("water.csv") if r["level"] == "central")
    water_levels = {r["level"]: r for r in read("water.csv")}
    meta = json.loads((ROOT / "code/gee/panel_meta.json").read_text(encoding="utf-8"))
    et_by_year = meta["reference_et_mm_per_year"]
    assert set(et_by_year) == {"2024", "2025"} and et_by_year["2025"] is None
    et0 = float(et_by_year["2024"])
    for row in water_levels.values():
        close(row["depth_m"], et0 * float(row["kc"]) / float(row["efficiency"]) / 1000)
        close(row["m3_per_ha_yr"], float(row["depth_m"]) * 10000)
    with (TABLES / "results.json").open(encoding="utf-8") as f:
        intensities = json.load(f)["config"]["kwh_per_m3"]
    for energy in read("energy.csv"):
        for level, factor in zip(("low", "central", "high"), intensities[energy["source"]]):
            close(energy[f"MWh_per_ha_yr_{level}"],
                  float(water_levels[level]["m3_per_ha_yr"]) * float(factor) / 1000, tol=1e-6)
    for row in read("decision_summary.csv"):
        close(row["water_m3"], float(row["pixels_mean"]) * .09 * float(central_water["m3_per_ha_yr"]), tol=1e-6)
        close(row["m3_per_degree"], float(row["water_m3"]) / abs(float(row["cooling_C"])), tol=1e-6)
    primary = [r for r in read("logic_primary_class_estimates.csv")
               if r["setting"] == outside and r["specification"] == "pre-treatment-only strata"]
    pooled = next(r for r in primary if r["dose_class"] == "all")
    classes = [r for r in primary if r["dose_class"] != "all"]
    close(sum(int(r["n_matched"]) for r in classes), 1006)
    mix = [r for r in read("reviewer_baseline_only_mix.csv") if r["specification"] == "baseline-only strata"]
    shares = {r["pixels"]: float(r["dose_squared_share"]) for r in mix}
    close(sum(shares.values()), 1)
    close(sum(float(r["per_pixel_C"]) * shares[r["dose_class"]] for r in classes), pooled["per_pixel_C"])
    fig2_est = read("logic_primary_figure2_estimates.csv")
    fig2_dose = [r for r in read("logic_primary_figure2_dose.csv") if r["setting"] == outside]
    fig2_pool = next(r for r in fig2_est if r["setting"] == outside and r["estimate"] == "per_pixel")
    fig2_placebo = next(r for r in fig2_est if r["setting"] == outside and r["estimate"] == "placebo_per_pixel")
    close(fig2_pool["estimate_C"], pooled["per_pixel_C"])
    close(fig2_pool["n_treated_matched"], 1006)
    close(fig2_placebo["n_treated_matched"], 375)
    close(sum(int(r["n_treated_matched"]) for r in fig2_dose), 1006)
    control_rules = read("followup_control_selection_primary.csv")
    assert len(control_rules) == 4
    outside_controls = {r["control_rule"]: r for r in control_rules if r["setting"] == outside}
    close(outside_controls["stable-surface controls"]["estimate_C"], fig2_pool["estimate_C"])
    close(outside_controls["stable-surface controls"]["n_treated_matched"], 1006)
    close(outside_controls["all eligible controls"]["n_treated_matched"], 1006)
    assert abs(float(outside_controls["stable-surface controls"]["estimate_C"])
               - float(outside_controls["all eligible controls"]["estimate_C"])) < .002
    for r in fig2_dose:
        close(r["estimate_C"], next(x["cell_C"] for x in classes if x["dose_class"] == r["pixels"]))
    primary_mixing = read("logic_primary_area_mixing.csv")
    assert len(primary_mixing) == 3
    for r in primary_mixing:
        close(r["measured_C"], float(r["benchmark_C"]) + float(r["difference_C"]))
        close(r["measured_C"], next(x["estimate_C"] for x in fig2_dose if x["pixels"] == r["pixels"]))
    primary_coast = read("logic_primary_coastal_difference.csv")
    assert len(primary_coast) == 2
    assert next(r for r in primary_coast if r["setting"] == outside)["n_far"] == "967"
    block = [r for r in read("logic_block_summary.csv") if r["setting"] == outside and r["dose_class"] == "all"]
    assert len(block) == 2 and all(int(r["n_deleted_blocks"]) == 34 for r in block)
    assert all(float(r["deletion_max_C"]) < 0 for r in block)
    proxy = [r for r in read("logic_panel_proximity_proxy.csv") if r["setting"] == outside
             and r["nearest_other_exported_greened_cell_centre_gt_m"] == "300"]
    assert len(proxy) == 2 and all(int(r["n_matched"]) == 34 for r in proxy)
    sio = read("logic_sio_integrity.csv")
    assert len(sio) == 17 and sum(r["inside_assumed_depth_band"] == "True" for r in sio) == 7
    assert not any(r["duplicate_branch_year_in_aggregated_file"] == "True" for r in sio)
    assert (ROOT / "figures_pdf/fig3_model_test.pdf").stat().st_size < 500_000
    png = {p.stem for p in (ROOT / "figures_png").glob("*.png")}
    pdf = {p.stem for p in (ROOT / "figures_pdf").glob("*.pdf")}
    assert png == pdf == {"fig1_sample", "fig2_cooling", "fig3_model_test", "fig4_water_energy", "fig5_decision"}
    figure_code = (ROOT / "code/src/fw_figures.py").read_text(encoding="utf-8")
    assert "Hypothetical irrigation conversion (744-cell sensitivity)" in figure_code
    assert "Outside (744-cell sensitivity)" in figure_code
    assert {p.name for p in ROOT.iterdir()} == {"figures_png", "figures_pdf", "tables", "code", "manuscript"}
    assert {p.name for p in (ROOT / "code").iterdir() if p.is_dir()} == {"gee", "src"}
    assert not any(p.is_dir() for folder in ("figures_png", "figures_pdf", "tables", "manuscript") for p in (ROOT / folder).iterdir())
    assert not any(p.is_dir() for folder in ("gee", "src") for p in (ROOT / "code" / folder).iterdir())
    annual = read("gee_month_composite_changes.csv")
    baseline = next(r for r in annual if r["year"] == "2014")
    close(baseline["n_shared_cells"], 25125)
    close(baseline["n_equal_1e_minus_6"], baseline["n_shared_cells"])
    close(baseline["max_abs_change_C"], 0)
    primary_windows = read("gee_common_month_primary_v2.csv")
    original = next(r for r in primary_windows if r["panel"] == "saved May–September"
                    and r["setting"] == outside and r["dose_class"] == "all")
    common = next(r for r in primary_windows if r["panel"] == "new June–September"
                  and r["setting"] == outside and r["dose_class"] == "all")
    close(original["n_matched"], 1006)
    close(original["per_pixel_C"], fig2_pool["estimate_C"])
    close(common["n_matched"], 971)
    paired = next(r for r in read("gee_common_month_paired_v2.csv")
                  if r["setting"] == outside and r["dose_class"] == "all")
    close(paired["n_matched_same_cells"], 957)
    close(paired["new_minus_old_C"], float(paired["new_per_pixel_C"]) - float(paired["old_per_pixel_C"]))
    assert float(paired["paired_lo_C"]) <= float(paired["new_minus_old_C"]) <= float(paired["paired_hi_C"])
    sparse = [r for r in primary_windows if r["setting"] == "built-up surroundings" and r["dose_class"] == "9"]
    assert len(sparse) == 2 and all(r["n_matched"] == "1" and not r["per_pixel_C"] for r in sparse)
    monthly = read("gee_monthly_raw_diagnostics_v2.csv")
    assert len(monthly) == 1
    m = monthly[0]
    close(m["n_exported"], 1333)
    close(m["n_source_joined"], m["n_exported"])
    close(m["n_own_cell_dose_disagreements"], 0)
    close(m["n_monthly_count_violations"], 0)
    close(m["n_full_valid_cell_months"], m["n_possible_cell_months"])
    close(m["n_neighbor_order_violations"], 1149)
    m_user = read("gee_monthly_raw_diagnostics_v3_oldrecipe.csv")
    assert len(m_user) == 1
    close(m_user[0]["n_neighbor_order_violations"], 1149)
    versions = read("gee_raw_version_comparison.csv")
    assert len(versions) == 1
    close(versions[0]["n_rows"], 1333)
    close(versions[0]["n_changed_rows"], 0)
    assert (TABLES / "gee_monthly_persistence_from_raw_v3_oldrecipe.csv").is_file()
    assert not (TABLES / "gee_exact_isolation_effects_from_raw_v3_oldrecipe.csv").exists()
    assert (TABLES / "gee_exact_isolation_effects_from_raw_v2_INVALID.csv").is_file()
    assert not (TABLES / "gee_exact_isolation_effects.csv").exists()
    tagged_meta = json.loads((ROOT / "code" / "gee" / "monthly_isolation_raw_v4.json").read_text(encoding="utf-8"))
    assert tagged_meta["recipe_revision"] == "native-30m-neighborhood-v4"
    close(tagged_meta["n_exported_cells"], 1333)
    close(tagged_meta["dose_count_disagreements_vs_source_panel"], 0)
    close(tagged_meta["unjoined_cells_vs_source_panel"], 0)
    close(tagged_meta["neighbor_order_violations"], 0)
    with (ROOT / "code" / "gee" / "monthly_isolation_raw_v4.csv").open(newline="", encoding="utf-8") as f:
        tagged_raw = list(csv.DictReader(f))
    assert len(tagged_raw) == 1333
    assert all(float(r["green_count_8_neighbor_cells"]) <=
               float(r["green_count_other_within_300m_center"]) + .25 for r in tagged_raw)
    tagged_diag = read("gee_monthly_raw_diagnostics_v4.csv")
    assert len(tagged_diag) == 1
    for field in ("n_own_cell_dose_disagreements", "n_neighbor_order_violations",
                  "n_monthly_count_violations"):
        close(tagged_diag[0][field], 0)
    close(tagged_diag[0]["n_exported"], 1333)
    close(tagged_diag[0]["n_source_joined"], 1333)
    close(tagged_diag[0]["n_full_valid_cell_months"], 13330)
    valid_isolation = read("gee_exact_isolation_effects_from_raw_v4.csv")
    assert len(valid_isolation) == 16
    primary_small = [r for r in valid_isolation if
                     r["setting"] == outside and r["specification"] == "pre-treatment-only strata"]
    full_small = next(r for r in primary_small if r["screen"] == "all 1–2-pixel cells")
    isolated_small = next(r for r in primary_small if
                          r["screen"] == "no other greened 30 m pixel within 300 m of cell centre")
    close(full_small["n_matched"], 419)
    close(isolated_small["n_matched"], 27)
    close(isolated_small["n_matched_blocks"], 16)
    close(isolated_small["difference_from_full_C"],
          float(isolated_small["per_pixel_C"]) - float(full_small["per_pixel_C"]))
    assert (float(isolated_small["difference_lo_C"]) <=
            float(isolated_small["difference_from_full_C"]) <=
            float(isolated_small["difference_hi_C"]))
    one_pixel = read("gee_isolation_dose1_comparison_v5_corrected.csv")
    assert len(one_pixel) == 2
    full_one, screened_one = one_pixel
    close(full_one["n_matched"], 274)
    close(screened_one["n_matched"], 23)
    close(screened_one["n_matched_blocks"], 16)
    close(screened_one["difference_from_full_C"],
          float(screened_one["per_pixel_C"]) - float(full_one["per_pixel_C"]))
    assert (float(screened_one["difference_lo_C"]) <=
            float(screened_one["difference_from_full_C"]) <=
            float(screened_one["difference_hi_C"]))
    for stem in ("primary", "paired", "overlap"):
        assert (TABLES / f"gee_common_month_{stem}_v6.csv").read_bytes() == (
            TABLES / f"gee_common_month_{stem}_v2.csv").read_bytes()
    composition = read("gee_common_month_composition_v6.csv")
    count = lambda pred: sum(int(r["n_cells"]) for r in composition if pred(r))
    close(count(lambda r: r["group_old"] == "greened" and r["group_new"] != "greened"), 94)
    close(count(lambda r: r["group_old"] != "greened" and r["group_new"] == "greened"), 34)
    close(count(lambda r: r["group_old"] == r["group_new"] == "greened"
                and r["old_dose"] != r["new_dose"]), 67)
    month_rows = read("gee_monthly_ndvi_by_month_belt_v6.csv")
    assert len(month_rows) == 150
    small_outside_belt = next(r for r in month_rows if r["setting"] == outside
                              and r["dose_class"] == "1–2" and r["southern_belt"] == "no"
                              and r["year"] == "2025" and r["month"] == "8")
    close(small_outside_belt["n_cells"], 217)
    close(small_outside_belt["fraction_observed_green_pixels_ndvi_ge_0_30"],
          0.5226480836236934, tol=1e-9)
    shared = read("gee_isolation_within_strata_v6.csv")
    close(len(shared), 11)
    close(sum(int(r["n_isolated"]) for r in shared), 12)
    close(sum(int(r["n_nonisolated"]) for r in shared), 103)
    persistence = read("gee_monthly_persistence_from_raw_v2.csv")
    outside_small = next(r for r in persistence if r["setting"] == outside and r["dose_class"] == "1–2")
    close(outside_small["n_cells"], 426)
    close(outside_small["median_fraction_green_pixel_months_ndvi_ge_0_30"], 0.9)
    assert next(r for r in persistence if r["setting"] == "built-up surroundings" and r["dose_class"] == "9/9")["reportable"] == "False"
    print("Primary Figure 2, block/proximity/SIO, common-month rerun and composition, monthly NDVI, tagged and invalid older isolation, population, model support, water ratios, figure pairs and code split passed.")


if __name__ == "__main__":
    main()
