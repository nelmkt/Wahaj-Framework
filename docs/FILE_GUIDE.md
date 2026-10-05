# File guide

Every file in `code/` and `tables/`, grouped by purpose and listed in reading order. The folders are kept flat because the scripts find each other and the tables by fixed relative paths.

Descriptions of scripts are the first sentence of each script's own docstring. Descriptions of tables are the population and reading note recorded in `tables/table_population_v8_followup10.csv`.

## code/src

### 1. Framework core (run these first)

| File | What it does |
|---|---|
| `fw_allocation.py` | Optional binary portfolio selection from externally supplied, comparable interventions. |
| `fw_config.py` | Analysis settings. Revision sensitivities are exploratory, not retrospectively prespecified. |
| `fw_contrast_checks.py` | Exploratory paired spatial contrasts; these do not identify physical mechanisms. |
| `fw_decision.py` | Scenario accounting, conditional prediction diagnostics and joint ranking sensitivity. |
| `fw_figures.py` | Five figures. Every label sits in its own space: legends outside the data, estimate values in a separate right-hand column, no text drawn on top of data. |
| `fw_followup.py` | Exploratory paired contrasts, tighter matching, and panel DR-DiD sensitivities. No specification here is retrospectively prespecified. DR targets cell ATT, not a dose slope. |
| `fw_ledger.py` | Water and energy of keeping one hectare green, set beside the cooling measured on it. |
| `fw_matching.py` | Measured cooling: matched difference-in-differences with a spatial block bootstrap, by setting and by how much of a cell greened. |
| `fw_model.py` | Machine-learning model of summer land surface temperature across the city (XGBoost). |
| `fw_panel.py` | Load the cell panel (remote sensing) and derive everything the later steps use. |
| `fw_pipeline.py` | Runs the framework in order; reads like the Methods section. |
| `fw_report.py` | Generate a claim-limited report from the current run; no narrative assertions of automatic validation. |
| `fw_validate.py` | Joint spatial effect diagnostics. Supported-subset inference is conditional on original-fit screening. |
| `run_framework.py` | Urban greening in Jeddah: matched surface cooling and conditional resource accounting. |
| `synthetic_panel.py` | A synthetic panel with the columns of code/gee/panel.csv and a known answer, for tests and dry runs. |

### 2. Matched-contrast and exposure analyses

| File | What it does |
|---|---|
| `analyze_combined_exposure_v8.py` | Analyze a verified *new* native-pixel external-count export. |
| `analyze_exposure_rings_v8_followup3.py` | Versioned saved-data ring regression and collinearity audit. |
| `analyze_isolation_reweighting_v8.py` | Saved-panel follow-up for outside 1–2-pixel isolation and class-slope spread. |
| `analyze_joint_exposure_v8_followup2.py` | Saved-data primary-match joint exposure and isolation follow-up. |
| `analyze_monthly_isolation.py` | Analyze a new GEE monthly-NDVI and complete 30 m neighborhood export. |
| `analyze_neighbor_gradient.py` | Pre-specified greening-count categories around outside one-pixel cells. |
| `analyze_pixel_centered_isolation.py` | Check a new pixel-centred Earth Engine export and compare selected sites. |
| `analyze_pixel_isolation_v8.py` | Dose-specific primary-match sensitivity using the verified 300 m pixel screen. |
| `analyze_pixel_subrings_v8_followup6.py` | Analyze a future pixel-centred 30/60 m export; never calls Earth Engine. |
| `analyze_ring_reduced_v8_followup5.py` | Saved-panel reduced-ring sensitivity for primary outside 1–2-pixel cells. |
| `analyze_ring_reduced_v8_followup6.py` | Versioned paired middle-ring change for primary outside 1–2-pixel cells. |
| `analyze_ring_robustness_v8_followup4.py` | Saved-panel ring collinearity and leave-one-spatial-block-out follow-up. |
| `analyze_subring_contrasts_v8_followup7.py` | Paired inner-subring coefficient contrasts on the saved primary match. |
| `check_common_month_composition.py` | Describe which NDVI-defined greened cells change under June–September export. |
| `check_control_selection.py` | Check post-period stable-surface control selection in the primary match. |
| `check_isolation_dose_mix.py` | Check whether the 300 m screen's dose mix alone explains attenuation. |
| `check_month_composites.py` | Record annual LST differences on cells present in both saved GEE exports. |
| `check_monthly_raw.py` | Cross-check own-cell, monthly, and nested-neighborhood counts in a GEE export. |
| `compare_common_month.py` | Compare saved May–September and newly exported June–September LST panels. |
| `compare_isolation_within_strata.py` | Feasibility and descriptive within-stratum isolation comparison. |
| `compare_monthly_raw_versions.py` | Check whether the user-labelled raw-v3 file changes any exported values. |
| `diagnose_populations.py` | Audit treatment retention and spatial support for the reported specifications. |
| `headline_block_v8.py` | Primary outside class slopes, bootstrap medians and dominant-block deletion. |
| `logic_sensitivity.py` | Recompute design and proximity diagnostics from the bundled annual cell panel. |
| `make_imagery_review_sample.py` | Create a blinded, reproducible high-resolution imagery review queue. |
| `peer_review_arithmetic.py` | (no docstring) |
| `rebuild_sio_v8.py` | Read the unedited SIO workbooks and reconcile branch-year supply depths. |
| `reviewer_diagnostics.py` | Reproduce added reviewer diagnostics from the observed panel and saved run. |
| `reviewer_lst_diagnostics.py` | Exploratory baseline-LST checks on the saved observed panel. |
| `run_contrast_checks.py` | Recompute paired mixing/coastal diagnostics without refitting the LST models. |
| `run_followup.py` | Refresh reviewer diagnostics from saved full results, without repeating 400 model refits. |
| `summarize_monthly_ndvi.py` | Summarize saved May–September endpoint NDVI by month, dose and latitude belt. |

### 3. Figures, maps and report text

| File | What it does |
|---|---|
| `make_population_map_v11.py` | Independent population resolver for the complete follow-up-10 inventory. |
| `make_population_map_v8.py` | Inventory table populations from provenance families; never read the blinded key. |
| `regenerate_fig1_v8.py` | Render all four exported panel classes and primary matched donors. |
| `regenerate_fig2.py` | Re-estimate the interpretation-leading pre-treatment-only Figure 2 and save its rows. |
| `regenerate_fig3.py` | Regenerate paired Figure 3 PNG/PDF with the saved A/B effect diagnostics. |
| `regenerate_fig4.py` | Replot paired Figure 4 PNG/PDF from the saved resource ledger. |
| `regenerate_fig5.py` | Regenerate paired Figure 5 PNG/PDF from the saved observed-panel results. |
| `regenerate_report.py` | Regenerate the narrative report from saved observed-panel results and current claim rules. |
| `write_exposure_manuscript_v8_followup3.py` | Create a versioned manuscript and supplementary spread note from saved tables. |
| `write_final_wording_v8_followup9.py` | Versioned final wording pass; no scientific output or estimate changes. |
| `write_ring_followup6.py` | Create a versioned manuscript update from the saved paired middle-ring result. |
| `write_ring_reduced_v8_followup5.py` | Create a versioned manuscript and validation note for reduced ring models. |
| `write_ring_robustness_v8_followup4.py` | Create a versioned manuscript with saved-data ring robustness limits. |
| `write_subring_results_v8_followup7.py` | Create versioned manuscript and provenance notes for observed subring results. |
| `write_subring_results_v8_followup8.py` | Versioned manuscript update with paired inner-subring comparisons. |

### 4. Registration, packaging and claim audit

| File | What it does |
|---|---|
| `audit_claims.py` | Cross-check reported populations, structural ratio, and flat figure pairs. |
| `hash_package.py` | Hash v8 package files except this manifest and the unopened blinded key. |
| `package_v8.py` | Archive v8 without inspecting the blinded sampling-key contents. |
| `register_population_v10.py` | Make the current complete table inventory without opening the blinded key. |
| `register_v8_followup2.py` | Versioned table-population inventory for the saved-data follow-ups. |
| `register_v8_followup3.py` | Write a new, exhaustive table-population registry without changing v8 predecessors. |
| `register_v8_followup4.py` | Inventory new saved-data robustness tables without changing older registries. |
| `register_v8_followup5.py` | Inventory saved-data reduced-ring tables without modifying prior registries. |
| `register_v8_followup6.py` | Inventory new paired middle-ring table without modifying earlier registries. |
| `register_v8_followup8.py` | Versioned inventory after observed 30/60 m pixel-subring export. |

### 5. Tests

| File | What it does |
|---|---|
| `test_combined_exposure_v8.py` | Local analyzer checks with a constructed zero-external case; no GEE output. |
| `test_core.py` | Regression tests for inference and resource-allocation failure modes. |
| `test_exposure_rings_v8_followup3.py` | Synthetic and saved-output checks for the versioned ring follow-up. |
| `test_external_pixel_counts_v8.py` | Synthetic native-pixel cases; these do not call Earth Engine. |
| `test_followup.py` | Reviewer-requested contrasts and nuisance-model checks. |
| `test_followup_v6.py` | Cross-artifact checks for the common-month and monthly-NDVI follow-up. |
| `test_followup_v7.py` | Cross-artifact population and provenance checks for the new follow-up. |
| `test_framework.py` | Invariant checks for the framework (python -m pytest tests/test_framework.py). |
| `test_isolation_reweighting_v8.py` | Small synthetic checks for the saved-panel reweighting arithmetic. |
| `test_joint_exposure_v8_followup2.py` | Synthetic tests for new joint-exposure and expanded-balance arithmetic. |
| `test_pixel_subring_observed_v8_followup8.py` | Independent saved-output reconciliation for the observed 30/60 m export. |
| `test_ring_followup6.py` | Tests for paired middle-ring change and prepared subring design. |
| `test_ring_reduced_v8_followup5.py` | Synthetic and saved-output tests for the versioned reduced-ring follow-up. |
| `test_ring_robustness_v8_followup4.py` | Synthetic and saved-output checks for ring robustness calculations. |
| `test_v8.py` | Tests for v8 source-workbook, exposure and population claims. |

### 6. Input files kept beside the code

| File | What it does |
|---|---|
| `6.xlsx` | Saudi Irrigation Organization open-data workbook (source for the irrigation context table) |
| `8.xlsx` | Saudi Irrigation Organization open-data workbook (source for the irrigation context table) |
| `allocation_options_template.csv` | empty template for optional allocation inputs |
| `jeddah_coastline_ne10m.csv` | coastline points (Natural Earth 10 m) used for distance to coast |
| `requirements.txt` | version pins recorded with the package (see the root requirements.txt) |
| `sio_irrigation_2020_2022.csv` | irrigation context table, earlier build |
| `sio_irrigation_2020_2022_v8.csv` | irrigation context table rebuilt from the two workbooks |

### 7. Added for the paper (release 2.1.0)

| File | What it does |
|---|---|
| `benchmark_models_v11.py` | Scores XGBoost, random forest, gradient boosting and linear regression on the same cells, predictors and spatial folds. |
| `tradeoff_illustration_v11.py` | Evaluates NEGI on measured contrasts for illustration and collects the hypothetical irrigation energy per degree of measured cooling. |

### 8. Added for the supervisor revision (release 2.2.0)

| File | What it does |
|---|---|
| `revision_r1_analyses.py` | Emissivity diagnostic and bound, dose-by-dose contrasts, slope with intercept, model sensitivity fits, residual variogram, NEGI weights, population counts. |
| `revision_r1_gate3.py` | Gates 2 and 3 on the primary match. |
| `revision_r1_psf_recovery.py` | Synthetic landscape blurred by a 150 m point-spread function; recovery of a known effect. |
| `revision_r1_figures.py` | The two revision figures. |
| `revision_r1_scene_counts.py` | Scenes per summer month from `code/gee/panel_meta.json`. |
| `revision_r1_reretrieved_contrasts.py` | Primary match repeated with re-retrieved LST (after the Earth Engine export). |

## code/gee

### Earth Engine export scripts

| File | What it does |
|---|---|
| `attach_landcover.js` | Earth Engine Code Editor script (JavaScript) |
| `export_external_pixel_counts_v8.py` | Export unique external 30 m greening counts for each classified 90 m cell. |
| `export_external_pixel_subrings_v8_followup5.py` | Export nested unique external greening counts at 30, 60, 90, 180, 300 m. |
| `export_panel.js` | Earth Engine Code Editor script (JavaScript) |
| `export_panel.py` | Export the cell panel for the framework using Earth Engine's Python API. |
| `export_pixel_centered_isolation.py` | Export an exact 30 m pixel-centred 300 m external-greening screen. |
| `fetch_landcover.py` | Fetch the land-cover labels straight from Earth Engine — no asset upload, no Drive export. |
| `gee_monthly_isolation.py` | Export endpoint monthly NDVI and a complete 30 m greening-neighborhood screen. |
| `landsat_scenes.py` | Find the Landsat scenes behind your LST CSV, and export the same columns for other dates. |

| `gee_r1_lst_reretrieval.py` | LST re-retrieval with a vegetation-following emissivity from the Collection 2 radiance layers. Not yet run. |

### Saved exports (inputs to the analysis)

| File | Bytes |
|---|---|
| `Jeddah_landcover.csv` | 2058355 |
| `Jeddah_points_for_gee.csv` | 811728 |
| `monthly_isolation_raw.csv` | 210483 |
| `monthly_isolation_raw.json` | 937 |
| `monthly_isolation_raw_v2.csv` | 211098 |
| `monthly_isolation_raw_v2.json` | 1101 |
| `monthly_isolation_raw_v3_oldrecipe.csv` | 211098 |
| `monthly_isolation_raw_v3_oldrecipe.json` | 1101 |
| `monthly_isolation_raw_v4.csv` | 212074 |
| `monthly_isolation_raw_v4.json` | 1188 |
| `panel.csv`  (stored compressed as `panel.csv.gz`; decompress before running) | 8592932 |
| `panel_jun_sep.csv`  (stored compressed as `panel_jun_sep.csv.gz`; decompress before running) | 9338018 |
| `panel_jun_sep_meta.json` | 1064 |
| `panel_meta.json` | 1270 |
| `pixel_external_counts_v8_raw.csv` | 84491 |
| `pixel_external_counts_v8_raw.json` | 1210 |
| `pixel_external_subrings_v8_followup5_raw.csv` | 110626 |
| `pixel_external_subrings_v8_followup5_raw.json` | 1636 |
| `pixel_isolation_raw.csv` | 46659 |
| `pixel_isolation_raw.json` | 598 |

## code/negi_original

| File | What it does |
|---|---|
| `NEGI_Framework.py` | Index module: surrogate pipeline, pathways and the Normalized Environmental Gain Index. The paper uses its definitions and algebra; no value from it is reported. |

## tables

### 1. Machine-learning model and counterfactual gates

| File | Population | How to read it |
|---|---|---|
| `model_cv.csv` | held-out non-greened model sample | Absolute-LST skill; not effect validation |
| `model_importance.csv` | held-out non-greened model sample | Absolute-LST skill; not effect validation |
| `test_by_coast.csv` | concurrent-change matched sensitivity: 744 outside / 310 built-up | Model support not computed for primary population |
| `test_by_dose.csv` | concurrent-change matched sensitivity: 744 outside / 310 built-up | Model support not computed for primary population |
| `test_by_setting.csv` | concurrent-change matched sensitivity: 744 outside / 310 built-up | Model support not computed for primary population |
| `test_region_holdouts.csv` | concurrent-change matched sensitivity: 744 outside / 310 built-up | Model support not computed for primary population |
| `test_support_coasts.csv` | concurrent-change matched sensitivity: 744 outside / 310 built-up | Model support not computed for primary population |
| `test_support_counts.csv` | concurrent-change matched sensitivity: 744 outside / 310 built-up | Model support not computed for primary population |
| `test_supported.csv` | concurrent-change matched sensitivity: 744 outside / 310 built-up | Model support not computed for primary population |
| `test_supported_by_dose.csv` | concurrent-change matched sensitivity: 744 outside / 310 built-up | Model support not computed for primary population |

### 2. Matched contrasts (primary estimates and leverage)

| File | Population | How to read it |
|---|---|---|
| `cells.csv` | exported cell panel or primary proximity subset | Read class and matching fields |
| `coastal_block_counts.csv` | concurrent-change matched sensitivity, with all-classified counts where labelled | Read n_treated versus n_treated_matched columns |
| `figure1_population_comparison_v8.csv` | primary pre-treatment-only matched panel | Setting and dose are identified by row |
| `headline_block_v8.csv` | primary pre-treatment-only matched panel | Setting and dose are identified by row |
| `logic_block_deletions.csv` | mixed matching sensitivities | Read specification, setting and subset columns; not interchangeable |
| `logic_block_summary.csv` | mixed matching sensitivities | Read specification, setting and subset columns; not interchangeable |
| `logic_panel_proximity_proxy.csv` | exported cell panel or primary proximity subset | Read class and matching fields |
| `logic_primary_area_mixing.csv` | primary pre-treatment-only matched panel | Use row's setting/dose/subset |
| `logic_primary_class_estimates.csv` | primary pre-treatment-only matched panel | Use row's setting/dose/subset |
| `logic_primary_coastal_difference.csv` | primary pre-treatment-only matched panel | Use row's setting/dose/subset |
| `logic_primary_figure2_dose.csv` | primary pre-treatment-only matched panel | Use row's setting/dose/subset |
| `logic_primary_figure2_estimates.csv` | primary pre-treatment-only matched panel | Use row's setting/dose/subset |
| `logic_primary_figure2_yearly.csv` | primary pre-treatment-only matched panel | Use row's setting/dose/subset |
| `logic_sio_integrity.csv` | agricultural SIO branch-years, not greening cells | Original join checked in v8 audit |
| `measured_balance.csv` | concurrent-change matched sensitivity, with all-classified counts where labelled | Read n_treated versus n_treated_matched columns |
| `measured_by_coast.csv` | concurrent-change matched sensitivity, with all-classified counts where labelled | Read n_treated versus n_treated_matched columns |
| `measured_cluster_diagnostics.csv` | concurrent-change matched sensitivity, with all-classified counts where labelled | Read n_treated versus n_treated_matched columns |
| `measured_dose.csv` | concurrent-change matched sensitivity, with all-classified counts where labelled | Read n_treated versus n_treated_matched columns |
| `measured_estimates.csv` | concurrent-change matched sensitivity, with all-classified counts where labelled | Read n_treated versus n_treated_matched columns |
| `measured_geographic_dose.csv` | concurrent-change matched sensitivity, with all-classified counts where labelled | Read n_treated versus n_treated_matched columns |
| `measured_region_influence.csv` | concurrent-change matched sensitivity, with all-classified counts where labelled | Read n_treated versus n_treated_matched columns |
| `measured_sensitivity.csv` | concurrent-change matched sensitivity, with all-classified counts where labelled | Read n_treated versus n_treated_matched columns |
| `measured_yearly.csv` | concurrent-change matched sensitivity, with all-classified counts where labelled | Read n_treated versus n_treated_matched columns |
| `population_retention.csv` | concurrent-change matched sensitivity, with all-classified counts where labelled | Read n_treated versus n_treated_matched columns |

### 3. Neighbour exposure, rings and isolation

| File | Population | How to read it |
|---|---|---|
| `combined_class_relative_spread_v8_followup2.csv` | primary outside matched four own-dose classes | Range divided by absolute mean class slope |
| `combined_class_spread_v8_followup.csv` | primary outside matched four own-dose classes | Historical absolute class-slope spread, superseded by relative table |
| `combined_exposure_effects_v8.csv` | primary pre-treatment-only matched cells by setting and own-dose class | Own and own-plus-external denominators are descriptive |
| `combined_exposure_fig5a_v8.csv` | 744/310 concurrent-change sensitivity matched population | Hypothetical irrigation ratios; no resource ranking |
| `combined_exposure_isolation_v8.csv` | primary outside matched isolation subsets by own-dose class | High-dose isolated samples are sparse or empty |
| `isolation_balance_v8_followup2.csv` | primary outside matched 1–2-pixel class | Expanded calibration on three continuous and two binary margins |
| `isolation_covariates_v8_followup.csv` | primary outside matched 1–2-pixel class | Baseline covariates by radius-defined isolation |
| `isolation_reweighted_slopes_v8_followup.csv` | primary outside matched 1–2-pixel class | Earlier three-continuous-covariate calibration |
| `isolation_subgroups_v8_followup2.csv` | primary outside matched 1–2-pixel class | Unweighted within-group comparisons; under-20 rows diagnostic |
| `isolation_subgroups_v8_followup3.csv` | primary outside matched 1–2-pixel cells by radius, own dose, and belt | Unweighted selected-site comparison; rows under 20 cells diagnostic only |
| `joint_exposure_collinearity_v8_followup3.csv` | all 15 earlier primary-outside joint own/external regression rows | Uncentred design geometry and zero-external share; full rank does not identify causal split |
| `joint_exposure_regression_v8_followup2.csv` | primary outside matched cells, all and specified own-dose classes | Zero-intercept conditional association, not causal decomposition |
| `pixel_isolation_by_dose_v8.csv` | primary pre-treatment-only matched panel | Setting and dose are identified by row |
| `pixel_subring_collinearity_v8_followup6.csv` | same 419 cells and 32 blocks | Pairwise uncentred cosines, Pearson correlations and zero-count shares |
| `pixel_subring_pair_differences_v8_followup7.csv` | same 419 cells and 2,000 paired block draws | Three direct within-90 m coefficient differences and intervals |
| `pixel_subring_regression_v8_followup6.csv` | primary outside matched 1–2-pixel treated cells (419, 32 blocks) | Fine 0–30/30–60/60–90 m and coarse 0–30/30–90 m joint fits |
| `ring_exposure_regression_v8_followup3.csv` | primary outside matched treated cells, all/1–2/belt/non-belt groups | Four-column zero-intercept matched-contrast regression; descriptive, not causal ring response |
| `ring_leave_one_block_out_v8_followup4.csv` | primary outside matched 1–2-pixel class, 32 separate treated-block deletions | Refitted matched contrasts; descriptive one-block leverage sensitivity |
| `ring_leave_one_block_summary_v8_followup4.csv` | same 32 primary outside 1–2-pixel block deletions | Range and largest change of the 0–90 m coefficient; not an interval |
| `ring_pair_collinearity_v8_followup4.csv` | primary outside matched treated, all/1–2/belt/non-belt groups | Pairwise external-ring geometry and three-/four-column condition numbers |
| `ring_reduced_coefficients_v8_followup5.csv` | primary outside matched 1–2-pixel treated cells (419, 32 blocks) | Full, far-ring-omitted and merged-outer-ring zero-intercept fits; paired block draws |
| `ring_reduced_coefficients_v8_followup6.csv` | primary outside matched 1–2-pixel treated cells (419, 32 blocks) | Full, far-ring-omitted and merged-outer-ring fits, reproduced from follow-up 5 |
| `ring_reduced_comparisons_v8_followup5.csv` | same 419 cells and 2,000 spatial-block draws | Paired own/near-ring changes relative to the full four-term model |
| `ring_reduced_comparisons_v8_followup6.csv` | same 419 cells and 2,000 paired spatial-block draws | Includes the middle-ring coefficient change after omitting the far ring |

### 4. Calendar, monthly and export-version checks

| File | Population | How to read it |
|---|---|---|
| `gee_common_month_composition_v6.csv` | May–September versus June–September panel | Row identifies full, shared or paired subset |
| `gee_common_month_overlap.csv` | May–September versus June–September panel | Row identifies full, shared or paired subset |
| `gee_common_month_overlap_v2.csv` | May–September versus June–September panel | Row identifies full, shared or paired subset |
| `gee_common_month_overlap_v6.csv` | May–September versus June–September panel | Row identifies full, shared or paired subset |
| `gee_common_month_paired.csv` | May–September versus June–September panel | Row identifies full, shared or paired subset |
| `gee_common_month_paired_v2.csv` | May–September versus June–September panel | Row identifies full, shared or paired subset |
| `gee_common_month_paired_v6.csv` | May–September versus June–September panel | Row identifies full, shared or paired subset |
| `gee_common_month_primary.csv` | May–September versus June–September panel | Row identifies full, shared or paired subset |
| `gee_common_month_primary_v2.csv` | May–September versus June–September panel | Row identifies full, shared or paired subset |
| `gee_common_month_primary_v6.csv` | May–September versus June–September panel | Row identifies full, shared or paired subset |
| `gee_exact_isolation_effects_from_raw_v2_INVALID.csv` | primary and narrower matched panels, cell-centre screen | Use row specification; INVALID file is preserved, not evidence |
| `gee_exact_isolation_effects_from_raw_v4.csv` | primary and narrower matched panels, cell-centre screen | Use row specification; INVALID file is preserved, not evidence |
| `gee_isolation_dose1_comparison_v5_INVALID.csv` | primary matched panel, neighbor-screen sensitivity | Filename marks preserved invalid variants |
| `gee_isolation_dose1_comparison_v5_corrected.csv` | primary matched panel, neighbor-screen sensitivity | Filename marks preserved invalid variants |
| `gee_isolation_within_strata_v6.csv` | primary matched panel, pixel-screen sensitivity | Selected isolated sites are not the full class |
| `gee_month_composite_changes.csv` | May–September versus June–September panel | Row identifies full, shared or paired subset |
| `gee_monthly_ndvi_by_month_belt_v6.csv` | all NDVI-classified cells, monthly export | Filename marks superseded or old-recipe variants |
| `gee_monthly_persistence_from_raw_v2.csv` | all NDVI-classified cells, monthly export | Filename marks superseded or old-recipe variants |
| `gee_monthly_persistence_from_raw_v2_prior.csv` | all NDVI-classified cells, monthly export | Filename marks superseded or old-recipe variants |
| `gee_monthly_persistence_from_raw_v3_oldrecipe.csv` | all NDVI-classified cells, monthly export | Filename marks superseded or old-recipe variants |
| `gee_monthly_persistence_from_raw_v4.csv` | all NDVI-classified cells, monthly export | Filename marks superseded or old-recipe variants |
| `gee_monthly_raw_diagnostics_v2.csv` | all NDVI-classified cells, monthly export | Filename marks superseded or old-recipe variants |
| `gee_monthly_raw_diagnostics_v3_oldrecipe.csv` | all NDVI-classified cells, monthly export | Filename marks superseded or old-recipe variants |
| `gee_monthly_raw_diagnostics_v4.csv` | all NDVI-classified cells, monthly export | Filename marks superseded or old-recipe variants |
| `gee_neighbor_count_gradient_lst_tercile_v7_PRINTFAIL.csv` | primary matched panel, neighbor-screen sensitivity | Filename marks preserved invalid variants |
| `gee_neighbor_count_gradient_lst_tercile_v7_verified.csv` | primary matched panel, neighbor-screen sensitivity | Filename marks preserved invalid variants |
| `gee_neighbor_count_gradient_v7.csv` | primary matched panel, neighbor-screen sensitivity | Filename marks preserved invalid variants |
| `gee_pixel_centered_isolation_effects_v7.csv` | primary matched panel, pixel-screen sensitivity | Selected isolated sites are not the full class |
| `gee_raw_version_comparison.csv` | all NDVI-classified cells, monthly export | Filename marks superseded or old-recipe variants |

### 5. Sensitivity and reviewer diagnostics

| File | Population | How to read it |
|---|---|---|
| `followup_area_mixing.csv` | mixed matching sensitivities | Read specification, setting and subset columns; not interchangeable |
| `followup_bias_scenarios.csv` | mixed matching sensitivities | Read specification, setting and subset columns; not interchangeable |
| `followup_bias_tipping.csv` | mixed matching sensitivities | Read specification, setting and subset columns; not interchangeable |
| `followup_coastal_difference.csv` | mixed matching sensitivities | Read specification, setting and subset columns; not interchangeable |
| `followup_control_selection_primary.csv` | primary pre-treatment-only matched panel | Use row's setting/dose/subset |
| `followup_matching_specifications.csv` | mixed matching sensitivities | Read specification, setting and subset columns; not interchangeable |
| `followup_specification_balance.csv` | mixed matching sensitivities | Read specification, setting and subset columns; not interchangeable |
| `followup_specification_comparison.csv` | mixed matching sensitivities | Read specification, setting and subset columns; not interchangeable |
| `reviewer_9pixel_profiles.csv` | mixed matching sensitivities | Read specification, setting and subset columns; not interchangeable |
| `reviewer_baseline_only_mix.csv` | mixed matching sensitivities | Read specification, setting and subset columns; not interchangeable |
| `reviewer_caliper_by_dose.csv` | mixed matching sensitivities | Read specification, setting and subset columns; not interchangeable |
| `reviewer_caliper_lst_interaction.csv` | mixed matching sensitivities | Read specification, setting and subset columns; not interchangeable |
| `reviewer_dose_leverage.csv` | mixed matching sensitivities | Read specification, setting and subset columns; not interchangeable |
| `reviewer_holdout_coverage.csv` | mixed matching sensitivities | Read specification, setting and subset columns; not interchangeable |
| `reviewer_placebo_lst_terciles.csv` | mixed matching sensitivities | Read specification, setting and subset columns; not interchangeable |

### 6. Hypothetical water and energy ledger

| File | Population | How to read it |
|---|---|---|
| `allocation_selected.csv` | hypothetical scenario on narrower matched panel or optional allocation | No municipal ranking established |
| `decision_bare_places_model.csv` | hypothetical scenario on narrower matched panel or optional allocation | No municipal ranking established |
| `decision_by_coast.csv` | hypothetical scenario on narrower matched panel or optional allocation | No municipal ranking established |
| `decision_pairwise.csv` | hypothetical scenario on narrower matched panel or optional allocation | No municipal ranking established |
| `decision_ranking.csv` | hypothetical scenario on narrower matched panel or optional allocation | No municipal ranking established |
| `decision_scenarios.csv` | hypothetical scenario on narrower matched panel or optional allocation | No municipal ranking established |
| `decision_summary.csv` | hypothetical scenario on narrower matched panel or optional allocation | No municipal ranking established |
| `energy.csv` | hypothetical resource factors or narrower-sensitivity conversion | Not measured irrigation or net energy |
| `sio_supply_depths.csv` | agricultural SIO branch-years, not greening cells | Original join checked in v8 audit |
| `sio_workbook_join_audit_v8.csv` | agricultural SIO branch-years, not greening cells | Original join checked in v8 audit |
| `structural_ratio_sensitivity.csv` | hypothetical resource factors or narrower-sensitivity conversion | Not measured irrigation or net energy |
| `water.csv` | hypothetical resource factors or narrower-sensitivity conversion | Not measured irrigation or net energy |

### 7. Inventories, review queue and checksums

| File | Population | How to read it |
|---|---|---|
| `SHA256SUMS.txt` | - | checksum list written with the package |
| `SHA256SUMS_v8.txt` | - | checksum list written with the package |
| `imagery_review_queue_v7.csv` | unlabelled stratified review queue, 200 cells | No land-use labels available |
| `results.json` | - | combined results of the main pipeline run |
| `table_population_v8.csv` | all table filenames in v8 | Inventory metadata only; blinded key not opened |
| `table_population_v8_followup10.csv` | all table CSV filenames in the v10 package | Current inventory metadata only; blinded key not opened |
| `table_population_v8_followup2.csv` | all table CSV filenames through follow-up 2 | Versioned inventory only; blinded key not opened |
| `table_population_v8_followup3.csv` | all table CSV filenames through follow-up 3 | Versioned inventory only; blinded imagery key not opened |
| `table_population_v8_followup4.csv` | all table CSV filenames through follow-up 4 | Versioned inventory only; blinded imagery key not opened |
| `table_population_v8_followup5.csv` | all table CSV filenames through follow-up 5 | Versioned table inventory; unrun subring export has no result table |
| `table_population_v8_followup6.csv` | all table CSV filenames through follow-up 6 | Versioned inventory; 30/60 m Earth Engine export still unrun |
| `table_population_v8_followup8.csv` | all table CSV filenames through follow-up 8 | Versioned one-to-one inventory |

`tables/imagery_sampling_key_v7.csv` is listed in the inventory but withheld from the repository (blinded review key).

## tables_revision_v11

| File | What it holds |
|---|---|
| `leverage_share_check_v11.csv` | dose-squared leverage shares recomputed from cell data (see `docs/MATH_CHECK_V11.md`) |
| `own_only_vs_joint_identity_v11.csv` | own-only slope reproduced from joint-ring coefficients, five specifications |
| `model_benchmark_v11.csv` | spatially blocked skill of four regressors on Model B's training set |
| `tradeoff_illustration_v11.csv` | illustrative NEGI on measured contrasts and hypothetical irrigation energy per degree, by dose class |

## tables_revision_r1

| File | Content |
|---|---|
| `r1_dose_contrasts.csv` | see `docs/REVISION_R1.md` |
| `r1_emissivity_bound.csv` | see `docs/REVISION_R1.md` |
| `r1_emissivity_diagnostic.csv` | see `docs/REVISION_R1.md` |
| `r1_gate3_primary.csv` | see `docs/REVISION_R1.md` |
| `r1_model_sensitivity.csv` | see `docs/REVISION_R1.md` |
| `r1_negi_weights.csv` | see `docs/REVISION_R1.md` |
| `r1_populations.csv` | see `docs/REVISION_R1.md` |
| `r1_psf_recovery.csv` | see `docs/REVISION_R1.md` |
| `r1_scene_counts.csv` | see `docs/REVISION_R1.md` |
| `r1_slope_intercept.csv` | see `docs/REVISION_R1.md` |
| `r1_variogram.csv` | see `docs/REVISION_R1.md` |

## figures_r1

| File | Content |
|---|---|
| `fig_r1_dose_contrasts.png` | Mean matched contrast for each dose, primary match |
| `fig_r1_negi_weights.png` | NEGI on measured contrasts over α/β and p |
