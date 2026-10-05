# REPRODUCE

How each number in the paper maps to a table, the script that writes that table, and the command. All paths are relative to the repository root.

Read this first:

- Commands are taken from the scripts' own docstrings and entry points. They were **not run** when this file was written.
- Decompress `code/gee/panel.csv.gz` and `code/gee/panel_jun_sep.csv.gz` first (see README.md).
- Most follow-up scripts refuse to overwrite an existing output. `run_framework.py` writes into `tables/` and the figure folders; run it in a copy to keep the shipped tables.
- Scripts under `code/gee/` that export from Google Earth Engine need an Earth Engine account and project. The saved exports they produced are in `code/gee/`.
- The number-by-number audit of the paper is `docs/Wahaj_Manuscript_numeric_audit.md`. The paper itself and its build script are not in the repository.
## A. Values from saved tables

Rows are file line numbers; the header is row 1.

| Printed | Table | Row | Column | Written by | Command |
|---|---|---|---|---|---|
| 0.795 | tables/model_cv.csv | row 2 | r2 | code/src/fw_pipeline.py:102 | python code/src/run_framework.py --panel code/gee/panel.csv --out . |
| 1.184 | tables/model_cv.csv | row 2 | rmse_C | code/src/fw_pipeline.py:102 | python code/src/run_framework.py --panel code/gee/panel.csv --out . |
| 19,650 | tables/model_cv.csv | row 2 | n | code/src/fw_pipeline.py:102 | python code/src/run_framework.py --panel code/gee/panel.csv --out . |
| 1,006 | tables/headline_block_v8.csv | row 2 | n_matched | code/src/headline_block_v8.py | python code/src/headline_block_v8.py |
| −1.181 | tables/headline_block_v8.csv | row 2 | point_C_per_pixel | code/src/headline_block_v8.py | python code/src/headline_block_v8.py |
| 55.7 | tables/headline_block_v8.csv | row 2 | dominant_pooled_block_leverage_share (shown x100 as a percentage) | code/src/headline_block_v8.py | python code/src/headline_block_v8.py |
| −1.925 | tables/headline_block_v8.csv | row 3 | point_C_per_pixel | code/src/headline_block_v8.py | python code/src/headline_block_v8.py |
| 14 | tables/test_support_counts.csv | rows 2, 3, 4, 5 (sum) | n_supported | code/src/run_followup.py | python code/src/run_followup.py |
| 744 | tables/test_support_counts.csv | rows 2, 3, 4, 5 (sum) | n_matched | code/src/run_followup.py | python code/src/run_followup.py |
| 1.01 | tables/test_by_dose.csv | row 9 | difference_C | code/src/fw_pipeline.py:105 | python code/src/run_framework.py --panel code/gee/panel.csv --out . |
| 7.24 | tables/test_by_dose.csv | row 12 | difference_C | code/src/fw_pipeline.py:105 | python code/src/run_framework.py --panel code/gee/panel.csv --out . |
| 313 | tables/logic_primary_class_estimates.csv | row 12 | n_matched | code/src/logic_sensitivity.py | python code/src/logic_sensitivity.py |
| 1.81 | tables/water.csv | row 3 | depth_m | code/src/fw_pipeline.py:109 | python code/src/run_framework.py --panel code/gee/panel.csv --out . |
| 419 | tables/headline_block_v8.csv | row 3 | n_matched | code/src/headline_block_v8.py | python code/src/headline_block_v8.py |
| 32 | tables/headline_block_v8.csv | row 3 | n_blocks | code/src/headline_block_v8.py | python code/src/headline_block_v8.py |
| −2.271 | tables/headline_block_v8.csv | row 3 | bootstrap_lo_C | code/src/headline_block_v8.py | python code/src/headline_block_v8.py |
| −1.477 | tables/headline_block_v8.csv | row 3 | bootstrap_hi_C | code/src/headline_block_v8.py | python code/src/headline_block_v8.py |
| 0.033 | tables/headline_block_v8.csv | row 3 | dose_squared_leverage_share | code/src/headline_block_v8.py | python code/src/headline_block_v8.py |
| −1.697 | tables/headline_block_v8.csv | row 3 | dominant_block_removed_C | code/src/headline_block_v8.py | python code/src/headline_block_v8.py |
| 183 | tables/headline_block_v8.csv | row 4 | n_matched | code/src/headline_block_v8.py | python code/src/headline_block_v8.py |
| 21 | tables/headline_block_v8.csv | row 4 | n_blocks | code/src/headline_block_v8.py | python code/src/headline_block_v8.py |
| −1.307 | tables/headline_block_v8.csv | row 4 | point_C_per_pixel | code/src/headline_block_v8.py | python code/src/headline_block_v8.py |
| −1.468 | tables/headline_block_v8.csv | row 4 | bootstrap_lo_C | code/src/headline_block_v8.py | python code/src/headline_block_v8.py |
| −0.998 | tables/headline_block_v8.csv | row 4 | bootstrap_hi_C | code/src/headline_block_v8.py | python code/src/headline_block_v8.py |
| 0.082 | tables/headline_block_v8.csv | row 4 | dose_squared_leverage_share | code/src/headline_block_v8.py | python code/src/headline_block_v8.py |
| −1.192 | tables/headline_block_v8.csv | row 4 | dominant_block_removed_C | code/src/headline_block_v8.py | python code/src/headline_block_v8.py |
| 244 | tables/headline_block_v8.csv | row 5 | n_matched | code/src/headline_block_v8.py | python code/src/headline_block_v8.py |
| 19 | tables/headline_block_v8.csv | row 5 | n_blocks | code/src/headline_block_v8.py | python code/src/headline_block_v8.py |
| −1.062 | tables/headline_block_v8.csv | row 5 | point_C_per_pixel | code/src/headline_block_v8.py | python code/src/headline_block_v8.py |
| −1.147 | tables/headline_block_v8.csv | row 5 | bootstrap_lo_C | code/src/headline_block_v8.py | python code/src/headline_block_v8.py |
| −0.779 | tables/headline_block_v8.csv | row 5 | bootstrap_hi_C | code/src/headline_block_v8.py | python code/src/headline_block_v8.py |
| 0.388 | tables/headline_block_v8.csv | row 5 | dose_squared_leverage_share | code/src/headline_block_v8.py | python code/src/headline_block_v8.py |
| −0.918 | tables/headline_block_v8.csv | row 5 | dominant_block_removed_C | code/src/headline_block_v8.py | python code/src/headline_block_v8.py |
| 160 | tables/headline_block_v8.csv | row 6 | n_matched | code/src/headline_block_v8.py | python code/src/headline_block_v8.py |
| 9 | tables/headline_block_v8.csv | row 6 | n_blocks | code/src/headline_block_v8.py | python code/src/headline_block_v8.py |
| −1.203 | tables/headline_block_v8.csv | row 6 | point_C_per_pixel | code/src/headline_block_v8.py | python code/src/headline_block_v8.py |
| −1.265 | tables/headline_block_v8.csv | row 6 | bootstrap_lo_C | code/src/headline_block_v8.py | python code/src/headline_block_v8.py |
| −0.881 | tables/headline_block_v8.csv | row 6 | bootstrap_hi_C | code/src/headline_block_v8.py | python code/src/headline_block_v8.py |
| 0.497 | tables/headline_block_v8.csv | row 6 | dose_squared_leverage_share | code/src/headline_block_v8.py | python code/src/headline_block_v8.py |
| −1.028 | tables/headline_block_v8.csv | row 6 | dominant_block_removed_C | code/src/headline_block_v8.py | python code/src/headline_block_v8.py |
| −1.170 | tables/headline_block_v8.csv | row 2 | bootstrap_median_C | code/src/headline_block_v8.py | python code/src/headline_block_v8.py |
| −1.252 | tables/headline_block_v8.csv | row 2 | bootstrap_lo_C | code/src/headline_block_v8.py | python code/src/headline_block_v8.py |
| −0.925 | tables/headline_block_v8.csv | row 2 | bootstrap_hi_C | code/src/headline_block_v8.py | python code/src/headline_block_v8.py |
| −1.465 | tables/headline_block_v8.csv | row 2 | jackknife_lo_C | code/src/headline_block_v8.py | python code/src/headline_block_v8.py |
| −0.896 | tables/headline_block_v8.csv | row 2 | jackknife_hi_C | code/src/headline_block_v8.py | python code/src/headline_block_v8.py |
| 641 | tables/headline_block_v8.csv | row 2 | n_matched_after_removal | code/src/headline_block_v8.py | python code/src/headline_block_v8.py |
| −1.042 | tables/headline_block_v8.csv | row 2 | dominant_block_removed_C | code/src/headline_block_v8.py | python code/src/headline_block_v8.py |
| −0.783 | tables/logic_primary_class_estimates.csv | row 12 | per_pixel_C | code/src/logic_sensitivity.py | python code/src/logic_sensitivity.py |
| −0.916 | tables/logic_primary_class_estimates.csv | row 12 | per_pixel_lo_C | code/src/logic_sensitivity.py | python code/src/logic_sensitivity.py |
| −0.659 | tables/logic_primary_class_estimates.csv | row 12 | per_pixel_hi_C | code/src/logic_sensitivity.py | python code/src/logic_sensitivity.py |
| 375 | tables/logic_primary_figure2_estimates.csv | row 4 | n_treated_matched | code/src/regenerate_fig2.py | python code/src/regenerate_fig2.py |
| −0.050 | tables/logic_primary_figure2_estimates.csv | row 4 | estimate_C | code/src/regenerate_fig2.py | python code/src/regenerate_fig2.py |
| −0.096 | tables/logic_primary_figure2_estimates.csv | row 4 | lo_C | code/src/regenerate_fig2.py | python code/src/regenerate_fig2.py |
| 0.016 | tables/logic_primary_figure2_estimates.csv | row 4 | hi_C | code/src/regenerate_fig2.py | python code/src/regenerate_fig2.py |
| −1.002 | tables/ring_reduced_coefficients_v8_followup6.csv | row 2 | beta_C_per_pixel | code/src/analyze_ring_reduced_v8_followup6.py | python code/src/analyze_ring_reduced_v8_followup6.py |
| −1.036 | tables/ring_reduced_coefficients_v8_followup6.csv | row 6 | beta_C_per_pixel | code/src/analyze_ring_reduced_v8_followup6.py | python code/src/analyze_ring_reduced_v8_followup6.py |
| −0.988 | tables/ring_reduced_coefficients_v8_followup6.csv | row 9 | beta_C_per_pixel | code/src/analyze_ring_reduced_v8_followup6.py | python code/src/analyze_ring_reduced_v8_followup6.py |
| −1.014 | tables/pixel_subring_regression_v8_followup6.csv | row 2 | beta_C_per_pixel | code/src/analyze_pixel_subrings_v8_followup6.py | python code/src/analyze_pixel_subrings_v8_followup6.py |
| −1.013 | tables/pixel_subring_regression_v8_followup6.csv | row 8 | beta_C_per_pixel | code/src/analyze_pixel_subrings_v8_followup6.py | python code/src/analyze_pixel_subrings_v8_followup6.py |
| −1.925 | tables_revision_v11/own_only_vs_joint_identity_v11.csv | row 2 | b_own_only | math check written for the paper (docs/MATH_CHECK_V11.md) | see MATH_CHECK_V11.md |
| −1.002 | tables_revision_v11/own_only_vs_joint_identity_v11.csv | row 2 | beta_own | math check written for the paper (docs/MATH_CHECK_V11.md) | see MATH_CHECK_V11.md |
| −0.923 | tables_revision_v11/own_only_vs_joint_identity_v11.csv | row 2 | sum_beta_ext_times_gamma | math check written for the paper (docs/MATH_CHECK_V11.md) | see MATH_CHECK_V11.md |
| −1.925 | tables_revision_v11/own_only_vs_joint_identity_v11.csv | row 2 | rhs_beta_own_plus_sum | math check written for the paper (docs/MATH_CHECK_V11.md) | see MATH_CHECK_V11.md |
| −1.036 | tables_revision_v11/own_only_vs_joint_identity_v11.csv | row 5 | beta_own | math check written for the paper (docs/MATH_CHECK_V11.md) | see MATH_CHECK_V11.md |
| −0.889 | tables_revision_v11/own_only_vs_joint_identity_v11.csv | row 5 | sum_beta_ext_times_gamma | math check written for the paper (docs/MATH_CHECK_V11.md) | see MATH_CHECK_V11.md |
| −1.925 | tables_revision_v11/own_only_vs_joint_identity_v11.csv | row 5 | rhs_beta_own_plus_sum | math check written for the paper (docs/MATH_CHECK_V11.md) | see MATH_CHECK_V11.md |
| −1.925 | tables_revision_v11/own_only_vs_joint_identity_v11.csv | row 5 | b_own_only | math check written for the paper (docs/MATH_CHECK_V11.md) | see MATH_CHECK_V11.md |
| −0.988 | tables_revision_v11/own_only_vs_joint_identity_v11.csv | row 7 | beta_own | math check written for the paper (docs/MATH_CHECK_V11.md) | see MATH_CHECK_V11.md |
| −0.937 | tables_revision_v11/own_only_vs_joint_identity_v11.csv | row 7 | sum_beta_ext_times_gamma | math check written for the paper (docs/MATH_CHECK_V11.md) | see MATH_CHECK_V11.md |
| −1.925 | tables_revision_v11/own_only_vs_joint_identity_v11.csv | row 7 | rhs_beta_own_plus_sum | math check written for the paper (docs/MATH_CHECK_V11.md) | see MATH_CHECK_V11.md |
| −1.925 | tables_revision_v11/own_only_vs_joint_identity_v11.csv | row 7 | b_own_only | math check written for the paper (docs/MATH_CHECK_V11.md) | see MATH_CHECK_V11.md |
| −1.014 | tables_revision_v11/own_only_vs_joint_identity_v11.csv | row 9 | beta_own | math check written for the paper (docs/MATH_CHECK_V11.md) | see MATH_CHECK_V11.md |
| −0.912 | tables_revision_v11/own_only_vs_joint_identity_v11.csv | row 9 | sum_beta_ext_times_gamma | math check written for the paper (docs/MATH_CHECK_V11.md) | see MATH_CHECK_V11.md |
| −1.925 | tables_revision_v11/own_only_vs_joint_identity_v11.csv | row 9 | rhs_beta_own_plus_sum | math check written for the paper (docs/MATH_CHECK_V11.md) | see MATH_CHECK_V11.md |
| −1.925 | tables_revision_v11/own_only_vs_joint_identity_v11.csv | row 9 | b_own_only | math check written for the paper (docs/MATH_CHECK_V11.md) | see MATH_CHECK_V11.md |
| −1.013 | tables_revision_v11/own_only_vs_joint_identity_v11.csv | row 14 | beta_own | math check written for the paper (docs/MATH_CHECK_V11.md) | see MATH_CHECK_V11.md |
| −0.912 | tables_revision_v11/own_only_vs_joint_identity_v11.csv | row 14 | sum_beta_ext_times_gamma | math check written for the paper (docs/MATH_CHECK_V11.md) | see MATH_CHECK_V11.md |
| −1.925 | tables_revision_v11/own_only_vs_joint_identity_v11.csv | row 14 | rhs_beta_own_plus_sum | math check written for the paper (docs/MATH_CHECK_V11.md) | see MATH_CHECK_V11.md |
| −1.925 | tables_revision_v11/own_only_vs_joint_identity_v11.csv | row 14 | b_own_only | math check written for the paper (docs/MATH_CHECK_V11.md) | see MATH_CHECK_V11.md |
| 0.832 | tables/ring_pair_collinearity_v8_followup4.csv | row 5 | uncentred_cosine | code/src/analyze_ring_robustness_v8_followup4.py | python code/src/analyze_ring_robustness_v8_followup4.py |
| 0.725 | tables/ring_pair_collinearity_v8_followup4.csv | row 6 | uncentred_cosine | code/src/analyze_ring_robustness_v8_followup4.py | python code/src/analyze_ring_robustness_v8_followup4.py |
| 0.897 | tables/ring_pair_collinearity_v8_followup4.csv | row 7 | uncentred_cosine | code/src/analyze_ring_robustness_v8_followup4.py | python code/src/analyze_ring_robustness_v8_followup4.py |
| 38.4 | tables/ring_reduced_coefficients_v8_followup6.csv | row 2 | design_condition | code/src/analyze_ring_reduced_v8_followup6.py | python code/src/analyze_ring_reduced_v8_followup6.py |
| 71.0 | tables/pixel_subring_regression_v8_followup6.csv | row 2 | design_condition | code/src/analyze_pixel_subrings_v8_followup6.py | python code/src/analyze_pixel_subrings_v8_followup6.py |
| 0.061 | tables/pixel_subring_pair_differences_v8_followup7.csv | row 2 | beta_a_minus_beta_b_C_per_pixel | code/src/analyze_subring_contrasts_v8_followup7.py | python code/src/analyze_subring_contrasts_v8_followup7.py |
| −0.377 | tables/pixel_subring_pair_differences_v8_followup7.csv | row 2 | lo_C | code/src/analyze_subring_contrasts_v8_followup7.py | python code/src/analyze_subring_contrasts_v8_followup7.py |
| 0.445 | tables/pixel_subring_pair_differences_v8_followup7.csv | row 2 | hi_C | code/src/analyze_subring_contrasts_v8_followup7.py | python code/src/analyze_subring_contrasts_v8_followup7.py |
| −1.136 | tables/isolation_balance_v8_followup2.csv | row 2 | isolated_slope_C_per_own_pixel | code/src/analyze_joint_exposure_v8_followup2.py | python code/src/analyze_joint_exposure_v8_followup2.py |
| −0.879 | tables/isolation_balance_v8_followup2.csv | row 3 | isolated_slope_C_per_own_pixel | code/src/analyze_joint_exposure_v8_followup2.py | python code/src/analyze_joint_exposure_v8_followup2.py |
| −0.891 | tables/isolation_balance_v8_followup2.csv | row 4 | isolated_slope_C_per_own_pixel | code/src/analyze_joint_exposure_v8_followup2.py | python code/src/analyze_joint_exposure_v8_followup2.py |
| −1.410 | tables/isolation_balance_v8_followup2.csv | row 4 | isolated_lo_C | code/src/analyze_joint_exposure_v8_followup2.py | python code/src/analyze_joint_exposure_v8_followup2.py |
| −0.425 | tables/isolation_balance_v8_followup2.csv | row 4 | isolated_hi_C | code/src/analyze_joint_exposure_v8_followup2.py | python code/src/analyze_joint_exposure_v8_followup2.py |
| 90 | tables/isolation_balance_v8_followup2.csv | row 2 | radius_m | code/src/analyze_joint_exposure_v8_followup2.py | python code/src/analyze_joint_exposure_v8_followup2.py |
| 180 | tables/isolation_balance_v8_followup2.csv | row 3 | radius_m | code/src/analyze_joint_exposure_v8_followup2.py | python code/src/analyze_joint_exposure_v8_followup2.py |
| 300 | tables/isolation_balance_v8_followup2.csv | row 4 | radius_m | code/src/analyze_joint_exposure_v8_followup2.py | python code/src/analyze_joint_exposure_v8_followup2.py |
| 76 | tables/isolation_balance_v8_followup2.csv | row 2 | isolated_cells | code/src/analyze_joint_exposure_v8_followup2.py | python code/src/analyze_joint_exposure_v8_followup2.py |
| 38 | tables/isolation_balance_v8_followup2.csv | row 3 | isolated_cells | code/src/analyze_joint_exposure_v8_followup2.py | python code/src/analyze_joint_exposure_v8_followup2.py |
| 26 | tables/isolation_balance_v8_followup2.csv | row 4 | isolated_cells | code/src/analyze_joint_exposure_v8_followup2.py | python code/src/analyze_joint_exposure_v8_followup2.py |
| 0.903 | tables/model_cv.csv | row 2 | mae_C | code/src/fw_pipeline.py:102 | python code/src/run_framework.py --panel code/gee/panel.csv --out . |
| 33.7 | tables/model_importance.csv | row 3 | gain_share (shown x100 as a percentage) | code/src/fw_pipeline.py:103 | python code/src/run_framework.py --panel code/gee/panel.csv --out . |
| 21.7 | tables/model_importance.csv | row 5 | gain_share (shown x100 as a percentage) | code/src/fw_pipeline.py:103 | python code/src/run_framework.py --panel code/gee/panel.csv --out . |
| 14.5 | tables/model_importance.csv | row 6 | gain_share (shown x100 as a percentage) | code/src/fw_pipeline.py:103 | python code/src/run_framework.py --panel code/gee/panel.csv --out . |
| 9.6 | tables/model_importance.csv | row 7 | gain_share (shown x100 as a percentage) | code/src/fw_pipeline.py:103 | python code/src/run_framework.py --panel code/gee/panel.csv --out . |
| 7.2 | tables/model_importance.csv | row 8 | gain_share (shown x100 as a percentage) | code/src/fw_pipeline.py:103 | python code/src/run_framework.py --panel code/gee/panel.csv --out . |
| 9.5 | tables/model_importance.csv | row 4 | gain_share (shown x100 as a percentage) | code/src/fw_pipeline.py:103 | python code/src/run_framework.py --panel code/gee/panel.csv --out . |
| 3.9 | tables/model_importance.csv | row 2 | gain_share (shown x100 as a percentage) | code/src/fw_pipeline.py:103 | python code/src/run_framework.py --panel code/gee/panel.csv --out . |
| 310 | tables/test_support_counts.csv | rows 6, 7, 8, 9 (sum) | n_matched | code/src/run_followup.py | python code/src/run_followup.py |
| 0 | tables/test_support_counts.csv | row 5 | n_supported | code/src/run_followup.py | python code/src/run_followup.py |
| 75 | tables/test_support_counts.csv | row 5 | n_matched | code/src/run_followup.py | python code/src/run_followup.py |
| 1.72 | tables/test_by_dose.csv | row 10 | difference_C | code/src/fw_pipeline.py:105 | python code/src/run_framework.py --panel code/gee/panel.csv --out . |
| 3.63 | tables/test_by_dose.csv | row 11 | difference_C | code/src/fw_pipeline.py:105 | python code/src/run_framework.py --panel code/gee/panel.csv --out . |
| −2.79 | tables/test_by_dose.csv | row 12 | model_C | code/src/fw_pipeline.py:105 | python code/src/run_framework.py --panel code/gee/panel.csv --out . |
| −10.03 | tables/test_by_dose.csv | row 12 | measured_C | code/src/fw_pipeline.py:105 | python code/src/run_framework.py --panel code/gee/panel.csv --out . |
| 58 | tables/test_supported_by_dose.csv | row 6 | n_cells | code/src/run_contrast_checks.py | python code/src/run_contrast_checks.py |
| 0.31 | tables/test_supported_by_dose.csv | row 6 | difference_C | code/src/run_contrast_checks.py | python code/src/run_contrast_checks.py |
| −0.21 | tables/test_supported_by_dose.csv | row 6 | difference_lo_C | code/src/run_contrast_checks.py | python code/src/run_contrast_checks.py |
| 0.77 | tables/test_supported_by_dose.csv | row 6 | difference_hi_C | code/src/run_contrast_checks.py | python code/src/run_contrast_checks.py |
| 971 | tables/gee_common_month_primary_v6.csv | row 12 | n_matched | code/src/compare_common_month.py [AUTHOR TO VERIFY: output name is not a literal in any script] | python code/src/compare_common_month.py |
| 957 | tables/gee_common_month_paired_v6.csv | row 2 | n_matched_same_cells | code/src/compare_common_month.py [AUTHOR TO VERIFY: output name is not a literal in any script] | python code/src/compare_common_month.py |
| −1.212 | tables/gee_common_month_primary_v6.csv | row 12 | per_pixel_C | code/src/compare_common_month.py [AUTHOR TO VERIFY: output name is not a literal in any script] | python code/src/compare_common_month.py |
| −1.181 | tables/gee_common_month_primary_v6.csv | row 2 | per_pixel_C | code/src/compare_common_month.py [AUTHOR TO VERIFY: output name is not a literal in any script] | python code/src/compare_common_month.py |
| 1,006 | tables/gee_common_month_primary_v6.csv | row 2 | n_matched | code/src/compare_common_month.py [AUTHOR TO VERIFY: output name is not a literal in any script] | python code/src/compare_common_month.py |
| −0.005 | tables/gee_common_month_paired_v6.csv | row 2 | new_minus_old_C | code/src/compare_common_month.py [AUTHOR TO VERIFY: output name is not a literal in any script] | python code/src/compare_common_month.py |
| −0.019 | tables/gee_common_month_paired_v6.csv | row 2 | paired_lo_C | code/src/compare_common_month.py [AUTHOR TO VERIFY: output name is not a literal in any script] | python code/src/compare_common_month.py |
| 0.005 | tables/gee_common_month_paired_v6.csv | row 2 | paired_hi_C | code/src/compare_common_month.py [AUTHOR TO VERIFY: output name is not a literal in any script] | python code/src/compare_common_month.py |
| 889 | tables/decision_summary.csv | row 2 | m3_per_degree | code/src/fw_pipeline.py:112 | python code/src/run_framework.py --panel code/gee/panel.csv --out . |
| 1,634 | tables/decision_summary.csv | row 4 | m3_per_degree | code/src/fw_pipeline.py:112 | python code/src/run_framework.py --panel code/gee/panel.csv --out . |
| 2.9 | tables/decision_summary.csv | row 2 | MWh_per_degree [desalinated seawater (SWRO)] | code/src/fw_pipeline.py:112 | python code/src/run_framework.py --panel code/gee/panel.csv --out . |
| 5.3 | tables/decision_summary.csv | row 4 | MWh_per_degree [desalinated seawater (SWRO)] | code/src/fw_pipeline.py:112 | python code/src/run_framework.py --panel code/gee/panel.csv --out . |
| 355 | tables/decision_summary.csv | row 2 | cells | code/src/fw_pipeline.py:112 | python code/src/run_framework.py --panel code/gee/panel.csv --out . |
| −2.46 | tables/decision_summary.csv | row 2 | cooling_C | code/src/fw_pipeline.py:112 | python code/src/run_framework.py --panel code/gee/panel.csv --out . |
| 2,186 | tables/decision_summary.csv | row 2 | water_m3 | code/src/fw_pipeline.py:112 | python code/src/run_framework.py --panel code/gee/panel.csv --out . |
| 147 | tables/decision_summary.csv | row 3 | cells | code/src/fw_pipeline.py:112 | python code/src/run_framework.py --panel code/gee/panel.csv --out . |
| −3.99 | tables/decision_summary.csv | row 3 | cooling_C | code/src/fw_pipeline.py:112 | python code/src/run_framework.py --panel code/gee/panel.csv --out . |
| 5,446 | tables/decision_summary.csv | row 3 | water_m3 | code/src/fw_pipeline.py:112 | python code/src/run_framework.py --panel code/gee/panel.csv --out . |
| 1,364 | tables/decision_summary.csv | row 3 | m3_per_degree | code/src/fw_pipeline.py:112 | python code/src/run_framework.py --panel code/gee/panel.csv --out . |
| 4.4 | tables/decision_summary.csv | row 3 | MWh_per_degree [desalinated seawater (SWRO)] | code/src/fw_pipeline.py:112 | python code/src/run_framework.py --panel code/gee/panel.csv --out . |
| 167 | tables/decision_summary.csv | row 4 | cells | code/src/fw_pipeline.py:112 | python code/src/run_framework.py --panel code/gee/panel.csv --out . |
| −6.27 | tables/decision_summary.csv | row 4 | cooling_C | code/src/fw_pipeline.py:112 | python code/src/run_framework.py --panel code/gee/panel.csv --out . |
| 10,251 | tables/decision_summary.csv | row 4 | water_m3 | code/src/fw_pipeline.py:112 | python code/src/run_framework.py --panel code/gee/panel.csv --out . |
| 75 | tables/decision_summary.csv | row 5 | cells | code/src/fw_pipeline.py:112 | python code/src/run_framework.py --panel code/gee/panel.csv --out . |
| −10.03 | tables/decision_summary.csv | row 5 | cooling_C | code/src/fw_pipeline.py:112 | python code/src/run_framework.py --panel code/gee/panel.csv --out . |
| 14,673 | tables/decision_summary.csv | row 5 | water_m3 | code/src/fw_pipeline.py:112 | python code/src/run_framework.py --panel code/gee/panel.csv --out . |
| 1,463 | tables/decision_summary.csv | row 5 | m3_per_degree | code/src/fw_pipeline.py:112 | python code/src/run_framework.py --panel code/gee/panel.csv --out . |
| 4.8 | tables/decision_summary.csv | row 5 | MWh_per_degree [desalinated seawater (SWRO)] | code/src/fw_pipeline.py:112 | python code/src/run_framework.py --panel code/gee/panel.csv --out . |

## B. Constants and settings quoted in the paper

| Printed | Source |
|---|---|
| 90 | code/gee/panel_meta.json cell_m |
| 1.0 | tables/ring_reduced_coefficients_v8_followup6.csv rows 2, 6, 9 and tables/pixel_subring_regression_v8_followup6.csv rows 2, 8 (own coefficients -1.036 to -0.988) |
| 2014 | code/src/fw_config.py:17 pre_years |
| 2025 | code/src/fw_config.py:19 post_years |
| 9 | code/src/fw_config.py:35 dose_bins |
| 30 | code/src/fw_config.py:34 (greened 30 m pixels of the cell's 9) |
| 2015 | code/src/fw_config.py:17 pre_years |
| 2018 | code/src/fw_config.py:18 mid_years |
| 2019 | code/src/fw_config.py:18 mid_years |
| 2024 | code/src/fw_config.py:19 post_years |
| 0.30 | code/gee/export_panel.py:82 |
| 1,333 | code/gee/panel_meta.json n_cells class 1 |
| 2,988 | code/gee/panel_meta.json n_cells class 2 |
| 150 | code/gee/export_panel.py:123 |
| 16,363 | code/gee/panel_meta.json n_cells class 3 |
| 300 | code/gee/export_panel.py:122 |
| 10 | code/gee/panel_meta.json control_fraction 0.1 |
| 4,892 | code/gee/panel_meta.json n_cells class 4 |
| 600 | code/src/fw_config.py:54 |
| 6 | code/src/fw_config.py:54 |
| 0.05 | code/src/fw_config.py:54 |
| 5 | code/src/fw_config.py:56 cv_folds |
| 200 | code/src/fw_config.py:57 n_refits |
| 500 | code/src/fw_config.py:21 |
| 0.10 | code/src/fw_config.py:26 |
| 3 | code/src/fw_config.py:31 |
| 0.5 | code/negi_original/NEGI_Framework.py:749 |
| 3.5 | code/negi_original/NEGI_Framework.py:748 |
| 95 | code/src/fw_config.py:59 |
| 2,000 | code/src/fw_config.py:40 n_boot |
| 2,264 | code/gee/panel_meta.json reference_et_mm_per_year 2024 = 2264.39 |
| 0.50 | code/src/fw_config.py:84 |
| 0.60 | code/src/fw_config.py:84 |
| 0.85 | code/src/fw_config.py:84 |
| 0.90 | code/src/fw_config.py:85 |
| 0.75 | code/src/fw_config.py:85 |
| 2.5 | code/src/fw_config.py:87 |
| 4.0 | code/src/fw_config.py:87 |
| 62 | author-observed local pytest result, 2026-10-02 (not run by the drafting agent) |
| 1–2 | code/src/fw_config.py:35 dose_bins |
| 3–4 | code/src/fw_config.py:35 dose_bins |
| 5–8 | code/src/fw_config.py:35 dose_bins |
| 9/9 | code/src/fw_config.py:35 dose_bins |
| Full: 0–90, 90–180, 180–300 m | ring bounds in metres: code/src/analyze_pixel_subrings_v8_followup6.py:27-30 and tables/ring_reduced_coefficients_v8_followup6.csv term names |
| Outer rings merged: 90–300 m | ring bounds in metres: code/src/analyze_pixel_subrings_v8_followup6.py:27-30 and tables/ring_reduced_coefficients_v8_followup6.csv term names |
| Fine inner: 0–30, 30–60, 60–90 m | ring bounds in metres: code/src/analyze_pixel_subrings_v8_followup6.py:27-30 and tables/ring_reduced_coefficients_v8_followup6.csv term names |
| Coarse inner: 0–30, 30–90 m | ring bounds in metres: code/src/analyze_pixel_subrings_v8_followup6.py:27-30 and tables/ring_reduced_coefficients_v8_followup6.csv term names |

## C. Tests

```
set PYTHONDONTWRITEBYTECODE=1
python -m pytest code/src -p no:cacheprovider
```

The author reports 62 tests passing on the full package (2026-10-02). In this release `tables/imagery_sampling_key_v7.csv` is withheld, so two tests are expected to fail here (see README.md).

## Tables added in release 2.1.0

| Table | Command (from the repository root, panels decompressed) |
|---|---|
| `tables_revision_v11/model_benchmark_v11.csv` | `python code/src/benchmark_models_v11.py .` |
| `tables_revision_v11/tradeoff_illustration_v11.csv` | `python code/src/tradeoff_illustration_v11.py .` |

Both scripts refuse to overwrite an existing table; work in a copy or remove the saved file first. The benchmark stops if its XGBoost row does not reproduce `tables/model_cv.csv`.

## Tables added in release 2.2.0

| Output | Command (from the repository root, panels decompressed) |
|---|---|
| most of `tables_revision_r1/` | `python code/src/revision_r1_analyses.py .` |
| `tables_revision_r1/r1_gate3_primary.csv` | `python code/src/revision_r1_gate3.py .` (about 20 minutes) |
| `tables_revision_r1/r1_psf_recovery.csv` | `python code/src/revision_r1_psf_recovery.py .` |
| `tables_revision_r1/r1_scene_counts.csv` | `python code/src/revision_r1_scene_counts.py .` |
| `figures_r1/` | `python code/src/revision_r1_figures.py .` |

`revision_r1_analyses.py` creates `tables_revision_r1/` and stops if it exists; work in a copy. It stops unless its full-model cross-validation reproduces `tables/model_cv.csv`. The LST re-retrieval needs an Earth Engine account: `python code/gee/gee_r1_lst_reretrieval.py --project YOUR-PROJECT`, then `python code/src/revision_r1_reretrieved_contrasts.py .`
