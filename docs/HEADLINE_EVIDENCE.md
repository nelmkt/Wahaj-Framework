# Headline result evidence map

This map records the Jeddah quantities used in the 13 October 2026 research summary and award drafts. `REPRODUCE.md` lists the full set of manuscript quantities and commands. The primary matched population and the concurrent-change model test are different samples.

| Claim as printed | Saved evidence | Producing command or check | Population and limit |
| --- | --- | --- | --- |
| Spatial CV R² 0.795, RMSE 1.184 °C, n = 19,650 | `tables/model_cv.csv`, first row | `python code/src/run_framework.py --panel code/gee/panel.csv --out RERUN` | Non-greened Model B training cells, held-out spatial blocks; absolute-LST accuracy only. |
| Primary outside-area slope −1.181 °C per greened pixel, n = 1,006 | `tables/headline_block_v8.csv`, `all` row | `python code/src/headline_block_v8.py` | Matched treated cells outside built-up area, zero-intercept dose-weighted summary. |
| Dominant block holds 55.7% of pooled squared-dose leverage | `tables/headline_block_v8.csv`, `all` row, `dominant_pooled_block_leverage_share` | Same headline-block analysis; independently checked in `tables_revision_v11/leverage_share_check_v11.csv` | Shows sensitivity of pooled slope and uncertain interval coverage. |
| Small-patch own-only slope −1.925 °C per pixel, n = 419 | `tables/headline_block_v8.csv`, `1–2` row | Same analysis; algebraic decomposition in `tables_revision_v11/own_only_vs_joint_identity_v11.csv` | Nearby greening covaries with own greening; not a new isolated-pixel effect. |
| Model B support 14 of 744 outside cells, 0 of 75 fully greened cells | `tables/test_support_counts.csv`, outside rows, summed by dose | `python code/src/run_framework.py --panel code/gee/panel.csv --out RERUN` | Narrower concurrent-change sensitivity population. |
| Model B prediction minus measured contrast +1.01 to +7.24 °C | `tables/test_by_dose.csv`, outside `test=B` rows, `difference_C` | Same framework run | Four outside dose classes on the concurrent-change population. |
| Conditional central desalination electricity 2.9–5.3 MWh/year per °C | `tables_revision_v11/tradeoff_illustration_v11.csv`, `swro_MWh_per_degree_central` for partial dose classes | `python code/src/tradeoff_illustration_v11.py RERUN` in a clean copy | Derived from measured contrasts plus assumed irrigation depth and energy intensity; no measured electricity saving. |
| Vegetation-following emissivity slope −1.246 vs product-LST slope −1.181 °C per pixel | `tables_revision_r1/r1_reretrieved_contrasts.csv` | `python code/src/revision_r1_reretrieved_contrasts.py .` after saved Earth Engine export | Sensitivity to LST retrieval; not independent field thermometry. |
| NDVI threshold slopes −1.339 to −0.993 °C per pixel | `tables_revision_r1/r1_ndvi_thresholds.csv` | `python code/src/revision_r1_ndvi_thresholds.py .` after re-exports | Alternative vegetation definitions. |

The linked scripts are recorded in `REPRODUCE.md`; where a script is an analysis follow-up, run it in a copy because several refuse to overwrite saved tables. The repository's `MANIFEST_SHA256.txt` allows the saved inputs and outputs to be checked by bytes and SHA256. The public test result and rerun comparison are described in `README.md` and `tables_revision_r1/r1_rerun_comparison.csv`.
