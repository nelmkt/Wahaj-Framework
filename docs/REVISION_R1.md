# Revision r1: response to the supervisor review of 5 October 2026

| Review item | Analysis | Output |
|---|---|---|
| M1 emissivity artifact | product emissivity before and after at greened cells; first-order bound on the slope | `r1_emissivity_diagnostic.csv`, `r1_emissivity_bound.csv`, `r1_reretrieved_contrasts.csv`, `r1_gate3_primary_reretrieved.csv` |
| M4 dose slope | mean contrast for each dose 1–9; slope with and without an intercept | `r1_dose_contrasts.csv`, `r1_slope_intercept.csv`, `fig_r1_dose_contrasts.png` |
| M5 provenance | gates 2 and 3 on the primary match; population counts | `r1_gate3_primary.csv`, `r1_populations.csv` |
| M6 thermal footprint | point-spread simulation | `r1_psf_recovery.csv` |
| M7 NEGI | α/β = 0.5, 1, 2 and p = 0.5, 1 on measured contrasts | `r1_negi_weights.csv`, `fig_r1_negi_weights.png` |
| M8 model | sensitivity fits, metrics in kelvin, residual variogram | `r1_model_sensitivity.csv`, `r1_variogram.csv` |
| m5 scenes | scenes per summer month | `r1_scene_counts.csv` |

All tables come from the saved panel `code/gee/panel.csv` with the settings of `code/src/fw_config.py`. The revised manuscript is not part of this repository.

Release 2.3.0 adds the LST re-retrieval, the NDVI-threshold sensitivity (`r1_ndvi_thresholds.csv`), the full rerun comparison (`r1_rerun_comparison.csv`) and the redrawn Fig. 1.
