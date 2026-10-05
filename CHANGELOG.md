# Changelog

## 2.3.1 (5 October 2026), release `v2.3.1`

- Archived on Zenodo: release v2.3.1, DOI [10.5281/zenodo.23168530](https://doi.org/10.5281/zenodo.23168530); all versions [10.5281/zenodo.23168529](https://doi.org/10.5281/zenodo.23168529).
- `.zenodo.json` archive metadata; README DOI badge; `CITATION.cff` DOI. No code, table or figure changed.

## 2.3.0 (5 October 2026), branch `v11-framework-ml`, release `v2.3.0`

The remaining supervisor items. No existing code, table or figure changed.

- LST re-retrieval run in Earth Engine (`code/gee/gee_r1_lst_reretrieval.py`, emissivity after Ermida et al. 2020); export `code/gee/panel_lst_reretrieved_r1.csv.gz`; contrasts and gates repeated on it (`r1_reretrieved_contrasts.csv`, `r1_gate3_primary_reretrieved.csv`).
- NDVI-threshold sensitivity: four re-exported panels (`code/gee/gee_r1_export_panel_thresholds.py`, `code/gee/panel_thr_*_r1.csv.gz`) and `r1_ndvi_thresholds.csv`.
- Full pipeline rerun into a separate folder and comparison with the saved tables (`r1_rerun_comparison.csv`).
- Fig. 1 redrawn with a scale bar, a north arrow and the coastline source (`figures_r1/fig_r1_cell_classes.png`).

## 2.2.0 (5 October 2026), branch `v11-framework-ml`, release `v2.2.0`

Added for the supervisor review of 5 October 2026 (see `docs/REVISION_R1.md`). No existing code, table or figure changed; the new scripts import the framework modules read-only and refuse to overwrite.

- `code/src/revision_r1_analyses.py`: emissivity diagnostic and first-order bound, contrast for each dose, slope with an intercept, model sensitivity fits (without coordinates, without emissivity, random folds), residual variogram, NEGI over α/β and p, population counts.
- `code/src/revision_r1_gate3.py`: gates 2 and 3 on the primary match (Models A and B, 200 joint refits).
- `code/src/revision_r1_psf_recovery.py`: point-spread extension of the synthetic recovery test.
- `code/src/revision_r1_figures.py`, `code/src/revision_r1_scene_counts.py`.
- `code/gee/gee_r1_lst_reretrieval.py` and `code/src/revision_r1_reretrieved_contrasts.py`: LST re-retrieval with a vegetation-following emissivity. **Not yet run**; parameters must be confirmed before use.
- `tables_revision_r1/` (12 tables) and `figures_r1/` (2 figures).

## 2.1.0 (2 October 2026), branch `v11-framework-ml`, release `v2.1.0`

Added

- `tables_revision_v11/model_benchmark_v11.csv` and `code/src/benchmark_models_v11.py`: XGBoost, random forest, gradient boosting and linear regression scored on the same cells, predictors and spatial folds. The XGBoost row reproduces `tables/model_cv.csv`.
- `tables_revision_v11/tradeoff_illustration_v11.csv` and `code/src/tradeoff_illustration_v11.py`: NEGI evaluated on measured contrasts for illustration, and the hypothetical irrigation energy per degree of measured cooling.

Changed

- Root `requirements.txt` records pandas 3.0.3, the version in the author's environment.
- `docs/Wahaj_Manuscript_numeric_audit.md` is the audit of the paper that includes the two additions.

No existing code, table or figure changed.

## 2.0.0 (2 October 2026), branch `v11-framework-ml`, release `v2.0.0`

Reorganised the repository around the framework described in the paper. The framework is named Wahaj and the repository `Wahaj-Framework`; NEGI remains the name of the index.

Added

- `code/gee`: Earth Engine export scripts and the saved cell panels (the two large panels are stored as `.gz`).
- `code/src`: framework modules (`fw_*.py`), matched-contrast and exposure analyses, figure scripts, tests.
- `code/negi_original/NEGI_Framework.py`: the index module.
- `tables/`, `tables_revision_v11/`, `figures_png/`, `figures_pdf/`: results.
- `docs/`: file guide, data notes, math check, NEGI logic audit, portability audit, numeric audit of the paper (`Wahaj_Manuscript_numeric_audit.md`).
- `REPRODUCE.md`, `MANIFEST_SHA256.txt`, new `README.md`, `CITATION.cff`, `requirements.txt`.

Removed from this branch (still on `main`)

- `src/`, `gee/`, `Data/`, `Results/`: the first version of the pipeline and its July 2026 outputs. Its scenario and NEGI values are withdrawn because the surrogate behind them was not tested against measured change.

Known gaps

- `tables/imagery_sampling_key_v7.csv` is withheld (blinded review key); two tests and one layout assertion are expected to fail for that reason and because of the added folders.
- The paper itself is not in the repository.
- The paper reference and DOI will be added to `README.md` and `CITATION.cff` after publication.

## 1.0.0 (July 2026), branch `main`

First public version: single-module surrogate pipeline with scenario analysis and NEGI.
