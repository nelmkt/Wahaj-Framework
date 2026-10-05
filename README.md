# Wahaj: remote sensing and machine learning for urban greening–energy trade-offs

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![XGBoost 3.4](https://img.shields.io/badge/XGBoost-3.4-EB5E28?style=flat)](https://xgboost.readthedocs.io/)
[![Google Earth Engine](https://img.shields.io/badge/Google%20Earth%20Engine-Landsat%208%20C2%20L2-4285F4?style=flat&logo=googleearth&logoColor=white)](https://earthengine.google.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat)](LICENSE)
[![Version 2.3.0](https://img.shields.io/badge/version-2.3.0-informational?style=flat)](https://github.com/nelmkt/Wahaj-Framework/releases/tag/v2.3.0)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0007--9887--0280-A6CE39?style=flat&logo=orcid&logoColor=white)](https://orcid.org/0009-0007-9887-0280)
[![Email](https://img.shields.io/badge/Email-nalmaktoum0001%40stu.kau.edu.sa-D14836?style=flat&logo=gmail&logoColor=white)](mailto:nalmaktoum0001@stu.kau.edu.sa)

Code, saved Earth Engine exports, figures and result tables for the paper

> **Wahaj: A Remote Sensing and Machine Learning Framework for Evaluating Urban Greening–Energy Trade-Offs in Desalination-Dependent Cities, with a Jeddah Case Study**
> Nelly F. Almaktoum

Wahaj is the framework described in the paper. It asks whether an XGBoost model that predicts land surface temperature (LST) can also predict the *change* associated with greening. It compares model diagnostics with matched satellite contrasts, then illustrates the water and energy implied by assumed irrigation. **Jeddah is the only city analysed.** This is neither a measured irrigation ledger nor a validated model for selecting a new greening site.

## Contents

- [Results and their populations](#results-and-their-populations)
- [Repository structure](#repository-structure)
- [What the code contains](#what-the-code-contains)
- [Interpretation and portability](#interpretation-and-portability)
- [Reproduction](#reproduction)
- [Tests and checks](#tests-and-checks)
- [Data availability](#data-availability)
- [Earlier version](#earlier-version)
- [Citation](#citation)
- [Licence](#licence)
- [Contact](#contact)

## Results and their populations

| Result | Saved value | Source and limit |
| --- | --- | --- |
| XGBoost absolute-LST prediction | Spatial GroupKFold R² **0.795**, RMSE **1.184 °C**, MAE **0.903 °C**, on **19,650** non-greened cells | [`model_cv.csv`](tables/model_cv.csv). Skill on LST levels does **not** validate predicted greening effects. |
| Benchmark of regressors | Same cells, predictors and folds: XGBoost R² **0.795**, random forest **0.775**, gradient boosting **0.773**, linear regression **0.711** | [`model_benchmark_v11.csv`](tables_revision_v11/model_benchmark_v11.csv). Fixed, untuned settings: the comparison ranks these settings, not the methods in general. |
| Primary matched outside-area slope | **−1.181 °C per greened pixel** among **1,006** matched cells | [`headline_block_v8.csv`](tables/headline_block_v8.csv), row `all`. A zero-intercept, dose-weighted summary in sampled, often contiguous blocks, not the marginal effect of a new isolated pixel. |
| Leverage concentration | One spatial block carries **55.7%** of pooled squared-dose leverage | Same table, row `all`. Bootstrap and jackknife coverage is unproven. |
| Small-patch own-only slope | **−1.925 °C per own pixel** among **419** primary matched outside cells with 1–2 greened pixels | Same table, row `1–2`; [algebraic check](tables_revision_v11/own_only_vs_joint_identity_v11.csv). The own-only coefficient absorbs co-varying neighbour greening within the specified regression (joint own coefficients −1.036 to −0.988); it is not evidence of a larger physical effect per pixel. |
| Model B outside support | **14/744** cells pass the feature-space screen, including **0/75** fully greened cells | [`test_support_counts.csv`](tables/test_support_counts.csv), outside rows. This is the **narrower concurrent-change sensitivity**, not the 1,006-cell primary population. |
| Model B minus measured outside contrast | **+1.01 to +7.24 °C** across dose classes | [`test_by_dose.csv`](tables/test_by_dose.csv), outside Model B rows. The gap grows with dose; the model does not corroborate that outside gradient. |
| Illustrative NEGI on measured contrasts | Cost exponent 0.5: **−0.127, −0.203, −0.213, 0**; linear cost: **+0.110, +0.034, −0.078, 0** for the four outside dose classes | [`tradeoff_illustration_v11.csv`](tables_revision_v11/tradeoff_illustration_v11.csv). An illustration on different cells, not a released index value; the sign is decided by the assumed cost shape. |
| Product emissivity at greened cells | Unchanged between 2014–15 and 2024–25 at **100%** of matched greened cells while NDVI rises; a first-order correction moves the primary slope from **−1.181** to **−1.452 °C per pixel** (largest case) | [`r1_emissivity_diagnostic.csv`](tables_revision_r1/r1_emissivity_diagnostic.csv), [`r1_emissivity_bound.csv`](tables_revision_r1/r1_emissivity_bound.csv). Collection 2 vegetation-adjustment anomaly; the measured cooling is conservative. Full re-retrieval script added, not yet run. |
| Thermal footprint (simulation) | Under a 150 m point-spread function the own-only slope recovers **0.26** (isolated cells) to **0.68** (5 × 5-cell patches) of a known effect | [`r1_psf_recovery.csv`](tables_revision_r1/r1_psf_recovery.csv). No physical spillover in the simulation: ring terms pick up optical blur. |
| Dose relation with an intercept | Slope **−0.966 °C per pixel**, intercept **−1.34 °C** (outside) | [`r1_slope_intercept.csv`](tables_revision_r1/r1_slope_intercept.csv), [`r1_dose_contrasts.csv`](tables_revision_r1/r1_dose_contrasts.csv). Dose-class means are the primary estimates. |
| Gates on the primary match | Model B support **15/419** (1–2 pixels), **0** for larger classes; Model B minus measured **+1.32 to +8.30 °C** | [`r1_gate3_primary.csv`](tables_revision_r1/r1_gate3_primary.csv). Same outcome as the concurrent-change match. |
| Model sensitivity | R² **0.741** without coordinates, **0.767** without emissivity, **0.881** with random folds | [`r1_model_sensitivity.csv`](tables_revision_r1/r1_model_sensitivity.csv), [`r1_variogram.csv`](tables_revision_r1/r1_variogram.csv). Random folds overstate skill for unsampled places. |
| Re-retrieved LST (vegetation-following emissivity) | Primary outside slope **−1.246 °C per pixel** against **−1.181** with the product LST | [`r1_reretrieved_contrasts.csv`](tables_revision_r1/r1_reretrieved_contrasts.csv), [`r1_gate3_primary_reretrieved.csv`](tables_revision_r1/r1_gate3_primary_reretrieved.csv). Emissivity after Ermida et al. (2020). |
| NDVI thresholds | Primary outside slope **−1.339 to −0.993 °C per pixel** across non-vegetated 0.10–0.20 and greened 0.25–0.35 | [`r1_ndvi_thresholds.csv`](tables_revision_r1/r1_ndvi_thresholds.csv); re-exported panels in `code/gee/*_r1.csv.gz`. |
| Full rerun | **30** tables rewritten by `run_framework.py` compared with the saved ones; largest relative difference **1.0e+12** | [`r1_rerun_comparison.csv`](tables_revision_r1/r1_rerun_comparison.csv). |

The 1,006-cell matched estimate and the 744-cell model diagnostics use different treated populations. They must not be read as one validation result. The small-patch exposure analyses are descriptive: the available data do not separate thermal-pixel blur, edge effects, neighbour cooling and neighbourhood confounding. The within-90 m ordering is **not resolved**.

![Model skill on held-out areas, and model-predicted against measured cooling by number of greened pixels](figures_png/fig3_model_test.png)

*Model test ([`fig3_model_test.png`](figures_png/fig3_model_test.png)). Panel a: predicted against observed summer LST with areas held out. Panels b and c: measured cooling and the two model versions by dose class. Outside the built-up area, Model B falls further short of the measured contrast as the dose rises.*

Because the model failed these checks, the framework released no model-based scenario and **no model-based value of the Normalized Environmental Gain Index (NEGI)**. The only NEGI numbers are the illustration on measured contrasts in the last row.

## Repository structure

```
README.md            this file
LICENSE              MIT licence
CITATION.cff         citation metadata
CHANGELOG.md         what changed and when
REPRODUCE.md         each reported number -> table, script, command
requirements.txt     Python packages and versions
MANIFEST_SHA256.txt  SHA256 and size of every file

code/
  gee/               Earth Engine export scripts and the saved exports (inputs)
  src/               framework modules, analysis scripts, figure scripts, tests
  negi_original/     earlier index module (NEGI definitions; not run for the paper)

tables/              result tables (CSV, JSON)
tables_revision_v11/ four tables added for the paper (two algebraic checks, model benchmark, trade-off illustration)
tables_revision_r1/  tables added for the supervisor revision (release 2.2.0)
figures_r1/          two figures added for the supervisor revision
figures_png/         figures (PNG)
figures_pdf/         figures (PDF)

docs/
  FILE_GUIDE.md                    every script and table, grouped by purpose, in reading order
  DATA.md                          data sources, panel columns, what is withheld
  MATH_CHECK_V11.md                leverage shares and the own-only vs joint identity
  NEGI_LOGIC_AUDIT_V11.md          what the index module computes, line by line
  PORTABILITY_AUDIT_V11.md         what is Jeddah-specific in the code
  Wahaj_Manuscript_numeric_audit.md  each number in the paper checked against its table
```

To find a script or table, start with [`docs/FILE_GUIDE.md`](docs/FILE_GUIDE.md). The paper itself is not in this repository.

## What the code contains

| Stage | Files | Role |
| --- | --- | --- |
| Satellite panel | [`export_panel.py`](code/gee/export_panel.py), [`panel.csv.gz`](code/gee/panel.csv.gz), [`panel_meta.json`](code/gee/panel_meta.json) | Cloud-screened Landsat 8 Collection 2 Level-2 optical and LST composites on aligned 90 m cells for 2014, 2015, 2018, 2019, 2024 and 2025. [`panel_jun_sep.csv.gz`](code/gee/panel_jun_sep.csv.gz) is a common-month sensitivity. |
| Matched contrasts | [`fw_panel.py`](code/src/fw_panel.py), [`fw_matching.py`](code/src/fw_matching.py) | Pre-treatment-strata matched comparison with never-vegetated controls and spatial-block resampling; follow-up scripts inspect leverage, isolation and exposure. |
| ML diagnostic | [`fw_model.py`](code/src/fw_model.py), [`fw_validate.py`](code/src/fw_validate.py) | XGBoost absolute-LST prediction, spatial cross-validation, feature-support screening and effect comparison. Absolute-LST skill alone does not license a counterfactual effect. |
| Conditional resource accounting | [`fw_ledger.py`](code/src/fw_ledger.py), [`fw_decision.py`](code/src/fw_decision.py) | Reference ET, assumed landscape coefficient and irrigation efficiency, and assumed water-source energy intensities. The per-degree ratios are largely rescalings of cooling per area under a shared irrigation depth, not a resource ranking or a net-energy benefit. |
| Earlier index module | [`NEGI_Framework.py`](code/negi_original/NEGI_Framework.py) | Historical pathway and NEGI definitions. Its earlier scenario and index values are not reported as validated findings here. This module is separate from `fw_decision.py`. |

The main entry point is [`code/src/run_framework.py`](code/src/run_framework.py).

## Interpretation and portability

- LST is **surface temperature**, not air temperature, thermal comfort, building electricity savings or population benefit. Satellite-overpass cooling and annual irrigation have different time bases.
- NDVI-defined greening is **not verified managed irrigation**. Persistent vegetation may also draw on groundwater, wadi flow or discharge. No water-meter, planting-history or building-energy observations establish the resource ledger.
- The model's outside support is sparse, its dose-gradient error grows, and no practical effect-error tolerance was prespecified. This repository presents no decision-ready model-based scenario or NEGI value.
- "Similar city" means a hot arid or semi-arid city with comparable optical and thermal satellite coverage, identifiable vegetation transitions, enough eligible bare controls and a relevant desalination or treated-wastewater supply context. That is a candidate for *adapting the workflow*, not for importing Jeddah's estimates. The seasonal windows, classification and matching thresholds, coastline, spatial blocks, model and support domain, and water–energy assumptions all need local justification. **The Jeddah estimates, fitted model and hypothetical ledger do not transfer.** See [`docs/PORTABILITY_AUDIT_V11.md`](docs/PORTABILITY_AUDIT_V11.md).

## Reproduction

Work **in a copy** of the repository: the main pipeline writes tables and figures, and several follow-up scripts refuse to overwrite saved outputs. Python 3.12 was used.

**1. Install.** [`requirements.txt`](requirements.txt) lists the packages. It differs from [`code/src/requirements.txt`](code/src/requirements.txt) in one pin (pandas 3.0.3, the version in the author's environment, against 3.0.1) and adds `openpyxl`, `threadpoolctl` and `earthengine-api`.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On macOS or Linux, activate with `source .venv/bin/activate`.

**2. Decompress the panels.** The two cell panels are stored as `.gz`. Decompress them once, in place:

```powershell
python -c "import gzip,shutil; [shutil.copyfileobj(gzip.open(f'code/gee/{n}.gz','rb'), open(f'code/gee/{n}','wb')) for n in ('panel.csv','panel_jun_sep.csv')]"
```

Their SHA256 values after decompression are in [`MANIFEST_SHA256.txt`](MANIFEST_SHA256.txt).

**3. Run the framework** from the repository root:

```powershell
python code/src/run_framework.py --panel code/gee/panel.csv --out .
```

Follow-up analyses are separate scripts in `code/src`. [`REPRODUCE.md`](REPRODUCE.md) gives the table, script and command behind each reported number.

Earth Engine is not needed to inspect or recompute the tables, because the exports are saved in `code/gee`. Re-exporting the panel with `code/gee/export_panel.py` needs an authenticated Earth Engine project.

## Tests and checks

```powershell
$env:PYTHONDONTWRITEBYTECODE = "1"
python -m pytest code/src -p no:cacheprovider
```

The author reported **62 passing tests and a passing claims audit** on the full local package on 2 October 2026. They were **not** rerun on this repository layout. Three checks are expected to fail here, for layout reasons only:

- `code/src/test_followup_v7.py` and `code/src/test_v8.py` look for `tables/imagery_sampling_key_v7.csv`, which is withheld.
- `code/src/audit_claims.py` asserts the earlier five-folder layout; this repository adds `docs/`, `tables_revision_v11/`, `code/negi_original/` and root files.

No test result by itself establishes parallel trends, irrigation identity, interval coverage or a physical cooling mechanism.

Two further checks are recorded in `docs/`: the [math check](docs/MATH_CHECK_V11.md) (leverage shares and the own-only against joint identity, residual at most 1.1e-15) and the [numeric audit](docs/Wahaj_Manuscript_numeric_audit.md) of the paper's numbers against these tables.

## Data availability

Landsat 8 Collection 2 Level-2 imagery is public through Google Earth Engine. The exports used here are in `code/gee/`; sources, identifiers and column meanings are in [`docs/DATA.md`](docs/DATA.md).

- **Assumed, not measured:** water and energy figures in `tables/water.csv`, `tables/energy.csv` and `tables/decision_*.csv`.
- **Withheld:** `tables/imagery_sampling_key_v7.csv`, the blinded key of an imagery review that is not complete. The unlabelled review queue is included.

## Earlier version

The [`main`](https://github.com/nelmkt/Wahaj-Framework/tree/main) branch holds the first version of this project (a surrogate pipeline with scenario and NEGI values from July 2026). Those scenario and index values are withdrawn: the surrogate behind them was never tested against measured change. They are kept for the record and are not used by the paper. See [`CHANGELOG.md`](CHANGELOG.md).

## Citation

If you use this code or these tables, please cite the repository. The paper reference and DOI will be added when available. Machine-readable metadata is in [`CITATION.cff`](CITATION.cff) (GitHub shows it under "Cite this repository").

```bibtex
@software{almaktoum_wahaj_2026,
  author  = {Almaktoum, Nelly F.},
  title   = {Wahaj: A Remote Sensing and Machine Learning Framework for Evaluating Urban Greening--Energy Trade-Offs in Desalination-Dependent Cities, with a Jeddah Case Study},
  year    = {2026},
  version = {2.3.0},
  url     = {https://github.com/nelmkt/Wahaj-Framework}
}
```

## Licence

Released under the [MIT License](LICENSE), © 2026 Nelly F. Almaktoum. The licence covers the code and the files in this repository. Landsat, Dynamic World, SRTM, TerraClimate, GHSL, Natural Earth and Saudi Irrigation Organization data remain under their providers' own terms; see [`docs/DATA.md`](docs/DATA.md).

## Contact

Nelly F. Almaktoum · [nalmaktoum0001@stu.kau.edu.sa](mailto:nalmaktoum0001@stu.kau.edu.sa) · [ORCID 0009-0007-9887-0280](https://orcid.org/0009-0007-9887-0280)

Questions and bug reports are also welcome through the repository's [issues](https://github.com/nelmkt/Wahaj-Framework/issues).
