# MATH_CHECK_V11 (Section C)

Written 2026-10-02. Saved data only. No bootstrap was run. No package script was run; the check script (kept outside the package, in the session scratchpad, `math_check_v11.py`) imports three loader and algebra functions from `code/src` with `PYTHONDONTWRITEBYTECODE=1` and created no `__pycache__`.

Functions imported: `load_new` and `subring_design` (`code/src/analyze_pixel_subrings_v8_followup6.py:33-82`), which call `load_saved` (`code/src/analyze_isolation_reweighting_v8.py:47-97`); `wls_no_intercept` (`code/src/analyze_exposure_rings_v8_followup3.py:25-37`).

Saved inputs read: `code/gee/panel.csv`, `code/gee/pixel_external_counts_v8_raw.csv` (+ `.json`), `code/gee/pixel_external_subrings_v8_followup5_raw.csv` (+ `.json`), `code/src/jeddah_coastline_ne10m.csv`, `tables/figure1_population_comparison_v8.csv`, `tables/headline_block_v8.csv`, `tables/ring_reduced_coefficients_v8_followup6.csv`, `tables/pixel_subring_regression_v8_followup6.csv`.

New files: `tables/revision_v11/leverage_share_check_v11.csv` (1,525 bytes), `tables/revision_v11/own_only_vs_joint_identity_v11.csv` (5,543 bytes). Nothing in `tables/` was touched.

## Population and outcome used

- Cells: outside-built-up greened cells (class 1) whose pre-treatment stratum holds at least 3 eligible controls. Count: 1,006. With 1-2 greened pixels: 419, in 32 blocks. These equal `tables/headline_block_v8.csv` rows 2 and 3 (`n_matched` 1006 and 419, `n_blocks` 32).
- Outcome: matched contrast tau_i = (LST change of cell i, post minus pre) minus (mean LST change of the eligible never-vegetated controls in cell i's stratum).
- Dose: d_i = number of greened 30 m pixels in the cell (1 to 9).

## C1. Leverage shares: reproduced

Formula: share(G) = sum over i in G of d_i^2, divided by the sum over all 1,006 matched outside cells of d_i^2. Denominator = 26058.

| Quantity | Cells | Numerator | Recomputed | Saved (`headline_block_v8.csv`) | Expected (printed) | Match at 4 d.p. |
|---|---|---|---|---|---|---|
| Block 786_426 | - | 14505 | 0.556642873590 | 0.5566428735896846 (row 2, `dominant_pooled_block_leverage_share`) | 0.5566 | yes |
| Class 1-2 | 419 | 854 | 0.032773044746 | 0.0327730447463351 (row 3) | 0.0328 | yes |
| Class 3-4 | 183 | 2130 | 0.081740732213 | 0.08174073221275616 (row 4) | 0.0817 | yes |
| Class 5-8 | 244 | 10114 | 0.388134162253 | 0.38813416225343467 (row 5) | 0.3881 | yes |
| Class 9/9 | 160 | 12960 | 0.497352060787 | 0.49735206078747407 (row 6) | 0.4974 | yes |

Largest absolute difference from the saved value: 5.6e-17. The block with the largest share is 786_426, the same label as saved. The four class numerators sum to 854 + 2130 + 10114 + 12960 = 26058.

Stop condition "a leverage share does not reproduce": **not triggered**.

## C2. Own-only versus joint identity on the 419 cells: holds

Step 1, same outcome and cells as the -1.925 estimate: the own-only slope recomputed here is sum(d_i tau_i) / sum(d_i^2) = -1.9253007042875268. The saved value is -1.9253007042875272 (`tables/headline_block_v8.csv` row 3, `point_C_per_pixel`). Difference 4.4e-16. The outcome and cell set are the same, so the identity may be tested.

Step 2, identity. For a joint zero-intercept least-squares fit tau = beta_own d + sum_k beta_k x_k + e on the same cells:

b_own_only = beta_own + sum_k beta_k gamma_k,   gamma_k = sum(d_i x_ik) / sum(d_i^2).

gamma_k is the zero-intercept slope of ring-k external count on own dose. The identity is exact algebra for least squares (the residual e is orthogonal to d). Tolerance used: 1e-10.

| Joint model | beta_own | sum beta_k gamma_k | Right-hand side | Residual | Holds |
|---|---|---|---|---|---|
| full (0-90, 90-180, 180-300 m) | -1.002390075275 | -0.922910629012 | -1.925300704288 | -2.2e-16 | yes |
| far ring omitted | -1.035835878551 | -0.889464825736 | -1.925300704288 | 4.4e-16 | yes |
| outer rings merged (90-300 m) | -0.987877689741 | -0.937423014547 | -1.925300704288 | 2.2e-16 | yes |
| fine inner (0-30, 30-60, 60-90 m) | -1.013772758630 | -0.911527945657 | -1.925300704288 | 2.2e-16 | yes |
| coarse inner (0-30, 30-90 m) | -1.013317040565 | -0.911983663722 | -1.925300704288 | -1.1e-15 | yes |

Every recomputed coefficient equals its saved value to within 4.4e-16 (`tables/ring_reduced_coefficients_v8_followup6.csv` rows 2-11; `tables/pixel_subring_regression_v8_followup6.csv` rows 2-12).

gamma_k values (external greened pixels per own greened pixel, zero-intercept): 0-30 m 0.741; 30-60 m 1.089; 60-90 m 1.830; 0-90 m 3.660; 30-90 m 2.919; 90-180 m 7.310; 180-300 m 13.232; 90-300 m 20.542. Per-term products are in the CSV.

Stop condition "identity residual is not zero": **not triggered** (largest |residual| 1.1e-15 against tolerance 1e-10).

### What the identity does and does not show

- It shows, as arithmetic, that the gap between -1.925 (own-only) and about -1.0 (joint own coefficient) equals the external coefficients times how strongly external counts rise with own dose. In the full model that term is -0.923 °C per own pixel.
- It is why the paper may say the own-only slope absorbs co-varying neighbour greening and is not a larger per-pixel effect.
- It is a property of least squares. It says nothing about cause: the split between own and external terms is not identified, the allocation among rings is specification sensitive, and confounders that co-vary with either count are absorbed in the same way.

## C3. Bootstrap coverage

No coverage simulation was run and none exists in the package. The paper states that interval coverage is unproven.
