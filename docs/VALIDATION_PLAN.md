# Prospective validation plan for greening-change predictions

This plan is recorded on 13 October 2026, before any new field observations are analysed. It is a protocol proposal for a future study, not a preregistration or a claim that Wahaj currently passes. Jeddah is the only city analysed in the released results.

## Decision question and estimand

For a proposed greening intervention on a previously non-vegetated 90 m cell, can a model predict the change in summer satellite-overpass land surface temperature (LST) relative to a suitable ungreened comparison? The outcome is LST change in degrees Celsius. Air temperature, thermal comfort, health, irrigation energy and building electricity are separate outcomes and require separate measurements.

## Study design to lock before acquiring outcomes

1. Recruit new sites in at least two independent spatial areas, with recorded planting date, species or cover, irrigation source, applied water and maintenance. Retain ungreened comparison sites with pre-intervention imagery.
2. Freeze site eligibility, matching variables, baseline window, outcome window, cloud and emissivity handling, feature list, model version and support screen before examining post-intervention LST.
3. Hold out whole spatial blocks and all sites from the chosen test areas during model fitting and threshold tuning. Report 1–2, 3–4, 5–8 and 9/9 greened-pixel dose classes separately; do not pool away a failing class.
4. Measure the matched-change benchmark independently of the model. Record pre-intervention trends and run a placebo on sites that had not yet been planted. Treat observational contrasts as uncertain if trend balance fails.
5. Audit the thermal point-spread function and rerun the outcome with vegetation-following emissivity. Report both product and re-retrieved LST.

## Proposed pass rules to finalize with a user before data collection

These numbers are conservative working thresholds, chosen for an interpretable research gate rather than inferred from the present data. Record their final values publicly before new outcomes are viewed.

| Gate | Proposed rule |
| --- | --- |
| Sample adequacy | At least 200 treated cells across at least 10 independent spatial blocks, with at least 30 cells and 5 blocks in every dose class used for a claim. |
| Feature support | At least 80% of candidate cells in each claimed dose class pass the frozen support screen; report the rate and confidence interval. |
| Effect agreement | Mean absolute error of predicted minus benchmark change at most 1.0 °C, absolute mean bias at most 0.5 °C, and no dose-class absolute mean bias above 1.0 °C. |
| Uncertainty | Nominal 90% prediction intervals cover 85–95% of held-out site contrasts overall and in every class with adequate sample size. |
| Robustness | The pass/fail conclusion is unchanged after removing the single highest-leverage block and under the emissivity re-retrieval. |

Any failure keeps the model-based scenario and NEGI gates closed. A future result that passes permits only claims for the tested population, time window and outcome. A separately measured irrigation ledger and decision study would still be needed for water and energy recommendations.

## Reporting

Publish the frozen protocol, site flow, exclusions, code commit, model artifact, support decisions, predictions made before outcome inspection, observed contrasts, error by dose, spatial block influence, uncertainty coverage and all deviations. A potential planning user should specify the decision and acceptable error before the final thresholds are locked.

## Current evidence that motivates the plan

The current Model B support screen accepts 14 of 744 outside validation cells and 0 of 75 fully greened cells (`tables/test_support_counts.csv`). One block carries 55.7% of the primary pooled squared-dose leverage (`tables/headline_block_v8.csv`). Thermal blur and emissivity sensitivities are in `tables_revision_r1/`. The 1,006-cell primary match and 744-cell concurrent-change test are different populations.
