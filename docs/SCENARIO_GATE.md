# Why model-based greening scenarios are withheld

The cross-validated Model B predicts absolute summer land surface temperature in held-out spatial blocks with R² 0.795 and RMSE 1.184 °C (`tables/model_cv.csv`). That test does not establish whether changing vegetation in a cell will cause the model's predicted temperature change. The latter question is tested separately against observed matched changes in `tables/test_by_dose.csv`.

For the outside-area concurrent-change test, 744 matched cells are available across four greening doses. The frozen feature-space screen supports only 14 cells (1.9%), all in the 1–2 pixel class; it supports 0 of the 75 fully greened cells (`tables/test_support_counts.csv`). Model B's predicted change is warmer than the matched observed change by 1.01, 1.72, 3.63 and 7.24 °C across the four dose classes (`tables/test_by_dose.csv`, `test=B`). These are discrepancies between two imperfect observational estimates, not proof of a causal effect size, but they rule out a credible current planting-scenario claim.

The primary 1,006-cell outside-area matched result in `tables/headline_block_v8.csv` is a different population and estimand. It can describe the observed contrast with its sensitivity bounds; it cannot repair the predictive gate. The irrigation-depth and desalination-energy figures in the tables are conditional arithmetic using assumptions, not measured resource savings. Thus the released framework does not rank planting locations or claim deployable intervention forecasts.

`docs/VALIDATION_PLAN.md` specifies prospective support, error, uncertainty, and influence checks to be frozen before new outcomes are observed. The scenario gate stays closed until those checks pass for a defined use population.
