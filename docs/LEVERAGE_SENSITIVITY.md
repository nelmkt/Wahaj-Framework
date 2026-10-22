# Spatial leverage sensitivity

The primary outside-area matched contrast uses 1,006 cells in 34 spatial blocks. Its zero-intercept, dose-weighted slope is -1.1806 °C per added greened pixel (`tables/headline_block_v8.csv`, `all`). This is an observational matched-change summary, not an intervention forecast.

Block `786_426` contributes 14,505 of 26,058 pooled squared-dose units, or 55.66%. The saved independent arithmetic check in `tables_revision_v11/leverage_share_check_v11.csv` reproduces that ratio exactly. Removing the block leaves 641 cells and changes the pooled slope to -1.0416 °C per pixel: a 0.1390 °C shift, or 11.8% of the original slope magnitude. The sign remains negative, but one block has enough influence to make a precise citywide effect claim unwarranted.

The pooled spatial-block bootstrap interval is -1.2520 to -0.9253 °C per pixel; its jackknife interval is -1.4650 to -0.8962 °C per pixel. These intervals quantify resampling uncertainty in the analysed blocks. They do not address unmeasured confounding, thermal mixing, site selection, or transfer to future sites. Class-specific slopes and block-removal values are reported in `tables/headline_block_v8.csv`; they should not be treated as independent isolated-pixel effects because nearby greening covaries with own greening.

For future field validation, freeze an influence analysis before outcomes are seen, report the full block-by-block distribution and repeat the pass/fail decision after excluding the highest-leverage block. `docs/VALIDATION_PLAN.md` states this proposed gate. Until a prospective change model passes, Wahaj withholds planting-scenario recommendations.
