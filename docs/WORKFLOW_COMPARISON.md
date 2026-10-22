# Where Wahaj adds to existing workflows

This comparison supports a narrow novelty claim: Wahaj combines several established practices in one gated Jeddah workflow and publishes a failure of model-based greening-change validation. It does not claim that spatial validation, matched contrasts, feature support or irrigation accounting were invented here.

| Existing approach | What it addresses | Wahaj's additional check | Remaining limit |
| --- | --- | --- | --- |
| Urban LST modelling with XGBoost, including Madrid and Beijing studies [1, 2] | Associations between observed land cover and observed LST | Separately compares predicted *change* under greening with matched observed change | Observational benchmark is not a randomized intervention |
| Spatial cross-validation and area-of-applicability methods [3, 4] | Spatial dependence and extrapolation | Applies both to the LST model, then screens hypothetical greened states | Good absolute-LST skill still cannot certify effect predictions |
| Matching and difference-in-differences methods [5, 6] | Change relative to observed comparisons | Uses a separate matched benchmark to decide whether the ML scenario can be released | Planting dates, parallel trends and dominant-block interval coverage remain unresolved |
| Urban irrigation and energy trade-off modelling [7] | Water and energy consequences of cooling strategies | Shows an explicit conditional ledger beside measured LST contrasts | The ledger uses assumed water and energy factors, not site meters or measured building savings |

The paper's contribution is therefore a **validation and reporting workflow**. In the present data, it rejects the model-based scenario rather than presenting a planting-location recommendation. This narrower framing is the one to use in abstracts, presentations and award applications.

## Sources

1. Bushenkova et al. (2024), *Urban Climate* 55, 101982. https://doi.org/10.1016/j.uclim.2024.101982
2. Cheng et al. (2025), *Energy and Buildings* 338, 115704. https://doi.org/10.1016/j.enbuild.2025.115704
3. Roberts et al. (2017), *Ecography* 40, 913–929. https://doi.org/10.1111/ecog.02881
4. Meyer and Pebesma (2021), *Methods in Ecology and Evolution* 12, 1620–1633. https://doi.org/10.1111/2041-210X.13650
5. Abadie (2005), *Review of Economic Studies* 72, 1–19. https://doi.org/10.1111/0034-6527.00321
6. Callaway, Goodman-Bacon and Sant'Anna (2024), NBER Working Paper 32117. https://doi.org/10.3386/w32117
7. Yang and Wang (2015), *Energy and Buildings* 107, 335–344. https://doi.org/10.1016/j.enbuild.2015.08.045

These references also appear in the companion manuscript. The table compares stated study aims and methods; it is not a systematic review of every implementation detail.
