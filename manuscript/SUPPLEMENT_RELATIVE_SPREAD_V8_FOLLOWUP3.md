# Supplement: descriptive class-slope convergence

The earlier relative-spread analysis is supplementary. It divides the range of four class slopes by the absolute value of their equally weighted mean. Because external-pixel counts enlarge the denominator, absolute slope spread shrinks partly mechanically. An external pixel can appear in several focal-cell exposures. Neither relative nor absolute convergence identifies spillover, an independent planted-pixel response, or a water/energy ranking.

| Exposure | Relative spread [95% spatial-block interval] |
|---|---:|
| own | 0.628 [0.556, 0.754] |
| combined_90m | 0.472 [0.275, 0.560] |
| combined_180m | 0.263 [0.087, 0.316] |
| combined_300m | 0.134 [0.066, 0.255] |

Source: [relative-spread table](../tables/combined_class_relative_spread_v8_followup2.csv) and the [saved-data script](../code/src/analyze_joint_exposure_v8_followup2.py).
