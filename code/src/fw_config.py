"""Analysis settings. Revision sensitivities are exploratory, not retrospectively prespecified."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Config:
    panel_path: Path = Path("code/gee/panel.csv")
    out_dir: Path = Path(".")
    seed: int = 42

    pre_years: tuple = (2014, 2015)
    mid_years: tuple = (2018, 2019)
    post_years: tuple = (2024, 2025)

    builtup_min: float = 0.05

    region_deg: float = 0.10
    coast_bins_km: tuple = (2.0, 5.0, 10.0)
    own_ghsl_bins: tuple = (0.05,)
    nb_change_bins: tuple = (0.02, 0.10)
    emis_bins: tuple = (0.96,)
    min_controls: int = 3
    stable_surface_max: float = 0.20

    dose_bins: tuple = ((1, 2), (3, 4), (5, 8), (9, 9))
    min_cells_reported: int = 10

    block_deg: float = 0.05
    n_boot: int = 2000

    far_control_m: float = 600.0
    arc_lat_bounds: tuple = (21.3, 21.4)

    lambda_b10_um: float = 10.9
    c2_umK: float = 14388.0

    sample_fraction: float = 0.10
    ml_features: tuple = ("ndvi", "ndbi", "elev", "emis", "coast_km", "lon", "lat")
    xgb_params: dict = field(default_factory=lambda: {
        "n_estimators": 600, "max_depth": 6, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8,
        "min_child_weight": 5, "reg_lambda": 1.0})
    cv_folds: int = 5
    n_refits: int = 200
    support_k: int = 5
    support_percentile: float = 95.0
    strict_exclusion_m: float = 300.0
    validation_margin_C: float | None = None
    min_validation_cells: int = 10
    min_validation_blocks: int = 5
    min_validation_refits: int = 200
    jobs: int = 4
    region_holdouts: bool = True

    def __post_init__(self):
        if self.n_boot < 2 or self.n_refits < 2 or self.n_refits > self.n_boot:
            raise ValueError("Require 2 <= refits <= bootstrap draws.")
        if self.validation_margin_C is not None and (not 0 < self.validation_margin_C < float('inf')):
            raise ValueError("Validation tolerance must be finite and positive.")

    decision_maker: str | None = None
    allocation_path: Path | None = None
    water_budget_m3: float | None = None
    wastewater_cap_m3: float | None = None
    energy_budget_mwh: float | None = None
    financial_budget: float | None = None

    kc: tuple = (0.50, 0.60, 0.85)
    efficiency: tuple = (0.90, 0.75, 0.60)
    kwh_per_m3: dict = field(default_factory=lambda: {
        "desalinated seawater (SWRO)": (2.5, 3.25, 4.0),
        "treated wastewater": (0.30, 0.60, 0.93),
    })
    sio_path: Path = HERE / "sio_irrigation_2020_2022_v8.csv"


SETTINGS = ("outside the built-up area", "built-up surroundings")

SOURCES = {
    "kc": "Landscape coefficient for warm-season turf 0.6 as used in landscape water budgets (WUCOLS IV, University "
          "of California Cooperative Extension, 2014); 0.5 for mixed moderate-water plantings; 0.85 is the FAO-56 "
          "value for unstressed warm-season turf (Allen et al. 1998, Table 12).",
    "efficiency": "Irrigation efficiency 0.75 for overhead spray and 0.9 for drip; 0.6 for poorly maintained systems "
                  "(California Model Water Efficient Landscape Ordinance, 2015).",
    "et0": "TerraClimate reference evapotranspiration (Penman–Monteith, Abatzoglou et al. 2018), annual total, mean "
           "over the greened land (code/gee/export_panel.py).",
    "swro": "Seawater reverse osmosis, whole plant: 2.5–4.0 kWh/m³ (Voutchkov 2018, Desalination 431, 2–14).",
    "tse": "Municipal wastewater treatment with reuse-grade polishing: 0.30–0.93 kWh/m³ (Plappally & Lienhard 2012, "
           "Renewable and Sustainable Energy Reviews 16, 4818–4848).",
    "sio": "Saudi Irrigation Organization open data, https://sio.gov.sa/OpenData/Primery_Data_en (irrigated area and "
           "water supplied by source, 2020–2022).",
    "ghsl": "GHSL built-up surface 2015 (JRC/GHSL/P2023A/GHS_BUILT_S, Pesaresi & Politis 2023).",
}
