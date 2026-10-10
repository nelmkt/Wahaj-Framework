from __future__ import annotations

import dataclasses
import datetime
import gc
import hashlib
import json
import logging
import os
import pickle
import sys
import textwrap
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import xgboost
from joblib import Parallel, delayed
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch
from scipy import stats as scipy_stats
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    RandomizedSearchCV,
    cross_validate,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import SplineTransformer, StandardScaler
from xgboost import XGBRegressor




FEATURES: list[str] = ["NDVI", "NDBI", "Elevation", "ST_EMIS", "ST_EMSD"]
SCENARIO_MANIPULATED_FEATURES: tuple[str, ...] = ("NDVI", "NDBI")
FIXED_REFERENCE_FEATURES: tuple[str, ...] = ("Elevation", "ST_EMIS", "ST_EMSD")
DEFAULT_MONOTONE_CONSTRAINTS: tuple[int, ...] = (0, 1, 0, 0, 0)

assert list(SCENARIO_MANIPULATED_FEATURES) + list(FIXED_REFERENCE_FEATURES) == FEATURES, (
    "FEATURES must be exactly the manipulated features followed by the "
    "fixed-reference features, in that order."
)
assert len(DEFAULT_MONOTONE_CONSTRAINTS) == len(FEATURES), (
    "DEFAULT_MONOTONE_CONSTRAINTS must have one entry per feature in FEATURES."
)


def fixed_reference_values(df: pd.DataFrame) -> dict[str, float]:
    """Reference (median) values of the fixed, non-intervention features.

    Computed from whichever frame the caller passes - scenario-defining
    callers pass the training-only frame (see the holdout-independence fix
    in run_scenario_trajectories); descriptive sweeps pass the full frame,
    matching the convention each of those sites already used for Elevation.
    """
    return {feat: float(df[feat].median()) for feat in FIXED_REFERENCE_FEATURES}


def build_feature_frame(
    ndvi, ndbi, fixed_values: dict[str, float],
) -> pd.DataFrame:
    """Assemble a model-input frame (columns == FEATURES, in order) from
    NDVI/NDBI values plus the fixed-reference feature values.

    `ndvi` / `ndbi` may be scalars or equal-length array-likes; each fixed
    feature is broadcast to the same length.
    """
    ndvi_arr = np.atleast_1d(np.asarray(ndvi, dtype=float))
    ndbi_arr = np.atleast_1d(np.asarray(ndbi, dtype=float))
    if ndvi_arr.shape != ndbi_arr.shape:
        raise ValueError(
            f"NDVI and NDBI must have the same length, got {ndvi_arr.shape} "
            f"and {ndbi_arr.shape}."
        )
    columns = {"NDVI": ndvi_arr, "NDBI": ndbi_arr}
    for feat in FIXED_REFERENCE_FEATURES:
        columns[feat] = np.full(len(ndvi_arr), float(fixed_values[feat]))
    return pd.DataFrame(columns)[FEATURES]


def validate_feature_configuration(monotone_constraints, feature_names) -> None:
    """Fail loudly if the configured feature list / monotone constraints
    disagree with the module-level FEATURES schema (a silent mismatch would
    apply constraints to the wrong columns)."""
    if list(feature_names) != FEATURES:
        raise ValueError(
            f"Config.feature_names {list(feature_names)} does not match the "
            f"module-level FEATURES schema {FEATURES}."
        )
    if len(monotone_constraints) != len(FEATURES):
        raise ValueError(
            f"Config.monotone_constraints has {len(monotone_constraints)} "
            f"entries but the model has {len(FEATURES)} features {FEATURES}. "
            "One constraint per feature, in FEATURES order, is required."
        )


_NEGI_BASE_DIR = Path(__file__).resolve().parent


def _default_data_path() -> Path:
    configured_path = os.environ.get("NEGI_DATA_PATH")
    if configured_path:
        return Path(configured_path).expanduser()
    candidates = (
        _NEGI_BASE_DIR / "data" / "jeddah_lst_data.csv",
        _NEGI_BASE_DIR.parent / "Downloads" / "Jeddah_LST_Dataset_2023_raw.csv",
        _NEGI_BASE_DIR.parent / "Desktop" / "Jeddah_LST_Dataset_2023_raw.csv",
        _NEGI_BASE_DIR / "Jeddah_LST_Dataset_2023_raw.csv",
        _NEGI_BASE_DIR.parent / "Downloads" / "Jeddah_LST_Dataset_2023_v4_with_purity_flags.csv",
        _NEGI_BASE_DIR.parent / "Downloads" / "Jeddah_LST_Dataset_2023.csv",
        _NEGI_BASE_DIR.parent / "Desktop" / "Jeddah_LST_Dataset_2023_v4_with_purity_flags.csv",
        _NEGI_BASE_DIR.parent / "Desktop" / "Jeddah_LST_Dataset_2023.csv",
    )
    return next((p for p in candidates if p.is_file()), candidates[0])


@dataclass
class Config:
    """Single source of truth for every configurable parameter.

    Avoid hard-coding any numeric constant outside this class.  All
    downstream functions receive a `cfg` argument and read from here.
    """

    base_dir: Path = field(default_factory=lambda: _NEGI_BASE_DIR)
    data_path: Path = field(default_factory=_default_data_path)

    random_seed: int = 42
    bootstrap_seed: int = 42

    verbose: bool = False
    debug: bool = False

    run_mode: str = field(
        default_factory=lambda: os.environ.get("NEGI_RUN_MODE", "development")
    )

    export_mode: str = field(
        default_factory=lambda: os.environ.get("NEGI_EXPORT_MODE", "standard")
    )

    vegetation_purity_filter: str = field(
        default_factory=lambda: os.environ.get(
            "NEGI_VEGETATION_PURITY_FILTER", "purified"
        )
    )
    purity_audit_ndvi_threshold: float = 0.2

    holdout_test_size: float = 0.20
    n_group_kfold_splits: int = 5
    n_random_search_iter: int = 80
    n_spatial_holdout_repeats: int = 10

    spatial_refit_mode: str = "publication"
    n_spatial_refits_quick: int = 20
    n_spatial_refits_publication: int = 500
    n_spatial_refits_previous_publication_ceiling: int = 500
    n_spatial_refits_percentile_sensitivity: int = 200
    n_spatial_refits: Optional[int] = None
    uncertainty_refit_seed: int = 43
    bootstrap_iterations: int = 1000

    spatial_refit_n_jobs: int = -1
    spatial_refit_max_batch_retries: int = 2

    search_fit_max_retries: int = 2

    n_permutation_outer_seeds: int = 10
    n_permutation_inner_repeats: int = 30

    xgb_tree_method: str = field(
        default_factory=lambda: os.environ.get("NEGI_XGB_TREE_METHOD", "hist")
    )
    feature_names: tuple[str, ...] = tuple(FEATURES)
    monotone_constraints: tuple[int, ...] = DEFAULT_MONOTONE_CONSTRAINTS
    param_distributions: dict = field(default_factory=lambda: {
        "n_estimators":     [100, 200, 300, 500],
        "max_depth":        [2, 3, 4, 5, 6],
        "learning_rate":    [0.01, 0.03, 0.05, 0.1],
        "subsample":        [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "min_child_weight": [1, 3, 5],
        "gamma":            [0.0, 0.1, 0.3],
        "reg_alpha":        [0.0, 0.1, 1.0],
        "reg_lambda":       [1, 2, 5],
    })

    n_benchmark_search_iter: int = 80
    n_benchmark_search_iter_gb: int = 15
    rf_param_distributions: dict = field(default_factory=lambda: {
        "n_estimators":      [200, 300, 500],
        "max_depth":         [None, 6, 10, 16],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf":  [1, 2, 4],
        "max_features":      [1.0, "sqrt", "log2"],
    })
    gb_param_distributions: dict = field(default_factory=lambda: {
        "n_estimators":     [100, 200, 300, 500],
        "max_depth":        [2, 3, 4, 5],
        "learning_rate":    [0.01, 0.03, 0.05, 0.1],
        "subsample":        [0.7, 0.8, 0.9, 1.0],
        "min_samples_leaf": [1, 2, 4],
    })

    scenario_step: float = 0.0025
    ndvi_target_percentile: float = 0.95
    ndvi_bin_edges: int = 40
    ndvi_bin_min_count: int = 20
    ndbi_sweep_points: int = 200
    ndbi_sweep_quantile_low: float = 0.01
    ndbi_sweep_quantile_high: float = 0.99

    continuous_adjustment_ndbi_percentiles: tuple = (10, 25, 50, 75, 90)

    varying_coef_ndbi_df_grid: tuple = (3, 4, 5, 6, 7, 8, 10, 12, 15)
    varying_coef_elev_df: int = 5
    varying_coef_cv_folds: int = 10
    varying_coef_n_boot: int = 1000
    varying_coef_grid_points: int = 200
    varying_coef_grid_percentile_lo: float = 1.0
    varying_coef_grid_percentile_hi: float = 99.0
    varying_coef_support_window_halfwidth: float = 0.01
    varying_coef_low_support_blocks: int = 200
    varying_coef_sensitivity_df_grid: tuple = (3, 5, 6, 7, 8, 10, 12, 15)


    s2_archetype_ndbi_percentile: float = 0.15
    s2_search_min_bin_count: int = 30
    s2_meaningful_cooling_threshold_c: float = 0.10
    s2_trajectory_n_bins: int = 40

    s2_sensitivity_percentiles: tuple = tuple(
        float(p) for p in np.round(np.arange(0.05, 0.401, 0.025), 4)
    )
    run_s2_percentile_sensitivity: bool = True


    tree_jump_threshold_c: float = 0.10
    savgol_smooth_target_window_pct: float = 5.0

    alpha_weight: float = 1.0
    beta_weight: float = 1.0
    reference_w0: float = 200.0
    desal_energy_intensity: float = 3.5
    sqrt_exponent: float = 0.5
    linear_exponent: float = 1.0
    reference_cooling_scale_factor: float = 1.0

    support_knn_k: int = 5
    support_percentile: float = 95.0
    moran_k_neighbors: int = 8
    moran_n_permutations: int = 999
    moran_permutation_seed: int = 42

    FEATURE_SPARSITY_PERCENTILE: float = 80.0
    DENSITY_MODERATE_PERCENTILE: float = 50.0

    joint_support_percentile: float = 95.0

    spatial_block_cell_size_deg: float = 0.09
    spatial_block_merge_factors: tuple[int, ...] = (1, 2, 3, 4)
    spatial_block_min_groups: int = 10
    correlogram_n_pairs: int = 400_000
    correlogram_max_dist_km: float = 20.0
    correlogram_n_bins: int = 20
    correlogram_seed: int = 42

    uncertainty_percentiles: tuple[float, ...] = (2.5, 25.0, 50.0, 75.0, 97.5)

    saturation_high_fraction: float = 0.90
    saturation_mid_fraction: float = 0.50
    saturation_near_zero_fraction: float = 0.05

    sensitivity_exponents: tuple[float, ...] = (0.5, 1.0, 1.25, 1.5, 1.75)
    sensitivity_alpha_values: tuple[float, ...] = tuple(np.arange(0.4, 2.21, 0.2))
    sensitivity_beta_values: tuple[float, ...] = tuple(np.arange(0.4, 2.21, 0.2))
    sensitivity_w0_values: tuple[float, ...] = tuple(np.arange(50.0, 401.0, 50.0))
    sensitivity_default_exponent: float = 0.5
    sensitivity_default_beta: float = 1.0

    cost_regime_min_gap_pct: float = 10.0
    cost_regime_min_cluster_fraction: float = 0.05
    cost_regime_dominant_jump_fraction: float = 0.5
    cost_regime_boundary_search_points: int = 4001

    qq_envelope_percentiles: tuple[float, float] = (95.0, 99.0)
    low_r2_threshold: float = 0.10
    variance_band_very_weak: float = 0.05
    variance_band_weak: float = 0.15
    variance_band_moderate: float = 0.30

    moran_i_magnitude_threshold: float = 0.30
    scenario_boundary_tolerance_pct: float = 5.0
    trajectory_stability_neighbor_steps: int = 2
    trajectory_stability_jump_threshold: float = 0.05

    trajectory_stability_cv_window_steps: int = 4
    trajectory_stability_max_derivative: float = 0.05

    local_density_z_high: float = 0.0
    local_density_z_moderate: float = 1.0

    robustness_score_weights: dict = field(default_factory=lambda: {
        "ci": 0.35, "support": 0.25, "density": 0.15,
        "boundary": 0.10, "stability": 0.15,
    })
    robustness_score_threshold: float = 0.99

    color_primary: str = "#1B6FA8"
    color_secondary: str = "#4A96C8"
    color_accent: str = "#C84A4A"
    color_neutral: str = "#3B3B3B"
    color_s1: str = "#2A9D8F"
    color_s2: str = "#E76F51"

    default_figsize: tuple[float, float] = (10.0, 6.0)

    spatial_x_candidates: tuple[str, ...] = (
        "x", "X", "longitude", "Longitude", "lon", "Lon", "easting", "Easting"
    )
    spatial_y_candidates: tuple[str, ...] = (
        "y", "Y", "latitude", "Latitude", "lat", "Lat", "northing", "Northing"
    )

    results_dir:            Path = field(init=False)
    data_dir:               Path = field(init=False)
    main_png_dir:           Path = field(init=False)
    main_pdf_dir:           Path = field(init=False)
    supplementary_png_dir:  Path = field(init=False)
    supplementary_pdf_dir:  Path = field(init=False)

    fast_dev: bool = field(
        default_factory=lambda: os.environ.get("NEGI_FAST_DEV", "") not in ("", "0", "false", "False")
    )

    cache_enabled: bool = field(
        default_factory=lambda: os.environ.get("NEGI_CACHE", "1") not in ("0", "false", "False")
    )
    run_model_tuning:          bool = True
    run_nested_validation:     bool = True
    run_benchmarks:            bool = True
    run_permutation_importance: bool = True
    run_uncertainty:            bool = True
    run_bootstrap:               bool = True
    run_moran_permutations:      bool = True
    regenerate_figures:          bool = True

    adaptive_uncertainty_convergence: bool = True
    uncertainty_convergence_min_refits: int = 25
    uncertainty_convergence_check_interval: int = 25
    uncertainty_convergence_required_stable_checkpoints: int = 3
    adaptive_convergence_tolerance: float = 0.02

    run_uncertainty_convergence_diagnostic: bool = True
    uncertainty_convergence_refit_counts: tuple[int, ...] = (25, 50, 100, 200)
    uncertainty_convergence_tolerance: float = 0.05

    uncertainty_convergence_lookback_checkpoints: int = 8

    def __post_init__(self) -> None:
        self.base_dir = Path(self.base_dir)
        self.data_path = Path(self.data_path)

        if self.run_mode not in ("development", "publication"):
            raise ValueError(
                "Config.run_mode must be 'development' or 'publication', "
                f"got {self.run_mode!r}."
            )

        if self.export_mode not in ("standard", "debug"):
            raise ValueError(
                "Config.export_mode must be 'standard' or 'debug', "
                f"got {self.export_mode!r}."
            )

        validate_feature_configuration(self.monotone_constraints, self.feature_names)

        if self.fast_dev:
            warnings.warn(
                "Config.fast_dev=True (or NEGI_FAST_DEV set): tuning/refit/"
                "bootstrap/permutation budgets have been drastically reduced "
                "for fast iteration. Do NOT use this run's output as a "
                "manuscript-facing or scientific result.",
                stacklevel=2,
            )
            self.n_random_search_iter        = min(self.n_random_search_iter, 8)
            self.n_benchmark_search_iter     = min(self.n_benchmark_search_iter, 8)
            self.n_benchmark_search_iter_gb  = min(self.n_benchmark_search_iter_gb, 4)
            self.n_group_kfold_splits        = min(self.n_group_kfold_splits, 3)
            self.spatial_refit_mode          = "quick"
            self.n_spatial_refits            = None
            self.n_spatial_refits_quick      = min(self.n_spatial_refits_quick, 10)
            self.bootstrap_iterations        = min(self.bootstrap_iterations, 100)
            self.n_permutation_outer_seeds   = min(self.n_permutation_outer_seeds, 2)
            self.n_permutation_inner_repeats = min(self.n_permutation_inner_repeats, 5)
            self.n_spatial_holdout_repeats   = min(self.n_spatial_holdout_repeats, 3)
            self.moran_n_permutations        = min(self.moran_n_permutations, 99)

        if self.n_spatial_refits is None:
            if self.spatial_refit_mode not in ("quick", "publication"):
                raise ValueError(
                    "Config.spatial_refit_mode must be 'quick' or 'publication', "
                    f"got {self.spatial_refit_mode!r}."
                )
            self.n_spatial_refits = (
                self.n_spatial_refits_publication
                if self.spatial_refit_mode == "publication"
                else self.n_spatial_refits_quick
            )
        if self.n_spatial_refits < 30 and self.run_mode != "publication":
            warnings.warn(
                f"Config.n_spatial_refits={self.n_spatial_refits} is very sparse for "
                "estimating 2.5th/97.5th percentiles; consider "
                "spatial_refit_mode='publication' (>=100 refits) for manuscript output.",
                stacklevel=2,
            )
        self.results_dir           = self.base_dir / "Results"
        self.data_dir              = self.base_dir / "Data"
        self.main_png_dir          = self.results_dir / "Main" / "PNG"
        self.main_pdf_dir          = self.results_dir / "Main" / "PDF"
        self.supplementary_png_dir = self.results_dir / "Supplementary" / "PNG"
        self.supplementary_pdf_dir = self.results_dir / "Supplementary" / "PDF"


CFG = Config()

CACHE_SCHEMA_VERSION = 6

PIPELINE_VERSION = "1.2.0"

JSON_INDENT = 2

DEFAULT_TIMEOUT_SECONDS = 5

_CACHE_STAGE_STATUS: dict[str, str] = {}


def _get_git_hash(cfg: Config) -> str:
    """Best-effort short git commit hash for reproducibility reporting.

    Never raises: returns "unavailable" if git isn't installed, the code
    isn't in a git repository, or any other error occurs. This is metadata
    only and has no bearing on any computed or exported scientific result.
    """
    import subprocess
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(cfg.base_dir),
            capture_output=True, text=True, timeout=DEFAULT_TIMEOUT_SECONDS, check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "unavailable"


def _dataset_fingerprint(cfg: Config) -> str:
    """Content hash of the *effective* dataset that `load_dataset()` /
    `build_dataset_bundle()` will produce, so any change that alters the
    resulting DataFrame invalidates every cache entry derived from it.

    BUGFIX: this used to hash only the raw CSV bytes. But `load_dataset()`
    also reshapes the DataFrame based on `cfg.vegetation_purity_filter`
    (drops rows) and `cfg.spatial_block_cell_size_deg` (changes the
    `spatial_block` grouping used for every split downstream). Neither
    knob touches the CSV file, so toggling either one between runs left
    the fingerprint - and therefore every cache key built from it -
    unchanged. A stale `xgb_tuning`/`nested_cv`/... cache entry built
    under the old setting would then be reused with a DatasetBundle whose
    row count doesn't match the freshly loaded `data` in main(), e.g.
    `data.y_test` (fresh) vs `validation.holdout_pred` (from the stale
    cached model's own DatasetBundle) ending up with different lengths -
    exactly the
        ValueError: operands could not be broadcast together with shapes (6782,) (5867,)
    raised from `compute_residual_diagnostics()`. Any config field that
    changes which/how many rows end up in the modelled DataFrame must be
    included here, not just the file bytes.

    Falls back to "unknown" (never raises) if the file can't be read here -
    a stage will still run correctly, it just won't be cacheable.
    """
    try:
        with open(cfg.data_path, "rb") as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()[:16]
    except OSError:
        file_hash = "unknown"
    effective = {
        "file_hash": file_hash,
        "vegetation_purity_filter": cfg.vegetation_purity_filter,
        "spatial_block_cell_size_deg": cfg.spatial_block_cell_size_deg,
        "features": list(FEATURES),
    }
    blob = json.dumps(effective, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def _cache_key(stage: str, cfg: Config, **extra) -> str:
    """Content-addressed cache key for `stage`. `extra` should include
    every stage-specific input that could change the result: the
    relevant Config fields (not the whole Config - unrelated fields
    like figure DPI must not cause spurious cache misses on model-fit
    stages), random seeds, hyperparameter grids, and feature/column
    names. json.dumps(..., default=str) keeps this robust to tuples,
    Paths, and numpy scalars without needing a custom encoder."""
    payload = {
        "stage": stage,
        "schema_version": CACHE_SCHEMA_VERSION,
        "dataset_fingerprint": _dataset_fingerprint(cfg),
        "feature_set": list(FEATURES),
        "monotone_constraints_schema": list(cfg.monotone_constraints),
        **extra,
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:24]


def _cache_path(cfg: Config, stage: str, key: str) -> Path:
    cache_dir = cfg.base_dir / ".negi_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{stage}__{key}.pkl"


def cache_load(cfg: Config, stage: str, key: str):
    """Return the cached object for (stage, key), or None on any miss.
    A corrupt/unreadable cache file is treated as a miss (logged, never
    raised) - it just gets recomputed and overwritten below."""
    if not cfg.cache_enabled:
        return None
    path = _cache_path(cfg, stage, key)
    if not path.is_file():
        return None
    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        log_info(f"  [cache] hit  '{stage}' -> {path.name}")
        return obj
    except Exception as exc:  # noqa: BLE001
        log_warning(f"  [cache] unreadable cache for '{stage}' ({exc}); recomputing.")
        return None


def cache_save(cfg: Config, stage: str, key: str, obj) -> None:
    """Best-effort save; a failed write never breaks the pipeline (it
    just means the next run recomputes this stage instead of loading it).
    Writes to a temp file and renames into place so a crash mid-write
    can never leave a half-written, falsely-"valid" cache file behind."""
    if not cfg.cache_enabled:
        return
    path = _cache_path(cfg, stage, key)
    tmp_path = path.with_suffix(".tmp")
    try:
        with open(tmp_path, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp_path.replace(path)
        log_info(f"  [cache] saved '{stage}' -> {path.name}")
    except Exception as exc:  # noqa: BLE001
        log_warning(f"  [cache] failed to save cache for '{stage}' ({exc}).")


def run_or_cached_stage(
    cfg: Config, stage: str, run_flag: bool, key: str, compute_fn: Callable[[], object],
    step_label: Optional[str] = None,
):
    """Run `compute_fn()`, or return its cached result for the exact same
    inputs (see `key`, from `_cache_key`).

    Precedence (item 3's "dependencies must be handled safely"):
    1. A valid cache entry for this exact key is ALWAYS used when present,
       regardless of `run_flag` - a cache hit is provably the same result
       a fresh run would give, so reusing it is never a scientific choice.
    2. If there's no cache entry and `run_flag` is True (the default),
       compute fresh and cache the result.
    3. If there's no cache entry and `run_flag` is False, the stage was
       explicitly marked "skip" for a dev run - but it CANNOT actually be
       skipped, because a downstream stage needs a real result and none
       exists yet. It runs anyway (with a warning) and is cached so the
       *next* run honours the skip request.

    Reporting only (engineering-refinement item 1/7): if `step_label` is
    given (the same text used for that stage's `log_step(...)` header),
    this explicitly states whether the stage was restored from cache or
    computed fresh, followed by the elapsed wall-clock time, e.g.:

        [STEP 2] XGBoost tuning (cache hit)
        Elapsed: 0.1 s

    or

        [STEP 2] XGBoost tuning (computed)
        Elapsed: 42.7 s

    This never affects which branch above runs or what is returned - it
    only changes what gets printed.
    """
    _t0 = time.monotonic()
    cached = cache_load(cfg, stage, key)
    if cached is not None:
        _CACHE_STAGE_STATUS[stage] = "cache hit"
        if step_label:
            report_development(cfg, f"{step_label} (cache hit)", level="headline")
            report_development(
                cfg, f"Elapsed: {time.monotonic() - _t0:.1f} s", level="headline"
            )
            _STEP_TIMER["suppress_next_report"] = True
        return cached
    if not run_flag:
        report_development(
            cfg,
            f"\n[{stage}] requested skip (run_flag=False) but no cached "
            "result exists yet for these inputs - running it once now "
            "(a stage can't be silently skipped when later stages depend "
            "on its result); it will be cached for next time.",
            level="warning",
        )
    result = compute_fn()
    cache_save(cfg, stage, key, result)
    _CACHE_STAGE_STATUS[stage] = "computed"
    if step_label:
        report_development(cfg, f"{step_label} (computed)", level="headline")
        report_development(
            cfg, f"Elapsed: {time.monotonic() - _t0:.1f} s", level="headline"
        )
        _STEP_TIMER["suppress_next_report"] = True
    return result


EPS            = 1e-12

TEMP_ZERO_GUARD = 1e-9

SCENARIO_AXIS_LABEL = "Scenario intensity (%)"


FIG_ACTUAL_VS_PREDICTED         = "Figure_01_Observed_vs_Predicted.png"
FIG_MODEL_COMPARISON            = "Figure_S01_Model_Comparison.png"
FIG_FEATURE_IMPORTANCE          = "Figure_02_Feature_Importance.png"
FIG_LST_RESPONSE_TO_NDBI        = "Figure_03_LST_Response_to_NDBI.png"
FIG_NEGI_SCENARIO_COMPARISON    = "Figure_04_NEGI_Scenario_Comparison.png"
FIG_NEGI_UNCERTAINTY_BANDS      = "Figure_05_NEGI_Uncertainty_Bands.png"
FIG_NDVI_CONDITIONAL_BY_NDBI    = "Figure_06_NDVI_Adjusted_Slope_by_NDBI.png"
FIG_NDVI_UNADJUSTED_STRATUM_SUPPLEMENT = "Figure_S19_Unadjusted_Within_NDBI_Stratum.png"
FIG_NDVI_VS_LST                 = "Figure_S02_NDVI_vs_LST.png"
FIG_NDVI_DECILE_ANALYSIS        = "Figure_S03_NDVI_Decile_Analysis.png"
FIG_NDBI_RESPONSE_UNCERTAINTY   = "Figure_S04_NDBI_Response_Uncertainty.png"
FIG_RESIDUALS_VS_PREDICTED      = "Figure_S05_Residuals_vs_Predicted.png"
FIG_RESIDUALS_QQ                = "Figure_S06_Residuals_QQ.png"
FIG_RESIDUALS_HISTOGRAM         = "Figure_S07_Residuals_Histogram.png"
FIG_RESIDUAL_SPATIAL_MAP        = "Figure_S08_Residual_Spatial_Map.png"
FIG_NEGI_SMOOTHING_ROBUSTNESS   = "Figure_S09_NEGI_Smoothing_Robustness.png"
FIG_NEGI_EXPONENT_SENSITIVITY   = "Figure_S10_NEGI_Exponent_Sensitivity.png"
FIG_NEGI_ENERGY_COST_COMPARISON = "Figure_S11_NEGI_Energy_Cost_Comparison.png"
FIG_RESPONSE_SURFACE            = "Figure_S12_Response_Surface.png"
FIG_SCENARIO2_DATA_SUPPORT      = "Figure_S13_Scenario2_Data_Support.png"
FIG_NEGI_SCENARIO2_ROBUSTNESS   = "Figure_S14_NEGI_Scenario2_Robustness.png"
FIG_SENSITIVITY_MAXIMUM_EVALUATED_NEGI = "Figure_S15_Sensitivity_Maximum_Evaluated_NEGI.png"
FIG_SENSITIVITY_OPTIMAL_NEGI    = FIG_SENSITIVITY_MAXIMUM_EVALUATED_NEGI
FIG_COOLING_SATURATION          = "Figure_S16_Cooling_Saturation_Diagnostic.png"
FIG_UNCERTAINTY_CONVERGENCE     = "Figure_S17_Uncertainty_Convergence.png"
FIG_NDVI_CONTINUOUS_ADJUSTMENT  = "Figure_S18_NDVI_LST_Continuous_NDBI_Adjustment.png"
FIG_SCENARIO2_PERCENTILE_SENSITIVITY = "Scenario2_percentile_sensitivity.png"
FIG_DECISION_ENVELOPE                = "Figure_S20_Decision_Envelope.png"
FIG_COST_REGIME_BOUNDARY             = "Figure_S21_Cost_Regime_Boundary.png"

MAIN_FIGURE_FILENAMES: frozenset[str] = frozenset({
    FIG_ACTUAL_VS_PREDICTED,
    FIG_FEATURE_IMPORTANCE,
    FIG_LST_RESPONSE_TO_NDBI,
    FIG_NEGI_SCENARIO_COMPARISON,
    FIG_NEGI_UNCERTAINTY_BANDS,
    FIG_NDVI_CONDITIONAL_BY_NDBI,
})

TITLE_SIZE   = 16
TITLE_WEIGHT = "normal" 
LABEL_SIZE   = 13
LABEL_WEIGHT = "normal"
TICK_SIZE    = 10
LEGEND_SIZE  = 11
LINE_WIDTH   = 2.0
MARKER_SIZE  = 8
GRID_ALPHA   = 0.28

STAT_BOX_STYLE: dict = {
    "boxstyle": "round,pad=0.35",
    "facecolor": "white",
    "edgecolor": "#666666",
    "alpha": 0.85,
    "linewidth": 1.0,
}


logger = logging.getLogger("negi_framework")
_LOG_FILE_PATH: Optional[Path] = None

HEADLINE_LEVEL = 25
logging.addLevelName(HEADLINE_LEVEL, "HEADLINE")


class _ConsoleFormatter(logging.Formatter):
    """Level-aware console formatting (presentation only).

    HEADLINE-level records (the principal stages/metrics that must read as
    an executive scientific summary - item 18) are printed as plain text,
    with no timestamp or level tag, so concise-mode output doesn't look
    like a debugging log. WARNING/ERROR records keep a plain textual
    prefix so genuine warnings/errors stay visually distinguishable from
    headline content. INFO/DEBUG records (only ever reach the console when
    cfg.verbose is True) keep the original timestamped '[LEVELNAME]' style,
    since that stream is explicitly the detailed, developer-facing log.
    This only changes how already-selected records are displayed; it does
    not change which records are selected (that's still handler level) or
    any computed value.
    """

    _detailed = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
    )

    def format(self, record: logging.LogRecord) -> str:
        if record.levelno == HEADLINE_LEVEL:
            return record.getMessage()
        if record.levelno >= logging.WARNING:
            return f"{record.levelname}: {record.getMessage()}"
        return self._detailed.format(record)


def configure_logging(cfg: Config) -> None:
    """Configure the module logger once, at pipeline start.

    Console handler level follows cfg.verbose: INFO (show every detailed
    message) when verbose, or HEADLINE (show only headline-level messages,
    plus warnings/errors) when quiet. Console formatting is level-aware
    (see _ConsoleFormatter): HEADLINE lines print as plain text so concise
    mode reads as an executive summary, not a timestamped debug log.
    A file handler always captures INFO/WARNING/ERROR to data/pipeline.log
    with full timestamped detail, and DEBUG as well when cfg.debug is True
    - full diagnostic detail is therefore always preserved in the log file
    regardless of console verbosity or console formatting.
    Idempotent - safe to call multiple times; clears previous handlers to
    avoid duplicate log lines.
    """
    global _LOG_FILE_PATH
    logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(_ConsoleFormatter())
    console_handler.setLevel(logging.INFO if cfg.verbose else HEADLINE_LEVEL)
    logger.addHandler(console_handler)

    try:
        cfg.data_dir.mkdir(parents=True, exist_ok=True)
        log_path = cfg.data_dir / "pipeline.log"
        file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        file_handler.setLevel(logging.DEBUG if cfg.debug else logging.INFO)
        logger.addHandler(file_handler)
        _LOG_FILE_PATH = log_path
    except OSError as exc:
        warnings.warn(f"Could not attach file log handler (pipeline.log): {exc}")

    logger.setLevel(logging.DEBUG)
    logger.propagate = False


def _assert_console_verbosity_invariant(cfg: Config) -> bool:
    """Structurally inspect the REAL configured handlers (not a synthetic
    probe) after configure_logging() has run, and verify the concise mode
    invariant holds:

        cfg.verbose is False  -> no console-writing StreamHandler reachable
                                  from the application logger (directly, via
                                  the root logger, or via any other
                                  instantiated logger) has a level low
                                  enough to admit an ordinary INFO (20)
                                  record.
        cfg.verbose is True   -> at least the app logger's own console
                                  handler admits INFO.

    This does not emit any log records itself (so it can't pollute
    console output) - it only reads handler.level / handler.stream off
    the handler objects actually attached at runtime. It also walks
    logging.Logger.manager.loggerDict so a stray handler attached to some
    other, unexpectedly-named logger elsewhere in the codebase or a
    dependency would still be caught.
    """
    problems: list[str] = []

    def _check(lg: logging.Logger) -> None:
        for h in lg.handlers:
            if not isinstance(h, logging.StreamHandler) or isinstance(h, logging.FileHandler):
                continue
            stream = getattr(h, "stream", None)
            if stream not in (sys.stdout, sys.stderr):
                continue
            if not cfg.verbose and h.level < HEADLINE_LEVEL:
                problems.append(
                    f"logger={lg.name!r} handler={type(h).__name__} "
                    f"level={h.level} stream={stream} "
                    f"(admits INFO while cfg.verbose=False)"
                )
            if cfg.verbose and lg is logger and h.level > logging.INFO:
                problems.append(
                    f"logger={lg.name!r} handler={type(h).__name__} "
                    f"level={h.level} stream={stream} "
                    f"(blocks INFO while cfg.verbose=True)"
                )

    _check(logger)
    _check(logging.getLogger())
    for name, lg in list(logging.Logger.manager.loggerDict.items()):
        if isinstance(lg, logging.PlaceHolder):
            continue
        if lg is logger:
            continue
        if lg.handlers:
            _check(lg)

    if problems:
        for p in problems:
            logger.warning("Console verbosity invariant violation: %s", p)
        return False
    return True


def log_info(*args) -> None:
    """Log `args` (space-joined) at INFO level."""
    logger.info(" ".join(str(a) for a in args))


def log_warning(*args) -> None:
    """Log `args` (space-joined) at WARNING level."""
    logger.warning(" ".join(str(a) for a in args))


def log_debug(*args) -> None:
    """Log `args` (space-joined) at DEBUG level (developer-only)."""
    logger.debug(" ".join(str(a) for a in args))


def log_headline(*args) -> None:
    """Log `args` (space-joined) at HEADLINE level.

    Use for the principal methodological stages, headline metrics,
    important scientific diagnostics, final maximum classification, QA
    summary, and runtime - the console content that must remain visible
    even when cfg.verbose is False. Everything else should use log_info
    (detailed, verbose-only on console) so it still reaches pipeline.log.
    """
    logger.log(HEADLINE_LEVEL, " ".join(str(a) for a in args))


_STEP_TIMER: dict = {"last_time": None, "last_label": None, "suppress_next_report": False}


def log_step(label: str) -> None:
    """Console step header. Prints how long the *previous* step took
    (wall-clock, via time.monotonic) before starting the new one -
    unless that previous step already reported its own elapsed time
    explicitly (see `_STEP_TIMER["suppress_next_report"]`), in which case
    this is skipped to avoid a duplicate timing line."""
    now = time.monotonic()
    if _STEP_TIMER["last_time"] is not None and not _STEP_TIMER["suppress_next_report"]:
        elapsed = now - _STEP_TIMER["last_time"]
        log_headline(f"  -> '{_STEP_TIMER['last_label']}' took {elapsed:.1f}s")
    _STEP_TIMER["last_time"]  = now
    _STEP_TIMER["last_label"] = label
    _STEP_TIMER["suppress_next_report"] = False
    log_info(f"\n{label}")


def log_step_finish() -> None:
    """Report elapsed time for the final step. Call once, right before
    the total-runtime line at the end of main(). Skipped if that final
    step already reported its own elapsed time explicitly (see
    `log_step`'s docstring)."""
    if _STEP_TIMER["last_time"] is not None and not _STEP_TIMER["suppress_next_report"]:
        elapsed = time.monotonic() - _STEP_TIMER["last_time"]
        log_headline(f"  -> '{_STEP_TIMER['last_label']}' took {elapsed:.1f}s")


def vlog_info(cfg: Config, *args) -> None:
    """Log only when cfg.verbose is True."""
    if cfg.verbose:
        log_info(*args)




def report_step(label: str) -> None:
    """Report a scientific-pipeline stage boundary (e.g. "[STEP 11] ...").

    Inputs
    ------
    label : the step header text to display.

    Side effects
    ------------
    Prints the previous step's elapsed time (if not already reported) and
    logs the new step header, via the existing log_step() primitive.
    Always shown in both "development" and "publication" run modes - stage
    boundaries are pipeline structure, not an engineering diagnostic.

    Returns
    -------
    None.
    """
    log_step(label)


def report_warning(message: str) -> None:
    """Report a scientific/QA warning that must be visible regardless of
    run_mode (e.g. a QA check failure, a data-quality concern).

    Inputs
    ------
    message : the warning text.

    Side effects
    ------------
    Logs `message` at WARNING level via the existing log_warning()
    primitive, unconditionally.

    Returns
    -------
    None.
    """
    log_warning(message)


def report_development(cfg: Config, message: str, level: str = "info") -> None:
    """Report an engineering/developer-only diagnostic, gated to
    "development" run mode.

    Use this for cache hit/miss status, elapsed cache/stage timings,
    uncertainty-convergence sweep detail, and configuration sanity
    warnings - anything that documents *how* the pipeline ran rather than
    *what it scientifically found*.

    Inputs
    ------
    cfg     : the active Config (only cfg.run_mode is consulted).
    message : the diagnostic text.
    level   : which underlying logger primitive to use when the message is
              shown - "info" (log_info, detail-only unless cfg.verbose),
              "headline" (log_headline, visible even without cfg.verbose),
              or "warning" (log_warning, always visible on console).
              Default "info".

    Side effects
    ------------
    When cfg.run_mode == "development": logs `message` via the primitive
    named by `level`. When cfg.run_mode == "publication": does nothing -
    the message never reaches the console/log in publication mode. This
    function never writes to any JSON/report export; callers that need the
    same detail preserved for run_manifest.json / qa_summary.json /
    uncertainty_convergence.json / the reproducibility report must write
    that separately (those exports are unconditional, independent of this
    gate).

    Returns
    -------
    None.
    """
    if cfg.run_mode != "development":
        return
    if level == "headline":
        log_headline(message)
    elif level == "warning":
        log_warning(message)
    else:
        log_info(message)


def format_p_value(p: float) -> str:
    """Return a presentation-safe 'p ...' clause for a p-value.

    When `p` has underflowed to exactly 0.0 in double precision (e.g. a
    normal-approximation two-sided p-value 2*(1 - norm.cdf(|z|)) rounding
    to 0.0 for very large |z|), state an accurate bounded value instead of
    the literal 'p = 0', which misrepresents a continuous statistic as
    exactly zero. This changes presentation only - the underlying computed
    p-value and z-score are never altered.
    """
    if np.isfinite(p) and p == 0.0:
        return f"p < {2 * np.finfo(float).eps:.1e} (underflow at machine precision)"
    return f"p = {p:.4g}"


def _float_arrays(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)


def isclose_mask(series: pd.Series, value: float, tol: float = 1e-6) -> np.ndarray:
    """Boolean mask of `series` entries within `tol` of `value`."""
    return np.isclose(series.to_numpy(dtype=float), value, atol=tol)


def safe_divide(numerator, denominator, fill: float = 0.0):
    """Division that returns `fill` wherever |denominator| <= EPS."""
    num, den = np.broadcast_arrays(
        np.asarray(numerator, dtype=float), np.asarray(denominator, dtype=float)
    )
    result = np.full(num.shape, fill, dtype=float)
    np.divide(num, den, out=result, where=np.abs(den) > EPS)
    return float(result) if result.ndim == 0 else result


def assert_finite(values, name: str = "values") -> None:
    """Raise ValueError if `values` contains any NaN or infinite entry."""
    if not np.all(np.isfinite(np.asarray(values, dtype=float))):
        raise ValueError(f"{name} contains NaN or infinite values.")


def ci95(values) -> float:
    """Half-width of the normal-approximation 95% CI for the mean of `values`."""
    values_arr = np.asarray(values, dtype=float)
    if values_arr.size < 2:
        return 0.0
    return float(1.96 * values_arr.std(ddof=0) / np.sqrt(values_arr.size))


def compute_rmse(y_true, y_pred) -> float:
    """Root-mean-squared error between `y_true` and `y_pred`."""
    y_true_a, y_pred_a = _float_arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true_a - y_pred_a) ** 2)))


def compute_mae(y_true, y_pred) -> float:
    """Mean absolute error between `y_true` and `y_pred`."""
    y_true_a, y_pred_a = _float_arrays(y_true, y_pred)
    return float(np.mean(np.abs(y_true_a - y_pred_a)))


def compute_bias(y_true, y_pred) -> float:
    """Mean signed bias (`y_true` minus `y_pred`)."""
    y_true_a, y_pred_a = _float_arrays(y_true, y_pred)
    return float(np.mean(y_true_a - y_pred_a))


def compute_r2(y_true, y_pred) -> float:
    """Coefficient of determination (R²) between `y_true` and `y_pred`."""
    y_true_a, y_pred_a = _float_arrays(y_true, y_pred)
    return float(r2_score(y_true_a, y_pred_a))


def compute_calibration(y_true, y_pred) -> tuple[float, float]:
    """Calibration slope and intercept from a linear fit of `y_true` on `y_pred`.

    Returns:
        (slope, intercept); (nan, nan) if there are fewer than two points or
        `y_pred` is constant.
    """
    y_true_a, y_pred_a = _float_arrays(y_true, y_pred)
    if y_true_a.size < 2 or np.ptp(y_pred_a) <= EPS:
        return float("nan"), float("nan")
    slope, intercept = np.polyfit(y_pred_a, y_true_a, 1)
    return float(slope), float(intercept)


def compute_pearson_r(y_true, y_pred) -> tuple[float, float]:
    """Pearson correlation coefficient and p-value between `y_true` and `y_pred`.

    Returns:
        (r, p_value); (nan, nan) if there are fewer than two points or either
        array is constant.
    """
    y_true_a, y_pred_a = _float_arrays(y_true, y_pred)
    if y_true_a.size < 2 or np.ptp(y_true_a) <= EPS or np.ptp(y_pred_a) <= EPS:
        return float("nan"), float("nan")
    result = scipy_stats.pearsonr(y_true_a, y_pred_a)
    return float(result.statistic), float(result.pvalue)


def compute_mean_absolute_calibration_error(y_true, y_pred) -> float:
    """Mean absolute calibration error (equivalent to MAE on the same pair)."""
    return compute_mae(y_true, y_pred)


def compute_residual_stats(residuals) -> dict:
    """Summary statistics (mean, median, std, skewness, kurtosis, Shapiro p) for residuals."""
    arr = np.asarray(residuals, dtype=float)
    shapiro_p = (
        float(scipy_stats.shapiro(arr).pvalue)
        if 3 <= arr.size <= 5000
        else float("nan")
    )
    return {
        "mean":     float(arr.mean()),
        "median":   float(np.median(arr)),
        "std":      float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "skewness": float(scipy_stats.skew(arr)),
        "kurtosis": float(scipy_stats.kurtosis(arr)),
        "shapiro_p": shapiro_p,
    }


def compute_metrics(y_true, y_pred) -> dict:
    """Bundle RMSE, MAE, bias, R², and calibration slope/intercept into one dict."""
    slope, intercept = compute_calibration(y_true, y_pred)
    return {
        "rmse":                 compute_rmse(y_true, y_pred),
        "mae":                  compute_mae(y_true, y_pred),
        "bias":                 compute_bias(y_true, y_pred),
        "r2":                   compute_r2(y_true, y_pred),
        "calibration_slope":    slope,
        "calibration_intercept": intercept,
    }


def _metric_rmse(y_true, y_pred) -> float:              return compute_rmse(y_true, y_pred)
def _metric_mae(y_true, y_pred) -> float:               return compute_mae(y_true, y_pred)
def _metric_bias(y_true, y_pred) -> float:              return compute_bias(y_true, y_pred)
def _metric_calibration_slope(y_true, y_pred) -> float: return compute_calibration(y_true, y_pred)[0]
def _metric_calibration_intercept(y_true, y_pred) -> float: return compute_calibration(y_true, y_pred)[1]
def _metric_r2(y_true, y_pred) -> float:                return compute_r2(y_true, y_pred)


def bootstrap_metric(
    y_true, y_pred, metric_fn: Callable, n_boot: int = 1000, seed: int = 42
) -> dict:
    """Bootstrap 95% CI for a scalar metric function."""
    y_true_a, y_pred_a = _float_arrays(y_true, y_pred)
    if y_true_a.size == 0:
        raise ValueError("Cannot bootstrap an empty array.")
    rng = np.random.default_rng(seed)
    samples = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, y_true_a.size, size=y_true_a.size)
        samples[i] = metric_fn(y_true_a[idx], y_pred_a[idx])
    return {
        "point_estimate": float(metric_fn(y_true_a, y_pred_a)),
        "median":  float(np.nanmedian(samples)),
        "lower":   float(np.nanpercentile(samples, 2.5)),
        "upper":   float(np.nanpercentile(samples, 97.5)),
        "n_boot":  int(n_boot),
    }


def spatial_block_bootstrap_metrics(
    y_true, y_pred, block_ids, metric_fns: dict[str, Callable],
    n_boot: int = 1000, seed: int = 42,
) -> dict[str, dict]:
    """Spatial-block bootstrap 95% CIs for one or more scalar metric functions.

    Resamples whole spatial blocks (the same ``spatial_block`` grouping used
    elsewhere in the pipeline for GroupKFold / GroupShuffleSplit) with
    replacement, rather than resampling individual holdout observations.
    Holdout residuals retain non-trivial spatial autocorrelation (see the
    Moran's I diagnostic), so block-level resampling better respects the
    dependence structure than i.i.d. observation-level resampling.

    Each bootstrap iteration samples the unique blocks present in the
    holdout set *with replacement* and includes every observation
    belonging to each selected block (preserving multiplicity when a
    block is drawn more than once).

    Returns a dict keyed by metric name, each value a dict with
    point_estimate / median / lower / upper / n_boot / n_unique_blocks.
    """
    y_true_a, y_pred_a = _float_arrays(y_true, y_pred)
    block_ids = np.asarray(block_ids)
    if y_true_a.size == 0:
        raise ValueError("Cannot bootstrap an empty array.")
    if len(block_ids) != y_true_a.size:
        raise ValueError("block_ids must have the same length as y_true/y_pred.")

    unique_blocks = np.unique(block_ids)
    n_unique_blocks = len(unique_blocks)

    block_to_positions = {
        b: np.where(block_ids == b)[0] for b in unique_blocks
    }

    rng = np.random.default_rng(seed)
    samples = {name: np.empty(n_boot, dtype=float) for name in metric_fns}

    for i in range(n_boot):
        sampled_blocks = rng.choice(unique_blocks, size=n_unique_blocks, replace=True)
        positions = np.concatenate([block_to_positions[b] for b in sampled_blocks])
        yt_boot, yp_boot = y_true_a[positions], y_pred_a[positions]
        for name, fn in metric_fns.items():
            samples[name][i] = fn(yt_boot, yp_boot)

    results = {}
    for name, fn in metric_fns.items():
        results[name] = {
            "point_estimate":  float(fn(y_true_a, y_pred_a)),
            "median":          float(np.nanmedian(samples[name])),
            "lower":           float(np.nanpercentile(samples[name], 2.5)),
            "upper":           float(np.nanpercentile(samples[name], 97.5)),
            "n_boot":          int(n_boot),
            "n_unique_blocks": int(n_unique_blocks),
        }
    return results


def apply_plot_style(cfg: Config = CFG) -> None:
    """Apply consistent rcParams for publication-quality figures."""
    mpl.rcParams.update({
        "figure.dpi":          120,
        "savefig.dpi":         300,
        "font.size":           TICK_SIZE,
        "axes.titlesize":      TITLE_SIZE,
        "axes.labelsize":      LABEL_SIZE,
        "legend.fontsize":     LEGEND_SIZE,
        "axes.spines.top":     False,
        "axes.spines.right":   False,
    })


apply_publication_style = apply_plot_style


def set_title(ax, title: str, *, fontsize: int = TITLE_SIZE,
              fontweight: str = TITLE_WEIGHT, **kw) -> None:
    """Set axis title with consistent size and weight (normal by default)."""
    ax.set_title(title, fontsize=fontsize, fontweight=fontweight, **kw)


def set_axis_labels(
    ax, xlabel: str = "", ylabel: str = "",
    *, fontsize: int = LABEL_SIZE, fontweight: str = LABEL_WEIGHT,
) -> None:
    """Set x- and y-axis labels with consistent typography."""
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=fontsize, fontweight=fontweight)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=fontsize, fontweight=fontweight)


def set_legend(ax, **kw) -> None:
    """Add a legend with consistent size and frame."""
    defaults = dict(fontsize=LEGEND_SIZE, frameon=True)
    defaults.update(kw)
    ax.legend(**defaults)


def stat_box(ax, text: str, x: float = 0.05, y: float = 0.97, **kw) -> None:
    """Render a statistics annotation box with the standard house style."""
    defaults = dict(
        transform=ax.transAxes, verticalalignment="top",
        fontsize=TICK_SIZE, bbox=STAT_BOX_STYLE,
    )
    defaults.update(kw)
    ax.text(x, y, text, **defaults)


def apply_grid(ax, alpha: float = GRID_ALPHA) -> None:
    """Apply a light grid and remove top/right spines consistently."""
    ax.grid(True, alpha=alpha)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def figure_footer(fig, text: str, cfg: Config = CFG, *, min_bottom: float = 0.16) -> None:
    """Add a small-font footer line at the bottom of a figure.

    Stores the required bottom margin on the figure object so that
    `save_fig` can re-apply it after `tight_layout()` - tight_layout only
    accounts for axes content, not free-floating `fig.text` footers, so
    without this the footer margin gets silently discarded and the footer
    text collides with the x-axis label/ticks.
    """
    fig.text(
        0.5, 0.01, text,
        ha="center", va="bottom",
        fontsize=7.5, color=cfg.color_neutral, wrap=True,
    )
    fig.subplots_adjust(bottom=max(fig.subplotpars.bottom, min_bottom))
    fig._footer_min_bottom = min_bottom


def generate_caption(description: str, statistics: Optional[dict] = None) -> str:
    """Build a figure caption string, optionally appending key statistics."""
    if not statistics:
        return description
    values = "; ".join(
        f"{k}={v:.4g}" if isinstance(v, (int, float, np.number)) and np.isfinite(v)
        else f"{k}={v}"
        for k, v in statistics.items()
    )
    return f"{description}. {values}" if values else description


class FigureRegistry:
    """Tracks every figure saved during a pipeline run for the figure list."""

    def __init__(self) -> None:
        self._rows: list[dict] = []

    def add(
        self, filename: str, caption: str = "",
        description: str = "", section: str = "",
    ) -> None:
        """Register one figure/table export with its caption, description, and section."""
        self._rows.append({
            "filename":    filename,
            "caption":     caption,
            "description": description,
            "section":     section,
        })

    def export(self, path: Path) -> Path:
        """Write the accumulated figure/table registry to a CSV at `path`."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            self._rows, columns=["filename", "caption", "description", "section"]
        ).to_csv(path, index=False)
        return path


FIGURE_REGISTRY = FigureRegistry()


def save_fig(
    fig,
    filename: str,
    cfg: Config = CFG,
    *,
    use_tight_layout: bool = True,
    caption: str = "",
    description: str = "",
    section: str = "",
) -> Path:
    """Save a figure to the correct Results subfolder as PNG and PDF.

    Routing logic
    -------------
    If `filename` is in MAIN_FIGURE_FILENAMES  -> Results/Main/{PNG,PDF}
    Otherwise                                  -> Results/Supplementary/{PNG,PDF}

    Parameters
    ----------
    fig : matplotlib Figure
    filename : str
        Canonical figure filename (e.g. FIG_ACTUAL_VS_PREDICTED).
    cfg : Config
    use_tight_layout : bool
        Call fig.tight_layout() before saving (default True).
    caption / description / section : str
        Metadata registered in FIGURE_REGISTRY.

    Returns
    -------
    Path
        Path of the saved PNG file.
    """
    filename_path = Path(filename)
    if not filename_path.suffix:
        filename_path = filename_path.with_suffix(".png")

    is_main = filename_path.name in MAIN_FIGURE_FILENAMES
    png_dir = cfg.main_png_dir if is_main else cfg.supplementary_png_dir
    pdf_dir = cfg.main_pdf_dir if is_main else cfg.supplementary_pdf_dir
    png_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)

    png_path = png_dir / filename_path.name
    pdf_path = pdf_dir / filename_path.with_suffix(".pdf").name

    if cfg.regenerate_figures or not png_path.is_file():
        if use_tight_layout:
            try:
                fig.tight_layout()
            except (ValueError, RuntimeError):
                pass
            footer_min_bottom = getattr(fig, "_footer_min_bottom", None)
            if footer_min_bottom is not None:
                fig.subplots_adjust(bottom=max(fig.subplotpars.bottom, footer_min_bottom))
        fig.savefig(png_path, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    FIGURE_REGISTRY.add(
        filename=png_path.name,
        caption=caption,
        description=description,
        section=section,
    )
    log_info(f"Saved: {png_path}")
    return png_path


export_fig = save_fig


def setup_directories(cfg: Config) -> None:
    """Create all output directories (idempotent)."""
    for d in (
        cfg.data_dir,
        cfg.main_png_dir, cfg.main_pdf_dir,
        cfg.supplementary_png_dir, cfg.supplementary_pdf_dir,
    ):
        d.mkdir(parents=True, exist_ok=True)


def confirm_saved(path: Path) -> None:
    """Log confirmation that a file was written to `path`."""
    log_info(f"Saved: {path}")


def save_csv(
    df: pd.DataFrame, path: Path, index: bool = False,
    *, cfg: Optional["Config"] = None, debug_only: bool = False,
) -> Optional[Path]:
    """Write `df` to `path` as CSV and log the confirmation.

    Export-mode gating (publication-readiness refactor): pass
    `debug_only=True` (with `cfg`) for developer/intermediate/checkpoint
    CSVs that should only reach disk when `cfg.export_mode == "debug"`.
    This never skips the CALCULATION that produced `df` - callers still
    build the DataFrame exactly as before; only the disk write is gated,
    so every downstream log line, figure, or summary-report row that
    reads `df` in memory is completely unaffected. When `debug_only=False`
    (the default) or `cfg` is None, behaviour is identical to before this
    refactor: the file is always written.
    """
    if debug_only and cfg is not None and cfg.export_mode != "debug":
        log_info(f"(export_mode='standard': skipped debug-only export {path.name})")
        return None
    df.to_csv(path, index=index)
    confirm_saved(path)
    return path


def save_json(
    data: dict, path: Path,
    *, cfg: Optional["Config"] = None, debug_only: bool = False,
) -> Optional[Path]:
    """Unified JSON export helper (item 6) - mirrors save_csv's behaviour
    so every JSON artifact in the pipeline goes through one function.

    Export-mode gating: see save_csv docstring - identical semantics,
    `debug_only=True` (with `cfg`) restricts this file to
    `cfg.export_mode == "debug"` runs without touching how `data` itself
    was computed.
    """
    if debug_only and cfg is not None and cfg.export_mode != "debug":
        log_info(f"(export_mode='standard': skipped debug-only export {Path(path).name})")
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=JSON_INDENT, default=str)
    confirm_saved(path)
    return path


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub-flavoured Markdown table."""
    cols = [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        cells = [str(row[c]).replace("|", "\\|") for c in df.columns]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


class SummaryLog:
    """Accumulates (Section, Quantity, Value) rows during the pipeline run.

    After the run, export() writes a CSV and a Markdown report with one
    section per heading, plus auto-populated Warnings and Conclusions
    sections built from detected diagnostics.
    """

    def __init__(self) -> None:
        self._rows: list[dict] = []
        self._warnings: list[str] = []
        self._conclusions: list[str] = []

    def log(self, section: str, quantity: str, value) -> None:
        """Record one (section, quantity, value) row for the summary report."""
        self._rows.append({"Section": section, "Quantity": quantity, "Value": value})

    def add_warning(self, text: str) -> None:
        """Register a diagnostic warning for the Warnings section."""
        self._warnings.append(text)

    def add_conclusion(self, text: str) -> None:
        """Register a conclusion statement for the Conclusions section."""
        self._conclusions.append(text)

    def to_dataframe(self) -> pd.DataFrame:
        """Combine logged rows, warnings, and conclusions into one DataFrame."""
        rows = list(self._rows)
        for i, w in enumerate(self._warnings, 1):
            rows.append({"Section": "Warnings", "Quantity": f"Warning {i}", "Value": w})
        for i, c in enumerate(self._conclusions, 1):
            rows.append({"Section": "Conclusions", "Quantity": f"Conclusion {i}", "Value": c})
        return pd.DataFrame(rows)

    def export(
        self, csv_path: Path, md_path: Path, cfg: Optional["Config"] = None,
        legacy_md_path: Optional[Path] = None,
    ) -> pd.DataFrame:
        """Write the full summary report as CSV and Markdown/text; return the DataFrame.

        Export-mode gating (publication-readiness refactor): the raw
        (Section, Quantity, Value) CSV form is a developer/tabular
        artifact and is gated `debug_only=True` when `cfg` is supplied;
        the human-readable text/Markdown form at `md_path` (the pipeline's
        master summary log) is always written regardless of export_mode,
        since it is the curated, publication-facing deliverable.

        `legacy_md_path`, if given, additionally writes the identical text
        under the pipeline's pre-refactor filename (full_summary_report.md)
        - but ONLY in `cfg.export_mode == "debug"` runs, preserving full
        backwards compatibility for anything that referenced the old name
        without duplicating the file in every standard-mode run.
        """
        df = self.to_dataframe()
        save_csv(df, csv_path, cfg=cfg, debug_only=True)
        report_text_parts = ["# Full Summary Report\n\n"]
        for section in df["Section"].unique():
            report_text_parts.append(f"## {section}\n\n")
            section_df = df[df["Section"] == section][["Quantity", "Value"]]
            report_text_parts.append(dataframe_to_markdown(section_df))
            report_text_parts.append("\n\n")
        report_text = "".join(report_text_parts)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(report_text)
        confirm_saved(md_path)
        if legacy_md_path is not None and (cfg is None or cfg.export_mode == "debug"):
            with open(legacy_md_path, "w", encoding="utf-8") as f:
                f.write(report_text)
            confirm_saved(legacy_md_path)
        log_info(
            f"\nFull summary report exported: {len(df)} rows across "
            f"{df['Section'].nunique()} sections (CSV / Markdown)."
        )
        return df



INTERPRETATION_LIBRARY: dict[str, str] = {
    "cost_parameter_scope": (
        "w0 is a resource-cost scaling coefficient: a relative scaling "
        "parameter of the NEGI cost term, normalised by its value at the "
        "reference w0. It is not a directly measured Jeddah water or energy "
        "quantity and is not calibrated to real-world data in this framework; "
        "alpha and beta are relative benefit and cost weights. Results that "
        "depend on these decision parameters are conditional on the "
        "resource-cost assumptions, not statements about measured resource use."
    ),
    "decision_envelope_scope": (
        "The Scenario 2 maximum is reported as a decision envelope, not as a "
        "single overall optimum. Physical quantities (observed NDVI, NDBI, "
        "Elevation, ST_EMIS, ST_EMSD, LST), model outputs (predicted LST and "
        "cooling from the five-predictor model), scenario assumptions "
        "(trajectory definitions and archetype endpoint) and decision "
        "parameters (alpha, beta, w0, exponent) are different kinds of "
        "quantity, so the reference-case maximum, the spatial-refit range, "
        "the archetype-percentile range and the cost-weight range are shown "
        "side by side and are never averaged or pooled."
    ),
    "spatial_dependence": (
        "Residual spatial dependence remains after spatially structured "
        "validation, indicating spatially structured variation not captured "
        "by the current predictor set. Spatial validation addresses "
        "geographic generalization of predictive performance but does not "
        "require residual spatial independence. Global predictive metrics "
        "should therefore be interpreted separately from local residual "
        "structure."
    ),
    "endpoint_sign_flip": (
        "NEGI changes sign along the evaluated trajectory near this point. "
        "This change should not be interpreted mechanistically (e.g. as "
        "\"cost overtakes benefit\"); the point is locally unstable and "
        "statistically unsupported, and the sign change is reported as a "
        "descriptive fact about the trajectory rather than a causal claim."
    ),
    "boundary_maximum": (
        "The highest evaluated NEGI occurs near the boundary of the explored "
        "scenario trajectory. This should be interpreted as the best-performing "
        "point within the evaluated scenario rather than evidence of a unique "
        "interior or global optimum."
    ),
    "scenario1_sign_convention": (
        "Under the isolated NDVI counterfactual (Scenario 1, NDBI held fixed "
        "at its observed baseline), increasing NDVI did not generate "
        "predicted cooling: predicted LST instead increased slightly, "
        "consistent with the conditional NDVI-LST association reported in "
        "Figure 6. Because negative temperature changes are assigned zero "
        "cooling benefit (max(\u0394T, 0)), Scenario 1's progressively "
        "declining NEGI reflects increasing intervention (irrigation/"
        "energy) cost accumulating against a benefit term held at zero - "
        "it is not a cooling trajectory. A more negative NEGI along this "
        "curve should be read as 'less favorable net environmental gain', "
        "not as 'more cooling'."
    ),
    "conditional_ndvi": (
        "The within-NDBI-stratum NDVI-LST slopes shown here are unadjusted "
        "(marginal) associations, not conditional, partial, or causal NDVI "
        "effects. Equal-count NDBI binning holds NDBI only approximately "
        "constant, so residual continuous NDBI variation can remain within "
        "a bin and bias these slopes - including producing an apparent "
        "sign reversal that does not reflect a real interaction between "
        "vegetation and urbanization. A continuous NDBI+Elevation "
        "adjustment (see the companion diagnostic) is required to assess "
        "the adjusted NDVI-LST association; neither analysis establishes "
        "a causal effect, and unmeasured confounding may remain."
    ),
    "calibration": (
        "Calibration slope, intercept, and mean bias describe agreement between "
        "observed and predicted LST without post-hoc recalibration. RMSE and MAE "
        "are reported separately in the performance section."
    ),
    "feature_support": (
        "Scenario points flagged as outside feature-space support extrapolate "
        "beyond the range of the training data and should be interpreted with "
        "additional caution."
    ),
    "smoothing": (
        "Smoothed curves are provided solely to improve visualization. All "
        "scientific interpretation and diagnostics are based on the underlying "
        "model predictions rather than the smoothed representation."
    ),
    "heteroscedasticity": (
        "Mild heteroscedasticity was detected, indicating that residual variance "
        "changes across the prediction range. This behavior is common in "
        "environmental prediction models and does not invalidate the model, but "
        "it should be considered when interpreting prediction uncertainty."
    ),
    "ndvi_ndbi_confound": (
        "Scenario 1 and Scenario 2 answer different causal questions and "
        "should not be read as replicates of the same effect. Scenario 1 "
        "holds NDBI fixed at its observed baseline while NDVI increases, "
        "isolating the model's NDVI effect net of urbanization. Scenario 2 "
        "is an empirical archetype trajectory toward existing "
        "lower-density, greener districts in Jeddah, defined entirely "
        "from observed cumulative-bin medians and independent of model "
        "predictions - not a continuation of Scenario 1's NDVI-only path, "
        "and not a synthetic linear blend of two endpoint values. Because "
        "NDBI is the dominant predictor of LST in this model, Scenario "
        "2's thermal response is expected to be driven substantially by "
        "its NDBI reduction as well as its NDVI increase. Scenario 1 is "
        "the more appropriate reference for an NDVI-specific claim; "
        "Scenario 2 should be described as a realistic, moderate "
        "municipal redevelopment programme rather than an idealized "
        "maximum-greening scenario."
    ),
    "predictor_scope": (
        "Five predictors (NDVI, NDBI, Elevation, and the two surface-"
        "emissivity covariates ST_EMIS and ST_EMSD) were intentionally used "
        "in this model. Only NDVI and NDBI are manipulated by the "
        "scenarios; Elevation, ST_EMIS, and ST_EMSD are held fixed at their "
        "reference (training-median) values at every scenario point, so the "
        "emissivity covariates condition the predicted LST but are not "
        "intervention variables. The remaining unexplained variance likely "
        "reflects additional environmental processes not captured by these "
        "variables. Omitted variables such as building morphology, wind "
        "exposure, and land-use type are presented only as potential future "
        "work and do not represent demonstrated deficiencies of the current "
        "model."
    ),
    "benchmark_comparison": (
        "XGBoost was optimized using RandomizedSearchCV with "
        "{xgb_iter} iterations over a {xgb_params}-parameter grid. "
        "Random Forest and Gradient Boosting benchmarks are tuned via "
        "RandomizedSearchCV with the same {bench_iter}-iteration budget on "
        "identical GroupKFold splits, groups, scoring metric, and "
        "random-state philosophy, so the comparison reflects each model's "
        "tuned best case under an equal tuning budget. Linear Regression is "
        "reported untuned as a simple baseline. Equal tuning effort does not "
        "by itself establish formal statistical superiority; see the paired "
        "fold-level comparison for whether any XGBoost advantage is "
        "consistent across folds or merely a small numerical difference."
    ),
    "scenario1_terminology": (
        "Scenario 1 — Isolated NDVI counterfactual (NDBI held fixed). "
        "Scenario 1 is a model-based hypothetical counterfactual and "
        "should not be interpreted as an independently identified causal "
        "effect of vegetation, and should not be described as showing "
        "that \"greening does or does not work\". NDVI and NDBI are "
        "correlated in the observed data (r \u2248 -0.637), which limits "
        "independent identification of the NDVI effect: holding NDBI "
        "fixed while varying NDVI evaluates the model along a combination "
        "of feature values that is less densely represented in the "
        "training data than the observed joint NDVI-NDBI relationship, "
        "not a randomized or otherwise causally identified intervention."
    ),
    "non_causal_safeguard": (
        "Scenario predictions (Scenario 1 and Scenario 2 alike) are "
        "model-based counterfactual evaluations of observational "
        "associations. They should not be interpreted as causal "
        "treatment-effect estimates. NEGI itself is not an empirically "
        "observed physical quantity: it is a decision metric conditional "
        "on the specified benefit weights (alpha/beta), energy-cost "
        "functional form, and scenario definition used to compute it - "
        "changing any of those assumptions changes NEGI's value without "
        "implying any change in the underlying physical cooling effect."
    ),
}


def interpretation_text(key: str, **format_kwargs) -> str:
    """Return the canonical interpretive sentence for `key`.

    Some entries (e.g. "benchmark_comparison") contain `{placeholders}`
    that must be filled in via `format_kwargs`; plain entries ignore
    `format_kwargs` entirely (`str.format` on a string with no
    placeholders is a no-op).

    Raises KeyError with a clear message for unknown keys so that a typo
    fails loudly rather than silently returning an empty caption.
    """
    try:
        text = INTERPRETATION_LIBRARY[key]
        return text.format(**format_kwargs) if format_kwargs else text
    except KeyError as exc:
        raise KeyError(
            f"Unknown interpretation key {key!r}. "
            f"Available keys: {sorted(INTERPRETATION_LIBRARY)}"
        ) from exc


def boundary_distance_fraction(idx: int, positions: Optional[np.ndarray], n_points: int) -> float:
    """Distance from trajectory index `idx` to its NEAREST endpoint,
    normalized by the total trajectory length/span - the quantity
    review recommendation 7 asked for, in place of a fixed index-count
    band.

    When `positions` (the actual scenario-position values, e.g.
    `scenario_pct`) is supplied, distance is measured in POSITION units
    and normalized by the full trajectory span (positions[-1] -
    positions[0]) - this is exact even when points are not evenly
    spaced. When `positions` is unavailable, falls back to an
    index-count fraction (equivalent under even spacing, which is the
    case for every trajectory in this pipeline today, but kept generic).
    Returns 0.0 at either endpoint, up to ~0.5 at the trajectory
    midpoint.
    """
    if n_points <= 1:
        return 0.0
    if positions is not None and len(positions) == n_points:
        span = float(positions[-1] - positions[0])
        if not np.isclose(span, 0.0):
            dist_to_start = abs(float(positions[idx]) - float(positions[0]))
            dist_to_end   = abs(float(positions[-1]) - float(positions[idx]))
            return float(min(dist_to_start, dist_to_end) / abs(span))
    dist_to_start = idx
    dist_to_end   = (n_points - 1) - idx
    return float(min(dist_to_start, dist_to_end) / (n_points - 1))


def is_near_trajectory_boundary(
    idx: int, n_points: int, tolerance_pct: float,
    positions: Optional[np.ndarray] = None,
) -> bool:
    """Interpretation-only helper (item 22): True when `idx` falls within
    `tolerance_pct`% of the nearest trajectory endpoint.

    Review-pass fix (recommendation 7): "near boundary" is now computed
    from `boundary_distance_fraction()` - the point's actual distance to
    its nearest endpoint, normalized by the trajectory's total span -
    rather than being tied to a fixed index-count band. Passing the
    trajectory's actual position array (`positions`, e.g. scenario_pct)
    makes this exact for non-uniformly-spaced trajectories; omitting it
    falls back to the original index-count behaviour (the two agree
    exactly for the evenly-spaced trajectories used throughout this
    pipeline, so no existing numeric result changes by default).

    This does not change how the optimum index itself is computed anywhere
    in the pipeline - it only widens the *reporting* rule from an exact
    endpoint match to a configurable near-boundary band, so that maxima one
    or two steps from the edge are still flagged as boundary-adjacent.
    """
    if n_points <= 1:
        return True
    return boundary_distance_fraction(idx, positions, n_points) <= (tolerance_pct / 100.0)



def interpret_model_comparison(cfg: Config = CFG) -> str:
    """Canonical benchmark-comparison disclaimer (item 23): the XGBoost
    surrogate was tuned via RandomizedSearchCV; benchmark models are shown
    for context only and the comparison is not intended to establish
    global optimality.

    Interpolates the actual runtime tuning-budget values from `cfg` into
    the "benchmark_comparison" template (xgb_iter, xgb_params, bench_iter).
    Values are read from existing Config fields only - nothing here is
    hard-coded or newly computed."""
    return interpretation_text(
        "benchmark_comparison",
        xgb_iter=cfg.n_random_search_iter,
        xgb_params=len(cfg.param_distributions),
        bench_iter=cfg.n_benchmark_search_iter,
    )


def interpret_support(feature_support_status: str) -> str:
    """Interpret a scenario's within/outside-support status (item 20)."""
    if feature_support_status.lower().startswith("ok"):
        return f"{feature_support_status}. Scenario points fall within training-data support."
    return f"{feature_support_status}. " + interpretation_text("feature_support")


class PredictionCache:
    """Caches model.predict() calls keyed on the rounded full feature
    vector - one entry per feature in FEATURES order (NDVI, NDBI,
    Elevation, ST_EMIS, ST_EMSD).

    Reuse never alters what is predicted or how the model was fit - it
    only avoids duplicate calls to model.predict() for feature vectors
    already evaluated elsewhere in the pipeline.
    """

    def __init__(self, precision: int = 10) -> None:
        self.precision = precision
        self._store: dict[tuple, tuple[float, bool]] = {}
        self.hits  = 0
        self.misses = 0

    def _key(self, *feature_values: float) -> tuple:
        if len(feature_values) != len(FEATURES):
            raise ValueError(
                f"PredictionCache key needs {len(FEATURES)} feature values "
                f"{FEATURES}, got {len(feature_values)}."
            )
        return tuple(round(float(v), self.precision) for v in feature_values)

    def get(self, *feature_values: float):
        """Return the cached prediction for this full feature vector, if any."""
        return self._store.get(self._key(*feature_values))

    def set(self, *feature_values_and_value) -> None:
        """Store a prediction: `set(*feature_values, value)` where `value`
        is the (prediction, out_of_bounds) tuple."""
        *feature_values, value = feature_values_and_value
        self._store[self._key(*feature_values)] = value

    def stats(self) -> dict:
        """Hit/miss counts, cache size, and hit rate (developer diagnostics only)."""
        total    = self.hits + self.misses
        hit_rate = self.hits / total if total else 0.0
        return {
            "hits": self.hits, "misses": self.misses,
            "size": len(self._store), "hit_rate": hit_rate,
        }

    def clear(self) -> None:
        """Reset the cache contents and hit/miss counters."""
        self._store.clear()
        self.hits  = 0
        self.misses = 0


PREDICTION_CACHE = PredictionCache()

def print_reproducibility_info(cfg: Config) -> dict:
    """Log and return the seed, library versions, and git commit for this run."""
    info = {
        "random_seed":     cfg.random_seed,
        "python_version":  sys.version.split()[0],
        "numpy_version":   np.__version__,
        "xgboost_version": xgboost.__version__,
        "sklearn_version": sklearn.__version__,
        "pandas_version":  pd.__version__,
        "scipy_version":   __import__("scipy").__version__,
    }
    info["timestamp_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    cfg_str = json.dumps({k: str(v) for k, v in cfg.__dict__.items()}, sort_keys=True)
    info["config_sha256"] = hashlib.sha256(cfg_str.encode()).hexdigest()[:16]

    log_info("REPRODUCIBILITY INFORMATION")
    for key, val in info.items():
        log_info(f"  {key:20s}: {val}")

    try:
        cfg.data_dir.mkdir(parents=True, exist_ok=True)
        with open(cfg.data_dir / "reproducibility_info.json", "w") as f:
            json.dump(info, f, indent=JSON_INDENT)
    except OSError as exc:
        warnings.warn(f"Could not write reproducibility_info.json: {exc}")

    return info


def write_reproducibility_report(
    cfg: Config,
    repro_info: dict,
    best_params: Optional[dict] = None,
    validation: Optional["ValidationResults"] = None,
    runtime_seconds: Optional[float] = None,
    convergence_status: Optional[str] = None,
) -> Path:
    """Write a human-readable Reproducibility_Report.txt."""
    lines = [
        "NEGI PIPELINE - REPRODUCIBILITY REPORT",
        "=" * 60,
        f"Date (UTC)         : "
        f"{datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}",
        f"Random seed        : {cfg.random_seed}",
        "",
        "Package versions",
        "-" * 60,
        f"  Python           : {repro_info.get('python_version')}",
        f"  numpy            : {repro_info.get('numpy_version')}",
        f"  pandas           : {repro_info.get('pandas_version')}",
        f"  scipy            : {repro_info.get('scipy_version')}",
        f"  scikit-learn     : {repro_info.get('sklearn_version')}",
        f"  xgboost          : {repro_info.get('xgboost_version')}",
        "",
    ]
    if best_params is not None:
        lines += ["Selected XGBoost hyperparameters (RandomizedSearchCV)", "-" * 60]
        for k, v in best_params.items():
            lines.append(f"  {k:20s}: {v}")
        lines.append("")
    if validation is not None:
        lines += [
            "Validation results",
            "-" * 60,
            f"  Nested GroupKFold CV R²            : {validation.nested_cv_r2:.4f}",
            f"  Repeated spatial-holdout R²        : "
            f"{validation.repeated_r2_mean:.4f} +/- {validation.repeated_r2_ci95:.4f}",
            f"  Untouched confirmatory holdout R²  : {validation.holdout_r2:.4f}",
            f"  Untouched confirmatory holdout RMSE: {validation.holdout_rmse:.4f}",
            f"  Untouched confirmatory holdout MAE : {validation.holdout_mae:.4f}",
            f"  Calibration slope                  : {validation.calibration_slope:.4f}",
            f"  Calibration intercept              : {validation.calibration_intercept:.4f}",
            f"  Calibration bias                   : {validation.calibration_bias:.4f}",
            "",
        ]
    if convergence_status is not None:
        lines.append(f"Uncertainty convergence diagnostic : {convergence_status}")
    if runtime_seconds is not None:
        lines.append(f"Total runtime              : {runtime_seconds:.1f} s")
    lines.append(f"Config SHA-256 (first 16)  : {repro_info.get('config_sha256', 'n/a')}")

    report_text = "\n".join(lines) + "\n"
    report_path = cfg.data_dir / "Reproducibility_Report.txt"
    try:
        cfg.data_dir.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report_text, encoding="utf-8")
        confirm_saved(report_path)
    except OSError as exc:
        warnings.warn(f"Could not write Reproducibility_Report.txt: {exc}")
    return report_path


def build_and_export_run_manifest(
    cfg: Config,
    repro_info: dict,
    *,
    runtime_seconds: Optional[float] = None,
    validation: Optional["ValidationResults"] = None,
    qa_summary: Optional[dict] = None,
    scenario_summary_df: Optional[pd.DataFrame] = None,
    n_train: Optional[int] = None,
    n_holdout: Optional[int] = None,
    convergence_criteria: Optional[dict] = None,
    decision_envelope: Optional[dict] = None,
    cost_regime: Optional[dict] = None,
) -> Path:
    """Export run_manifest.json: the single machine-readable file describing
    *how* this run was produced and *what it found*, for tooling (CI,
    regression tests, audit) to consume programmatically.

    IMPORTANT: this is the canonical machine-readable export for automated
    consumers. Human-readable artifacts (Reproducibility_Report.txt,
    full_summary_report.md) are formatted for people and their wording,
    layout, or rounding may change without notice - they must never be
    parsed by tests or other tooling. Anything a script needs to read back
    programmatically (validation metrics, QA counts, scenario-maximum
    position, provenance/hashes/versions) is written here, in a stable
    key structure, instead.

    This function is documentation/metadata export only (engineering-
    refinement item 3) - it reports facts already computed elsewhere and
    never performs, alters, or re-derives any scientific computation.

    Fields
    ------
    timestamp          : UTC ISO-8601 timestamp (from repro_info, so it is
                          identical to reproducibility_info.json's
                          timestamp for this same run - not a second,
                          independent clock read).
    config_hash         : SHA-256 (first 16 hex chars) of the full Config,
                          as already computed by print_reproducibility_info().
    git_hash            : short git commit hash, or "unavailable".
    software_versions   : Python / numpy / pandas / scipy / scikit-learn /
                          xgboost versions in use for this run.
    random_seeds        : every seed that affects any stochastic stage.
    run_mode            : "development" or "publication" (Config.run_mode).
    dataset_hash        : content hash of the input CSV (see
                          _dataset_fingerprint) - so any edit to the input
                          data is detectable from the manifest alone.
    pipeline_version    : PIPELINE_VERSION constant (engineering/scaffolding
                          version, distinct from CACHE_SCHEMA_VERSION).
    cache_status        : {stage_name: "cache hit" | "computed"} for every
                          cacheable stage that ran this session (see
                          _CACHE_STAGE_STATUS, populated by
                          run_or_cached_stage()).
    validation_metrics  : nested-CV R², holdout R²/RMSE/MAE, calibration
                          slope/intercept/bias - read directly off the
                          already-computed ValidationResults object
                          (no recomputation).
    qa_summary          : {n_total, n_passed, n_failed} - read directly
                          off run_qa_checks()'s already-computed return
                          value (no recomputation).
    scenario2_maximum   : Scenario 2's NEGI-maximizing position (percent)
                          and NEGI value - read directly off the
                          already-exported scenario_summary_df row (no
                          recomputation).
    """
    manifest = {
        "timestamp": repro_info.get("timestamp_utc"),
        "config_hash": repro_info.get("config_sha256"),
        "git_hash": _get_git_hash(cfg),
        "software_versions": {
            "python": repro_info.get("python_version"),
            "numpy": repro_info.get("numpy_version"),
            "pandas": repro_info.get("pandas_version"),
            "scipy": repro_info.get("scipy_version"),
            "scikit_learn": repro_info.get("sklearn_version"),
            "xgboost": repro_info.get("xgboost_version"),
        },
        "random_seeds": {
            "random_seed": cfg.random_seed,
            "bootstrap_seed": cfg.bootstrap_seed,
            "uncertainty_refit_seed": cfg.uncertainty_refit_seed,
            "moran_permutation_seed": cfg.moran_permutation_seed,
        },
        "run_mode": cfg.run_mode,
        "dataset_hash": _dataset_fingerprint(cfg),
        "pipeline_version": PIPELINE_VERSION,
        "cache_status": dict(_CACHE_STAGE_STATUS),
    }
    if runtime_seconds is not None:
        manifest["runtime_seconds"] = round(runtime_seconds, 3)

    if validation is not None:
        manifest["validation_metrics"] = {
            "nested_cv_r2":         float(validation.nested_cv_r2),
            "repeated_r2_mean":     float(validation.repeated_r2_mean),
            "holdout_r2":           float(validation.holdout_r2),
            "holdout_rmse":         float(validation.holdout_rmse),
            "holdout_mae":          float(validation.holdout_mae),
            "calibration_slope":    float(validation.calibration_slope),
            "calibration_intercept": float(validation.calibration_intercept),
            "calibration_bias":     float(validation.calibration_bias),
        }

    if qa_summary is not None:
        manifest["qa_summary"] = {
            "n_total":  qa_summary.get("n_total"),
            "n_passed": qa_summary.get("n_passed"),
            "n_failed": qa_summary.get("n_failed"),
        }

    if scenario_summary_df is not None and len(scenario_summary_df) > 0:
        row = scenario_summary_df.iloc[0]
        manifest["scenario2_maximum"] = {
            "position_pct": float(row["s2_max_negi_pct"]),
            "negi_value":   float(row["s2_max_negi"]),
        }

    if n_train is not None and n_holdout is not None:
        manifest["dataset_size"] = {
            "n_total":   int(n_train) + int(n_holdout),
            "n_train":   int(n_train),
            "n_holdout": int(n_holdout),
        }

    if convergence_criteria is not None:
        manifest["uncertainty_convergence"] = dict(convergence_criteria)

    manifest["model_features"] = {
        "features": list(FEATURES),
        "scenario_manipulated": list(SCENARIO_MANIPULATED_FEATURES),
        "fixed_at_reference_values": list(FIXED_REFERENCE_FEATURES),
        "monotone_constraints": list(cfg.monotone_constraints),
    }
    if decision_envelope is not None:
        manifest["decision_envelope"] = {
            "reference_case": decision_envelope.get("reference_case"),
            "layers": [
                {k: l[k] for k in ("layer", "kind", "position_low_pct", "position_high_pct")}
                for l in decision_envelope.get("layers", [])
            ],
            "assessment": decision_envelope.get("assessment"),
        }
    if cost_regime is not None:
        manifest["cost_regime"] = {
            k: cost_regime.get(k) for k in (
                "verdict", "regime_detected", "is_sharp", "is_bimodal_on_evaluated_grid",
                "regime_split_position_pct", "reference_case",
            )
        }

    manifest_path = save_json(manifest, cfg.data_dir / "run_manifest.json")

    flat_row = {
        "timestamp":               manifest.get("timestamp"),
        "git_hash":                manifest.get("git_hash"),
        "config_hash":             manifest.get("config_hash"),
        "dataset_hash":            manifest.get("dataset_hash"),
        "pipeline_version":        manifest.get("pipeline_version"),
        "run_mode":                manifest.get("run_mode"),
        "random_seed":             cfg.random_seed,
        "runtime_seconds":         manifest.get("runtime_seconds"),
    }
    if n_train is not None and n_holdout is not None:
        flat_row["n_total"]   = int(n_train) + int(n_holdout)
        flat_row["n_train"]   = int(n_train)
        flat_row["n_holdout"] = int(n_holdout)
    if qa_summary is not None:
        flat_row["qa_n_total"]  = qa_summary.get("n_total")
        flat_row["qa_n_passed"] = qa_summary.get("n_passed")
    if convergence_criteria is not None:
        flat_row["configured_refits"]       = convergence_criteria.get("configured_refits")
        flat_row["completed_refits"]        = convergence_criteria.get("completed_refits")
        flat_row["adaptive_stop"]           = convergence_criteria.get("adaptive_stop")
        flat_row["publication_convergence"] = convergence_criteria.get("publication_convergence")
    save_csv(pd.DataFrame([flat_row]), cfg.data_dir / "run_metadata.csv", cfg=cfg)

    return manifest_path


def export_cache_status_json(
    cfg: Config,
    prediction_cache_stats: Optional[dict] = None,
) -> Path:
    """Export cache_status.json: structured, machine-readable record of
    per-stage cache hit/computed status, plus prediction-cache hit/miss
    counters if prediction caching was active.

    This is the same information console cache-hit/elapsed messages would
    otherwise carry (see run_or_cached_stage() / report_development()) -
    exporting it here means the detail is preserved even in "publication"
    run_mode, where those console messages are suppressed.

    Inputs
    ------
    cfg : active Config - only cfg.data_dir (output location) is read.
    prediction_cache_stats : optional dict as returned by
        PREDICTION_CACHE.stats() ("hits", "misses", "hit_rate"), or None
        if prediction caching was inactive for this run.

    Side effects
    ------------
    Writes cache_status.json into cfg.data_dir. Reads the existing
    _CACHE_STAGE_STATUS dict only - performs no computation and mutates no
    cache state.

    Returns
    -------
    Path to the written cache_status.json file.
    """
    payload = {
        "stage_cache_status": dict(_CACHE_STAGE_STATUS),
        "prediction_cache": prediction_cache_stats,
    }
    return save_json(payload, cfg.data_dir / "cache_status.json")


def export_runtime_summary_json(cfg: Config, runtime_seconds: float) -> Path:
    """Export runtime_summary.json: total pipeline wall-clock runtime as a
    structured, machine-readable record (the same figure already printed
    at the end of a development-mode console run and written into
    Reproducibility_Report.txt / run_manifest.json - this is simply a
    dedicated, minimal file for tooling that only wants runtime, without
    parsing the larger manifest or the human-readable report).

    Inputs
    ------
    cfg             : active Config - only cfg.data_dir is read.
    runtime_seconds : total end-to-end wall-clock runtime in seconds, as
                      already measured by main() via time.monotonic() -
                      not recomputed or re-timed here.

    Side effects
    ------------
    Writes runtime_summary.json into cfg.data_dir.

    Returns
    -------
    Path to the written runtime_summary.json file.
    """
    payload = {"runtime_seconds": round(float(runtime_seconds), 3)}
    return save_json(payload, cfg.data_dir / "runtime_summary.json")


def run_engineering_qa_checks(cfg: Config) -> dict:
    """Engineering-consistency QA - separate from, and additive to, the
    scientific QA performed by run_qa_checks().

    This does NOT touch, extend, or recompute anything in run_qa_checks()'s
    scientific checklist - it is a distinct check, exported to its own file,
    so the scientific QA pass/fail count in qa_summary.json is completely
    unaffected by this function's existence or result.

    Must be called after run_manifest.json, runtime_summary.json,
    cache_status.json, and qa_summary.json have already been written (i.e.
    at the very end of main()), since it only verifies those files exist
    and parse as valid JSON - it performs no scientific computation.

    Inputs
    ------
    cfg : active Config - only cfg.data_dir (where the four files are
          expected) is read.

    Side effects
    ------------
    None beyond reading the four files from disk to confirm they exist and
    parse as valid JSON. Does not write engineering_qa.json itself - the
    caller (main()) does that via save_json, matching every other export.

    Returns
    -------
    dict with one entry per checked file: {"exists": bool, "valid_json": bool}
    plus "n_total"/"n_passed"/"n_failed" summarizing this engineering-only
    checklist (kept in its own namespace, never merged into the scientific
    qa_summary.json counts).
    """
    expected_files = (
        "run_manifest.json",
        "runtime_summary.json",
        "cache_status.json",
        "qa_summary.json",
    )
    checks = {}
    n_passed = 0
    for fname in expected_files:
        fpath = cfg.data_dir / fname
        exists = fpath.is_file()
        valid_json = False
        if exists:
            try:
                json.loads(fpath.read_text(encoding="utf-8"))
                valid_json = True
            except (OSError, ValueError):
                valid_json = False
        checks[fname] = {"exists": exists, "valid_json": valid_json}
        if exists and valid_json:
            n_passed += 1

    result = {
        "checks": checks,
        "n_total": len(expected_files),
        "n_passed": n_passed,
        "n_failed": len(expected_files) - n_passed,
    }
    report_development(
        cfg,
        f"Engineering QA: {n_passed}/{len(expected_files)} expected "
        f"manifest/report files present and valid JSON.",
    )
    return result


def build_publication_summary(
    cfg: Config,
    repro_info: dict,
    validation: "ValidationResults",
    calibration_table_df: Optional[pd.DataFrame],
    resid: Optional["ResidualDiagnostics"],
    spatial_diagnostic: Optional[dict],
    comparison_df: Optional[pd.DataFrame],
    fi: Optional["FeatureImportanceResult"],
    scenario_summary_df: Optional[pd.DataFrame],
    maximum_diagnostics_df: Optional[pd.DataFrame],
    endpoint_diagnostics_df: Optional[pd.DataFrame],
    qa_summary: Optional[dict] = None,
    runtime_seconds: Optional[float] = None,
) -> dict:
    """Assemble the unified, machine-readable publication_summary.json
    (engineering-refinement item 5): one file collecting every table this
    pipeline already exports separately elsewhere, for a single-file
    programmatic read of validation, calibration, residual diagnostics,
    model comparison, feature importance, scenario summaries, maximum
    diagnostics, endpoint diagnostics, QA status, software versions,
    configuration/dataset hashes, and the pipeline timestamp. Every value
    here is read from an already-computed object/DataFrame - nothing is
    recalculated, and no existing export (run_manifest.json,
    scenario_summary.csv, maximum_diagnostics.csv, etc.) is altered or
    replaced by this addition.

    `qa_summary` is intentionally optional: this function is called twice
    in main() - once BEFORE run_qa_checks() so the file already exists for
    run_qa_checks' own "publication_summary.json exists" check, and once
    more AFTER run_qa_checks() returns, to fold the QA result itself into
    the final version of the file. Only the second (post-QA) write is
    authoritative; a reader should treat the pre-QA write as a transient
    intermediate that this same run immediately overwrites.
    """
    summary_dict = {
        "pipeline_version":    PIPELINE_VERSION,
        "pipeline_timestamp":  repro_info.get("timestamp_utc"),
        "configuration_hash":  repro_info.get("config_sha256"),
        "dataset_hash":        _dataset_fingerprint(cfg),
        "run_mode":            cfg.run_mode,
        "software_versions": {
            "python":       repro_info.get("python_version"),
            "numpy":        repro_info.get("numpy_version"),
            "pandas":       repro_info.get("pandas_version"),
            "scipy":        repro_info.get("scipy_version"),
            "scikit_learn": repro_info.get("sklearn_version"),
            "xgboost":      repro_info.get("xgboost_version"),
        },
        "validation_metrics": {
            "nested_cv_r2":     float(validation.nested_cv_r2),
            "repeated_r2_mean": float(validation.repeated_r2_mean),
            "repeated_r2_ci95": float(validation.repeated_r2_ci95),
            "holdout_r2":       float(validation.holdout_r2),
            "holdout_rmse":     float(validation.holdout_rmse),
            "holdout_mae":      float(validation.holdout_mae),
        } if validation is not None else None,
        "calibration_metrics": (
            calibration_table_df.iloc[0].to_dict()
            if calibration_table_df is not None and len(calibration_table_df) else None
        ),
        "residual_diagnostics": {
            "skewness":                float(resid.skewness),
            "kurtosis":                float(resid.kurtosis),
            "bp_statistic":             float(resid.bp_statistic),
            "bp_p_value":              float(resid.bp_p_value),
            "spearman_rho":            float(resid.spearman_rho),
            "spearman_p":              float(resid.spearman_p),
            "morans_i": (
                float(spatial_diagnostic["moran"]["morans_i"])
                if spatial_diagnostic is not None else None
            ),
            "spatial_dependence_flag": (
                bool(spatial_diagnostic["spatial_dependence_flag"])
                if spatial_diagnostic is not None else None
            ),
        } if resid is not None else None,
        "model_comparison": (
            comparison_df.to_dict(orient="records") if comparison_df is not None else None
        ),
        "feature_importance": (
            fi.table.to_dict(orient="records") if fi is not None else None
        ),
        "scenario_summary": (
            scenario_summary_df.iloc[0].to_dict()
            if scenario_summary_df is not None and len(scenario_summary_df) else None
        ),
        "maximum_diagnostics": (
            maximum_diagnostics_df.to_dict(orient="records") if maximum_diagnostics_df is not None else None
        ),
        "endpoint_diagnostics": (
            endpoint_diagnostics_df.to_dict(orient="records") if endpoint_diagnostics_df is not None else None
        ),
        "qa_summary": qa_summary,
    }
    if runtime_seconds is not None:
        summary_dict["runtime_seconds"] = round(runtime_seconds, 3)
    return summary_dict



def run_qa_checks(
    cfg: Config,
    validation: "ValidationResults",
    fi: "FeatureImportanceResult",
    repro_info: dict,
    scenario_summary_df: pd.DataFrame,
    scenario_points_df: pd.DataFrame,
    summary: Optional["SummaryLog"] = None,
    maximum_diagnostics_df: Optional[pd.DataFrame] = None,
    maximum_diagnostics_s2: Optional[dict] = None,
    zero_crossing_df: Optional[pd.DataFrame] = None,
    negi: Optional["NEGIResults"] = None,
    support: Optional["SupportDiagnostics"] = None,
    maximum_diagnostics_s1: Optional[dict] = None,
    calibration_table_df: Optional[pd.DataFrame] = None,
    data: Optional["DatasetBundle"] = None,
    uncertainty: Optional["UncertaintyResults"] = None,
    comparison_df: Optional[pd.DataFrame] = None,
    spatial_diagnostic: Optional[dict] = None,
    endpoint_diagnostics_s2: Optional[dict] = None,
    endpoint_diagnostics_s1: Optional[dict] = None,
    endpoint_diagnostics_df: Optional[pd.DataFrame] = None,
    spatial_autocorr_df: Optional[pd.DataFrame] = None,
    publication_summary_path: Optional[Path] = None,
    joint_support_df: Optional[pd.DataFrame] = None,
    block_size_sensitivity_df: Optional[pd.DataFrame] = None,
    publication_convergence_status: Optional[str] = None,
) -> dict:
    """Run all automated cross-artifact consistency checks, print each
    check's real name and PASS/FAIL result plus an overall count, then
    raise a descriptive AssertionError if any check failed. Existing checks
    are unchanged; the new checks below (item 13) add scientific-consistency
    verification without altering any prior check or any scientific result -
    a QA failure identifies the inconsistency rather than silently
    "fixing" the underlying numbers."""
    checks = []

    for label, df in (
        ("scenario_summary.csv", scenario_summary_df),
        ("scenario_points.csv", scenario_points_df),
    ):
        if "calibration_slope" in df.columns:
            csv_val = float(df["calibration_slope"].iloc[0])
            passed = bool(np.isclose(csv_val, validation.calibration_slope, equal_nan=True))
            checks.append((
                f"Calibration slope matches {label}", passed,
                f"calibration slope in {label} ({csv_val:.6f}) does not match "
                f"ValidationResults ({validation.calibration_slope:.6f})."
            ))

    non_finite = [
        name for name, value in (
            ("nested_cv_r2", validation.nested_cv_r2),
            ("holdout_r2", validation.holdout_r2),
            ("holdout_rmse", validation.holdout_rmse),
            ("calibration_slope", validation.calibration_slope),
        )
        if not np.isfinite(value)
    ]
    checks.append((
        "Validation metrics finite", not non_finite,
        f"ValidationResults.{', '.join(non_finite)} not finite." if non_finite else "",
    ))

    importances = fi.table["Importance"].to_numpy()
    fi_sorted = bool(np.all(np.diff(importances) <= 1e-12))
    checks.append((
        "Feature-importance table sorted descending", fi_sorted,
        "feature-importance table is not sorted in descending order at export time.",
    ))

    filenames = [row["filename"] for row in FIGURE_REGISTRY._rows]
    dupes = {f for f in filenames if filenames.count(f) > 1}
    checks.append((
        "No duplicate export filenames", not dupes,
        f"duplicate export filenames detected: {dupes}",
    ))

    has_hash = bool(repro_info.get("config_sha256"))
    checks.append((
        "Configuration hash present", has_hash,
        "configuration hash missing from repro_info.",
    ))


    placeholder_sources = [scenario_summary_df, scenario_points_df]
    if maximum_diagnostics_df is not None:
        placeholder_sources.append(maximum_diagnostics_df)
    if zero_crossing_df is not None:
        placeholder_sources.append(zero_crossing_df)
    if summary is not None:
        placeholder_sources.append(summary.to_dataframe())
    unresolved = find_unresolved_placeholders(*placeholder_sources)
    checks.append((
        "No unresolved report placeholders", not unresolved,
        f"unresolved template placeholder(s) found in exported reports: {unresolved}",
    ))

    _deprecated_selector_absent = "select_scenario2_empirical_endpoint" not in globals()
    checks.append((
        "Deprecated model-guided S2 endpoint selector removed from module",
        _deprecated_selector_absent,
        "select_scenario2_empirical_endpoint() still exists in this module; "
        "it was deleted as superseded, circular-by-construction dead code "
        "and must not be reintroduced.",
    ))

    if maximum_diagnostics_s2 is not None and negi is not None:
        d = maximum_diagnostics_s2
        s2_idx = negi.s2_optimum_idx

        actual_max_negi = float(negi.negi_s2[s2_idx])
        actual_max_pct  = float(negi.scenario_pct[s2_idx])
        checks.append((
            "S2 maximum_evaluated_negi matches trajectory argmax",
            bool(np.isclose(d["maximum_evaluated_negi"], actual_max_negi)),
            f"maximum_diagnostics maximum_evaluated_negi "
            f"({d['maximum_evaluated_negi']}) != negi.negi_s2[s2_optimum_idx] "
            f"({actual_max_negi}).",
        ))
        checks.append((
            "S2 maximum_scenario_position matches argmax position",
            bool(np.isclose(d["maximum_scenario_position"], actual_max_pct)),
            f"maximum_diagnostics maximum_scenario_position "
            f"({d['maximum_scenario_position']}) != negi.scenario_pct[s2_optimum_idx] "
            f"({actual_max_pct}).",
        ))

        if zero_crossing_df is not None:
            zc_row = zero_crossing_df[
                (zero_crossing_df["scenario"] == "Scenario 2")
                & (zero_crossing_df["scenario_position"] == "maximum_evaluated_negi")
            ]
            if len(zc_row) == 1:
                zc_low  = float(zc_row["uncertainty_lower"].iloc[0])
                zc_high = float(zc_row["uncertainty_upper"].iloc[0])
                agree = (
                    d["maximum_uncertainty_low"] is not None
                    and np.isclose(zc_low, d["maximum_uncertainty_low"])
                    and np.isclose(zc_high, d["maximum_uncertainty_high"])
                )
                checks.append((
                    "S2 maximum uncertainty agrees across reporting paths", agree,
                    f"zero_crossing_df 95% spatial-refit uncertainty band [{zc_low}, {zc_high}] "
                    f"does not match maximum_diagnostics 95% spatial-refit uncertainty band "
                    f"[{d['maximum_uncertainty_low']}, {d['maximum_uncertainty_high']}] "
                    "at the same scenario position.",
                ))

        checks.append((
            "S2 maximum uncertainty type explicitly identified",
            bool(d["maximum_uncertainty_type"]) and d["maximum_uncertainty_type"] != "unavailable",
            "maximum_uncertainty_type is empty or 'unavailable' for Scenario 2.",
        ))

        if d["maximum_q25"] is not None and d["maximum_uncertainty_low"] is not None:
            iqr_not_mislabeled = not (
                np.isclose(d["maximum_q25"], d["maximum_uncertainty_low"])
                and np.isclose(d["maximum_q75"], d["maximum_uncertainty_high"])
            )
            checks.append((
                "S2 IQR not mislabeled as 95% interval", iqr_not_mislabeled,
                "maximum_q25/q75 are identical to maximum_uncertainty_low/high; "
                "the 95% interval and the IQR must be distinct quantities.",
            ))

        if d["maximum_ci_includes_zero"] is True:
            checks.append((
                "S2 zero-crossing maximum not classified as robust",
                d["maximum_robust"] is not True,
                "Scenario 2 maximum's uncertainty interval includes zero but "
                "maximum_robust is True.",
            ))
        if d["maximum_local_stability"] is False:
            checks.append((
                "S2 locally unstable maximum not classified as supported optimum",
                d["maximum_supports_interior_optimum"] is not True,
                "Scenario 2 maximum is locally unstable but "
                "maximum_supports_interior_optimum is True.",
            ))
        if d["maximum_near_boundary"] or d["maximum_at_boundary"]:
            checks.append((
                "S2 near-boundary maximum not classified as interior optimum",
                d["maximum_supports_interior_optimum"] is not True,
                "Scenario 2 maximum is at/near the trajectory boundary but "
                "maximum_supports_interior_optimum is True.",
            ))
        if d["maximum_knn_support"] is False:
            checks.append((
                "S2 unsupported maximum not classified as robust",
                d["maximum_robust"] is not True,
                "Scenario 2 maximum fails feature-space kNN support but "
                "maximum_robust is True.",
            ))

        if "s2_max_negi" in scenario_summary_df.columns:
            csv_negi = float(scenario_summary_df["s2_max_negi"].iloc[0])
            checks.append((
                "scenario_summary.csv s2_max_negi matches maximum_diagnostics",
                bool(np.isclose(csv_negi, d["maximum_evaluated_negi"])),
                f"scenario_summary.csv s2_max_negi ({csv_negi}) != "
                f"maximum_diagnostics maximum_evaluated_negi "
                f"({d['maximum_evaluated_negi']}).",
            ))

    has_25 = 2.5 in cfg.uncertainty_percentiles
    has_975 = 97.5 in cfg.uncertainty_percentiles
    checks.append((
        "Config.uncertainty_percentiles defines the 95% spatial-refit uncertainty band (2.5/97.5)",
        bool(has_25 and has_975),
        f"Config.uncertainty_percentiles = {cfg.uncertainty_percentiles} does not "
        "contain both 2.5 and 97.5; any '95% spatial-refit uncertainty band' "
        "label would not match its construction.",
    ))

    _test_vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    _expected_ci95 = float(1.96 * _test_vals.std(ddof=0) / np.sqrt(_test_vals.size))
    checks.append((
        "ci95() formula matches its documented 95% normal-approximation definition",
        bool(np.isclose(ci95(_test_vals), _expected_ci95)),
        f"ci95() returned {ci95(_test_vals)} for a known test array; expected "
        f"{_expected_ci95} under the documented 1.96*SD/sqrt(n) definition.",
    ))

    boot_ci = getattr(validation, "spatial_block_bootstrap_ci", None) or {}
    mismatched_n_boot = [
        name for name, info in boot_ci.items()
        if info.get("n_boot") != cfg.bootstrap_iterations
    ]
    if boot_ci:
        checks.append((
            "Spatial-block bootstrap 95% CIs use the configured resample count",
            not mismatched_n_boot,
            f"metrics with n_boot != Config.bootstrap_iterations "
            f"({cfg.bootstrap_iterations}): {mismatched_n_boot}.",
        ))

    wrong_n_repeats = [
        (row["Feature"], row["N_Repeats"])
        for _, row in fi.table.iterrows()
        if row["N_Repeats"] != cfg.n_permutation_outer_seeds
    ]
    checks.append((
        "Permutation-importance CI uses independent outer-seed replication",
        not wrong_n_repeats,
        f"expected N_Repeats == Config.n_permutation_outer_seeds "
        f"({cfg.n_permutation_outer_seeds}) for every feature; found: {wrong_n_repeats}.",
    ))


    if support is not None:
        obs_derived = (
            support.sparsity_threshold_k is not None
            and not np.isclose(support.sparsity_threshold_k, float(np.median(support.mean_knn_dist_s2)))
        )
        checks.append((
            "Empirical sparsity threshold is not the Scenario 2 trajectory median",
            obs_derived,
            f"sparsity_threshold_k ({support.sparsity_threshold_k}) equals the "
            f"Scenario 2 trajectory's own median NN distance - it must be "
            "derived from the observed/model-reference feature space instead.",
        ))
        if hasattr(support, "mean_knn_dist_s1") and support.mean_knn_dist_s1 is not None:
            not_s1_median = not np.isclose(
                support.sparsity_threshold_k, float(np.median(support.mean_knn_dist_s1))
            )
            checks.append((
                "Empirical sparsity threshold is not the Scenario 1 trajectory median",
                not_s1_median,
                f"sparsity_threshold_k ({support.sparsity_threshold_k}) equals the "
                f"Scenario 1 trajectory's own median NN distance.",
            ))

        checks.append((
            "Configured FEATURE_SPARSITY_PERCENTILE is applied",
            bool(np.isclose(support.sparsity_percentile, cfg.FEATURE_SPARSITY_PERCENTILE)),
            f"SupportDiagnostics.sparsity_percentile ({support.sparsity_percentile}) "
            f"!= Config.FEATURE_SPARSITY_PERCENTILE ({cfg.FEATURE_SPARSITY_PERCENTILE}).",
        ))

        checks.append((
            "Sparsity threshold is strictly below the formal support threshold (non-vacuous)",
            bool(support.sparsity_threshold_k < support.support_threshold_k),
            f"sparsity_threshold_k ({support.sparsity_threshold_k}) is not strictly below "
            f"support_threshold_k ({support.support_threshold_k}); the local-density "
            "diagnostic would be mathematically unable to ever report LOW density for a "
            "point that also passes formal feature support.",
        ))
        checks.append((
            "Density-moderate threshold is strictly below the sparsity threshold (non-vacuous)",
            bool(support.density_moderate_threshold_k < support.sparsity_threshold_k),
            f"density_moderate_threshold_k ({support.density_moderate_threshold_k}) is not "
            f"strictly below sparsity_threshold_k ({support.sparsity_threshold_k}); the "
            "MODERATE local-density band would be empty by construction.",
        ))

        recomputed_mask = sparse_region_mask(
            support.mean_knn_dist_s2, support.in_support_s2, support.sparsity_threshold_k
        )
        recomputed_n = int(recomputed_mask.sum()) if recomputed_mask is not None else 0
        recomputed_frac = (
            recomputed_n / len(recomputed_mask) if recomputed_mask is not None and len(recomputed_mask) else 0.0
        )
        if "feature_support_sparse_count" in scenario_summary_df.columns:
            csv_n = int(scenario_summary_df["feature_support_sparse_count"].iloc[0])
            checks.append((
                "Sparse-point count matches recomputed trajectory diagnostic",
                csv_n == recomputed_n,
                f"scenario_summary.csv feature_support_sparse_count ({csv_n}) != "
                f"recomputed count ({recomputed_n}).",
            ))
        if "feature_support_sparse_fraction" in scenario_summary_df.columns:
            csv_frac = float(scenario_summary_df["feature_support_sparse_fraction"].iloc[0])
            checks.append((
                "Sparse-point fraction matches recomputed trajectory diagnostic",
                bool(np.isclose(csv_frac, recomputed_frac)),
                f"scenario_summary.csv feature_support_sparse_fraction ({csv_frac}) != "
                f"recomputed fraction ({recomputed_frac}).",
            ))

        if recomputed_mask is not None:
            sparse_implies_supported = bool(np.all(support.in_support_s2[recomputed_mask]))
            checks.append((
                "Locally-sparse points remain within formal kNN support (not reclassified as extrapolation)",
                sparse_implies_supported,
                "one or more points flagged locally sparse are not also "
                "formally within kNN support - formal support and local "
                "sparsity must remain separate, non-overriding concepts.",
            ))

        if maximum_diagnostics_s2 is not None and negi is not None:
            s2_idx = negi.s2_optimum_idx
            expected_sparse_at_max = bool(
                support.in_support_s2[s2_idx]
                and support.mean_knn_dist_s2[s2_idx] > support.sparsity_threshold_k
            )
            checks.append((
                "maximum_locally_sparse corresponds to the NN distance at the actual NEGI argmax",
                maximum_diagnostics_s2["maximum_sparse_region"] == expected_sparse_at_max,
                f"maximum_diagnostics maximum_sparse_region "
                f"({maximum_diagnostics_s2['maximum_sparse_region']}) does not match "
                f"a direct recomputation at negi.s2_optimum_idx "
                f"({expected_sparse_at_max}).",
            ))

    if "spatial_refit_count" in scenario_summary_df.columns and uncertainty is not None:
        csv_refits = int(scenario_summary_df["spatial_refit_count"].iloc[0])
        actual_refits = int(uncertainty.n_refits_used)
        checks.append((
            "Spatial-refit count is exported and matches the actual completed refit count",
            csv_refits == actual_refits,
            f"scenario_summary.csv spatial_refit_count ({csv_refits}) != "
            f"actual completed refits, uncertainty.n_refits_used ({actual_refits}).",
        ))
        checks.append((
            "Exported spatial-refit count never exceeds the configured ceiling",
            csv_refits <= cfg.n_spatial_refits,
            f"scenario_summary.csv spatial_refit_count ({csv_refits}) exceeds "
            f"Config.n_spatial_refits ceiling ({cfg.n_spatial_refits}).",
        ))
    else:
        checks.append((
            "Spatial-refit count is exported and matches the actual completed refit count",
            False, "spatial_refit_count column missing from scenario_summary.csv "
            "(or uncertainty results unavailable).",
        ))

    if {"spatial_refit_lower_percentile", "spatial_refit_upper_percentile"} <= set(scenario_summary_df.columns):
        lo = float(scenario_summary_df["spatial_refit_lower_percentile"].iloc[0])
        hi = float(scenario_summary_df["spatial_refit_upper_percentile"].iloc[0])
        checks.append((
            "Spatial-refit percentiles match Config.uncertainty_percentiles (2.5/97.5)",
            bool(np.isclose(lo, 2.5) and np.isclose(hi, 97.5)
                 and 2.5 in cfg.uncertainty_percentiles and 97.5 in cfg.uncertainty_percentiles),
            f"exported spatial-refit percentiles ({lo}, {hi}) do not match the "
            f"2.5/97.5 percentiles defined in Config.uncertainty_percentiles "
            f"({cfg.uncertainty_percentiles}).",
        ))

    if "spatial_refit_uncertainty_type" in scenario_summary_df.columns:
        label_text = str(scenario_summary_df["spatial_refit_uncertainty_type"].iloc[0]).lower()
        not_overclaimed = ("confidence interval" not in label_text) and ("empirical" in label_text or "band" in label_text)
        checks.append((
            "Spatial-refit uncertainty label does not overclaim a formal confidence interval",
            not_overclaimed,
            f"spatial_refit_uncertainty_type ('{label_text}') either omits the "
            "empirical/band framing or claims a formal confidence interval.",
        ))

    _mislabeled_fields = []
    if "spatial_refit_uncertainty_type" in scenario_summary_df.columns:
        txt = str(scenario_summary_df["spatial_refit_uncertainty_type"].iloc[0]).lower()
        if "bootstrap" in txt:
            _mislabeled_fields.append(("scenario_summary.csv spatial_refit_uncertainty_type", txt))
        if "spatial-refit" not in txt and "spatial refit" not in txt:
            _mislabeled_fields.append(("scenario_summary.csv spatial_refit_uncertainty_type (missing own label)", txt))
    if calibration_table_df is not None and "Intercept CI note" in calibration_table_df.columns:
        cal_txt = str(calibration_table_df["Intercept CI note"].iloc[0]).lower()
        if "spatial-refit" in cal_txt or "spatial refit" in cal_txt:
            _mislabeled_fields.append(("calibration_metrics.csv Intercept CI note", cal_txt))
        if "bootstrap" not in cal_txt:
            _mislabeled_fields.append(("calibration_metrics.csv Intercept CI note (missing own label)", cal_txt))
    boot_ci_obj = getattr(validation, "spatial_block_bootstrap_ci", None)
    if boot_ci_obj:
        overlap_keys = [k for k in boot_ci_obj if "refit" in str(k).lower()]
        if overlap_keys:
            _mislabeled_fields.append(("validation.spatial_block_bootstrap_ci keys", str(overlap_keys)))
    checks.append((
        "Spatial-block bootstrap CI and spatial-refit uncertainty band are not interchanged in exports",
        not _mislabeled_fields,
        f"found field(s) using the other quantity's terminology (or missing "
        f"their own): {_mislabeled_fields}.",
    ))

    checks.append((
        "Calibration slope is finite and unaffected by the wording edit",
        bool(np.isfinite(validation.calibration_slope)),
        "validation.calibration_slope is not finite after the calibration-note wording edit.",
    ))

    if maximum_diagnostics_s1 is not None:
        checks.append((
            "Scenario 1 baseline maximum is not classified as a supported intervention optimum",
            maximum_diagnostics_s1["maximum_supports_interior_optimum"] is not True,
            "Scenario 1's maximum_supports_interior_optimum is True; the "
            "baseline (0% intervention) argmax must not be reported as a "
            "supported intervention optimum.",
        ))

    if support is not None:
        checks.append((
            "Empirical kNN reference distances exclude self-neighbours",
            support.empirical_knn_self_excluded,
            "the empirical kNN reference distribution's dropped column "
            "(intended to be each point's own row, at distance 0) did not "
            "match every point's own index - a possible self-neighbour bug "
            "in the empirical feature-support reference construction.",
        ))

    if maximum_diagnostics_s2 is not None:
        d2 = maximum_diagnostics_s2
        if d2["maximum_ci_includes_zero"] and (d2["maximum_near_boundary"] or d2["maximum_at_boundary"]):
            checks.append((
                "Scenario 2's zero-crossing near-boundary maximum remains non-robust",
                d2["maximum_robust"] is not True,
                "Scenario 2's maximum crosses zero and sits at/near the "
                "trajectory boundary, but maximum_robust is True.",
            ))


    if data is not None:
        train_groups = set(data.groups_train.unique())
        holdout_groups = set(data.groups.iloc[data.test_idx].unique())
        overlap_train_holdout = train_groups & holdout_groups
        checks.append((
            "Zero group overlap: training vs. confirmatory holdout",
            len(overlap_train_holdout) == 0,
            f"{len(overlap_train_holdout)} spatial group(s) appear in both "
            f"the training partition and the confirmatory holdout: "
            f"{sorted(overlap_train_holdout)[:10]}.",
        ))

        outer_cv = GroupKFold(n_splits=cfg.n_group_kfold_splits)
        fold_leak_details = []
        for fold_id, (tr_idx, te_idx) in enumerate(
            outer_cv.split(data.X, data.y, groups=data.groups), start=1
        ):
            tr_groups = set(data.groups.iloc[tr_idx].unique())
            te_groups = set(data.groups.iloc[te_idx].unique())
            fold_overlap = tr_groups & te_groups
            if fold_overlap:
                fold_leak_details.append(f"fold {fold_id}: {sorted(fold_overlap)[:5]}")
        checks.append((
            "Zero group overlap: outer GroupKFold train vs. test (all folds)",
            not fold_leak_details,
            "spatial group(s) shared between outer-train and outer-test in: "
            + "; ".join(fold_leak_details),
        ))

        train_min_match = data.feature_min.equals(data.X_train.min())
        train_max_match = data.feature_max.equals(data.X_train.max())
        checks.append((
            "Scenario admissibility bounds derived from training partition only",
            bool(train_min_match and train_max_match),
            "data.feature_min/feature_max do not match data.X_train's own "
            "min/max - the confirmatory holdout may be contaminating the "
            "feature-space support bounds used to admit scenario points.",
        ))

        if support is not None:
            checks.append((
                "Support kNN reference distribution sized to training partition",
                len(support.reference_mean_knn_dist) == len(data.X_train),
                f"support.reference_mean_knn_dist has "
                f"{len(support.reference_mean_knn_dist)} entries but "
                f"data.X_train has {len(data.X_train)} rows - the empirical "
                "support reference population does not match the training "
                "partition (possible holdout contamination).",
            ))

    if uncertainty is not None and uncertainty.n_unique_partitions is not None:
        n_refits_configured = len(uncertainty.partition_hashes)
        checks.append((
            "Spatial-refit uncertainty partitions are unique",
            uncertainty.n_unique_partitions == n_refits_configured,
            f"only {uncertainty.n_unique_partitions}/{n_refits_configured} "
            "spatial-refit partitions are unique (duplicate held-out group "
            "sets detected).",
        ))

    if comparison_df is not None and "Tuning iterations" in comparison_df.columns:
        tuned = comparison_df.set_index("Model")["Tuning iterations"]
        rf_gb_tuned = (
            "Random Forest" in tuned.index and "Gradient Boosting" in tuned.index
            and tuned["Random Forest"] == cfg.n_benchmark_search_iter
            and tuned["Gradient Boosting"] == cfg.n_benchmark_search_iter_gb
            and tuned["Random Forest"] > 0 and tuned["Gradient Boosting"] > 0
        )
        checks.append((
            "Random Forest and Gradient Boosting are genuinely tuned with their configured budgets",
            bool(rf_gb_tuned),
            f"expected RF to show {cfg.n_benchmark_search_iter} and GB to show "
            f"{cfg.n_benchmark_search_iter_gb} tuning iterations in "
            f"benchmark_comparison.csv; got {tuned.to_dict()}.",
        ))
        finite_metrics = comparison_df[["R2", "RMSE (°C)", "MAE (°C)"]].to_numpy()
        checks.append((
            "Model comparison table has no NaN/inf values",
            bool(np.all(np.isfinite(finite_metrics))),
            "benchmark_comparison.csv contains non-finite R2/RMSE/MAE values.",
        ))

    if summary is not None:
        p_eq_zero_pattern = _re.compile(r"p\s*=\s*0(?![\.\d])")
        summary_text_blob = " ".join(
            str(v) for v in summary.to_dataframe().to_numpy().ravel().tolist()
        )
        checks.append((
            "No literal 'p = 0' printed anywhere in the summary report",
            not bool(p_eq_zero_pattern.search(summary_text_blob)),
            "found a literal unqualified 'p = 0' in the exported summary "
            "text; underflowed p-values must use format_p_value()'s "
            "bounded 'p < ...' form instead.",
        ))

    if spatial_diagnostic is not None and spatial_diagnostic.get("moran"):
        m = spatial_diagnostic["moran"]
        if m.get("n_permutations", 0) > 0:
            floor = 1.0 / (m["n_permutations"] + 1)
            checks.append((
                "Moran's I permutation p-value respects its achievable floor",
                bool(m["permutation_p_value"] >= floor - 1e-12),
                f"permutation_p_value ({m['permutation_p_value']}) is below "
                f"the achievable floor 1/(n_permutations+1) ({floor}) for "
                f"n_permutations={m['n_permutations']}.",
            ))


    if endpoint_diagnostics_s2 is not None and negi is not None:
        de = endpoint_diagnostics_s2
        actual_endpoint_negi = float(negi.negi_s2[-1])
        actual_endpoint_pct  = float(negi.scenario_pct[-1])
        checks.append((
            "S2 endpoint_evaluated_negi matches trajectory last point",
            bool(np.isclose(de["endpoint_evaluated_negi"], actual_endpoint_negi)),
            f"endpoint_diagnostics endpoint_evaluated_negi "
            f"({de['endpoint_evaluated_negi']}) != negi.negi_s2[-1] "
            f"({actual_endpoint_negi}).",
        ))
        checks.append((
            "S2 endpoint_scenario_position matches trajectory last point",
            bool(np.isclose(de["endpoint_scenario_position"], actual_endpoint_pct)),
            f"endpoint_diagnostics endpoint_scenario_position "
            f"({de['endpoint_scenario_position']}) != negi.scenario_pct[-1] "
            f"({actual_endpoint_pct}).",
        ))
        stability_fields_present = all(
            k in de for k in
            ("endpoint_local_change", "endpoint_local_sign_change", "endpoint_local_stability")
        )
        checks.append((
            "S2 endpoint stability fields present",
            stability_fields_present,
            "endpoint_diagnostics_s2 is missing one or more of "
            "endpoint_local_change / endpoint_local_sign_change / endpoint_local_stability.",
        ))
        if uncertainty is not None and "negi_s2" in uncertainty.percentiles:
            ci_lo_actual = float(uncertainty.percentiles["negi_s2"][2.5][-1])
            ci_hi_actual = float(uncertainty.percentiles["negi_s2"][97.5][-1])
            checks.append((
                "S2 endpoint uncertainty band matches spatial-refit percentiles",
                bool(
                    np.isclose(de["endpoint_uncertainty_low"], ci_lo_actual)
                    and np.isclose(de["endpoint_uncertainty_high"], ci_hi_actual)
                ),
                f"endpoint_diagnostics uncertainty band "
                f"[{de['endpoint_uncertainty_low']}, {de['endpoint_uncertainty_high']}] "
                f"does not match uncertainty.percentiles['negi_s2'] "
                f"[{ci_lo_actual}, {ci_hi_actual}] at the last index.",
            ))

        if endpoint_diagnostics_df is not None and len(endpoint_diagnostics_df) > 0:
            row = endpoint_diagnostics_df.iloc[0]
            mismatches = [
                col for col in ("endpoint_evaluated_negi", "endpoint_ci_includes_zero", "endpoint_robust")
                if col in row and not (
                    row[col] == de[col]
                    or (isinstance(de[col], float) and np.isclose(row[col], de[col], equal_nan=True))
                )
            ]
            checks.append((
                "endpoint_diagnostics.csv matches endpoint robustness report",
                not mismatches,
                f"endpoint_diagnostics.csv disagrees with the in-memory endpoint "
                f"diagnostics dict on: {mismatches}.",
            ))

    if endpoint_diagnostics_s1 is not None:
        checks.append((
            "S1 endpoint diagnostics present (symmetry)",
            "endpoint_evaluated_negi" in endpoint_diagnostics_s1,
            "endpoint_diagnostics_s1 missing (Scenario 1 endpoint symmetry not computed).",
        ))

    _SYMMETRY_FIELD_MAP = [
        ("ci_low",          "maximum_uncertainty_low",      "endpoint_uncertainty_low"),
        ("ci_high",         "maximum_uncertainty_high",     "endpoint_uncertainty_high"),
        ("q25",             "maximum_q25",                  "endpoint_q25"),
        ("q75",             "maximum_q75",                  "endpoint_q75"),
        ("ci_includes_zero","maximum_ci_includes_zero",     "endpoint_ci_includes_zero"),
        ("uncertainty_type","maximum_uncertainty_type",     "endpoint_uncertainty_type"),
        ("locally_stable",  "maximum_local_stability",      "endpoint_local_stability"),
        ("biggest_jump",    "maximum_local_largest_change", "endpoint_local_change"),
        ("sign_flip",       "maximum_local_sign_change",    "endpoint_local_sign_change"),
        ("knn_distance",    "maximum_knn_distance",         "endpoint_knn_distance"),
        ("knn_support",     "maximum_knn_support",          "endpoint_knn_support"),
        ("sparse_region",   "maximum_sparse_region",        "endpoint_sparse_region"),
        ("local_density",   "maximum_local_density",        "endpoint_local_density"),
        ("robust",          "maximum_robust",               "endpoint_robust"),
    ]

    def _symmetry_eq(a, b) -> bool:
        if a is None or b is None:
            return a is b
        if isinstance(a, (float, np.floating)) or isinstance(b, (float, np.floating)):
            return bool(np.isclose(float(a), float(b), equal_nan=True))
        return a == b

    if (
        negi is not None and uncertainty is not None and support is not None
        and maximum_diagnostics_s2 is not None and endpoint_diagnostics_s2 is not None
    ):
        ref_max = evaluate_negi_candidate(
            negi.s2_optimum_idx, negi.negi_s2, negi.scenario_pct,
            uncertainty.percentiles.get("negi_s2"), support.in_support_s2,
            support.mean_knn_dist_s2, cfg, sparsity_threshold=support.sparsity_threshold_k,
            density_moderate_threshold=support.density_moderate_threshold_k,
            reference_knn_distribution=support.reference_mean_knn_dist,
        
            actual_n_refits=uncertainty.n_refits_used,
        )
        max_divergences = [
            core_key for core_key, max_key, _ in _SYMMETRY_FIELD_MAP
            if not _symmetry_eq(ref_max[core_key], maximum_diagnostics_s2[max_key])
        ]
        checks.append((
            "S2 maximum diagnostics use identical CI extraction as evaluate_negi_candidate",
            "ci_low" not in max_divergences and "ci_high" not in max_divergences
            and "q25" not in max_divergences and "q75" not in max_divergences,
            f"build_maximum_diagnostics CI fields diverge from evaluate_negi_candidate: {max_divergences}.",
        ))
        checks.append((
            "S2 maximum diagnostics use identical stability routine as evaluate_negi_candidate",
            "locally_stable" not in max_divergences and "biggest_jump" not in max_divergences
            and "sign_flip" not in max_divergences,
            f"build_maximum_diagnostics stability fields diverge from evaluate_negi_candidate: {max_divergences}.",
        ))
        checks.append((
            "S2 maximum diagnostics use identical support checks as evaluate_negi_candidate",
            "knn_distance" not in max_divergences and "knn_support" not in max_divergences
            and "sparse_region" not in max_divergences and "local_density" not in max_divergences,
            f"build_maximum_diagnostics support fields diverge from evaluate_negi_candidate: {max_divergences}.",
        ))
        checks.append((
            "S2 maximum diagnostics use identical robustness decision rule as evaluate_negi_candidate",
            "robust" not in max_divergences,
            f"build_maximum_diagnostics robust flag diverges from evaluate_negi_candidate: {max_divergences}.",
        ))

        ref_end = evaluate_negi_candidate(
            -1, negi.negi_s2, negi.scenario_pct,
            uncertainty.percentiles.get("negi_s2"), support.in_support_s2,
            support.mean_knn_dist_s2, cfg, sparsity_threshold=support.sparsity_threshold_k,
            density_moderate_threshold=support.density_moderate_threshold_k,
            reference_knn_distribution=support.reference_mean_knn_dist,
        
            actual_n_refits=uncertainty.n_refits_used,
        )
        end_divergences = [
            core_key for core_key, _, end_key in _SYMMETRY_FIELD_MAP
            if not _symmetry_eq(ref_end[core_key], endpoint_diagnostics_s2[end_key])
        ]
        checks.append((
            "S2 endpoint diagnostics use identical CI/stability/support/robustness core as evaluate_negi_candidate",
            not end_divergences,
            f"build_endpoint_diagnostics diverges from evaluate_negi_candidate: {end_divergences}.",
        ))

        _cross_idx = len(negi.negi_s2) - 1
        via_endpoint_wrapper = build_endpoint_diagnostics(
            "symmetry-cross-check", negi.negi_s2, negi.scenario_pct, _cross_idx,
            uncertainty.percentiles.get("negi_s2"), support.in_support_s2,
            support.mean_knn_dist_s2, cfg, sparsity_threshold=support.sparsity_threshold_k,
            density_moderate_threshold=support.density_moderate_threshold_k,
            reference_knn_distribution=support.reference_mean_knn_dist,
        
            actual_n_refits=uncertainty.n_refits_used,
        )
        via_maximum_core = evaluate_negi_candidate(
            _cross_idx, negi.negi_s2, negi.scenario_pct,
            uncertainty.percentiles.get("negi_s2"), support.in_support_s2,
            support.mean_knn_dist_s2, cfg, sparsity_threshold=support.sparsity_threshold_k,
            density_moderate_threshold=support.density_moderate_threshold_k,
            reference_knn_distribution=support.reference_mean_knn_dist,
        
            actual_n_refits=uncertainty.n_refits_used,
        )
        cross_mismatches = [
            core_key for core_key, _, end_key in _SYMMETRY_FIELD_MAP
            if not _symmetry_eq(via_maximum_core[core_key], via_endpoint_wrapper[end_key])
        ]
        checks.append((
            "Maximum-path and endpoint-path decision rules agree at a shared index",
            not cross_mismatches,
            f"evaluate_negi_candidate (as used by the maximum path) disagrees with "
            f"build_endpoint_diagnostics at index {_cross_idx} on: {cross_mismatches}.",
        ))

    if spatial_autocorr_df is not None:
        expected_series = {"Raw observed LST", "Linear Regression residuals", "XGBoost residuals"}
        present_series = set(spatial_autocorr_df["Series"]) if "Series" in spatial_autocorr_df.columns else set()
        checks.append((
            "spatial_autocorrelation_summary.csv contains all three series",
            expected_series.issubset(present_series),
            f"spatial_autocorrelation_summary.csv is missing series: "
            f"{expected_series - present_series}.",
        ))
        if "Moran's I" in spatial_autocorr_df.columns:
            checks.append((
                "spatial_autocorrelation_summary.csv Moran's I values finite",
                bool(np.all(np.isfinite(spatial_autocorr_df["Moran's I"].to_numpy(dtype=float)))),
                "spatial_autocorrelation_summary.csv contains a non-finite Moran's I value.",
            ))

    if joint_support_df is not None and len(joint_support_df):
        checks.append((
            "Joint-support diagnostics computed for both scenarios",
            set(joint_support_df["scenario"]) == {"Scenario 1", "Scenario 2"},
            f"joint_support_df covers scenarios {sorted(joint_support_df['scenario'].unique())}, expected both Scenario 1 and Scenario 2.",
        ))
        pct_col = joint_support_df["pct_points_within_threshold"].to_numpy(dtype=float)
        checks.append((
            "Joint-support percentages within [0, 100]",
            bool(np.all((pct_col >= 0.0) & (pct_col <= 100.0))),
            f"pct_points_within_threshold out of [0,100] range: {pct_col.tolist()}",
        ))
        pct_fields = joint_support_df[["worst_percentile", "endpoint_percentile"]].to_numpy(dtype=float)
        checks.append((
            "Joint-support density percentiles are valid",
            bool(np.all(np.isfinite(pct_fields)) and np.all(pct_fields >= 0.0)),
            "worst_percentile/endpoint_percentile contain non-finite or negative values.",
        ))

    if (joint_support_df is not None and len(joint_support_df)
            and maximum_diagnostics_s2 is not None and endpoint_diagnostics_s2 is not None):
        _s2_joint_status = joint_support_df.loc[
            joint_support_df["scenario"] == "Scenario 2", "joint_support_status"
        ].iloc[0]
        _max_block = format_maximum_classification(
            maximum_diagnostics_s2, joint_support_status=_s2_joint_status,
            publication_convergence_status=publication_convergence_status,
        )
        _end_block = format_endpoint_classification(
            endpoint_diagnostics_s2, joint_support_status=_s2_joint_status,
            publication_convergence_status=publication_convergence_status,
        )
        checks.append((
            "Joint multivariate support propagated into maximum/endpoint classification text",
            (f"Joint multivariate support" in _max_block and _s2_joint_status in _max_block
             and f"Joint multivariate support" in _end_block and _s2_joint_status in _end_block
             and "Feature support" not in _max_block and "Feature support" not in _end_block),
            "Rendered maximum/endpoint classification blocks do not correctly surface the "
            "Step 10b joint multivariate support status, or still contain the old combined "
            "'Feature support' wording.",
        ))

        _max_needs_caveat = (
            maximum_diagnostics_s2.get("maximum_ci_includes_zero") is False
            and publication_convergence_status not in (None, "STABILIZED")
        )
        _end_needs_caveat = (
            endpoint_diagnostics_s2.get("endpoint_ci_includes_zero") is False
            and publication_convergence_status not in (None, "STABILIZED")
        )
        if _max_needs_caveat or _end_needs_caveat:
            checks.append((
                "Non-stabilized zero-excluding bands carry the convergence caveat in exported text",
                (("Caveat (convergence)" in _max_block) if _max_needs_caveat else True)
                and (("Caveat (convergence)" in _end_block) if _end_needs_caveat else True),
                "A Scenario 2 maximum/endpoint band excludes zero while publication convergence "
                f"has not passed ({publication_convergence_status}), but the exported classification "
                "text is missing the provisional-band caveat.",
            ))

    if block_size_sensitivity_df is not None and len(block_size_sensitivity_df):
        evaluated_mask = block_size_sensitivity_df["robustness_classification"] != "SKIPPED"
        n_evaluated = int(evaluated_mask.sum())
        checks.append((
            "Spatial block-size sensitivity completed for >=2 valid block sizes (or all-skipped is explained)",
            n_evaluated >= 2 or n_evaluated == len(block_size_sensitivity_df) - int((~evaluated_mask).sum()),
            f"only {n_evaluated} of {len(block_size_sensitivity_df)} block-size settings were evaluated "
            "(rest skipped for having too few spatial groups).",
        ))
        primary_row = block_size_sensitivity_df[block_size_sensitivity_df["merge_factor"] == 1]
        checks.append((
            "Primary block size (merge_factor=1) present and unaltered by sensitivity sweep",
            len(primary_row) == 1,
            "merge_factor=1 (the unaltered primary spatial grouping) is missing from "
            "the block-size sensitivity table.",
        ))

    if publication_summary_path is not None:
        checks.append((
            "publication_summary.json exists",
            Path(publication_summary_path).is_file(),
            f"expected publication_summary.json at {publication_summary_path} "
            "but the file does not exist.",
        ))

    n_passed = sum(1 for _, passed, _ in checks if passed)
    n_failed = len(checks) - n_passed
    vlog_info(cfg, "QA checks:")
    for name, passed, _ in checks:
        vlog_info(cfg, f"  {name}: {'PASS' if passed else 'FAIL'}")

    if n_failed == 0:
        log_headline(
            f"Engineering/data-consistency QA: {n_passed}/{len(checks)} checks passed "
            "(implementation consistency only - does not certify scientific robustness, "
            "which is reported separately)"
        )
    else:
        failed_names = [name for name, passed, _ in checks if not passed]
        log_headline(
            f"Engineering/data-consistency QA: {n_passed}/{len(checks)} checks passed "
            f"({n_failed} FAILED) - implementation consistency only, not a scientific "
            "robustness certification"
        )
        log_headline("Failed checks: " + "; ".join(failed_names))

    qa_summary = {
        "n_total": len(checks),
        "n_passed": n_passed,
        "n_failed": n_failed,
        "check_names": [name for name, _, _ in checks],
        "failed_names": [name for name, passed, _ in checks if not passed],
    }
    try:
        save_json(qa_summary, cfg.data_dir / "qa_summary.json")
    except OSError as exc:  # noqa: BLE001
        log_warning(f"Could not write qa_summary.json: {exc}")

    qa_summary_df = pd.DataFrame(
        [{"check": name, "passed": passed, "detail": detail} for name, passed, detail in checks]
    )
    try:
        save_csv(qa_summary_df, cfg.data_dir / "qa_summary.csv")
    except OSError as exc:  # noqa: BLE001
        log_warning(f"Could not write qa_summary.csv: {exc}")

    failures = [detail for _, passed, detail in checks if not passed]
    if failures:
        raise AssertionError("QA check(s) failed: " + " | ".join(failures))

    return qa_summary


@dataclass
class DatasetBundle:
    df:           pd.DataFrame
    X:            pd.DataFrame
    y:            pd.Series
    groups:       pd.Series
    X_train:      pd.DataFrame
    X_test:       pd.DataFrame
    y_train:      pd.Series
    y_test:       pd.Series
    groups_train: pd.Series
    train_idx:    np.ndarray
    test_idx:     np.ndarray
    feature_min:  pd.Series
    feature_max:  pd.Series
    df_train:     pd.DataFrame


@dataclass
class ModelResults:
    model:       XGBRegressor
    best_params: dict
    search:      RandomizedSearchCV
    data:        DatasetBundle


@dataclass
class ValidationResults:
    nested_cv_r2:           float
    inner_tuning_r2:        float
    repeated_r2_scores:     np.ndarray
    repeated_rmse_scores:   np.ndarray
    repeated_mae_scores:    np.ndarray
    holdout_pred:           np.ndarray
    holdout_rmse:           float
    holdout_r2:             float
    holdout_mae:            float
    train_r2:               float
    test_r2:                float
    calibration_slope:      float
    calibration_intercept:  float
    calibration_bias:       float
    spatial_validation_df:  pd.DataFrame
    spatial_block_bootstrap_ci: dict = field(default_factory=dict)
    bootstrap_ci:           dict = field(default_factory=dict)

    @property
    def repeated_r2_mean(self) -> float:
        """Mean R² across repeated spatial-block validation folds."""
        return float(self.repeated_r2_scores.mean())

    @property
    def repeated_r2_ci95(self) -> float:
        """95% CI half-width of R² across repeated spatial-block validation folds."""
        return ci95(self.repeated_r2_scores)

    @property
    def repeated_rmse_mean(self) -> float:
        """Mean RMSE across repeated spatial-block validation folds."""
        return float(self.repeated_rmse_scores.mean())

    @property
    def repeated_rmse_ci95(self) -> float:
        """95% CI half-width of RMSE across repeated spatial-block validation folds."""
        return ci95(self.repeated_rmse_scores)

    @property
    def repeated_mae_mean(self) -> float:
        """Mean MAE across repeated spatial-block validation folds."""
        return float(self.repeated_mae_scores.mean())

    @property
    def repeated_mae_ci95(self) -> float:
        """95% CI half-width of MAE across repeated spatial-block validation folds."""
        return ci95(self.repeated_mae_scores)

    @property
    def train_test_r2_gap(self) -> float:
        """Difference between train R² and test R² (overfitting indicator)."""
        return self.train_r2 - self.test_r2


@dataclass
class ResidualDiagnostics:
    residuals:    np.ndarray
    skewness:     float
    kurtosis:     float
    bp_statistic: float
    bp_p_value:   float
    spearman_rho: float
    spearman_p:   float
    bp_method:    str = "auxiliary-regression approximation (statsmodels unavailable)"


@dataclass
class FeatureImportanceResult:
    table: pd.DataFrame


@dataclass
class ScenarioTrajectory:
    name:              str
    scenario_fraction: np.ndarray
    ndvi:              np.ndarray
    ndbi:              np.ndarray
    elevation:         float
    st_emis:           float
    st_emsd:           float
    lst:               np.ndarray
    out_of_bounds:     np.ndarray

    @property
    def scenario_pct(self) -> np.ndarray:
        """Scenario fraction expressed as a percentage (0-100)."""
        return self.scenario_fraction * 100

    def feature_frame(self) -> pd.DataFrame:
        """Assemble scenario percent plus every model feature into a DataFrame.

        NDVI/NDBI vary along the trajectory; Elevation, ST_EMIS and ST_EMSD
        are constant (fixed reference values) at every scenario point.
        """
        return pd.DataFrame({
            "Scenario (%)": self.scenario_pct,
            "NDVI":         self.ndvi,
            "NDBI":         self.ndbi,
            "Elevation":    self.elevation,
            "ST_EMIS":      self.st_emis,
            "ST_EMSD":      self.st_emsd,
        })

    def model_frame(self) -> pd.DataFrame:
        """The trajectory's model-input columns only, in FEATURES order."""
        return self.feature_frame()[FEATURES]

    def cooling(self, baseline_lst: float) -> np.ndarray:
        """Predicted cooling relative to `baseline_lst` at each scenario point."""
        return baseline_lst - self.lst


@dataclass
class ScenarioBaseline:
    baseline_lst:   float
    baseline_ndvi:  float
    baseline_ndbi:  float
    ndvi_target:    float
    fixed_elevation: float
    ndvi_ndbi_corr: float
    fixed_st_emis:  float
    fixed_st_emsd:  float

    def fixed_feature_values(self) -> dict[str, float]:
        """Reference values of every fixed (non-intervention) model feature."""
        return {
            "Elevation": self.fixed_elevation,
            "ST_EMIS":   self.fixed_st_emis,
            "ST_EMSD":   self.fixed_st_emsd,
        }


@dataclass
class NEGIResults:
    scenarios:            np.ndarray
    baseline_lst:         float
    delta_t_s1:           np.ndarray
    delta_t_s2:           np.ndarray
    reference_benefit_scale: float
    desal_energy_sqrt:    np.ndarray
    desal_energy_linear:  np.ndarray
    energy_norm_sqrt:     np.ndarray
    energy_norm_linear:   np.ndarray
    negi_s1:              np.ndarray
    negi_s2:              np.ndarray
    negi_s1_linear:       np.ndarray
    negi_s2_linear:       np.ndarray
    warming_fraction_s2:  float
    cooling_fraction_s2:  float
    max_cooling_s2:       float
    max_warming_s2:       float
    first_cooling_pct:    float

    @property
    def scenario_pct(self) -> np.ndarray:
        """Scenario fraction expressed as a percentage (0-100)."""
        return self.scenarios * 100

    @property
    def s1_optimum_idx(self) -> int:
        """Index of the maximum NEGI value along the Scenario 1 trajectory."""
        return int(np.argmax(self.negi_s1))

    @property
    def s2_optimum_idx(self) -> int:
        """Index of the maximum NEGI value along the Scenario 2 trajectory."""
        return int(np.argmax(self.negi_s2))


@dataclass
class SupportDiagnostics:
    scaler:              StandardScaler
    nn_dist_s1:          np.ndarray
    nn_p95_s1:           float
    nn_beyond_p95_s1:    np.ndarray
    nn_dist_s2:          np.ndarray
    nn_p95_s2:           float
    nn_beyond_p95_s2:    np.ndarray
    support_threshold_k: float
    mean_knn_dist_s1:    np.ndarray
    mean_knn_dist_s2:    np.ndarray
    in_support_s1:       np.ndarray
    in_support_s2:       np.ndarray
    extrap_s1_ok:        bool
    extrap_s2_ok:        bool
    sparsity_percentile: float
    sparsity_threshold_k: float
    density_moderate_percentile: float
    density_moderate_threshold_k: float
    empirical_knn_self_excluded: bool
    reference_mean_knn_dist: np.ndarray


@dataclass
class UncertaintyResults:
    negi_s1_folds:    np.ndarray
    negi_s2_folds:    np.ndarray
    lst_s1_folds:     np.ndarray
    lst_s2_folds:     np.ndarray
    cooling_s1_folds: np.ndarray
    cooling_s2_folds: np.ndarray
    negi_s1_mean:     np.ndarray
    negi_s1_std:      np.ndarray
    negi_s2_mean:     np.ndarray
    negi_s2_std:      np.ndarray
    percentiles:      dict
    table:            pd.DataFrame
    ndbi_range:       np.ndarray = None
    ndbi_sweep_folds: np.ndarray = None
    partition_hashes:      list = None
    n_unique_partitions:   int  = None
    model_stochasticity_varied: bool = False
    n_refits_used:            int  = None
    converged_early:          bool = None
    convergence_checkpoints:  list = None


@dataclass
class SensitivityResults:
    table:      pd.DataFrame
    plot_table: pd.DataFrame


@dataclass
class ConditionalBinResult:
    label:       str
    ndvi:        np.ndarray
    lst:         np.ndarray
    n_pixels:    int
    slope:       Optional[float]
    intercept:   Optional[float]
    slope_se:    Optional[float]
    slope_ci95:  Optional[float]
    r_squared:   Optional[float]
    p_value:     Optional[float]


@dataclass
class ContinuousAdjustmentResult:
    """Result of compute_ndvi_ndbi_continuous_adjustment(): the
    continuously-adjusted companion diagnostic to the NDBI-stratified
    (binned) NDVI-LST slope figures (Figure 6 / Figure S18).

    coef_* are the fitted LST ~ NDVI + NDBI + Elevation + NDVI:NDBI
    coefficients on data.df_train. implied_ndvi_slope[p] is
    d(LST)/d(NDVI) = coef_ndvi + coef_interaction * NDBI_at_percentile[p],
    i.e. the adjusted NDVI-LST slope evaluated at the p-th percentile of
    the observed training NDBI distribution. slope_ci_lower/upper[p] and
    slope_frac_positive[p] are the matching spatial-block-bootstrap 95%
    CI and fraction-of-bootstrap-draws-positive (resampling whole
    spatial_block groups and refitting the interaction model each draw - see
    spatial_block_bootstrap_ndvi_slope()), reported alongside the point
    estimate exactly as the underlying audit did. This is NOT a causal
    effect estimate - see the accompanying caveat text - only an
    observational association with continuous (rather than binned)
    control for NDBI and Elevation.
    """
    n_obs:              int
    r_squared:          float
    coef_ndvi:          float
    coef_ndbi:          float
    coef_elevation:     float
    coef_interaction:   float
    intercept:          float
    ndbi_percentiles:       dict
    implied_ndvi_slope:     dict
    slope_ci_lower:         dict = field(default_factory=dict)
    slope_ci_upper:         dict = field(default_factory=dict)
    slope_frac_positive:    dict = field(default_factory=dict)
    n_boot:                 int  = 0
    n_unique_blocks:        int  = 0


@dataclass
class VaryingCoefficientResult:
    """Result of compute_ndvi_ndbi_varying_coefficient_gam(): the estimator
    behind the redesigned Figure 6.

    Replaces the four-independent-regressions estimator with ONE continuous
    spline-basis varying-coefficient model fit once on data.df_train:

        LST = f_NDBI(NDBI) + f_Elevation(Elevation) + NDVI * g(NDBI) + error
        g(NDBI) = b_ndvi + sum_k gamma_k * BSpline_k(NDBI; df=ndbi_df_selected)

    f_NDBI and f_Elevation are themselves cubic B-spline smooths (this is
    the GAM-style additive control). g(NDBI) is the continuously-varying
    adjusted NDVI-LST slope - a single fitted curve, not four disconnected
    per-bin slopes, so there are no artificial discontinuities at the
    quartile bin edges.

    ndbi_grid / g_hat / ci_lower / ci_upper are the fitted curve and its
    spatial-block-bootstrap (spatial_block) 95% CI, evaluated on ndbi_grid
    (1st-99th percentile of training NDBI). obs_support / block_support are
    the local observation/unique-block counts in a +-0.01 NDBI window
    around each grid point, and low_support flags grid points with fewer
    than low_support_block_threshold unique blocks nearby - tail behaviour
    at those points should NOT be presented as reliable (requirement 6).

    ndbi_df_selected was chosen by 10-fold GroupKFold(spatial_block) cross-
    validated RMSE over ndbi_df_grid (an objective procedure - never hand-
    tuned to produce a desired curve shape). cv_table records the full
    grid search. sensitivity_table re-fits at neighbouring df values and
    reports whether the peak location and zero-crossing location persist.

    This is purely a diagnostic/reporting computation, exactly like
    ContinuousAdjustmentResult before it: it does not touch, refit, or
    otherwise influence the production XGBoost model, the NEGI formula, or
    either scenario trajectory. It is also not a causal estimate.
    """
    n_obs:                     int
    r_squared:                 float
    ndbi_df_selected:          int
    elev_df_fixed:             int
    cv_table:                  pd.DataFrame
    ndbi_grid:                 np.ndarray
    g_hat:                     np.ndarray
    ci_lower:                  np.ndarray
    ci_upper:                  np.ndarray
    obs_support:               np.ndarray
    block_support:             np.ndarray
    low_support_block_threshold: int
    low_support:               np.ndarray
    zero_crossings:            list
    n_boot:                    int
    n_unique_blocks:           int
    sensitivity_table:         pd.DataFrame


@dataclass
class ScenarioResults:
    """Per-scenario-point arrays bundled for CSV export."""
    scenario_percent:   np.ndarray
    ndvi:               np.ndarray
    ndbi:               np.ndarray
    predicted_lst:      np.ndarray
    deltaT:             np.ndarray
    clipped_cooling:    np.ndarray
    normalized_cooling: np.ndarray
    irrigation_cost:    np.ndarray
    negi:               np.ndarray

    def to_dataframe(self) -> pd.DataFrame:
        """Assemble the per-scenario-point arrays into a single export DataFrame."""
        return pd.DataFrame({
            "Scenario (%)":          self.scenario_percent,
            "NDVI":                  self.ndvi,
            "NDBI":                  self.ndbi,
            "Predicted LST":         self.predicted_lst,
            "DeltaT":                self.deltaT,
            "Clipped Cooling":       self.clipped_cooling,
            "Normalized Cooling":    self.normalized_cooling,
            "Irrigation/Energy Cost": self.irrigation_cost,
            "NEGI":                  self.negi,
        })


def compute_scenario_results(
    traj: ScenarioTrajectory,
    negi_arr: np.ndarray,
    energy_norm: np.ndarray,
    desal_energy: np.ndarray,
    baseline_lst: float,
) -> ScenarioResults:
    """Assemble a ScenarioResults from already-computed arrays (no new computation)."""
    delta_t = traj.cooling(baseline_lst)
    clipped_cooling = np.maximum(delta_t, 0.0)
    assert len(traj.scenario_pct) == len(traj.ndvi) == len(traj.ndbi) \
        == len(traj.lst) == len(negi_arr), "Scenario arrays must have equal length."
    return ScenarioResults(
        scenario_percent=traj.scenario_pct,
        ndvi=traj.ndvi,
        ndbi=traj.ndbi,
        predicted_lst=traj.lst,
        deltaT=delta_t,
        clipped_cooling=clipped_cooling,
        normalized_cooling=energy_norm,
        irrigation_cost=desal_energy,
        negi=negi_arr,
    )


def load_dataset(cfg: Config) -> pd.DataFrame:
    """Load, validate, and clean the input LST dataset from `cfg.data_path`."""
    if not cfg.data_path.is_file():
        raise FileNotFoundError(
            f"Input CSV was not found: {cfg.data_path}\n"
            "Set the NEGI_DATA_PATH environment variable or update Config.data_path."
        )
    df_file = pd.read_csv(cfg.data_path)
    n_file_rows = len(df_file)
    _model_cols = [c for c in list(FEATURES) + ["LST", "block_id"] if c in df_file.columns]
    _nan_rows = df_file.isna().any(axis=1)
    n_dropna_removed = int(_nan_rows.sum())
    n_dropna_aux_only = int((_nan_rows & df_file[_model_cols].notna().all(axis=1)).sum())
    df = df_file.dropna().reset_index(drop=True)
    del df_file
    print(f"[DATA PROVENANCE] file loaded = {cfg.data_path}")
    print(f"[DATA PROVENANCE] rows in file = {n_file_rows}")
    print(f"[DATA PROVENANCE] rows loaded = {len(df)} (after dropna; "
          f"{n_dropna_removed} removed, {n_dropna_aux_only} of them only via non-model columns)")
    if n_dropna_aux_only:
        log_warning(
            f"  [DATA] dropna() removed {n_dropna_aux_only} rows whose model columns "
            "were complete, because of NaNs in auxiliary columns. These rows never "
            "reach the purity filter. Run negi_provenance_audit.py for details."
        )
    required = set(FEATURES) | {"LST", "block_id"}
    missing  = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(
            f"Input CSV is missing required columns: {', '.join(missing)}. "
            f"The model requires the five predictors {FEATURES} plus LST and block_id."
        )
    if df.empty:
        raise ValueError("Input CSV has no complete rows after missing values are removed.")
    for _feat in FEATURES:
        if not pd.api.types.is_numeric_dtype(df[_feat]):
            raise ValueError(f"Predictor column {_feat!r} must be numeric, got dtype {df[_feat].dtype}.")
        if float(df[_feat].std(ddof=0)) == 0.0:
            log_warning(
                f"  [DATA] Predictor {_feat!r} has zero variance in the loaded "
                "data; it carries no information for the model."
            )
    print(f"[DATA PROVENANCE] model predictors = {FEATURES}")

    n_before_purity_filter = len(df)
    purity_accounting: Optional[dict] = None
    if "clean_vegetation_flag" in df.columns:
        _thr = float(cfg.purity_audit_ndvi_threshold)
        _clean = df["clean_vegetation_flag"] == 1
        _veg = df["NDVI"] > _thr
        _n_clean = int(_clean.sum())
        _n_veg = int(_veg.sum())
        _n_veg_clean = int((_veg & _clean).sum())
        purity_accounting = {
            "n_file_rows": n_file_rows,
            "n_dropna_removed": n_dropna_removed,
            "n_dropna_removed_aux_only": n_dropna_aux_only,
            "n_valid_before_purity_filter": n_before_purity_filter,
            "n_retained_clean": _n_clean,
            "n_rejected": n_before_purity_filter - _n_clean,
            "retention_rate_pct": 100.0 * _n_clean / max(n_before_purity_filter, 1),
            "rejection_rate_pct": 100.0 * (n_before_purity_filter - _n_clean) / max(n_before_purity_filter, 1),
            "ndvi_threshold": _thr,
            "n_ndvi_gt_threshold": _n_veg,
            "n_vegetated_retained": _n_veg_clean,
            "n_vegetated_rejected": _n_veg - _n_veg_clean,
            "vegetated_rejection_rate_pct": (100.0 * (_n_veg - _n_veg_clean) / _n_veg) if _n_veg else float("nan"),
            "n_nonvegetated_rejected": int(((~_veg) & (~_clean)).sum()),
            "vegetation_purity_filter": cfg.vegetation_purity_filter,
            "flag_semantics": ("clean_vegetation_flag=1 means non-vegetated OR vegetation "
                               "passing homogeneity + patch-size criteria; it does NOT mean vegetation"),
        }
    if "clean_vegetation_flag" not in df.columns:
        log_info(
            "  [DATA] clean_vegetation_flag column not found in "
            f"{cfg.data_path.name} (older/pre-v4 export?). "
            "Proceeding UNFILTERED regardless of "
            f"Config.vegetation_purity_filter={cfg.vegetation_purity_filter!r}. "
            "Re-run the v4 GEE script and export "
            "Jeddah_LST_Dataset_2023_v4_with_purity_flags.csv to enable "
            "mixed-pixel filtering."
        )
    elif cfg.vegetation_purity_filter == "purified":
        df = df[df["clean_vegetation_flag"] == 1].reset_index(drop=True)
        n_after = len(df)
        log_info(
            f"  [DATA] Vegetation purity filter APPLIED "
            f"(Config.vegetation_purity_filter='purified'): "
            f"{n_before_purity_filter} -> {n_after} rows "
            f"({n_before_purity_filter - n_after} dropped, "
            f"{100 * (n_before_purity_filter - n_after) / max(n_before_purity_filter, 1):.1f}% "
            "removed as likely mixed/isolated sub-pixel vegetation: villa "
            "courtyard gardens, isolated street trees, narrow road-median "
            "landscaping)."
        )
        if df.empty:
            raise ValueError(
                "Vegetation purity filter removed every row. Check "
                "clean_vegetation_flag values in the input CSV, or set "
                "Config(vegetation_purity_filter='raw') / "
                "NEGI_VEGETATION_PURITY_FILTER=raw to bypass."
            )
    elif cfg.vegetation_purity_filter == "raw":
        log_info(
            "  [DATA] Vegetation purity filter DISABLED "
            "(Config.vegetation_purity_filter='raw'): using all "
            f"{n_before_purity_filter} rows, including flagged mixed-pixel "
            "vegetation. This reproduces the pre-fix numbers for "
            "side-by-side comparison; it is not the default."
        )
    else:
        raise ValueError(
            "Config.vegetation_purity_filter must be 'purified' or 'raw', "
            f"got {cfg.vegetation_purity_filter!r}."
        )
    if purity_accounting is not None:
        pa = purity_accounting
        pa["flag_has_no_rejections"] = bool(pa["n_rejected"] == 0)
        if pa["flag_has_no_rejections"]:
            _dtype = ("PURIFIED UPSTREAM (flag has no 0 values - filtered in GEE "
                      "before sampling; rejection counts NOT observable from this CSV)")
        elif cfg.vegetation_purity_filter == "purified":
            _dtype = "PURIFIED (filtered in pipeline)"
        else:
            _dtype = "RAW (filter disabled)"
        pa["dataset_type"] = _dtype
        _w = 34
        _lines = [
            f"{'Valid pixels before purity filter':<{_w}}= {pa['n_valid_before_purity_filter']}",
            f"{'Pixels retained':<{_w}}= {pa['n_retained_clean']}",
            f"{'Pixels rejected':<{_w}}= {pa['n_rejected']}",
            f"{'Retention rate':<{_w}}= {pa['retention_rate_pct']:.2f}%",
            f"{'Rejection rate':<{_w}}= {pa['rejection_rate_pct']:.2f}%",
            f"{'NDVI > ' + format(pa['ndvi_threshold'], 'g') + ' pixels':<{_w}}= {pa['n_ndvi_gt_threshold']}",
            f"{'Vegetated pixels retained':<{_w}}= {pa['n_vegetated_retained']}",
            f"{'Vegetated pixels rejected':<{_w}}= {pa['n_vegetated_rejected']}",
            f"{'Vegetated rejection rate':<{_w}}= {pa['vegetated_rejection_rate_pct']:.2f}%",
            f"{'Non-vegetated pixels rejected':<{_w}}= {pa['n_nonvegetated_rejected']}",
            f"{'Dataset type':<{_w}}= {_dtype}",
        ]
        for _ln in _lines:
            print(f"[DATA PROVENANCE] {_ln}")
        if pa["flag_has_no_rejections"]:
            log_warning(
                "  [DATA] clean_vegetation_flag is 1 for EVERY row: this CSV was sampled "
                "AFTER the purity mask was applied (GEE v5 purified export). The "
                "'Pixels rejected = 0' above is an artifact of that, not a result. "
                + ("vegetation_purity_filter='raw' does NOT reproduce unfiltered data "
                   "from this file - load Jeddah_LST_Dataset_2023_raw.csv instead. "
                   if cfg.vegetation_purity_filter == "raw" else "")
                + "For audit items #5-#7 run negi_provenance_audit.py on the raw export."
            )
        if pa["n_nonvegetated_rejected"]:
            log_warning(
                f"  [DATA] {pa['n_nonvegetated_rejected']} pixels with NDVI <= "
                f"{pa['ndvi_threshold']} were rejected by clean_vegetation_flag. "
                "Under the documented semantics (non-vegetated pixels always pass) "
                "this should be 0 - check the NDVI threshold used in the GEE v4 script."
            )
        df.attrs["purity_accounting"] = pa
        try:
            save_json(pa, cfg.data_dir / "purity_filter_accounting.json", cfg=cfg)
        except Exception as exc:
            log_warning(f"Could not write purity_filter_accounting.json: {exc}")
    else:
        print("[DATA PROVENANCE] clean_vegetation_flag column absent — "
              "confirmed dataset type = RAW (pre-v4 export)")

    correlogram = estimate_response_correlogram(df, cfg)
    df["spatial_block"] = build_primary_spatial_blocks(df, cfg)
    n_groups = df["spatial_block"].nunique()
    log_info(
        f"  [DATA] Primary spatial blocking: {n_groups} blocks at "
        f"{cfg.spatial_block_cell_size_deg} deg cells (re-gridded from raw "
        f"lat/lon; replaces block_id, which encoded a ~0.01 deg/~1km grid)."
    )
    if n_groups < cfg.n_group_kfold_splits:
        raise ValueError(
            f"Found only {n_groups} spatial blocks, but GroupKFold requires at least "
            f"{cfg.n_group_kfold_splits}.  Reduce Config.n_group_kfold_splits, use a "
            "smaller Config.spatial_block_cell_size_deg, or supply more data."
        )
    df.attrs["response_correlogram"] = correlogram
    return df


def build_dataset_bundle(df: pd.DataFrame, cfg: Config) -> DatasetBundle:
    """Split features/target/groups into the confirmatory holdout and training sets."""
    X      = df[FEATURES]
    y      = df["LST"]
    groups = df["spatial_block"].astype(str)

    holdout_splitter = GroupShuffleSplit(
        n_splits=1, test_size=cfg.holdout_test_size, random_state=cfg.random_seed
    )
    train_idx, test_idx = next(holdout_splitter.split(X, y, groups=groups))

    X_train = X.iloc[train_idx]
    return DatasetBundle(
        df=df, X=X, y=y, groups=groups,
        X_train=X_train, X_test=X.iloc[test_idx],
        y_train=y.iloc[train_idx], y_test=y.iloc[test_idx],
        groups_train=groups.iloc[train_idx],
        train_idx=train_idx, test_idx=test_idx,
        feature_min=X_train.min(), feature_max=X_train.max(),
        df_train=df.loc[train_idx],
    )


def _fit_search_with_worker_retry(
    build_and_fit: Callable[[int], object],
    cfg: Config,
    label: str,
) -> object:
    """Run ``build_and_fit(n_jobs)`` - which must construct a FRESH
    estimator/search object using that ``n_jobs`` and return it already
    fitted - retrying with fewer parallel workers if a loky/joblib worker
    dies mid-fit (the same "A worker stopped while some jobs were given
    to the executor... too short worker timeout or ... memory leak"
    failure mode handled for the spatial-refit uncertainty loop, see
    Config.spatial_refit_n_jobs). A RandomizedSearchCV (or similar) fit
    cannot resume partway through, so a retry always restarts the WHOLE
    fit from scratch under a smaller worker pool - this changes only how
    many workers run concurrently, never the data, the search space, or
    the resulting best estimator (the search is re-run to completion with
    an identical random_state, so a successful retry reproduces the same
    result a memory-unconstrained run would have given).

    Used by fit_model(), run_nested_group_kfold_cv(),
    run_nested_group_kfold_cv_for_benchmark(), compare_models(), and the
    Step 6 permutation-importance call - every other n_jobs=-1 call site
    in the pipeline that hands parallelism off to sklearn/joblib
    internally rather than dispatching our own batch loop.
    """
    n_jobs = -1
    attempt = 0
    while True:
        try:
            result = build_and_fit(n_jobs)
            gc.collect()
            return result
        except Exception as exc:
            attempt += 1
            gc.collect()
            if attempt > cfg.search_fit_max_retries:
                raise
            current = n_jobs if n_jobs and n_jobs > 0 else (os.cpu_count() or 4)
            reduced = max(1, current // 2)
            log_info(
                f"    [{label}] hit {type(exc).__name__} ({exc}); retrying with "
                f"n_jobs={reduced} (was {n_jobs}), attempt "
                f"{attempt}/{cfg.search_fit_max_retries}. Refitting from scratch "
                "under the smaller pool - no partial result is reused, and no "
                "other step is affected."
            )
            n_jobs = reduced


def fit_model(
    data: DatasetBundle, cfg: Config, summary: SummaryLog
) -> ModelResults:
    """Fit the deployed model via RandomizedSearchCV(GroupKFold) hyperparameter
    tuning on the training split. This produces the single model used
    throughout the rest of the pipeline (scenarios, NEGI, uncertainty bands).

    IMPORTANT (Refactor item 1): ``search.best_score_`` here is the mean
    *inner tuning* score (RandomizedSearchCV's own GroupKFold mean_test_score
    for the winning hyperparameter combination). It is an inner-loop score,
    not an outer-fold nested-CV score, so it must never be reported as the
    "Nested GroupKFold CV" headline metric. The genuine nested estimate is
    computed separately and independently by ``run_nested_group_kfold_cv()``
    below, which never touches this model.
    """
    base_model = XGBRegressor(
        objective="reg:squarederror",
        random_state=cfg.random_seed,
        monotone_constraints=cfg.monotone_constraints,
        tree_method=cfg.xgb_tree_method,
        n_jobs=1,
    )

    def _build_and_fit(n_jobs: int) -> RandomizedSearchCV:
        s = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=cfg.param_distributions,
            n_iter=cfg.n_random_search_iter,
            scoring="r2",
            cv=GroupKFold(n_splits=cfg.n_group_kfold_splits),
            random_state=cfg.random_seed,
            n_jobs=n_jobs,
        )
        s.fit(data.X_train, data.y_train, groups=data.groups_train)
        return s

    search = _fit_search_with_worker_retry(_build_and_fit, cfg, "STEP 2 XGBoost tuning")
    log_info("\nBest Parameters:", search.best_params_)
    save_csv(pd.DataFrame(search.cv_results_), cfg.data_dir / "randomized_search_results.csv", cfg=cfg, debug_only=True)
    log_info(
        f"\n[1] Inner tuning score - RandomizedSearchCV mean_test_score "
        f"(GroupKFold, winning params): {search.best_score_:.3f}"
    )
    summary.log(
        "Training performance",
        "Inner tuning score (RandomizedSearchCV mean_test_score, GroupKFold)",
        f"{search.best_score_:.4f}",
    )
    return ModelResults(
        model=search.best_estimator_,
        best_params=search.best_params_,
        search=search,
        data=data,
    )


@dataclass
class NestedCVResults:
    """Genuine outer-fold nested GroupKFold CV results (Refactor items 1-2).

    Each outer fold refits a *fresh* RandomizedSearchCV (with its own inner
    GroupKFold) on the outer-training partition only, then scores the
    resulting model on the untouched outer-test partition. This is
    independent of, and never modifies, the model produced by fit_model().
    """
    fold_summary:  pd.DataFrame
    predictions:   pd.DataFrame
    outer_r2_mean: float
    outer_r2_std:  float


def run_nested_group_kfold_cv(
    data: DatasetBundle, cfg: Config, summary: SummaryLog
) -> NestedCVResults:
    """Compute the true Nested GroupKFold CV R² from outer-fold predictions.

    Outer loop: GroupKFold(cfg.n_group_kfold_splits) over the full dataset
    (X, y, groups) -- the same splitter class/fold count already used as the
    *inner* loop inside fit_model()'s RandomizedSearchCV, but instantiated
    independently here as the *outer* loop.
    Inner loop: for each outer-training partition, a fresh
    RandomizedSearchCV(GroupKFold) performs hyperparameter tuning exactly as
    in fit_model() (identical param_distributions, n_iter, scoring, seed),
    but using only that partition's data and groups.
    The outer-fold R²/RMSE/MAE are computed exclusively from predictions on
    each outer-test partition, which the corresponding inner search never saw.
    This function does not alter the model or hyperparameters used anywhere
    else in the pipeline; it exists solely to report an honest, leakage-free
    nested validation number (Refactor items 1, 2, 6).
    """
    X, y, groups = data.X, data.y, data.groups
    outer_cv = GroupKFold(n_splits=cfg.n_group_kfold_splits)

    fold_rows, pred_rows = [], []
    outer_r2_scores = []

    for fold_id, (outer_train_idx, outer_test_idx) in enumerate(
        outer_cv.split(X, y, groups=groups), start=1
    ):
        log_info(
            f"    Starting outer fold {fold_id}/{cfg.n_group_kfold_splits} "
            f"(inner search: {cfg.n_random_search_iter} iterations x "
            f"{cfg.n_group_kfold_splits}-fold CV = "
            f"{cfg.n_random_search_iter * cfg.n_group_kfold_splits} fits)..."
        )
        X_out_train, y_out_train = X.iloc[outer_train_idx], y.iloc[outer_train_idx]
        groups_out_train         = groups.iloc[outer_train_idx]
        X_out_test,  y_out_test  = X.iloc[outer_test_idx],  y.iloc[outer_test_idx]

        inner_model = XGBRegressor(
            objective="reg:squarederror",
            random_state=cfg.random_seed,
            monotone_constraints=cfg.monotone_constraints,
            tree_method=cfg.xgb_tree_method,
            n_jobs=1,
        )

        def _build_and_fit(n_jobs: int, _im=inner_model, _Xt=X_out_train, _yt=y_out_train, _gt=groups_out_train):
            s = RandomizedSearchCV(
                estimator=_im,
                param_distributions=cfg.param_distributions,
                n_iter=cfg.n_random_search_iter,
                scoring="r2",
                cv=GroupKFold(n_splits=cfg.n_group_kfold_splits),
                random_state=cfg.random_seed,
                n_jobs=n_jobs,
            )
            s.fit(_Xt, _yt, groups=_gt)
            return s

        inner_search = _fit_search_with_worker_retry(
            _build_and_fit, cfg, f"STEP 3a outer fold {fold_id}",
        )

        outer_pred = inner_search.best_estimator_.predict(X_out_test)
        fold_r2   = compute_r2(y_out_test, outer_pred)
        fold_rmse = compute_rmse(y_out_test, outer_pred)
        fold_mae  = compute_mae(y_out_test, outer_pred)
        outer_r2_scores.append(fold_r2)

        fold_rows.append({
            "outer_fold":               fold_id,
            "validation_r2":            fold_r2,
            "validation_rmse_c":        fold_rmse,
            "validation_mae_c":         fold_mae,
            "n_observations":           len(outer_test_idx),
            "selected_hyperparameters": json.dumps(inner_search.best_params_),
            "inner_tuning_score":       inner_search.best_score_,
        })
        for obs, pred_val in zip(y_out_test.to_numpy(), outer_pred):
            pred_rows.append({
                "observed": obs, "predicted": pred_val, "outer_fold": fold_id,
            })

        log_info(
            f"    Outer fold {fold_id}/{cfg.n_group_kfold_splits}: "
            f"nested R² = {fold_r2:.3f}, RMSE = {fold_rmse:.2f} °C "
            f"(n={len(outer_test_idx)})"
        )

    fold_summary = pd.DataFrame(fold_rows)
    predictions  = pd.DataFrame(pred_rows)
    outer_r2_scores = np.array(outer_r2_scores)

    save_csv(fold_summary, cfg.data_dir / "nested_cv_summary.csv", cfg=cfg, debug_only=True)
    save_csv(predictions,  cfg.data_dir / "nested_cv_predictions.csv", cfg=cfg, debug_only=True)

    outer_r2_mean = float(outer_r2_scores.mean())
    outer_r2_std  = float(outer_r2_scores.std())

    log_info(
        f"\n[1] Headline metric - Nested GroupKFold CV R² "
        f"(outer-fold predictions): {outer_r2_mean:.3f} ± {outer_r2_std:.3f}"
    )
    summary.log("Training performance", "Nested GroupKFold CV R² (headline, outer-fold)",
                f"{outer_r2_mean:.4f} ± {outer_r2_std:.4f}")
    summary.log("Training performance", "Nested GroupKFold CV outer folds",
                str(cfg.n_group_kfold_splits))

    return NestedCVResults(
        fold_summary=fold_summary,
        predictions=predictions,
        outer_r2_mean=outer_r2_mean,
        outer_r2_std=outer_r2_std,
    )


def run_nested_group_kfold_cv_for_benchmark(
    data: DatasetBundle, cfg: Config, summary: SummaryLog,
    model_name: str, base_model_factory: Callable[[], object],
    param_distributions: dict, n_iter: Optional[int] = None,
) -> NestedCVResults:
    """Same outer/inner nested GroupKFold procedure as
    ``run_nested_group_kfold_cv``, generalised to an arbitrary sklearn
    regressor (item 2 audit finding: paired model-comparison diagnostics).

    ``n_iter`` overrides ``cfg.n_benchmark_search_iter`` for the inner
    RandomizedSearchCV budget when the caller needs a model-specific
    budget (e.g. Gradient Boosting, which is far slower per fit than
    XGBoost/RandomForest at the same iteration count - see
    ``cfg.n_benchmark_search_iter_gb``). Defaults to
    ``cfg.n_benchmark_search_iter`` when not given, preserving prior
    behaviour for any other caller.

    ``GroupKFold.split()`` is deterministic given identical (X, y, groups)
    passed in identical order and takes no ``random_state`` - so calling it
    again here, with the same ``cfg.n_group_kfold_splits`` and the same
    ``data.X, data.y, data.groups``, reproduces *exactly* the same outer
    fold assignment already used for XGBoost in
    ``run_nested_group_kfold_cv``. That is what makes the resulting
    fold-level metrics genuinely paired (same held-out spatial blocks per
    fold index) rather than merely two independent CV runs.
    """
    n_iter = cfg.n_benchmark_search_iter if n_iter is None else n_iter
    X, y, groups = data.X, data.y, data.groups
    outer_cv = GroupKFold(n_splits=cfg.n_group_kfold_splits)

    fold_rows, pred_rows = [], []
    outer_r2_scores = []

    for fold_id, (outer_train_idx, outer_test_idx) in enumerate(
        outer_cv.split(X, y, groups=groups), start=1
    ):
        X_out_train, y_out_train = X.iloc[outer_train_idx], y.iloc[outer_train_idx]
        groups_out_train         = groups.iloc[outer_train_idx]
        X_out_test,  y_out_test  = X.iloc[outer_test_idx],  y.iloc[outer_test_idx]

        def _build_and_fit(n_jobs: int, _Xt=X_out_train, _yt=y_out_train, _gt=groups_out_train):
            s = RandomizedSearchCV(
                estimator=base_model_factory(),
                param_distributions=param_distributions,
                n_iter=n_iter,
                scoring="r2",
                cv=GroupKFold(n_splits=cfg.n_group_kfold_splits),
                random_state=cfg.random_seed,
                n_jobs=n_jobs,
            )
            s.fit(_Xt, _yt, groups=_gt)
            return s

        inner_search = _fit_search_with_worker_retry(
            _build_and_fit, cfg, f"{model_name} nested CV outer fold {fold_id}",
        )

        outer_pred = inner_search.best_estimator_.predict(X_out_test)
        fold_r2   = compute_r2(y_out_test, outer_pred)
        fold_rmse = compute_rmse(y_out_test, outer_pred)
        fold_mae  = compute_mae(y_out_test, outer_pred)
        outer_r2_scores.append(fold_r2)

        fold_rows.append({
            "outer_fold":               fold_id,
            "validation_r2":            fold_r2,
            "validation_rmse_c":        fold_rmse,
            "validation_mae_c":         fold_mae,
            "n_observations":           len(outer_test_idx),
            "selected_hyperparameters": json.dumps(inner_search.best_params_),
            "inner_tuning_score":       inner_search.best_score_,
        })
        for obs, pred_val in zip(y_out_test.to_numpy(), outer_pred):
            pred_rows.append({
                "observed": obs, "predicted": pred_val, "outer_fold": fold_id,
            })

    fold_summary = pd.DataFrame(fold_rows)
    predictions  = pd.DataFrame(pred_rows)
    outer_r2_scores = np.array(outer_r2_scores)
    outer_r2_mean = float(outer_r2_scores.mean())
    outer_r2_std  = float(outer_r2_scores.std())

    save_csv(
        fold_summary, cfg.data_dir / f"nested_cv_summary_{model_name.lower().replace(' ', '_')}.csv",
        cfg=cfg, debug_only=True,
    )

    summary.log(
        "Model comparison", f"{model_name} nested GroupKFold CV R² (outer-fold)",
        f"{outer_r2_mean:.4f} \u00b1 {outer_r2_std:.4f}",
    )

    return NestedCVResults(
        fold_summary=fold_summary,
        predictions=predictions,
        outer_r2_mean=outer_r2_mean,
        outer_r2_std=outer_r2_std,
    )


def paired_fold_comparison(
    nested_a: NestedCVResults, nested_b: NestedCVResults,
    name_a: str, name_b: str, cfg: Config, summary: SummaryLog,
) -> pd.DataFrame:
    """Descriptive paired fold-level differences between two models scored
    on the SAME outer GroupKFold folds (item 2). Reports mean and
    variability of the per-fold differences; does NOT construct a formal
    significance test - with only ``cfg.n_group_kfold_splits`` folds there
    are too few paired observations to justify one (item 2 explicitly asks
    that a formal test not be manufactured here)."""
    a = nested_a.fold_summary.set_index("outer_fold")
    b = nested_b.fold_summary.set_index("outer_fold")
    common_folds = a.index.intersection(b.index)
    rows = []
    for fold_id in common_folds:
        rows.append({
            "outer_fold": fold_id,
            f"r2_{name_a}":   a.loc[fold_id, "validation_r2"],
            f"r2_{name_b}":   b.loc[fold_id, "validation_r2"],
            "delta_r2":       a.loc[fold_id, "validation_r2"] - b.loc[fold_id, "validation_r2"],
            f"rmse_{name_a}": a.loc[fold_id, "validation_rmse_c"],
            f"rmse_{name_b}": b.loc[fold_id, "validation_rmse_c"],
            "delta_rmse":     a.loc[fold_id, "validation_rmse_c"] - b.loc[fold_id, "validation_rmse_c"],
            f"mae_{name_a}":  a.loc[fold_id, "validation_mae_c"],
            f"mae_{name_b}":  b.loc[fold_id, "validation_mae_c"],
            "delta_mae":      a.loc[fold_id, "validation_mae_c"] - b.loc[fold_id, "validation_mae_c"],
        })
    paired_df = pd.DataFrame(rows)

    mean_dr2, std_dr2   = paired_df["delta_r2"].mean(),   paired_df["delta_r2"].std()
    mean_drmse, std_drmse = paired_df["delta_rmse"].mean(), paired_df["delta_rmse"].std()
    mean_dmae, std_dmae   = paired_df["delta_mae"].mean(),  paired_df["delta_mae"].std()
    consistent_sign = bool(np.all(paired_df["delta_r2"] > 0) or np.all(paired_df["delta_r2"] < 0))

    summary.log(
        "Model comparison",
        f"Paired \u0394R\u00b2 ({name_a} - {name_b}) across {len(common_folds)} folds",
        f"{mean_dr2:.4f} \u00b1 {std_dr2:.4f} (sign consistent across all folds: {consistent_sign})",
    )
    summary.log(
        "Model comparison",
        f"Paired \u0394RMSE / \u0394MAE ({name_a} - {name_b})",
        f"{mean_drmse:+.4f} \u00b0C / {mean_dmae:+.4f} \u00b0C (mean across folds)",
    )
    log_info(
        f"\nPaired fold-level comparison ({name_a} vs {name_b}), "
        f"{len(common_folds)} matched outer folds:\n"
        f"  mean \u0394R\u00b2   = {mean_dr2:+.4f} \u00b1 {std_dr2:.4f}\n"
        f"  mean \u0394RMSE = {mean_drmse:+.4f} \u00b0C\n"
        f"  mean \u0394MAE  = {mean_dmae:+.4f} \u00b0C\n"
        f"  sign of \u0394R\u00b2 consistent across all folds: {consistent_sign}\n"
        "  (descriptive only - no formal significance test is claimed with "
        f"only {len(common_folds)} paired folds.)"
    )
    return paired_df


def evaluate_model(
    model_results: ModelResults, cfg: Config, summary: SummaryLog,
    nested_cv: "NestedCVResults",
) -> ValidationResults:
    """Run repeated spatial-block validation and the confirmatory holdout evaluation."""
    model, data = model_results.model, model_results.data
    X, y, groups = data.X, data.y, data.groups

    assert np.all(np.isfinite(X.values)),    "Feature matrix contains non-finite values."
    assert np.all(np.isfinite(y.values)),    "Target vector contains non-finite values."
    assert len(X) == len(y) == len(groups),  "X, y, and groups must have the same length."

    spatial_repeats = GroupShuffleSplit(
        n_splits=cfg.n_spatial_holdout_repeats,
        test_size=cfg.holdout_test_size,
        random_state=cfg.random_seed,
    )
    def _build_and_fit(n_jobs: int):
        return cross_validate(
            model, X, y, groups=groups, cv=spatial_repeats,
            scoring={
                "r2":   "r2",
                "rmse": "neg_root_mean_squared_error",
                "mae":  "neg_mean_absolute_error",
            },
            n_jobs=n_jobs,
        )

    cv_results = _fit_search_with_worker_retry(
        _build_and_fit, cfg, "repeated spatial-holdout cross_validate",
    )
    cv_scores      = cv_results["test_r2"]
    cv_rmse_scores = -cv_results["test_rmse"]
    cv_mae_scores  = -cv_results["test_mae"]

    log_info(f"\n[2] Repeated spatial-holdout R²   = "
             f"{cv_scores.mean():.3f} ± {ci95(cv_scores):.3f} (95% CI)")
    log_info(f"    Repeated spatial-holdout RMSE = "
             f"{cv_rmse_scores.mean():.2f} ± {ci95(cv_rmse_scores):.2f} °C (95% CI)")
    log_info(f"    Repeated spatial-holdout MAE  = "
             f"{cv_mae_scores.mean():.2f} ± {ci95(cv_mae_scores):.2f} °C (95% CI)")
    summary.log("Nested spatial CV performance", "Repeated spatial-holdout R² (mean ± 95% CI)",
                f"{cv_scores.mean():.4f} ± {ci95(cv_scores):.4f}")
    summary.log("Nested spatial CV performance", "Repeated spatial-holdout RMSE (mean ± 95% CI, °C)",
                f"{cv_rmse_scores.mean():.4f} ± {ci95(cv_rmse_scores):.4f}")
    summary.log("Nested spatial CV performance", "Repeated spatial-holdout MAE (mean ± 95% CI, °C)",
                f"{cv_mae_scores.mean():.4f} ± {ci95(cv_mae_scores):.4f}")

    pred = model.predict(data.X_test)
    assert np.all(np.isfinite(pred)),       "Model produced non-finite holdout predictions."
    assert len(pred) == len(data.y_test),   "Predictions and observations must have the same length."

    metrics  = compute_metrics(data.y_test, pred)
    rmse, r2, mae = metrics["rmse"], metrics["r2"], metrics["mae"]

    log_info("\n[3] Single untouched spatial-block holdout (confirmatory):")
    log_info(f"    RMSE: {rmse:.4f} °C")
    log_info(f"    R²:   {r2:.3f}")
    log_info(f"    MAE:  {mae:.4f} °C")
    summary.log("Holdout performance", "Single holdout RMSE (°C)", f"{rmse:.4f}")
    summary.log("Holdout performance", "Single holdout R²",         f"{r2:.4f}")
    summary.log("Holdout performance", "Single holdout MAE (°C)",   f"{mae:.4f}")

    train_r2 = model.score(data.X_train, data.y_train)
    test_r2  = model.score(data.X_test,  data.y_test)
    log_info(f"\nOverfitting check: Train R² = {train_r2:.3f}, Test R² = {test_r2:.3f}")
    if train_r2 - test_r2 > 0.08:
        log_info("Note: Train-test gap exceeds 0.08; report in Methods.")
    summary.log("Holdout performance", "Train R²",            f"{train_r2:.4f}")
    summary.log("Holdout performance", "Train-test R² gap",   f"{train_r2 - test_r2:.4f}")

    nested_fold_r2 = nested_cv.fold_summary["validation_r2"].to_numpy()
    nested_fold_rmse = nested_cv.fold_summary["validation_rmse_c"].to_numpy()
    nested_fold_mae = nested_cv.fold_summary["validation_mae_c"].to_numpy()

    spatial_validation_df = pd.DataFrame([
        {
            "Validation": "Nested GroupKFold CV (headline metric, outer-fold predictions)",
            "R2_Holdout": np.nan, "RMSE_C_Holdout": np.nan, "MAE_C_Holdout": np.nan,
            "R2_Mean": nested_cv.outer_r2_mean, "R2_95CI": ci95(nested_fold_r2),
            "RMSE_Mean": nested_fold_rmse.mean(), "RMSE_95CI": ci95(nested_fold_rmse),
            "MAE_Mean": nested_fold_mae.mean(), "MAE_95CI": ci95(nested_fold_mae),
            "n_test_pixels": nested_cv.fold_summary["n_observations"].sum(),
            "n_test_blocks": np.nan,
        },
        {
            "Validation": "RandomizedSearchCV inner tuning score (NOT nested; informational only)",
            "R2_Holdout": np.nan, "RMSE_C_Holdout": np.nan, "MAE_C_Holdout": np.nan,
            "R2_Mean": model_results.search.best_score_, "R2_95CI": np.nan,
            "RMSE_Mean": np.nan, "RMSE_95CI": np.nan, "MAE_Mean": np.nan, "MAE_95CI": np.nan,
            "n_test_pixels": np.nan, "n_test_blocks": np.nan,
        },
        {
            "Validation": "Repeated spatial-block validation",
            "R2_Holdout": np.nan, "RMSE_C_Holdout": np.nan, "MAE_C_Holdout": np.nan,
            "R2_Mean": cv_scores.mean(), "R2_95CI": ci95(cv_scores),
            "RMSE_Mean": cv_rmse_scores.mean(), "RMSE_95CI": ci95(cv_rmse_scores),
            "MAE_Mean": cv_mae_scores.mean(), "MAE_95CI": ci95(cv_mae_scores),
            "n_test_pixels": np.nan, "n_test_blocks": np.nan,
        },
        {
            "Validation": "Single untouched spatial-block holdout (confirmatory)",
            "R2_Holdout": r2, "RMSE_C_Holdout": rmse, "MAE_C_Holdout": mae,
            "R2_Mean": np.nan, "R2_95CI": np.nan, "RMSE_Mean": np.nan, "RMSE_95CI": np.nan,
            "MAE_Mean": np.nan, "MAE_95CI": np.nan,
            "n_test_pixels": len(data.X_test),
            "n_test_blocks": groups.iloc[data.test_idx].nunique(),
        },
    ])
    save_csv(spatial_validation_df, cfg.data_dir / "spatial_validation_summary.csv", cfg=cfg, debug_only=True)

    slope, intercept, bias = (
        metrics["calibration_slope"], metrics["calibration_intercept"], metrics["bias"]
    )
    assert np.isfinite(slope) and 0.1 < slope < 2.0, \
        f"Calibration slope {slope:.3f} is outside the expected range [0.1, 2.0]."

    log_info(
        f"\nCalibration slope = {slope:.3f} "
        f"(no post-hoc recalibration applied)."
    )
    log_info(f"Calibration intercept   : {intercept:.4f} °C")
    log_info(f"Mean bias               : {bias:.4f} °C")
    log_info(
        "Calibration diagnostics are reported descriptively only; "
        "no post-hoc recalibration was applied."
    )
    summary.log("Calibration", "Calibration slope",             f"{slope:.4f}")
    summary.log("Calibration", "Calibration intercept (°C)",    f"{intercept:.4f}")
    summary.log("Calibration", "Mean bias (°C)",                f"{bias:.4f}")
    summary.log("Calibration", "Recalibration applied", "False")
    summary.log(
        "Calibration", "Reporting note",
        "Calibration diagnostics are reported descriptively only; "
        "no post-hoc recalibration was applied.  RMSE and MAE are "
        "reported once in the Confirmatory Holdout section and are not "
        "repeated here.  MACE is omitted because it is identical to MAE.",
    )

    y_test_arr = (
        data.y_test.to_numpy() if hasattr(data.y_test, "to_numpy")
        else np.asarray(data.y_test)
    )
    boot_specs = {
        "rmse":                  _metric_rmse,
        "mae":                   _metric_mae,
        "bias":                  _metric_bias,
        "calibration_slope":     _metric_calibration_slope,
        "calibration_intercept": _metric_calibration_intercept,
        "r2":                    _metric_r2,
    }

    holdout_block_ids = data.groups.iloc[data.test_idx].to_numpy()

    def _compute_bootstrap_cis():
        spatial_block_ci = spatial_block_bootstrap_metrics(
            y_test_arr, pred, holdout_block_ids, boot_specs,
            n_boot=cfg.bootstrap_iterations, seed=cfg.bootstrap_seed,
        )
        obs_level_ci = {
            name: bootstrap_metric(
                y_test_arr, pred, fn,
                n_boot=cfg.bootstrap_iterations, seed=cfg.bootstrap_seed,
            )
            for name, fn in boot_specs.items()
        }
        return spatial_block_ci, obs_level_ci

    spatial_block_bootstrap_ci, bootstrap_ci = run_or_cached_stage(
        cfg, "bootstrap_ci", cfg.run_bootstrap,
        _cache_key(
            "bootstrap_ci", cfg,
            bootstrap_iterations=cfg.bootstrap_iterations,
            bootstrap_seed=cfg.bootstrap_seed,
            holdout_test_size=cfg.holdout_test_size,
            random_seed=cfg.random_seed,
            monotone_constraints=cfg.monotone_constraints,
            param_distributions=cfg.param_distributions,
            n_random_search_iter=cfg.n_random_search_iter,
            n_group_kfold_splits=cfg.n_group_kfold_splits,
        ),
        _compute_bootstrap_cis,
    )

    log_info(
        "\nPrimary holdout confidence intervals use spatial-block bootstrap "
        "resampling to account for within-block spatial dependence. "
        "Observation-level bootstrap intervals, if reported, are secondary "
        "descriptive estimates."
    )
    log_info(
        f"\nSpatial-block bootstrap 95% CIs ({cfg.bootstrap_iterations} resamples, "
        f"{spatial_block_bootstrap_ci['rmse']['n_unique_blocks']} unique holdout blocks) "
        "- PRIMARY:"
    )
    for name, ci_info in spatial_block_bootstrap_ci.items():
        log_info(
            f"  {name:22s}: {ci_info['point_estimate']:.4f}  "
            f"95% CI: {ci_info['lower']:.4f}-{ci_info['upper']:.4f}"
        )
        summary.log("Spatial-block bootstrap CI (primary)",
                    f"{name} (point estimate)", f"{ci_info['point_estimate']:.4f}")
        summary.log("Spatial-block bootstrap CI (primary)",
                    f"{name} 95% CI", f"{ci_info['lower']:.4f}-{ci_info['upper']:.4f}")

    log_info(
        f"\nObservation-level bootstrap 95% CIs ({cfg.bootstrap_iterations} resamples, "
        "holdout predictions) - SECONDARY descriptive estimates:"
    )
    for name, ci_info in bootstrap_ci.items():
        log_info(
            f"  {name:22s}: {ci_info['point_estimate']:.4f}  "
            f"95% CI: {ci_info['lower']:.4f}-{ci_info['upper']:.4f}"
        )
        summary.log("Observation-level bootstrap CI (secondary)",
                    f"{name} (point estimate)", f"{ci_info['point_estimate']:.4f}")
        summary.log("Observation-level bootstrap CI (secondary)",
                    f"{name} 95% CI", f"{ci_info['lower']:.4f}-{ci_info['upper']:.4f}")

    save_csv(
        pd.DataFrame([
            {
                "metric": name,
                "estimate": ci_info["point_estimate"],
                "ci_lower": ci_info["lower"],
                "ci_upper": ci_info["upper"],
                "n_bootstrap": ci_info["n_boot"],
                "resampling_unit": "spatial_block",
                "n_unique_blocks": ci_info["n_unique_blocks"],
            }
            for name, ci_info in spatial_block_bootstrap_ci.items()
        ]),
        cfg.data_dir / "spatial_block_bootstrap_summary.csv",
        cfg=cfg, debug_only=True,
    )
    save_csv(
        pd.DataFrame([
            {
                "metric": name, "point_estimate": ci_info["point_estimate"],
                "median": ci_info["median"],
                "ci_lower_2.5": ci_info["lower"], "ci_upper_97.5": ci_info["upper"],
                "n_boot": ci_info["n_boot"],
            }
            for name, ci_info in bootstrap_ci.items()
        ]),
        cfg.data_dir / "bootstrap_summary.csv",
        cfg=cfg, debug_only=True,
    )

    return ValidationResults(
        nested_cv_r2=nested_cv.outer_r2_mean,
        inner_tuning_r2=model_results.search.best_score_,
        repeated_r2_scores=cv_scores,
        repeated_rmse_scores=cv_rmse_scores,
        repeated_mae_scores=cv_mae_scores,
        holdout_pred=pred,
        holdout_rmse=rmse, holdout_r2=r2, holdout_mae=mae,
        train_r2=train_r2, test_r2=test_r2,
        calibration_slope=slope, calibration_intercept=intercept, calibration_bias=bias,
        spatial_validation_df=spatial_validation_df,
        spatial_block_bootstrap_ci=spatial_block_bootstrap_ci,
        bootstrap_ci=bootstrap_ci,
    )


def compute_residual_diagnostics(
    validation: ValidationResults, data: DatasetBundle
) -> ResidualDiagnostics:
    """Compute holdout residual skewness, kurtosis, Breusch-Pagan, and Spearman stats."""
    log_info("  [TIMING] compute_residual_diagnostics: entered")
    residuals = data.y_test - validation.holdout_pred
    residuals_arr = (
        residuals.values if hasattr(residuals, "values") else np.asarray(residuals)
    )
    pred = validation.holdout_pred

    skewness = scipy_stats.skew(residuals_arr)
    kurtosis = scipy_stats.kurtosis(residuals_arr)

    bp = _breusch_pagan_test(np.zeros_like(pred), pred, residuals_arr)

    spearman_corr, spearman_p = scipy_stats.spearmanr(np.abs(residuals_arr), pred)

    return ResidualDiagnostics(
        residuals=residuals_arr,
        skewness=skewness, kurtosis=kurtosis,
        bp_statistic=bp["statistic"], bp_p_value=bp["p_value"], bp_method=bp["method"],
        spearman_rho=spearman_corr, spearman_p=spearman_p,
    )


def classify_heteroscedasticity(resid: "ResidualDiagnostics") -> str:
    """Return the short heteroscedasticity classification label.

    Logic mirrors report_residual_diagnostics's existing branching exactly
    (same 0.05 / |rho| < 0.15 thresholds); extracted only so the concise
    headline and the detailed report read the same classification instead
    of maintaining two copies of the same test.
    """
    if resid.bp_p_value < 0.05 or resid.spearman_p < 0.05:
        if abs(resid.spearman_rho) < 0.15:
            return "weak heteroscedasticity"
        return "heteroscedasticity"
    return "no strong evidence of heteroscedasticity"


def report_residual_diagnostics(
    resid: ResidualDiagnostics, cfg: Config, summary: SummaryLog
) -> None:
    """Log and record residual skewness, kurtosis, Breusch-Pagan, and interpretation text."""
    log_info(f"\nResidual skewness       : {resid.skewness:.4f}")
    log_info(f"Residual kurtosis       : {resid.kurtosis:.4f} (excess; 0 = normal)")
    summary.log("Residual diagnostics", "Residual skewness",          f"{resid.skewness:.4f}")
    summary.log("Residual diagnostics", "Residual kurtosis (excess)", f"{resid.kurtosis:.4f}")

    log_info(f"\nBreusch-Pagan test ({resid.bp_method}): "
             f"statistic = {resid.bp_statistic:.4f}, p = {resid.bp_p_value:.4g}")
    log_info(f"Spearman |resid| vs pred: rho = {resid.spearman_rho:.4f}, "
             f"p = {resid.spearman_p:.4g}")

    if resid.bp_p_value < 0.05 or resid.spearman_p < 0.05:
        if abs(resid.spearman_rho) < 0.15:
            log_info(
                "  -> Statistically detectable but practically weak heteroscedasticity "
                "(|Spearman rho| < 0.15); likely driven by large sample size rather "
                "than a severe modelling problem."
            )
        else:
            log_info(f"  -> {interpretation_text('heteroscedasticity')}")
            summary.add_conclusion(interpretation_text("heteroscedasticity"))
    else:
        log_info("  -> No strong evidence of heteroscedasticity at the 0.05 level.")

    summary.log("Residual diagnostics", "Breusch-Pagan statistic",
                f"{resid.bp_statistic:.4f}")
    summary.log("Residual diagnostics", "Breusch-Pagan p-value",
                f"{resid.bp_p_value:.4g}")
    summary.log("Residual diagnostics", "Breusch-Pagan implementation", resid.bp_method)
    summary.log("Residual diagnostics", "Spearman |residual| vs predicted (rho)",
                f"{resid.spearman_rho:.4f}")
    summary.log("Residual diagnostics", "Spearman |residual| vs predicted (p)",
                f"{resid.spearman_p:.4g}")


def _find_spatial_columns(
    df: pd.DataFrame, cfg: Config
) -> Optional[tuple[str, str]]:
    """Return (x_col, y_col) or None when no coordinate columns are found."""
    x_col = next((c for c in cfg.spatial_x_candidates if c in df.columns), None)
    y_col = next((c for c in cfg.spatial_y_candidates if c in df.columns), None)
    return (x_col, y_col) if (x_col is not None and y_col is not None) else None


def build_spatial_grid_blocks(
    x: np.ndarray, y: np.ndarray, cell_size_deg: float,
) -> np.ndarray:
    """Grid raw coordinates into square cells `cell_size_deg` degrees on a
    side, returning one string block id per point ("<x_cell>_<y_cell>").

    This is a genuine geographic re-grid of the ORIGINAL coordinates - not
    an ordinal merge of a pre-existing label column - so it can be
    evaluated at any cell size (coarser or finer) and always produces
    spatially contiguous blocks: two points fall in the same block iff
    they fall in the same physical grid cell.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if cell_size_deg <= 0:
        raise ValueError(f"cell_size_deg must be positive, got {cell_size_deg}.")
    x_idx = np.floor(x / cell_size_deg).astype(np.int64)
    y_idx = np.floor(y / cell_size_deg).astype(np.int64)
    return np.array([f"{a}_{b}" for a, b in zip(x_idx, y_idx)])


def build_primary_spatial_blocks(df: pd.DataFrame, cfg: Config) -> pd.Series:
    """Derive the PRIMARY spatial-block grouping directly from the
    dataset's raw lat/lon coordinates, at `cfg.spatial_block_cell_size_deg`.

    Replaces the previous `block_id` column as the single grouping fed to
    every GroupKFold split, GroupShuffleSplit holdout, and spatial-block
    bootstrap in this pipeline (see the Config.spatial_block_cell_size_deg
    docstring for why: block_id turned out to be a ~1km re-encoding of
    these same coordinates, too fine relative to the empirical spatial
    autocorrelation range to serve as the blocking unit).
    """
    coord_cols = _find_spatial_columns(df, cfg)
    if coord_cols is None:
        raise ValueError(
            "build_primary_spatial_blocks: no coordinate columns found "
            f"(looked for {cfg.spatial_x_candidates} / {cfg.spatial_y_candidates}). "
            "The primary spatial-block grouping requires raw coordinates."
        )
    x_col, y_col = coord_cols
    blocks = build_spatial_grid_blocks(
        df[x_col].to_numpy(), df[y_col].to_numpy(), cfg.spatial_block_cell_size_deg,
    )
    return pd.Series(blocks, index=df.index, name="spatial_block")


def regrid_spatial_blocks(df: pd.DataFrame, cfg: Config, size_multiplier: float) -> pd.Series:
    """Re-grid the ORIGINAL lat/lon at `cfg.spatial_block_cell_size_deg *
    size_multiplier`, for the block-size sensitivity sweep
    (run_spatial_block_size_sensitivity). `size_multiplier=1` reproduces
    `build_primary_spatial_blocks` exactly. Unlike the old
    `_coarsen_spatial_blocks` (an ordinal merge of block_id labels in
    sorted order, which does not track geographic adjacency), every
    multiplier here is a genuine coarser/finer physical grid.
    """
    coord_cols = _find_spatial_columns(df, cfg)
    if coord_cols is None:
        raise ValueError(
            "regrid_spatial_blocks: no coordinate columns found "
            f"(looked for {cfg.spatial_x_candidates} / {cfg.spatial_y_candidates})."
        )
    x_col, y_col = coord_cols
    cell_size = cfg.spatial_block_cell_size_deg * float(size_multiplier)
    blocks = build_spatial_grid_blocks(df[x_col].to_numpy(), df[y_col].to_numpy(), cell_size)
    return pd.Series(blocks, index=df.index, name="spatial_block")


def estimate_response_correlogram(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Empirical distance-binned spatial-autocorrelation estimate
    (a "correlogram") computed on the RAW response (LST) directly -
    deliberately not on any model's residuals, so that the choice of
    `spatial_block_cell_size_deg` (made from this function's output) can
    never be circular with a model that itself depends on the blocking it
    justifies (the same Model->Blocking->Model circularity already guarded
    against elsewhere in this pipeline for the Scenario 2 residual check).

    Method: draws `cfg.correlogram_n_pairs` random pairs of rows (with
    replacement; Monte-Carlo estimate, not an exhaustive O(n^2) computation
    - infeasible at this n), computes each pair's planar great-circle-ish
    distance (flat-earth approximation using local km-per-degree scaling;
    adequate at the <=cfg.correlogram_max_dist_km extent considered here),
    bins pairs by distance, and reports the mean standardized cross-product
    z_i * z_j per bin (z = (LST - mean) / std) - a Moran's-I-style spatial
    autocorrelation estimate as a function of distance, without requiring a
    full spatial weights matrix.

    Returns a DataFrame with one row per distance bin: bin edges, bin
    midpoint (km), n_pairs actually landing in that bin, and the
    autocorrelation estimate. Values near 1 at short distances that decay
    toward ~0 mark the range past which two points' LST values are
    effectively spatially independent.
    """
    coord_cols = _find_spatial_columns(df, cfg)
    if coord_cols is None:
        raise ValueError(
            "estimate_response_correlogram: no coordinate columns found "
            f"(looked for {cfg.spatial_x_candidates} / {cfg.spatial_y_candidates})."
        )
    x_col, y_col = coord_cols
    lon = df[x_col].to_numpy(dtype=float)
    lat = df[y_col].to_numpy(dtype=float)
    lst = df["LST"].to_numpy(dtype=float)
    n = len(df)

    z = (lst - lst.mean()) / lst.std(ddof=0)

    km_per_deg_lat = 110.574
    km_per_deg_lon = 111.320 * np.cos(np.radians(lat.mean()))

    rng = np.random.default_rng(cfg.correlogram_seed)
    n_pairs = int(cfg.correlogram_n_pairs)
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n, size=n_pairs)
    keep = i != j
    i, j = i[keep], j[keep]

    dx_km = (lon[i] - lon[j]) * km_per_deg_lon
    dy_km = (lat[i] - lat[j]) * km_per_deg_lat
    dist_km = np.sqrt(dx_km**2 + dy_km**2)

    max_dist = float(cfg.correlogram_max_dist_km)
    in_range = dist_km <= max_dist
    dist_km, zi_zj = dist_km[in_range], (z[i] * z[j])[in_range]

    n_bins = int(cfg.correlogram_n_bins)
    edges = np.linspace(0.0, max_dist, n_bins + 1)
    bin_idx = np.clip(np.digitize(dist_km, edges) - 1, 0, n_bins - 1)

    rows = []
    for b in range(n_bins):
        mask = bin_idx == b
        n_in_bin = int(mask.sum())
        rows.append({
            "bin_lo_km": edges[b],
            "bin_hi_km": edges[b + 1],
            "bin_mid_km": 0.5 * (edges[b] + edges[b + 1]),
            "n_pairs": n_in_bin,
            "mean_dist_km": float(dist_km[mask].mean()) if n_in_bin else np.nan,
            "autocorrelation": float(zi_zj[mask].mean()) if n_in_bin else np.nan,
        })
    table = pd.DataFrame(rows)

    below = table[(table["autocorrelation"] <= 0.1) & table["n_pairs"].gt(0)]
    decorrelation_km = float(below["bin_mid_km"].iloc[0]) if len(below) else np.nan
    log_info(
        f"  [CORRELOGRAM] LST spatial autocorrelation from {len(dist_km)} sampled "
        f"pairs (<= {max_dist:.0f} km): "
        f"{table['autocorrelation'].iloc[0]:.3f} at ~{table['bin_mid_km'].iloc[0]:.2f} km -> "
        f"first drops <= 0.1 at ~{decorrelation_km:.1f} km"
        if np.isfinite(decorrelation_km) else
        f"  [CORRELOGRAM] LST spatial autocorrelation does not drop to <= 0.1 "
        f"within {max_dist:.0f} km of sampled pairs."
    )
    table.attrs["decorrelation_km"] = decorrelation_km
    return table


def global_morans_i(
    values: np.ndarray, x: np.ndarray, y: np.ndarray, k: int = 8,
    n_permutations: int = 0, permutation_seed: int = 42,
    max_n_for_permutation: int = 8000,
) -> dict:
    """Global Moran's I via a k-nearest-neighbour row-standardised weight
    matrix, with an optional permutation-based inference procedure
    (item 3 audit finding).

    The normal-approximation morans_i/expected_i/z_score/p_value
    calculation below is unchanged from the original implementation -
    preserving the existing scientific result. Permutation inference
    (preferred by the audit over the normal approximation alone) is added
    as additional fields when ``n_permutations > 0``.
    """
    values = np.asarray(values, dtype=float)
    coords = np.column_stack([np.asarray(x, dtype=float), np.asarray(y, dtype=float)])
    n    = len(values)
    k_eff = min(k, n - 1)
    if n < 4 or k_eff < 1:
        return {
            "morans_i": float("nan"), "expected_i": float("nan"),
            "z_score":  float("nan"), "p_value":    float("nan"), "n": n,
            "weighting_method": "k-nearest-neighbour", "k": k, "k_effective": k_eff,
            "row_standardized": True, "n_connected_components": None,
            "n_islands": None, "n_permutations": 0,
            "permutation_p_value": float("nan"),
        }

    _t_knn = time.perf_counter()
    nn = NearestNeighbors(n_neighbors=k_eff + 1, algorithm="kd_tree").fit(coords)
    _, neighbor_idx = nn.kneighbors(coords)
    neighbor_idx = neighbor_idx[:, 1:]
    log_info(f"  [TIMING] global_morans_i: kNN fit+query (n={n}): "
             f"{time.perf_counter() - _t_knn:.2f}s")

    z = values - values.mean()

    _t_w = time.perf_counter()
    row_idx = np.repeat(np.arange(n), k_eff)
    col_idx = neighbor_idx.ravel()
    W = csr_matrix(
        (np.full(row_idx.shape, 1.0 / k_eff), (row_idx, col_idx)), shape=(n, n)
    )
    s0 = float(W.sum())

    numerator   = float(z @ (W @ z))
    denominator = float(np.sum(z ** 2))
    morans_i    = (n / s0) * safe_divide(
        np.array([numerator]), np.array([denominator]), fill=0.0
    )[0]
    log_info(f"  [TIMING] global_morans_i: weight matrix + I statistic: "
             f"{time.perf_counter() - _t_w:.2f}s")

    expected_i = -1.0 / (n - 1)
    s1 = 0.5 * float((W + W.T).power(2).sum())
    row_sums = np.asarray(W.sum(axis=1)).ravel()
    col_sums = np.asarray(W.sum(axis=0)).ravel()
    s2 = float(np.sum((row_sums + col_sums) ** 2))
    b2 = (np.sum(z ** 4) / n) / ((np.sum(z ** 2) / n) ** 2) if denominator > 0 else np.nan
    var_i = (
        (
            n * ((n ** 2 - 3 * n + 3) * s1 - n * s2 + 3 * s0 ** 2)
            - b2 * ((n ** 2 - n) * s1 - 2 * n * s2 + 6 * s0 ** 2)
        )
        / ((n - 1) * (n - 2) * (n - 3) * s0 ** 2)
    ) - expected_i ** 2

    z_score = (
        safe_divide(
            np.array([morans_i - expected_i]),
            np.array([np.sqrt(max(var_i, 0.0))]),
            fill=0.0,
        )[0]
        if var_i > 0 else float("nan")
    )
    p_value = (
        float(2 * (1 - scipy_stats.norm.cdf(abs(z_score))))
        if np.isfinite(z_score) else float("nan")
    )

    _t_cc = time.perf_counter()
    adjacency = csr_matrix(
        (np.ones(row_idx.shape, dtype=np.int8), (row_idx, col_idx)), shape=(n, n)
    )
    adjacency = (adjacency + adjacency.T).astype(bool).astype(np.int8)
    n_components, component_labels = connected_components(
        adjacency, directed=False
    )
    component_sizes = np.bincount(component_labels)
    n_islands = int(np.sum(component_sizes == 1))
    log_info(f"  [TIMING] global_morans_i: connected_components: "
             f"{time.perf_counter() - _t_cc:.2f}s")

    result = {
        "morans_i":  float(morans_i), "expected_i": float(expected_i),
        "z_score":   float(z_score),  "p_value":    p_value, "n": n,
        "weighting_method": "k-nearest-neighbour", "k": k, "k_effective": k_eff,
        "row_standardized": True,
        "n_connected_components": int(n_components), "n_islands": n_islands,
        "n_permutations": 0, "permutation_p_value": float("nan"),
    }

    if n_permutations > 0:
        if n > max_n_for_permutation:
            result["n_permutations"] = 0
            result["permutation_p_value"] = float("nan")
            result["permutation_skipped_reason"] = (
                f"n={n} exceeds max_n_for_permutation={max_n_for_permutation}; "
                "permutation inference skipped, normal-approximation p-value only."
            )
        else:
            _t_perm = time.perf_counter()
            rng = np.random.default_rng(permutation_seed)
            perm_i = np.empty(n_permutations, dtype=float)
            heartbeat_every = max(1, n_permutations // 10)
            for p_idx in range(n_permutations):
                z_perm = rng.permutation(z)
                neighbor_sum = z_perm[neighbor_idx].sum(axis=1)
                numerator_perm = float(np.dot(z_perm, neighbor_sum)) / k_eff
                perm_i[p_idx] = (n / s0) * safe_divide(
                    np.array([numerator_perm]), np.array([denominator]), fill=0.0
                )[0]
                if (p_idx + 1) % heartbeat_every == 0:
                    log_info(
                        f"  [TIMING] global_morans_i: permutation "
                        f"{p_idx + 1}/{n_permutations} "
                        f"({time.perf_counter() - _t_perm:.2f}s elapsed)"
                    )
            log_info(f"  [TIMING] global_morans_i: permutation loop "
                     f"({n_permutations} perms, n={n}): "
                     f"{time.perf_counter() - _t_perm:.2f}s")
            observed_extremity = abs(morans_i - expected_i)
            perm_extremity = np.abs(perm_i - expected_i)
            perm_p = (1 + np.sum(perm_extremity >= observed_extremity)) / (n_permutations + 1)
            result["n_permutations"] = int(n_permutations)
            result["permutation_p_value"] = float(perm_p)
            result["permutation_reference_distribution_mean"] = float(perm_i.mean())
            result["permutation_reference_distribution_std"] = float(perm_i.std())

    return result


def compute_effective_sample_size(n: int, morans_i: float) -> dict:
    """Approximate effective sample size under positive spatial
    autocorrelation, n_eff = n * (1 - I) / (1 + I).

    This is a simple, commonly used interpretive approximation (e.g.
    Griffith, 2005) for how much residual spatial autocorrelation shrinks
    the amount of independent information in a holdout sample.  It is
    reported alongside - not in place of - the existing bootstrap-based
    uncertainty quantification.
    """
    if not np.isfinite(morans_i) or n <= 0:
        return {
            "n": n, "morans_i": float(morans_i) if np.isfinite(morans_i) else float("nan"),
            "n_eff": float("nan"), "reduction_pct": float("nan"),
        }
    i_clipped = float(np.clip(morans_i, -0.999, 0.999))
    n_eff = n * safe_divide(
        np.array([1.0 - i_clipped]), np.array([1.0 + i_clipped]), fill=float(n)
    )[0]
    n_eff = float(max(n_eff, 0.0))
    reduction_pct = float(safe_divide(
        np.array([n - n_eff]), np.array([n]), fill=0.0
    )[0] * 100.0)
    return {"n": n, "morans_i": float(morans_i), "n_eff": n_eff, "reduction_pct": reduction_pct}


def compute_residual_spatial_diagnostic(
    validation: ValidationResults, data: DatasetBundle,
    cfg: Config, summary: SummaryLog,
) -> Optional[dict]:
    """Compute Moran's I on holdout residuals, skipping gracefully when
    no coordinate columns are present."""
    df_test    = data.df.loc[data.test_idx]
    coord_cols = _find_spatial_columns(df_test, cfg)
    if coord_cols is None:
        log_warning(
            "\nResidual spatial diagnostic skipped: no recognisable coordinate "
            f"columns found (looked for {cfg.spatial_x_candidates} / "
            f"{cfg.spatial_y_candidates})."
        )
        return None

    x_col, y_col = coord_cols
    residuals = data.y_test.values - validation.holdout_pred
    x_vals    = df_test[x_col].values
    y_vals    = df_test[y_col].values

    moran = global_morans_i(
        residuals, x_vals, y_vals, k=cfg.moran_k_neighbors,
        n_permutations=(cfg.moran_n_permutations if cfg.run_moran_permutations else 0),
        permutation_seed=cfg.moran_permutation_seed,
    )
    log_info(
        f"\nGlobal Moran's I (holdout residuals): "
        f"I = {moran['morans_i']:.4f} "
        f"(expected under CSR = {moran['expected_i']:.4f}), "
        f"z = {moran['z_score']:.3f}, {format_p_value(moran['p_value'])} (normal approx.)"
    )
    log_info(
        f"  Weighting: {moran['weighting_method']} (k={moran['k']}, "
        f"row-standardised={moran['row_standardized']}), n={moran['n']}, "
        f"connected components={moran['n_connected_components']} "
        f"(islands={moran['n_islands']})"
    )
    if moran["n_permutations"] > 0:
        log_info(
            f"  Permutation inference: {format_p_value(moran['permutation_p_value'])} "
            f"({moran['n_permutations']} permutations; preferred over the normal "
            "approximation per the audit)."
        )
    elif moran.get("permutation_skipped_reason"):
        log_info(f"  Permutation inference skipped: {moran['permutation_skipped_reason']}")
    inference_p = (
        moran["permutation_p_value"] if moran["n_permutations"] > 0
        else moran["p_value"]
    )
    significant_p = np.isfinite(inference_p) and inference_p < 0.05
    exceeds_magnitude = (
        np.isfinite(moran["morans_i"])
        and moran["morans_i"] > cfg.moran_i_magnitude_threshold
    )
    if significant_p or exceeds_magnitude:
        log_info(f"  -> {interpretation_text('spatial_dependence')}")
        summary.add_warning(
            f"Significant residual spatial autocorrelation detected "
            f"(Moran's I = {moran['morans_i']:.4f}, {format_p_value(inference_p)}).  "
            "Confidence intervals should be interpreted accordingly."
        )
        if exceeds_magnitude:
            summary.add_conclusion(interpretation_text("spatial_dependence"))
    else:
        log_info(
            "  -> No strong evidence of spatially clustered residual error "
            "at the 0.05 level."
        )

    spatial_dependence_flag = bool(significant_p or exceeds_magnitude)
    summary.log("Residual diagnostics", "Global Moran's I (holdout residuals)",
                f"{moran['morans_i']:.4f}")
    summary.log("Residual diagnostics", "Moran's I p-value (normal approx.)",
                format_p_value(moran["p_value"]).replace("p = ", "").replace("p < ", "< "))
    if moran["n_permutations"] > 0:
        summary.log("Residual diagnostics", "Moran's I p-value (permutation)",
                    format_p_value(moran["permutation_p_value"]).replace("p = ", "").replace("p < ", "< "))
        summary.log("Residual diagnostics", "Moran's I permutations", str(moran["n_permutations"]))
    summary.log("Residual diagnostics", "Moran's I weighting",
                f"{moran['weighting_method']}, k={moran['k']}, "
                f"row-standardised={moran['row_standardized']}, n={moran['n']}, "
                f"connected components={moran['n_connected_components']} "
                f"(islands={moran['n_islands']})")

    n_eff_result = compute_effective_sample_size(len(residuals), moran["morans_i"])
    if cfg.debug and np.isfinite(n_eff_result["n_eff"]):
        log_debug(
            f"\n[debug-only, not manuscript-facing] "
            f"Holdout observations                : {n_eff_result['n']}\n"
            f"Residual Moran's I                   : {n_eff_result['morans_i']:.3f}\n"
            f"Approximate effective sample size    : {n_eff_result['n_eff']:.0f}\n"
            f"Approximate reduction                : {n_eff_result['reduction_pct']:.0f}%"
        )

    return {
        "x": x_vals, "y": y_vals,
        "residuals": residuals, "moran": moran,
        "n_eff": n_eff_result,
        "x_col": x_col, "y_col": y_col,
        "spatial_dependence_flag": spatial_dependence_flag,
    }


def _moran_row(moran: dict) -> dict:
    """Flatten a global_morans_i() result dict into the columns used by
    compute_spatial_autocorrelation_summary's comparison table. Prefers the
    permutation p-value when available, exactly as
    compute_residual_spatial_diagnostic already does for its own reporting."""
    inference_p = (
        moran["permutation_p_value"] if moran.get("n_permutations", 0) > 0
        else moran["p_value"]
    )
    return {
        "Moran's I":        moran["morans_i"],
        "Expected I (CSR)": moran["expected_i"],
        "z-score":          moran["z_score"],
        "p-value":          inference_p,
        "n":                moran["n"],
        "k (neighbours)":   moran["k"],
    }


def compute_spatial_autocorrelation_summary(
    data: DatasetBundle, spatial_diagnostic: Optional[dict], cfg: Config,
) -> Optional[pd.DataFrame]:
    """Contextual Moran's I comparison table (engineering-refinement item 2):
    raw observed LST, Linear Regression residuals, and XGBoost residuals -
    all on the SAME holdout points, SAME coordinate columns, and the SAME
    weighting matrix / permutation settings already used by
    compute_residual_spatial_diagnostic (k=cfg.moran_k_neighbors,
    n_permutations=cfg.moran_n_permutations when cfg.run_moran_permutations,
    seed=cfg.moran_permutation_seed). global_morans_i() itself is not
    modified; this only calls it two more times for context.

    This is purely contextual/interpretive reporting, as requested: the
    XGBoost residual Moran's I is read verbatim from `spatial_diagnostic`
    (never recomputed), and the Linear Regression fit here is a fresh,
    untuned LinearRegression() on the SAME data.X_train/y_train split used
    everywhere else in the pipeline, used only to obtain holdout residuals
    for this one comparison row - it is independent of, and does not
    affect, the separately-tuned Linear Regression benchmark reported in
    compare_models()/benchmark_comparison.csv.

    Returns None (and logs why) when compute_residual_spatial_diagnostic
    itself returned None (no coordinate columns found), so this table never
    silently disagrees with the primary residual diagnostic about whether
    spatial analysis was possible for this dataset.
    """
    if spatial_diagnostic is None:
        log_warning(
            "\nSpatial autocorrelation comparison skipped: residual spatial "
            "diagnostic is unavailable (no coordinate columns found)."
        )
        return None

    df_test    = data.df.loc[data.test_idx]
    coord_cols = _find_spatial_columns(df_test, cfg)
    if coord_cols is None:
        return None
    x_col, y_col = coord_cols
    x_vals = df_test[x_col].values
    y_vals = df_test[y_col].values

    k         = cfg.moran_k_neighbors
    n_perm    = cfg.moran_n_permutations if cfg.run_moran_permutations else 0
    perm_seed = cfg.moran_permutation_seed

    moran_raw = global_morans_i(
        data.y_test.values, x_vals, y_vals, k=k,
        n_permutations=n_perm, permutation_seed=perm_seed,
    )

    lr_model = LinearRegression()
    lr_model.fit(data.X_train, data.y_train)
    lr_residuals = data.y_test.values - lr_model.predict(data.X_test)
    moran_lr = global_morans_i(
        lr_residuals, x_vals, y_vals, k=k,
        n_permutations=n_perm, permutation_seed=perm_seed,
    )

    moran_xgb = spatial_diagnostic["moran"]

    out = pd.DataFrame([
        {"Series": "Raw observed LST",           **_moran_row(moran_raw)},
        {"Series": "Linear Regression residuals", **_moran_row(moran_lr)},
        {"Series": "XGBoost residuals",           **_moran_row(moran_xgb)},
    ])
    log_info("\nSpatial autocorrelation comparison (contextual reporting only):")
    log_info(out[["Series", "Moran's I", "p-value"]].to_string(index=False))
    if (
        np.isfinite(moran_raw["morans_i"]) and moran_raw["morans_i"] > 0
        and np.isfinite(moran_xgb["morans_i"])
    ):
        reduction_pct = 100.0 * (1.0 - moran_xgb["morans_i"] / moran_raw["morans_i"])
        log_info(
            f"  XGBoost residual Moran's I is {reduction_pct:.0f}% lower than raw "
            "observed LST Moran's I (same holdout points and weighting matrix)."
        )
    return out


def compute_calibration_table(
    y_true: np.ndarray, y_pred: np.ndarray,
    intercept_ci: Optional[dict] = None,
) -> pd.DataFrame:
    """Produce a single calibration metrics table (descriptive only;
    no recalibration is applied).

    Columns reported:
      Calibration slope, Calibration intercept, Mean bias (MBE),
      Pearson r and p-value.

    RMSE and MAE are omitted here because they are already reported in
    the Confirmatory Holdout section.  MACE was removed because it is
    mathematically identical to MAE (compute_mean_absolute_calibration_error
    delegates to compute_mae) and therefore duplicates an existing column.
    Bias and Mean Bias Error (MBE) are the same quantity; only 'Mean bias'
    is retained.

    `intercept_ci`, if supplied, is the "calibration_intercept" entry from
    `ValidationResults.spatial_block_bootstrap_ci` (keys: lower/upper/...).
    The reported note states only what was actually computed - the slope
    value, the intercept value, its CI, and whether that CI spans zero -
    with no causal claim about *why* (e.g. no unverified "regression
    toward the mean" mechanism is asserted, since this function never
    checks whether that mechanism actually applies in a given run).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m            = compute_metrics(y_true, y_pred)
    pearson_r, pearson_p = compute_pearson_r(y_true, y_pred)
    slope        = m["calibration_slope"]
    intercept    = m["calibration_intercept"]

    if intercept_ci is not None:
        ci_low, ci_high = intercept_ci["lower"], intercept_ci["upper"]
        spans_zero = bool(ci_low <= 0.0 <= ci_high)
        intercept_note = (
            f"Calibration slope = {slope:.3f}; calibration intercept = "
            f"{intercept:.3f} (spatial-block bootstrap 95% CI "
            f"[{ci_low:.3f}, {ci_high:.3f}]). "
            + ("This interval includes zero." if spans_zero else
               "This interval excludes zero.")
        )
    else:
        intercept_note = (
            f"Calibration slope = {slope:.3f}; calibration intercept = "
            f"{intercept:.3f}. No bootstrap interval was supplied for the "
            "intercept in this call."
        )

    return pd.DataFrame([{
        "Calibration slope":      slope,
        "Calibration intercept":  intercept,
        "Mean bias (°C)":         m["bias"],
        "Pearson r":              pearson_r,
        "Pearson p":              pearson_p,
        "Intercept CI note":      intercept_note,
    }])


def _replay_benchmark_reporting(
    comparison_df: pd.DataFrame, cfg: Config, summary: SummaryLog,
) -> None:
    """Re-emit compare_models()'s export and summary rows from a cached
    comparison table (audit #10). Reporting only - no refitting."""
    save_csv(comparison_df, cfg.data_dir / "benchmark_comparison.csv")
    summary.log("Model comparison", "Benchmark note", interpret_model_comparison(cfg))
    ranked = comparison_df.sort_values("R2", ascending=False).reset_index(drop=True)
    best = ranked.iloc[0]
    xgb = comparison_df[comparison_df["Model"].str.contains("XGBoost")].iloc[0]
    runner_up = ranked.iloc[1] if len(ranked) > 1 else best
    gap = float(best["R2"] - runner_up["R2"])
    summary.log(
        "Model comparison", "Top model under equal-budget comparison",
        f"{best['Model']} (R2={best['R2']:.4f}); XGBoost R2={xgb['R2']:.4f}; "
        f"gap to runner-up={gap:.4f}",
    )
    for _, row in comparison_df.iterrows():
        summary.log(
            "Model comparison", f"{row['Model']} holdout metrics",
            f"R2={row['R2']:.4f}; RMSE={row['RMSE (°C)']:.4f} °C; MAE={row['MAE (°C)']:.4f} °C",
        )
    if best["Model"] != xgb["Model"]:
        summary.add_warning(
            f"Under the equal-budget benchmark comparison, {best['Model']} "
            f"(R2={best['R2']:.4f}) outperformed XGBoost (R2={xgb['R2']:.4f}) "
            "on the confirmatory holdout (restored from cache)."
        )


def compare_models(
    model_results: ModelResults, cfg: Config, summary: SummaryLog,
    nested_cv: "NestedCVResults",
) -> pd.DataFrame:
    """Fit Linear Regression, Random Forest, and Gradient Boosting
    benchmarks and compare them with XGBoost on the confirmatory holdout.

    Random Forest and Gradient Boosting are each tuned via
    RandomizedSearchCV(GroupKFold) on the training split, using the same
    scoring/CV scheme as XGBoost's own tuning in fit_model(). They do NOT
    share a single iteration count with each other: RandomForestRegressor
    parallelizes across trees/cores and keeps the full
    cfg.n_benchmark_search_iter budget (same as XGBoost), while plain
    GradientBoostingRegressor builds trees one at a time on a single core
    and is dramatically slower per fit, so it uses the smaller, dedicated
    cfg.n_benchmark_search_iter_gb budget to keep runtime reasonable. Each
    model's "fair" budget is therefore genuinely spent (Tuning iterations
    > 0 for both), just not identical between RF and GB - see the QA check
    below, which validates each model against its own configured budget
    rather than requiring RF and GB to match each other.
    """
    log_info("\nNote: " + interpret_model_comparison(cfg))
    data = model_results.data
    group_cv = GroupKFold(n_splits=cfg.n_group_kfold_splits)

    vlog_info(cfg, "Running Linear Regression (untuned baseline)...")
    lr_model = LinearRegression()
    lr_model.fit(data.X_train, data.y_train)

    vlog_info(
        cfg,
        f"Running Random Forest (RandomizedSearchCV, "
        f"{cfg.n_benchmark_search_iter} iterations)...",
    )

    def _build_and_fit_rf(n_jobs: int) -> RandomizedSearchCV:
        s = RandomizedSearchCV(
            estimator=RandomForestRegressor(random_state=cfg.random_seed, n_jobs=1),
            param_distributions=cfg.rf_param_distributions,
            n_iter=cfg.n_benchmark_search_iter,
            scoring="r2",
            cv=group_cv,
            random_state=cfg.random_seed,
            n_jobs=n_jobs,
        )
        s.fit(data.X_train, data.y_train, groups=data.groups_train)
        return s

    rf_search = _fit_search_with_worker_retry(_build_and_fit_rf, cfg, "STEP 5 Random Forest tuning")
    rf_model = rf_search.best_estimator_

    vlog_info(
        cfg,
        f"Running Gradient Boosting (RandomizedSearchCV, "
        f"{cfg.n_benchmark_search_iter_gb} iterations)...",
    )

    def _build_and_fit_gb(n_jobs: int) -> RandomizedSearchCV:
        s = RandomizedSearchCV(
            estimator=GradientBoostingRegressor(random_state=cfg.random_seed),
            param_distributions=cfg.gb_param_distributions,
            n_iter=cfg.n_benchmark_search_iter_gb,
            scoring="r2",
            cv=group_cv,
            random_state=cfg.random_seed,
            n_jobs=n_jobs,
        )
        s.fit(data.X_train, data.y_train, groups=data.groups_train)
        return s

    gb_search = _fit_search_with_worker_retry(_build_and_fit_gb, cfg, "STEP 5 Gradient Boosting tuning")
    gb_model = gb_search.best_estimator_

    fitted_models = {
        "Linear Regression": (lr_model, None, None, None),
        "Random Forest":     (rf_model, cfg.n_benchmark_search_iter,
                               len(cfg.rf_param_distributions), rf_search.best_score_),
        "Gradient Boosting": (gb_model, cfg.n_benchmark_search_iter_gb,
                               len(cfg.gb_param_distributions), gb_search.best_score_),
        "XGBoost":           (model_results.model, cfg.n_random_search_iter,
                               len(cfg.param_distributions), model_results.search.best_score_),
    }
    best_params_by_model = {
        "Linear Regression": {},
        "Random Forest":     rf_search.best_params_,
        "Gradient Boosting": gb_search.best_params_,
        "XGBoost":           model_results.best_params,
    }

    rows = []
    for name, (mdl, tuning_iter, search_space_size, cv_score) in fitted_models.items():
        y_pred_model = mdl.predict(data.X_test)
        rows.append({
            "Model":              name,
            "Tuning iterations":  tuning_iter if tuning_iter is not None else 0,
            "Search space size":  search_space_size if search_space_size is not None else 0,
            "CV score (inner tuning R2)": cv_score if cv_score is not None else float("nan"),
            "Best parameters":   json.dumps(best_params_by_model[name]),
            "R2":       r2_score(data.y_test, y_pred_model),
            "RMSE (°C)": np.sqrt(mean_squared_error(data.y_test, y_pred_model)),
            "MAE (°C)": mean_absolute_error(data.y_test, y_pred_model),
        })

    comparison_df = pd.DataFrame(rows).sort_values("R2", ascending=False).reset_index(drop=True)

    best_row_preview = comparison_df.iloc[0]
    xgb_row_preview   = comparison_df.loc[comparison_df["Model"] == "XGBoost"].iloc[0]
    xgb_is_best_preview = bool(best_row_preview["Model"] == "XGBoost")
    r2_gap_preview = float(
        comparison_df["R2"].iloc[0] - comparison_df["R2"].iloc[1]
    ) if len(comparison_df) > 1 else float("nan")

    benchmark_factories = {
        "Random Forest": (
            lambda: RandomForestRegressor(random_state=cfg.random_seed, n_jobs=1),
            cfg.rf_param_distributions, cfg.n_benchmark_search_iter,
        ),
        "Gradient Boosting": (
            lambda: GradientBoostingRegressor(random_state=cfg.random_seed),
            cfg.gb_param_distributions, cfg.n_benchmark_search_iter_gb,
        ),
    }
    runner_up_name = (
        best_row_preview["Model"] if not xgb_is_best_preview
        else comparison_df.iloc[1]["Model"]
    )
    comparison_df["Paired_vs_XGBoost_mean_dR2"]   = float("nan")
    comparison_df["Paired_vs_XGBoost_mean_dRMSE"] = float("nan")
    comparison_df["Paired_vs_XGBoost_mean_dMAE"]  = float("nan")
    comparison_df["Paired_vs_XGBoost_sign_consistent"] = None
    if (
        (not xgb_is_best_preview or r2_gap_preview < 0.01)
        and runner_up_name in benchmark_factories
    ):
        factory, param_dist, n_iter = benchmark_factories[runner_up_name]
        vlog_info(cfg, f"Running paired fold-level comparison (XGBoost vs {runner_up_name})...")
        benchmark_nested = run_nested_group_kfold_cv_for_benchmark(
            data, cfg, summary, runner_up_name, factory, param_dist, n_iter=n_iter,
        )
        paired_df = paired_fold_comparison(
            nested_cv, benchmark_nested, "XGBoost", runner_up_name, cfg, summary,
        )
        row_idx = comparison_df.index[comparison_df["Model"] == runner_up_name][0]
        comparison_df.loc[row_idx, "Paired_vs_XGBoost_mean_dR2"]   = paired_df["delta_r2"].mean()
        comparison_df.loc[row_idx, "Paired_vs_XGBoost_mean_dRMSE"] = paired_df["delta_rmse"].mean()
        comparison_df.loc[row_idx, "Paired_vs_XGBoost_mean_dMAE"]  = paired_df["delta_mae"].mean()
        comparison_df.loc[row_idx, "Paired_vs_XGBoost_sign_consistent"] = bool(
            np.all(paired_df["delta_r2"] > 0) or np.all(paired_df["delta_r2"] < 0)
        )

    if cfg.run_mode == "publication":
        log_info("\nBenchmark comparison completed.")
    else:
        log_info("\nModel comparison:")
        log_info(comparison_df[["Model", "Tuning iterations", "R2", "RMSE (°C)", "MAE (°C)"]])
    save_csv(comparison_df, cfg.data_dir / "benchmark_comparison.csv")
    summary.log(
        "Model comparison", "Benchmark note",
        interpret_model_comparison(cfg),
    )

    best_row = best_row_preview
    xgb_row  = xgb_row_preview
    xgb_is_best = xgb_is_best_preview
    r2_gap_to_next = r2_gap_preview
    summary.log(
        "Model comparison", "Top model under equal-budget comparison",
        f"{best_row['Model']} (R2={best_row['R2']:.4f}); "
        f"XGBoost R2={xgb_row['R2']:.4f}; gap to runner-up={r2_gap_to_next:.4f}",
    )

    if not xgb_is_best:
        summary.add_warning(
            f"Under the equal-budget benchmark comparison, {best_row['Model']} "
            f"(R2={best_row['R2']:.4f}) outperformed XGBoost "
            f"(R2={xgb_row['R2']:.4f}) on the confirmatory holdout. XGBoost "
            "remains the deployed model for scenario/NEGI analysis because "
            "it was selected and validated via nested GroupKFold CV before "
            "this benchmark comparison was run; this is disclosed as a "
            "benchmark-comparison finding rather than silently omitted."
        )
    elif r2_gap_to_next < 0.01:
        summary.add_conclusion(
            f"XGBoost has the highest holdout R2 ({xgb_row['R2']:.4f}) under "
            f"the equal-budget benchmark comparison, but the margin over the "
            f"runner-up ({r2_gap_to_next:.4f}) is small; see the paired "
            "fold-level comparison for whether this advantage is consistent "
            "across spatial folds."
        )

    return comparison_df


import re as _re

_PLACEHOLDER_PATTERN = _re.compile(r"\{[A-Za-z_][A-Za-z0-9_]{1,}\}")


def find_unresolved_placeholders(*sources) -> list[str]:
    """Scan strings / DataFrames for literal, un-interpolated template
    placeholders such as ``{xgb_iter}`` (item 1 / item 13.1).

    Only flags ``{identifier}``-shaped tokens (a leading letter/underscore
    followed by 2+ word characters), which distinguishes genuine unresolved
    `str.format` placeholders from short mathematical notation such as
    ``{x}`` or dict/set literals containing commas, spaces, or quotes -
    none of which match this pattern. This performs no interpolation and
    changes no report content; it only detects tokens that should already
    have been filled in by the point a report is exported.
    """
    hits: list[str] = []
    for source in sources:
        if source is None:
            continue
        if isinstance(source, pd.DataFrame):
            texts = [str(v) for v in source.to_numpy().ravel().tolist()]
        elif isinstance(source, (list, tuple)):
            texts = [str(v) for v in source]
        else:
            texts = [str(source)]
        for text in texts:
            hits.extend(_PLACEHOLDER_PATTERN.findall(text))
    seen: list[str] = []
    for h in hits:
        if h not in seen:
            seen.append(h)
    return seen


def sparse_region_mask(
    knn_dist: Optional[np.ndarray],
    in_support: Optional[np.ndarray],
    sparsity_threshold: Optional[float],
) -> Optional[np.ndarray]:
    """Single, shared definition of 'locally sparse' (corrected per this
    request): a trajectory point that passes the existing formal kNN
    support threshold (`in_support`, already computed in
    compute_support_diagnostics from support_threshold_k, the
    `support_percentile`-th percentile of observed kNN distances) AND whose
    own kNN distance exceeds `sparsity_threshold` - an EMPIRICAL REFERENCE
    threshold computed once from the observed/model-reference feature-space
    kNN-distance distribution (same standardized predictor space, same k,
    as scenario support assessment; see compute_support_diagnostics /
    Config.FEATURE_SPARSITY_PERCENTILE).

    This threshold is never derived from the scenario trajectory itself
    (not its median, not any trajectory-based percentile). A point that
    fails formal support is never additionally labelled "sparse" here -
    formal support and local sparsity are reported as separate concepts.
    Used identically by report_data_support_check (trajectory-wide and
    at-the-maximum reporting) and build_maximum_diagnostics, so the two
    never disagree on which points count as sparse.
    """
    if knn_dist is None or in_support is None or sparsity_threshold is None:
        return None
    return in_support & (knn_dist > sparsity_threshold)


def classify_local_density(
    knn_dist: Optional[float],
    reference_knn_distribution: Optional[np.ndarray],
    cfg: Config,
    moderate_threshold: Optional[float] = None,
    sparse_threshold: Optional[float] = None,
) -> Optional[str]:
    """Three-level local-density classification for a single trajectory
    point (endpoint-density diagnostic, highest priority): "HIGH",
    "MODERATE", or "LOW".

    Review-pass fix (recommendation 8): the label is now read off a
    Z-SCORE of `knn_dist` against the mean/std of
    `reference_knn_distribution` (the same reference kNN-distance sample
    used everywhere else in this diagnostic), compared against
    `cfg.local_density_z_high` / `cfg.local_density_z_moderate` - "how
    many standard deviations from typical is this point's local
    density" - rather than comparing the raw distance against two FIXED
    threshold VALUES. This is mathematically cleaner (continuous,
    scale-aware) and was the reviewer's own suggested alternative to
    percentile lookups.

    Falls back to the original threshold-VALUE comparison
    (`moderate_threshold` / `sparse_threshold`, still derived elsewhere
    from FEATURE_SPARSITY_PERCENTILE / DENSITY_MODERATE_PERCENTILE) only
    when no usable reference distribution is available (e.g. a
    degenerate zero-variance reference sample) - this keeps the
    diagnostic available rather than silently returning "N/A" whenever
    the z-score path can't be computed.

    This answers a genuinely different question than the formal PASS/FAIL
    support check: "PASS" only says the point is not classified as
    extrapolation; it does not say whether the point sits in a
    well-populated or a sparse part of the training data. Reported and
    exported strictly alongside, never in place of, the formal PASS/FAIL
    result - the two lines should never be combined into one sentence.

    Returns None (not a string) when no classification can be made at
    all, so callers can render "N/A" rather than guessing.
    """
    if knn_dist is None:
        return None
    if reference_knn_distribution is not None and len(reference_knn_distribution) > 1:
        mean_ref = float(np.mean(reference_knn_distribution))
        std_ref  = float(np.std(reference_knn_distribution))
        if std_ref > 0:
            z = (knn_dist - mean_ref) / std_ref
            if z <= cfg.local_density_z_high:
                return "HIGH"
            if z <= cfg.local_density_z_moderate:
                return "MODERATE"
            return "LOW"
    if moderate_threshold is None or sparse_threshold is None:
        return None
    if knn_dist <= moderate_threshold:
        return "HIGH"
    if knn_dist <= sparse_threshold:
        return "MODERATE"
    return "LOW"


def local_density_z_score(
    knn_dist: Optional[float], reference_knn_distribution: Optional[np.ndarray],
) -> Optional[float]:
    """The raw z-score classify_local_density() classifies against -
    exposed separately so evaluate_negi_candidate() can also fold a
    continuous density signal into the internal robustness score
    (recommendation 4) without recomputing mean/std twice."""
    if (
        knn_dist is None or reference_knn_distribution is None
        or len(reference_knn_distribution) <= 1
    ):
        return None
    mean_ref = float(np.mean(reference_knn_distribution))
    std_ref  = float(np.std(reference_knn_distribution))
    if std_ref <= 0:
        return None
    return (knn_dist - mean_ref) / std_ref


def compute_robustness_score(
    components: dict, weights: dict,
) -> Optional[float]:
    """Weighted combination of component scores (recommendation 4) into
    one internal robustness score in [0, 1].

    Each entry of `components` is either a float in [0, 1] (1.0 = fully
    supports robustness on that dimension, 0.0 = fully against it) or
    None (that dimension's input was unavailable for this candidate).
    Weights for any None-valued components are dropped and the
    remaining weights re-normalized, so a missing input never silently
    counts as either "pass" or "fail" - it is simply excluded from the
    average. Returns None only when every component is unavailable.

    INTERNAL bookkeeping only, per the reviewer's own framing ("not to
    report") - never surfaced to a reader as a headline number; see the
    call sites in evaluate_negi_candidate() for how it does (`robust`)
    and does not (`robustness_score`, boundary/density folded in) gate
    published classifications.
    """
    used = {k: v for k, v in components.items() if v is not None}
    if not used:
        return None
    w_sum = sum(weights[k] for k in used)
    if w_sum <= 0:
        return None
    return float(sum(weights[k] * used[k] for k in used) / w_sum)


def _compute_local_stability(negi_arr: np.ndarray, idx: int, cfg: Config) -> dict:
    """Local-stability perturbation check around trajectory index `idx`.

    Extracted from build_maximum_diagnostics's original inline block
    (engineering-refinement item 1: endpoint robustness symmetry) so that
    both the evaluated maximum and the trajectory endpoint are assessed
    with exactly the same neighbour window, jump threshold, and sign-flip
    rule - no new stability criterion is introduced anywhere by this
    extraction; the arithmetic is unchanged from the original inline code.

    Review-pass addition (recommendation 6): the original biggest-jump /
    sign-flip check (over `trajectory_stability_neighbor_steps` on each
    side) is kept EXACTLY as before - `biggest_jump`, `sign_flip`, and
    their contribution to `locally_stable` are unchanged, so no existing
    numeric export changes. A second, independent signal is now also
    computed over a WIDER neighbourhood
    (`trajectory_stability_cv_window_steps`): the maximum absolute local
    derivative (steepest consecutive-point slope, i.e.
    |NEGI[i+1]-NEGI[i]| / |scenario step|-equivalent, here simply the
    per-step NEGI difference since evaluated points are equally spaced)
    across that window. A derivative-based check is used rather than the
    coefficient of variation the review also suggested, because NEGI
    legitimately crosses zero along the trajectory - CV = std/|mean| is
    undefined/explosive there and would flag ordinary near-zero-crossing
    behaviour as "unstable" for reasons unrelated to actual local
    instability; a derivative has no such singularity. `locally_stable`
    now requires BOTH the original check AND this wider derivative check
    to pass - it can only newly flag something as unstable that the
    original check missed, never override an original UNSTABLE verdict
    back to stable.
    """
    n     = len(negi_arr)
    steps = cfg.trajectory_stability_neighbor_steps
    neighbor_idxs = [i for i in range(idx - steps, idx + steps + 1) if 0 <= i < n and i != idx]
    if not neighbor_idxs:
        return {
            "biggest_jump": None, "sign_flip": False, "locally_stable": None,
            "max_local_derivative": None,
        }
    val           = float(negi_arr[idx])
    neighbor_vals = negi_arr[neighbor_idxs]
    biggest_jump  = float(np.max(np.abs(neighbor_vals - val)))
    sign_flip     = bool(
        not np.isclose(val, 0.0)
        and np.any(np.sign(neighbor_vals) != np.sign(val))
    )
    original_stable = not (sign_flip or (biggest_jump > cfg.trajectory_stability_jump_threshold))

    cv_steps = max(steps, int(cfg.trajectory_stability_cv_window_steps))
    lo = max(0, idx - cv_steps)
    hi = min(n - 1, idx + cv_steps)
    window_vals = negi_arr[lo:hi + 1]
    if len(window_vals) >= 2:
        max_local_derivative = float(np.max(np.abs(np.diff(window_vals))))
        derivative_stable = max_local_derivative <= cfg.trajectory_stability_max_derivative
    else:
        max_local_derivative = None
        derivative_stable = True

    locally_stable = bool(original_stable and derivative_stable)
    return {
        "biggest_jump": biggest_jump, "sign_flip": sign_flip,
        "locally_stable": locally_stable,
        "max_local_derivative": max_local_derivative,
    }


def evaluate_negi_candidate(
    idx: int,
    negi_arr: np.ndarray,
    scenario_pct: np.ndarray,
    uncertainty_percentiles: Optional[dict],
    support_in_array: Optional[np.ndarray],
    support_knn_array: Optional[np.ndarray],
    cfg: Config,
    sparsity_threshold: Optional[float] = None,
    density_moderate_threshold: Optional[float] = None,
    reference_knn_distribution: Optional[np.ndarray] = None,
    actual_n_refits: Optional[int] = None,
) -> dict:
    """SINGLE SHARED DECISION CORE for evaluating any one trajectory index -
    whether it is the discovered numerical maximum or a fixed trajectory
    endpoint - against uncertainty, local stability, and feature support.

    (Robustness-symmetry audit, item 1.) Prior to this extraction,
    build_maximum_diagnostics and build_endpoint_diagnostics each contained
    their own copy of the CI-extraction block, the feature-support block,
    and the robustness decision rule (`not ci_includes_zero and
    locally_stable is not False and knn_support is not False`); only the
    local-stability arithmetic itself had already been factored out into
    `_compute_local_stability`. That was a genuine (if silent) duplication:
    the two copies happened to be byte-for-byte identical today, but nothing
    enforced that they would STAY identical after a future edit to one of
    them. This function is now the only place any of that logic lives -
    build_maximum_diagnostics and build_endpoint_diagnostics both call it
    and only differ in (a) which index they pass in, and (b) the
    maximum-only "interior optimum" bookkeeping / the endpoint-only
    "always at trajectory boundary by construction" bookkeeping, both of
    which are genuinely different concepts between a discovered optimum and
    a fixed boundary point, not reimplementations of shared logic.

    No arithmetic differs from the pre-extraction inline code in either
    original function - every formula below is copied verbatim, so
    numerical outputs (maximum_evaluated_negi, endpoint_evaluated_negi,
    every uncertainty/stability/support/robust field, and every downstream
    CSV/JSON export built from them) are unchanged by this refactor.

    Negative `idx` is resolved the same way Python indexing would (e.g.
    -1 = last point), matching the pre-refactor endpoint convention.

    Returns a dict of generic (unprefixed) fields:
      idx, value, position, at_edge, near_boundary,
      ci_low, ci_high, q25, q75, ci_includes_zero, uncertainty_type, n_refits,
      locally_stable, biggest_jump, sign_flip,
      knn_distance, knn_support, sparse_region, local_density,
      robust
    """
    n = len(negi_arr)
    idx = int(idx)
    if idx < 0:
        idx = n + idx

    at_edge       = idx == 0 or idx == n - 1
    near_boundary = is_near_trajectory_boundary(
        idx, n, cfg.scenario_boundary_tolerance_pct, positions=scenario_pct,
    )

    stability            = _compute_local_stability(negi_arr, idx, cfg)
    biggest_jump         = stability["biggest_jump"]
    sign_flip            = stability["sign_flip"]
    locally_stable       = stability["locally_stable"]
    max_local_derivative = stability["max_local_derivative"]

    if uncertainty_percentiles is not None:
        p        = uncertainty_percentiles
        ci_low   = float(p[2.5][idx])
        ci_high  = float(p[97.5][idx])
        q25      = float(p[25.0][idx])
        q75      = float(p[75.0][idx])
        ci_includes_zero = bool(ci_low <= 0.0 <= ci_high)
        n_refits = actual_n_refits if actual_n_refits is not None else cfg.n_spatial_refits
        uncertainty_type  = (
            f"empirical spatial-refit 95% interval (2.5th-97.5th percentile "
            f"of NEGI across {n_refits} independent spatial-block "
            "holdout refits, seeded from Config.uncertainty_refit_seed)"
        )
    else:
        ci_low = ci_high = q25 = q75 = None
        ci_includes_zero = None
        uncertainty_type  = "unavailable"
        n_refits = None

    if support_in_array is not None:
        knn_support   = bool(support_in_array[idx])
        knn_dist      = float(support_knn_array[idx]) if support_knn_array is not None else None
        sparse_mask   = sparse_region_mask(support_knn_array, support_in_array, sparsity_threshold)
        sparse_region = bool(sparse_mask[idx]) if sparse_mask is not None else None
        local_density = classify_local_density(
            knn_dist, reference_knn_distribution, cfg,
            moderate_threshold=density_moderate_threshold,
            sparse_threshold=sparsity_threshold,
        )
        knn_distance_percentile = (
            float((reference_knn_distribution <= knn_dist).mean() * 100.0)
            if (knn_dist is not None and reference_knn_distribution is not None)
            else None
        )
        density_z = local_density_z_score(knn_dist, reference_knn_distribution)
    else:
        knn_support, knn_dist, sparse_region, local_density = None, None, None, None
        knn_distance_percentile = None
        density_z = None

    ci_component        = None if ci_includes_zero is None else (0.0 if ci_includes_zero else 1.0)
    support_component   = None if knn_support is None else (1.0 if knn_support else 0.0)
    stability_component = None if locally_stable is None else (1.0 if locally_stable else 0.0)
    core_components = {
        "ci": ci_component, "support": support_component, "stability": stability_component,
    }
    core_weights = {k: cfg.robustness_score_weights[k] for k in core_components}
    core_score = compute_robustness_score(core_components, core_weights)
    robust = None if core_score is None else bool(core_score >= cfg.robustness_score_threshold)

    boundary_component = 0.0 if near_boundary else 1.0
    if density_z is None:
        density_component = None
    else:
        z_high, z_mod = cfg.local_density_z_high, cfg.local_density_z_moderate
        density_component = float(np.clip((z_mod - density_z) / (z_mod - z_high), 0.0, 1.0))
    all_components = {**core_components, "boundary": boundary_component, "density": density_component}
    robustness_score = compute_robustness_score(all_components, cfg.robustness_score_weights)

    return {
        "idx":             idx,
        "value":           float(negi_arr[idx]),
        "position":        float(scenario_pct[idx]),
        "at_edge":         bool(at_edge),
        "near_boundary":   bool(near_boundary),
        "ci_low":          ci_low,
        "ci_high":         ci_high,
        "q25":             q25,
        "q75":             q75,
        "ci_includes_zero": ci_includes_zero,
        "uncertainty_type": uncertainty_type,
        "n_refits":        n_refits,
        "locally_stable":  locally_stable,
        "biggest_jump":    biggest_jump,
        "sign_flip":       sign_flip,
        "max_local_derivative": max_local_derivative,
        "knn_distance":    knn_dist,
        "knn_distance_percentile": knn_distance_percentile,
        "knn_support":     knn_support,
        "sparse_region":   sparse_region,
        "local_density":   local_density,
        "local_density_z": density_z,
        "robust":          robust,
        "robustness_score": robustness_score,
    }



def build_maximum_diagnostics(
    label: str,
    negi_arr: np.ndarray,
    scenario_pct: np.ndarray,
    optimum_idx: int,
    uncertainty_percentiles: Optional[dict],
    support_in_array: Optional[np.ndarray],
    support_knn_array: Optional[np.ndarray],
    cfg: Config,
    sparsity_threshold: Optional[float] = None,
    density_moderate_threshold: Optional[float] = None,
    reference_knn_distribution: Optional[np.ndarray] = None,
    actual_n_refits: Optional[int] = None,
) -> dict:
    """Single source of truth for maximum-evaluated-NEGI diagnostics (item 10).

    Every quantity here is read from arrays already computed elsewhere
    (NEGIResults.s1_optimum_idx / s2_optimum_idx, UncertaintyResults.percentiles,
    SupportDiagnostics.in_support_s2 / mean_knn_dist_s2, the same boundary
    tolerance used by is_near_trajectory_boundary, and the same neighbour
    window used by check_trajectory_stability). No new modelling, refitting,
    or NEGI computation happens here - this only assembles one consistent
    dict that every downstream reporter (console, CSV, Markdown, QA) should
    read from, instead of each recomputing its own copy.

    All CI-extraction / local-stability / feature-support / robustness
    decision logic is delegated to `evaluate_negi_candidate` (robustness-
    symmetry audit, item 1) - this function only relabels that shared
    result into the `maximum_*`-prefixed dict shape and adds the two
    maximum-only concepts that have no endpoint equivalent: the
    single-source-of-truth argmax assertion below, and
    `maximum_supports_interior_optimum`.

    The only interpretation-only addition beyond existing diagnostics is a
    "sparse region" flag: a trajectory point can pass the formal kNN support
    threshold (support_threshold_k, the `support_percentile`-th percentile
    of observed kNN distances - already computed in
    compute_support_diagnostics) while its own kNN distance still exceeds
    the separate EMPIRICAL reference threshold (`sparsity_threshold`,
    SupportDiagnostics.sparsity_threshold_k) derived once from the
    observed/model-reference kNN-distance distribution. Formal support and
    local sparsity remain distinct concepts; a formally-supported-but-sparse
    point is never reclassified as extrapolation here.
    """
    idx = int(optimum_idx)
    assert idx == int(np.argmax(negi_arr)), (
        f"{label}: supplied optimum_idx does not match np.argmax(negi_arr); "
        "single-source-of-truth invariant violated."
    )

    r = evaluate_negi_candidate(
        idx, negi_arr, scenario_pct, uncertainty_percentiles,
        support_in_array, support_knn_array, cfg, sparsity_threshold,
        density_moderate_threshold, reference_knn_distribution,
        actual_n_refits=actual_n_refits,
    )

    if r["robust"] is None:
        supports_interior_optimum = None
    else:
        supports_interior_optimum = bool(
            r["robust"] and not r["near_boundary"] and not r["at_edge"]
        )

    return {
        "label":                            label,
        "maximum_evaluated_negi":           r["value"],
        "maximum_scenario_position":        r["position"],
        "maximum_at_boundary":              r["at_edge"],
        "maximum_near_boundary":            r["near_boundary"],
        "maximum_uncertainty_low":          r["ci_low"],
        "maximum_uncertainty_high":         r["ci_high"],
        "maximum_uncertainty_type":         r["uncertainty_type"],
        "maximum_uncertainty_n_refits":     r["n_refits"],
        "maximum_q25":                      r["q25"],
        "maximum_q75":                      r["q75"],
        "maximum_ci_includes_zero":         r["ci_includes_zero"],
        "maximum_local_stability":          r["locally_stable"],
        "maximum_local_largest_change":     r["biggest_jump"],
        "maximum_local_sign_change":        r["sign_flip"],
        "maximum_knn_distance":             r["knn_distance"],
        "maximum_knn_distance_percentile":  r["knn_distance_percentile"],
        "maximum_knn_k":                    cfg.support_knn_k,
        "maximum_knn_support":              r["knn_support"],
        "maximum_sparse_region":            r["sparse_region"],
        "maximum_local_density":            r["local_density"],
        "maximum_robust":                   r["robust"],
        "maximum_supports_interior_optimum": supports_interior_optimum,
    }


def build_endpoint_diagnostics(
    label: str,
    negi_arr: np.ndarray,
    scenario_pct: np.ndarray,
    endpoint_idx: int,
    uncertainty_percentiles: Optional[dict],
    support_in_array: Optional[np.ndarray],
    support_knn_array: Optional[np.ndarray],
    cfg: Config,
    sparsity_threshold: Optional[float] = None,
    density_moderate_threshold: Optional[float] = None,
    reference_knn_distribution: Optional[np.ndarray] = None,
    actual_n_refits: Optional[int] = None,
) -> dict:
    """Single source of truth for trajectory-ENDPOINT diagnostics
    (engineering-refinement item 1: endpoint robustness symmetry).

    Delegates all CI-extraction / local-stability / feature-support /
    robustness decision logic to `evaluate_negi_candidate` - the SAME
    function build_maximum_diagnostics calls - so the two can never
    silently diverge; this function only relabels that shared result into
    the `endpoint_*`-prefixed dict shape.

    Unlike build_maximum_diagnostics, `endpoint_idx` is NOT required to
    equal np.argmax(negi_arr): an endpoint is a fixed position (first or
    last evaluated scenario point), not a candidate optimum, so that
    invariant does not apply and is intentionally not asserted here.

    Per the boundary-vs-extrapolation distinction raised during review:
    `endpoint_at_trajectory_boundary` is always True by construction (the
    endpoint IS the edge of the evaluated scenario range) and is reported
    separately from `endpoint_knn_support` / `endpoint_sparse_region`,
    which answer the distinct empirical-data-support question. A point can
    be at the trajectory boundary while remaining well supported by
    observed data, and this function never conflates the two.
    """
    r = evaluate_negi_candidate(
        endpoint_idx, negi_arr, scenario_pct, uncertainty_percentiles,
        support_in_array, support_knn_array, cfg, sparsity_threshold,
        density_moderate_threshold, reference_knn_distribution,
        actual_n_refits=actual_n_refits,
    )

    return {
        "label":                             label,
        "endpoint_evaluated_negi":           r["value"],
        "endpoint_scenario_position":        r["position"],
        "endpoint_at_trajectory_boundary":   True,
        "endpoint_uncertainty_low":          r["ci_low"],
        "endpoint_uncertainty_high":         r["ci_high"],
        "endpoint_uncertainty_type":         r["uncertainty_type"],
        "endpoint_uncertainty_n_refits":     r["n_refits"],
        "endpoint_q25":                      r["q25"],
        "endpoint_q75":                      r["q75"],
        "endpoint_ci_includes_zero":         r["ci_includes_zero"],
        "endpoint_local_stability":          r["locally_stable"],
        "endpoint_local_change":             r["biggest_jump"],
        "endpoint_local_sign_change":        r["sign_flip"],
        "endpoint_knn_distance":             r["knn_distance"],
        "endpoint_knn_distance_percentile":  r["knn_distance_percentile"],
        "endpoint_knn_k":                    cfg.support_knn_k,
        "endpoint_knn_support":              r["knn_support"],
        "endpoint_sparse_region":            r["sparse_region"],
        "endpoint_local_density":            r["local_density"],
        "endpoint_robust":                   r["robust"],
    }


def classify_endpoint_interpretation(d: dict) -> str:
    """One-line interpretation label for an endpoint-diagnostics dict,
    parallel to classify_maximum_interpretation. An endpoint has no
    "interior optimum" concept - it is a boundary point by construction -
    so the hierarchy is simply robust / not robust / undetermined."""
    if d["endpoint_robust"] is True:
        sign = "positive" if d["endpoint_evaluated_negi"] > 0 else "negative"
        return f"Statistically supported {sign} endpoint"
    if d["endpoint_robust"] is False:
        reasons = []
        if d["endpoint_ci_includes_zero"]:
            reasons.append("uncertainty band crosses zero")
        if d["endpoint_local_stability"] is False:
            reasons.append(
                "locally unstable, including a sign change across neighbouring "
                "evaluated points"
                if d["endpoint_local_sign_change"] else "locally unstable"
            )
        if d["endpoint_knn_support"] is False:
            reasons.append("fails formal feature support")
        reason_text = ", ".join(reasons) if reasons else "reason undetermined"
        return f"Endpoint not statistically supported ({reason_text})"
    return "Endpoint robustness undetermined (uncertainty band unavailable)"


def format_endpoint_classification(
    d: dict, joint_support_status: Optional[str] = None,
    publication_convergence_status: Optional[str] = None,
) -> str:
    """Render an endpoint-diagnostics dict as a fixed-width classification
    block, mirroring format_maximum_classification. Every value is read
    directly from `d` - nothing is recalculated here.

    `joint_support_status` is the already-computed Step 10b classification
    (`compute_joint_multivariate_support`) - read here, never recomputed.
    None (the default) reports "NOT AVAILABLE" rather than conflating it
    with the marginal-support value below.

    `publication_convergence_status` is the already-computed status from
    compute_uncertainty_convergence_diagnostic (STABILIZED /
    NOT YET STABILIZED / None) - read here, never recomputed. Same
    caveat-appending rule as format_maximum_headline: whenever the band
    excludes zero AND convergence has not been confirmed, an explicit
    caveat is appended here, at the point the band is shown - this is the
    machine-readable, persisted export, so it must carry the caveat even
    if a reader never sees the console output.
    """
    def _yn(v: Optional[bool]) -> str:
        return "N/A" if v is None else ("YES" if v else "NO")

    stability = (
        "N/A" if d["endpoint_local_stability"] is None
        else ("STABLE" if d["endpoint_local_stability"] else "UNSTABLE")
    )
    support = (
        "N/A" if d["endpoint_knn_support"] is None
        else ("PASS" if d["endpoint_knn_support"] else "FAIL")
    )
    joint_support_line = (
        joint_support_status if joint_support_status is not None else "NOT AVAILABLE"
    )
    band = (
        f"[{d['endpoint_uncertainty_low']:.4f}, {d['endpoint_uncertainty_high']:.4f}]"
        if d["endpoint_uncertainty_low"] is not None else "N/A"
    )
    density = d.get("endpoint_local_density") or "N/A"
    lines = [
        f"{d['label']} - Endpoint",
        f"Endpoint evaluated NEGI       : {d['endpoint_evaluated_negi']:.4f}",
        f"Endpoint position             : {d['endpoint_scenario_position']:.1f}%",
        f"At trajectory boundary        : {_yn(d['endpoint_at_trajectory_boundary'])}",
        f"95% spatial-refit uncertainty band : {band}",
        f"Band includes zero            : {_yn(d['endpoint_ci_includes_zero'])}",
        f"Marginal feature support       : {support}",
        f"Joint multivariate support     : {joint_support_line}",
        f"Local endpoint density        : {density}",
        f"Local stability               : {stability}",
        f"Local sign change (neighbours): {_yn(d['endpoint_local_sign_change'])}",
        f"Robust endpoint                : {_yn(d['endpoint_robust'])}",
        f"Interpretation                : {classify_endpoint_interpretation(d)}",
        "Note                           : the endpoint sits at the edge of "
        "the EVALUATED scenario trajectory by construction; this is distinct "
        "from being outside the region of empirical data support (see "
        "'Marginal feature support' / 'Joint multivariate support' above) and distinct from local data "
        "density (see 'Local endpoint density' above) - feature support, "
        "local density, and trajectory-boundary position are three separate "
        "questions and are never combined into one PASS/FAIL statement.",
    ]
    if d["endpoint_local_sign_change"]:
        lines.append("Note (sign change)             : " + interpretation_text("endpoint_sign_flip"))
    if (d["endpoint_uncertainty_low"] is not None
            and d["endpoint_ci_includes_zero"] is False
            and publication_convergence_status not in (None, "STABILIZED")):
        lines.append(
            "Caveat (convergence)           : this band excludes zero, but "
            f"the publication convergence check has not passed "
            f"({publication_convergence_status}) - treat this band as "
            "provisional, not confirmed, until convergence is reached."
        )
    return "\n".join(lines)


def format_endpoint_headline(
    d: dict, joint_support_status: Optional[str] = None,
    publication_convergence_status: Optional[str] = None,
) -> list[str]:
    """Condensed endpoint-assessment console lines, parallel to
    format_maximum_headline - omits nothing scientifically material and
    drops only fields that are None/not applicable.

    `joint_support_status` is the ALREADY-COMPUTED classification from
    Step 10b (`compute_joint_multivariate_support`) - it is read here, not
    recomputed. Passing None (the default) reports it as "NOT AVAILABLE"
    rather than silently treating the marginal-support value below as if
    it were the joint result.

    `publication_convergence_status`: same caveat rule as
    format_maximum_headline - appended whenever the band excludes zero
    and convergence has not been confirmed.
    """
    lines = [f"{d['label']} (endpoint):"]
    lines.append(
        f"    Endpoint evaluated NEGI = {d['endpoint_evaluated_negi']:.3f} "
        f"at {d['endpoint_scenario_position']:.0f}%"
    )
    if d["endpoint_uncertainty_low"] is not None:
        lines.append(
            f"    95% spatial-refit uncertainty band = "
            f"[{d['endpoint_uncertainty_low']:.3f}, {d['endpoint_uncertainty_high']:.3f}]"
        )
        lines.append(f"    Includes zero = {'YES' if d['endpoint_ci_includes_zero'] else 'NO'}")
        if (d["endpoint_ci_includes_zero"] is False
                and publication_convergence_status not in (None, "STABILIZED")):
            lines.append(
                "    Caveat: this band excludes zero, but the publication "
                f"convergence check has not passed ({publication_convergence_status}) "
                "- treat this band as provisional, not confirmed, until "
                "convergence is reached."
            )
    if d["endpoint_knn_support"] is not None:
        support_label = "PASS" if d["endpoint_knn_support"] else "FAIL"
        lines.append(f"    Marginal feature support = {support_label}")
        lines.append(
            f"    Joint multivariate support = "
            f"{joint_support_status if joint_support_status is not None else 'NOT AVAILABLE'}"
        )
    if d.get("endpoint_local_density") is not None:
        if (d.get("endpoint_knn_distance") is not None
                and d.get("endpoint_knn_distance_percentile") is not None):
            lines.append(
                f"    Local endpoint density = {d['endpoint_local_density']} "
                f"(mean neighbour distance = {d['endpoint_knn_distance']:.4f}; "
                f"{d['endpoint_knn_distance_percentile']:.0f}th percentile of "
                "training-density distribution)"
            )
        else:
            lines.append(f"    Local endpoint density = {d['endpoint_local_density']}")
    if d["endpoint_local_stability"] is not None:
        lines.append(
            f"    Local stability = {'STABLE' if d['endpoint_local_stability'] else 'UNSTABLE'}"
        )
    if d["endpoint_robust"] is not None:
        lines.append(f"    Robust endpoint = {'YES' if d['endpoint_robust'] else 'NO'}")
    lines.append(f"    Interpretation = {classify_endpoint_interpretation(d).upper()}")
    return lines


def classify_maximum_interpretation(d: dict) -> str:
    """One-line interpretation label consistent with the Level 1/2/3
    hierarchy already enforced by build_maximum_diagnostics."""
    if d["maximum_supports_interior_optimum"]:
        return "Supported interior optimum"
    if d["maximum_robust"]:
        return "Robust maximum (directional evidence only, not an interior optimum)"
    return "Numerical maximum only"


def format_maximum_classification(
    d: dict, joint_support_status: Optional[str] = None,
    publication_convergence_status: Optional[str] = None,
) -> str:
    """Render the single-source-of-truth maximum-diagnostics dict as a
    fixed-width, machine-readable classification block. Every value here
    is read directly from `d` - nothing is recalculated.

    `joint_support_status` is the already-computed Step 10b classification
    (`compute_joint_multivariate_support`) - read here, never recomputed.
    None (the default) reports "NOT AVAILABLE" rather than conflating it
    with the marginal-support value below.

    `publication_convergence_status`: same caveat rule as
    format_maximum_headline - appended whenever the band excludes zero
    and convergence has not been confirmed. This is the persisted,
    machine-readable export (maximum_classification.txt), so the caveat
    must be here too, not only in the console headline.
    """
    def _yn(v: Optional[bool]) -> str:
        return "N/A" if v is None else ("YES" if v else "NO")

    stability = (
        "N/A" if d["maximum_local_stability"] is None
        else ("STABLE" if d["maximum_local_stability"] else "UNSTABLE")
    )
    if d["maximum_knn_support"] is None:
        support = (
            "N/A \u2014 baseline condition"
            if np.isclose(d["maximum_scenario_position"], 0.0)
            else "N/A"
        )
    else:
        support = "PASS" if d["maximum_knn_support"] else "FAIL"
    joint_support_line = (
        joint_support_status if joint_support_status is not None else "NOT AVAILABLE"
    )
    band = (
        f"[{d['maximum_uncertainty_low']:.4f}, {d['maximum_uncertainty_high']:.4f}]"
        if d["maximum_uncertainty_low"] is not None else "N/A"
    )
    near_boundary = d["maximum_near_boundary"] or d["maximum_at_boundary"]
    density = d.get("maximum_local_density") or "N/A"

    lines = [
        f"{d['label']}",
        f"Maximum evaluated NEGI       : {d['maximum_evaluated_negi']:.4f}",
        f"Maximum position             : {d['maximum_scenario_position']:.1f}%",
        f"Near trajectory boundary     : {_yn(near_boundary)}",
        f"95% spatial-refit uncertainty band : {band}",
        f"Band includes zero           : {_yn(d['maximum_ci_includes_zero'])}",
        f"Marginal feature support     : {support}",
        f"Joint multivariate support   : {joint_support_line}",
        f"Local density at maximum     : {density}",
        f"Local stability              : {stability}",
        f"Robust maximum                : {_yn(d['maximum_robust'])}",
        f"Supported interior optimum   : {_yn(d['maximum_supports_interior_optimum'])}",
        f"Interpretation               : {classify_maximum_interpretation(d)}",
    ]
    if near_boundary:
        lines.append(
            "Note                          : Maximum occurs near the evaluated "
            "scenario boundary and should not be interpreted as evidence of "
            "an interior optimum."
        )
    if d.get("maximum_local_sign_change"):
        lines.append("Note (sign change)            : " + interpretation_text("endpoint_sign_flip"))
    if (d["maximum_uncertainty_low"] is not None
            and d["maximum_ci_includes_zero"] is False
            and publication_convergence_status not in (None, "STABILIZED")):
        lines.append(
            "Caveat (convergence)          : this band excludes zero, but "
            f"the publication convergence check has not passed "
            f"({publication_convergence_status}) - treat this band as "
            "provisional, not confirmed, until convergence is reached."
        )
    return "\n".join(lines)


def format_maximum_headline(
    d: dict, joint_support_status: Optional[str] = None,
    publication_convergence_status: Optional[str] = None,
) -> list[str]:
    """Condensed [8] NEGI assessment lines for the console headline stream
    (HEADLINE level - visible in both verbose and concise console modes).

    Reads the same single-source-of-truth dict as format_maximum_classification
    and omits nothing scientifically material (near-boundary status,
    uncertainty band, feature support, local stability, robustness) for any
    scenario whose evaluated maximum is an actual candidate intervention
    point - it only drops fields that are None/not applicable for a given
    scenario (e.g. Scenario 1 has no per-point kNN support array).

    When the reported maximum sits at the trajectory baseline (0%
    intervention - the case for Scenario 1, where NEGI = 0 by construction
    and no interior optimum is identifiable at the reference parameter
    values), the headline is reduced to the three fields that are actually
    informative for a baseline condition: the (zero) maximum value merged
    with its position, whether it's a robust maximum (no), and the
    numerical-only interpretation. This is presentation only - it does not
    change any computed value, and the complete classification (uncertainty
    band, feature support, local stability, etc.) remains available via
    format_maximum_classification() and the exported
    maximum_classification.txt / maximum_diagnostics.csv.

    `joint_support_status` is the already-computed Step 10b classification
    (`compute_joint_multivariate_support`) - read here, never recomputed.
    None (the default) reports "NOT AVAILABLE" rather than conflating it
    with the marginal-support value below.

    `publication_convergence_status` is the already-computed status from
    compute_uncertainty_convergence_diagnostic (STABILIZED /
    NOT YET STABILIZED / None) - read here, never recomputed. Whenever the
    reported band excludes zero (the one condition that makes a maximum
    look like a candidate non-null finding) AND convergence has not been
    confirmed, an explicit caveat is appended here, at the point the band
    is shown - not left to a separate section a reader could miss - so the
    band can't be read or cited as confirmed without its own caveat
    attached.
    """
    position = d["maximum_scenario_position"]
    is_baseline = np.isclose(position, 0.0)
    position_phrase = "at baseline" if is_baseline else f"at {position:.0f}%"

    lines = [f"{d['label']}:"]
    lines.append(
        f"    Maximum evaluated NEGI = {d['maximum_evaluated_negi']:.3f} {position_phrase}"
    )

    if is_baseline:
        if d["maximum_robust"] is not None:
            lines.append(f"    Robust maximum = {'YES' if d['maximum_robust'] else 'NO'}")
        lines.append(f"    Interpretation = {classify_maximum_interpretation(d).upper()}")
        return lines

    if d["maximum_uncertainty_low"] is not None:
        lines.append(
            f"    95% spatial-refit uncertainty band = "
            f"[{d['maximum_uncertainty_low']:.3f}, {d['maximum_uncertainty_high']:.3f}]"
        )
        if (d["maximum_ci_includes_zero"] is False
                and publication_convergence_status not in (None, "STABILIZED")):
            lines.append(
                "    Caveat: this band excludes zero, but the publication "
                f"convergence check has not passed ({publication_convergence_status}) "
                "- treat this band as provisional, not confirmed, until "
                "convergence is reached."
            )
    if d["maximum_knn_support"] is not None:
        support_label = "PASS" if d["maximum_knn_support"] else "FAIL"
        lines.append(f"    Marginal feature support = {support_label}")
        lines.append(
            f"    Joint multivariate support = "
            f"{joint_support_status if joint_support_status is not None else 'NOT AVAILABLE'}"
        )
    if d.get("maximum_local_density") is not None:
        if (d.get("maximum_knn_distance") is not None
                and d.get("maximum_knn_distance_percentile") is not None):
            lines.append(
                f"    Local maximum density = {d['maximum_local_density']} "
                f"(mean neighbour distance = {d['maximum_knn_distance']:.4f}; "
                f"{d['maximum_knn_distance_percentile']:.0f}th percentile of "
                "training-density distribution)"
            )
        else:
            lines.append(f"    Local maximum density = {d['maximum_local_density']}")
    near_boundary = d["maximum_near_boundary"] or d["maximum_at_boundary"]
    if near_boundary:
        lines.append("    Near scenario-trajectory boundary = YES")
    if d["maximum_local_stability"] is not None:
        lines.append(
            f"    Local stability = {'STABLE' if d['maximum_local_stability'] else 'UNSTABLE'}"
        )
    if d["maximum_robust"] is not None:
        lines.append(f"    Robust maximum = {'YES' if d['maximum_robust'] else 'NO'}")
    interpretation = classify_maximum_interpretation(d)
    if interpretation == "Numerical maximum only":
        if d["maximum_ci_includes_zero"] is False:
            band = (
                f" [{d['maximum_uncertainty_low']:.3f}, {d['maximum_uncertainty_high']:.3f}]"
                if d["maximum_uncertainty_low"] is not None else ""
            )
            lines.append(
                f"    -> NEGI at the numerical maximum is positive under "
                f"the current spatial-refit uncertainty band{band}, but "
                "the location of the maximum is non-robust because it is "
                "near the trajectory boundary and/or locally unstable. Do "
                "not interpret it as an optimal intervention."
            )
        else:
            lines.append(
                "    -> Not statistically distinguishable from zero. Do "
                "not interpret as an optimal intervention."
            )
    else:
        lines.append(f"    Interpretation = {interpretation.upper()}")
    return lines


def compute_feature_importance(
    model_results: ModelResults, cfg: Config, summary: SummaryLog
) -> FeatureImportanceResult:
    """Compute permutation feature importance for every model predictor
    (NDVI, NDBI, Elevation, ST_EMIS, ST_EMSD) across repeated seeds."""
    model, data = model_results.model, model_results.data
    perm_seeds  = np.arange(cfg.random_seed, cfg.random_seed + cfg.n_permutation_outer_seeds)

    runs: dict[str, list[float]] = {feat: [] for feat in data.X.columns}
    for seed in perm_seeds:
        def _build_and_fit(n_jobs: int, _seed=seed):
            return permutation_importance(
                model, data.X_test, data.y_test,
                n_repeats=cfg.n_permutation_inner_repeats,
                random_state=int(_seed), scoring="r2", n_jobs=n_jobs,
            )

        perm_run = _fit_search_with_worker_retry(
            _build_and_fit, cfg, f"STEP 6 permutation importance (seed={int(seed)})",
        )
        for i, feat in enumerate(data.X.columns):
            runs[feat].append(perm_run.importances_mean[i])

    rows = []
    for feat in data.X.columns:
        vals     = np.array(runs[feat])
        feat_mean = vals.mean()
        feat_std  = vals.std(ddof=1)
        t_crit    = float(scipy_stats.t.ppf(0.975, df=len(vals) - 1)) if len(vals) > 1 else float("nan")
        feat_ci95 = t_crit * feat_std / np.sqrt(len(vals))
        rows.append({
            "Feature":          feat,
            "Importance":       feat_mean,
            "Importance_Std":   feat_std,
            "Importance_CI95":  feat_ci95,
            "Importance_CI_Low":  feat_mean - feat_ci95,
            "Importance_CI_High": feat_mean + feat_ci95,
            "N_Repeats":        len(vals),
        })
    table = pd.DataFrame(rows).sort_values("Importance", ascending=False)

    log_info(
        f"\nPermutation feature importance "
        f"(mean ± SD over {cfg.n_permutation_outer_seeds} outer seeds, "
        f"{cfg.n_permutation_inner_repeats} inner repeats each):"
    )
    log_info(table[["Feature", "Importance", "Importance_Std",
                     "Importance_CI_Low", "Importance_CI_High"]])
    ndvi_ndbi_corr = data.df["NDVI"].corr(data.df["NDBI"])
    log_info(
        "Note: Permutation importance reflects marginal predictive importance "
        f"and may be influenced by correlated predictors (NDVI-NDBI r = "
        f"{ndvi_ndbi_corr:.3f}).  Values are not additive."
    )
    save_csv(table, cfg.data_dir / "feature_importance.csv")
    for _, row in table.iterrows():
        summary.log(
            "Feature importance",
            f"{row['Feature']} permutation importance (mean ± 95% CI)",
            f"{row['Importance']:.4f} ± {row['Importance_CI95']:.4f}",
        )
    return FeatureImportanceResult(table=table)


def predict_lst(
    model,
    ndvi: float, ndbi: float, elevation: float,
    st_emis: float, st_emsd: float,
    feature_min: pd.Series, feature_max: pd.Series,
    cache: Optional[PredictionCache] = None,
) -> tuple[float, bool]:
    """Predict LST for a single five-feature vector, using the module-level
    cache. `st_emis` / `st_emsd` are model predictors held at their fixed
    reference values by every scenario - they are inputs here, not
    quantities the scenarios manipulate."""
    cache = PREDICTION_CACHE if cache is None else cache
    values = (ndvi, ndbi, elevation, st_emis, st_emsd)
    cached = cache.get(*values) if cache is not None else None
    if cached is not None:
        cache.hits += 1
        return cached

    row   = pd.DataFrame([[float(v) for v in values]], columns=FEATURES)
    raw   = model.predict(row)[0]
    ood   = ((row.iloc[0] < feature_min) | (row.iloc[0] > feature_max)).any()
    result = (float(raw), bool(ood))
    if cache is not None:
        cache.misses += 1
        cache.set(*values, result)
    return result


def _empirical_cumulative_bin_stats(
    df: pd.DataFrame, ndbi_percentile: float
) -> tuple[float, float, int, object, float]:
    """Nearest REAL, observed (NDVI, NDBI) pixel for the cumulative subset
    of pixels at or below a given NDBI percentile.

    Integrity-pass fix: the previous implementation returned the NDVI
    median and the NDBI median of the cumulative subset SEPARATELY. Those
    are two independent marginal statistics - nothing guarantees any real
    pixel actually has that joint (NDVI, NDBI) combination, so the
    resulting "archetype" could be a synthetic joint point even though
    each coordinate, on its own, was a real observed value. That
    contradicted the documented claim that the endpoint is "never
    synthetic."

    Fixed here: the subset medians are computed only as a TEMPORARY
    target. NDVI/NDBI within the cumulative subset are standardized
    (z-scored using the subset's own mean/std) and the Euclidean distance
    from every real pixel in the subset to the (standardized) target is
    computed; the SINGLE closest real observation is returned. The
    archetype is therefore always an actual observed pixel - guaranteed
    to lie inside the observed joint feature-space support - never a
    component-wise composite. No model prediction is used anywhere in
    this selection.

    Returns
    -------
    ndvi_endpoint, ndbi_endpoint : the nearest real pixel's actual NDVI/NDBI
    n : cumulative subset size
    archetype_pixel_index : the df index label of the selected pixel
    distance_to_target : standardized Euclidean distance from the selected
        pixel to the (temporary) subset-median target
    """
    ndbi_threshold = df["NDBI"].quantile(ndbi_percentile)
    bin_mask = df["NDBI"] <= ndbi_threshold
    n = int(bin_mask.sum())
    if n == 0:
        return float("nan"), float("nan"), 0, None, float("nan")

    subset = df.loc[bin_mask]
    ndvi_target = float(subset["NDVI"].median())
    ndbi_target = float(subset["NDBI"].median())

    ndvi_std = float(subset["NDVI"].std(ddof=0))
    ndbi_std = float(subset["NDBI"].std(ddof=0))
    ndvi_std = ndvi_std if ndvi_std > EPS else 1.0
    ndbi_std = ndbi_std if ndbi_std > EPS else 1.0

    z_ndvi = (subset["NDVI"].to_numpy() - ndvi_target) / ndvi_std
    z_ndbi = (subset["NDBI"].to_numpy() - ndbi_target) / ndbi_std
    dist = np.sqrt(z_ndvi ** 2 + z_ndbi ** 2)

    nearest_pos = int(np.argmin(dist))
    archetype_pixel_index = subset.index[nearest_pos]
    ndvi_endpoint = float(subset["NDVI"].iloc[nearest_pos])
    ndbi_endpoint = float(subset["NDBI"].iloc[nearest_pos])
    distance_to_target = float(dist[nearest_pos])

    return ndvi_endpoint, ndbi_endpoint, n, archetype_pixel_index, distance_to_target


def select_scenario2_archetype_endpoint(
    df: pd.DataFrame, cfg: Config, summary: SummaryLog,
) -> dict:
    """Define the Scenario 2 endpoint from a fixed, a-priori urban
    archetype - WITHOUT consulting the model in any way.

    Scenario 2 represents redevelopment toward the built-form profile of
    Jeddah's existing lower-density, more vegetated residential/park
    areas. That target NDBI percentile
    (``cfg.s2_archetype_ndbi_percentile``) is a documented constant fixed
    independently of the fitted model, before this function ever runs.
    The corresponding endpoint is the NEAREST OBSERVED ARCHETYPE PIXEL to
    that percentile's (NDVI, NDBI) target profile - a real observation,
    never a synthetic combination - via ``_empirical_cumulative_bin_stats``
    (integrity-pass fix: the subset's NDVI/NDBI medians are used only as a
    temporary target; the returned endpoint is the single closest real
    pixel to that target, not the two medians combined independently).

    This deliberately preserves causal separation between scenario
    definition and model evaluation. An earlier version of this pipeline
    (``select_scenario2_empirical_endpoint``, since deleted as dead code -
    see version control history) instead walked down NDBI percentiles and
    accepted the first one the FITTED MODEL predicted enough cooling at.
    On review that is circular: it turns
    Scenario -> Model -> NEGI into Model -> Scenario -> Model -> NEGI,
    i.e. it optimises the intervention against the very model that then
    evaluates it - a reviewer could reasonably say the scenario is no
    longer independent of the evaluation. This function removes the
    model from the selection step entirely; the model is used only
    downstream (in ``run_scenario_trajectories``) to evaluate predicted
    LST/NEGI along the trajectory this function defines, exactly as it
    would for any other externally-specified scenario.
    """
    ndvi_endpoint, ndbi_endpoint, n, pixel_index, distance_to_target = (
        _empirical_cumulative_bin_stats(df, cfg.s2_archetype_ndbi_percentile)
    )
    if n < cfg.s2_search_min_bin_count:
        raise RuntimeError(
            f"Scenario 2 archetype percentile "
            f"{cfg.s2_archetype_ndbi_percentile:.2f} has only {n} observed "
            "pixels at/below it - too sparse for a stable empirical "
            "endpoint. Choose a less extreme cfg.s2_archetype_ndbi_percentile."
        )
    summary.log(
        "Scenario construction", "Scenario 2 archetype endpoint",
        "The evaluated archetype is the nearest observed pixel to the "
        "target percentile profile. Endpoint fixed a priori, independent "
        f"of the model, at NDBI percentile "
        f"{cfg.s2_archetype_ndbi_percentile:.2f} - representing "
        "observed characteristics of Jeddah's existing lower-density, more "
        f"vegetated residential/park areas: pixel index={pixel_index}, "
        f"NDVI={ndvi_endpoint:.4f}, NDBI={ndbi_endpoint:.4f} "
        f"(distance to target medians={distance_to_target:.4f} standardized "
        f"units; n={n} observed pixels in the cumulative subset searched). "
        "The model plays no role in selecting this endpoint; it is used "
        "only afterward to evaluate predicted LST/NEGI along the "
        "resulting trajectory, preserving Scenario -> Model -> NEGI causal "
        "separation.",
    )
    log_info(
        f"\nScenario 2 archetype endpoint (nearest observed pixel, defined "
        f"independently of the model): NDBI percentile "
        f"{cfg.s2_archetype_ndbi_percentile:.2f} -> pixel index={pixel_index}, "
        f"NDVI={ndvi_endpoint:.4f}, NDBI={ndbi_endpoint:.4f} (distance to "
        f"target medians={distance_to_target:.4f}; n={n} observed pixels)."
    )
    return {
        "ndvi_endpoint": ndvi_endpoint,
        "ndbi_endpoint": ndbi_endpoint,
        "endpoint_percentile": cfg.s2_archetype_ndbi_percentile,
        "n_pixels": n,
        "archetype_pixel_index": pixel_index,
        "distance_to_target": distance_to_target,
    }


def build_scenario2_empirical_trajectory(
    df: pd.DataFrame, baseline_percentile: float, endpoint_percentile: float,
    scenarios: np.ndarray, cfg: Config,
    baseline_ndvi: float, baseline_ndbi: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the Scenario 2 (NDVI, NDBI) path as a sequence of NEAREST
    OBSERVED ARCHETYPE PIXELS between the baseline and the chosen endpoint
    percentile, resampled onto the shared scenario-fraction axis.

    This directly follows the model's learned response through the
    observed NDVI-NDBI joint co-variation (moving through real,
    increasingly-vegetated / less-built-up observed pixels) rather than
    moving NDVI and NDBI independently and linearly, which is not
    representative of how urban redevelopment actually co-varies these
    two quantities. Every intermediate point is the single nearest real
    pixel to that bin's (temporary) percentile target - never a
    component-wise composite or a synthetic linear blend (see
    ``_empirical_cumulative_bin_stats``) - so no additional feature-space
    clipping is required.

    Bug fix (baseline anchor): ``_empirical_cumulative_bin_stats`` targets
    the MEDIAN of the CUMULATIVE subset of pixels at/below a percentile -
    i.e. at ``baseline_percentile`` (~0.50) it targets the median of the
    bottom HALF of the NDBI distribution, not the pixel AT the 50th
    percentile itself. Because NDVI and NDBI are anti-correlated in this
    dataset, the median of that bottom half is systematically greener /
    less built-up than the true (median NDVI, median NDBI) baseline point
    used everywhere else (``baseline_lst``, Scenario 1's start, the
    ``ScenarioBaseline`` object). Previously this meant Scenario 2's own
    "0% intervention" point was NOT the same point as the baseline it is
    measured against - it was already partway toward the archetype
    endpoint, so the trajectory was credited with cooling benefit (and a
    nonzero NEGI, e.g. the widely-visible ~0.50 "maximum evaluated NEGI")
    that did not correspond to any actual modeled intervention. This was
    confirmed empirically: the pre-fix 0% point sat at NDBI~0.055 versus
    the true median NDBI of ~0.083, already predicting ~2.1 degC of the
    ~4.2 degC max trajectory cooling before any scenario intensity was
    applied.

    Fixed here: the FIRST bin (bin_fraction = 0.0, i.e. the trajectory's
    starting point) is pinned to the actual, already-known baseline
    (``baseline_ndvi``, ``baseline_ndbi``) rather than the nearest pixel
    to the cumulative-subset-median target. Every other bin (including
    the endpoint) is unchanged - still the single nearest real observed
    pixel to its own percentile target, exactly as before. This
    guarantees delta_t_s2[0] == 0 (matching Scenario 1 and the "NEGI = 0
    at baseline" invariant already documented elsewhere in this module),
    without altering how the endpoint or any interior archetype point is
    selected.
    """
    bin_percentiles = np.linspace(
        baseline_percentile, endpoint_percentile, cfg.s2_trajectory_n_bins
    )
    bin_ndvi, bin_ndbi = [], []
    for pct in bin_percentiles:
        ndvi_val, ndbi_val, n, _pixel_idx, _dist = _empirical_cumulative_bin_stats(
            df, float(pct)
        )
        if n < cfg.s2_search_min_bin_count:
            ndvi_val, ndbi_val = bin_ndvi[-1], bin_ndbi[-1]
        bin_ndvi.append(ndvi_val)
        bin_ndbi.append(ndbi_val)
    bin_ndvi, bin_ndbi = np.array(bin_ndvi), np.array(bin_ndbi)

    bin_ndvi[0] = baseline_ndvi
    bin_ndbi[0] = baseline_ndbi

    bin_fraction = np.linspace(0.0, 1.0, cfg.s2_trajectory_n_bins)
    ndvi_traj = np.interp(scenarios, bin_fraction, bin_ndvi)
    ndbi_traj = np.interp(scenarios, bin_fraction, bin_ndbi)
    return ndvi_traj, ndbi_traj


def run_scenario_trajectories(
    model_results: ModelResults, cfg: Config, summary: SummaryLog
) -> tuple[ScenarioBaseline, ScenarioTrajectory, ScenarioTrajectory]:
    """Build the Scenario 1 (isolated NDVI counterfactual, NDBI fixed) and Scenario 2 (realistic
    municipal greening intervention) NDVI/NDBI trajectories.

    Scenario 1 (UNCHANGED): NDBI held fixed at its observed baseline while
    NDVI increases linearly from the observed baseline (median) to the
    observed ``cfg.ndvi_target_percentile`` quantile - isolating the
    model's NDVI effect net of urbanization.

    Scenario 2 (REVISED v2) represents a realistic municipal greening
    intervention for Jeddah, with its endpoint fixed a priori from an
    urban archetype - deliberately WITHOUT reference to the model's
    predictions, to preserve Scenario -> Model -> NEGI causal separation.

    Background: the original fixed +20% NDVI / -8% NDBI relative-change
    endpoint, although formally inside observed feature-space support,
    fell inside a flat/saturated region of the learned LST response
    surface (predicted endpoint cooling only about -0.035 degC against a
    local finite-difference gradient of about -1.05 degC at the
    baseline; 93.5% of consecutive trajectory points shared identical
    predictions). The issue was the trajectory's definition, not
    extrapolation, and not the NEGI equation. A first revision addressed
    this by searching NDBI percentiles for one the model predicted
    enough cooling at - but that construction is circular
    (Model -> Scenario -> Model -> NEGI: it optimises the intervention
    against the very model that then evaluates it), so that selector has
    since been deleted as dead code (see version control history for the
    prior implementation).

    The endpoint used now is instead chosen by
    ``select_scenario2_archetype_endpoint``: a fixed, documented NDBI
    percentile (``cfg.s2_archetype_ndbi_percentile``, default 0.15),
    representing the observed characteristics of Jeddah's existing
    lower-density, more vegetated residential/park districts - a target
    profile decided on urban-form grounds before the model is ever
    consulted. The corresponding endpoint is the NEAREST OBSERVED
    ARCHETYPE PIXEL to that percentile's (NDVI, NDBI) target (never a
    synthetic combination of separately-computed marginal medians - see
    ``_empirical_cumulative_bin_stats``). The model is used only
    AFTERWARD, in the loop below, to evaluate predicted LST/NEGI along
    the resulting trajectory - exactly as it would for any other
    externally-specified scenario.

    The path FROM the baseline TO that fixed endpoint is built by
    ``build_scenario2_empirical_trajectory`` as a sequence of nearest
    observed archetype pixels spanning the intermediate percentiles (i.e.
    following the OBSERVED NDVI-NDBI joint co-variation), resampled onto
    the same scenario-fraction axis Scenario 1 uses - rather than moving
    NDVI and NDBI independently and linearly, which does not represent
    how redevelopment actually co-varies the two quantities. This
    construction, like the endpoint choice, uses only observed data
    statistics - the model plays no role in it either. Elevation is left
    unchanged, as in Scenario 1 (both scenarios evaluate at the fixed
    observed median elevation), and so are the two emissivity covariates
    ST_EMIS / ST_EMSD (five-feature model): they are model predictors held
    at their training-median reference values, never scenario variables.
    Every point on the path is a real observed statistic, so no additional
    feature-space clipping is required or applied.

    After the trajectory is evaluated, ``cfg.s2_meaningful_cooling_threshold_c``
    is used purely DESCRIPTIVELY (see the endpoint-cooling log line below)
    to report whether this independently-chosen archetype happens to fall
    in a region the model predicts meaningful cooling in - this is
    reporting, not a selection criterion, and does not feed back into the
    trajectory in any way.

    This again replaces the earlier "greening + densification" spline
    co-variation definition, the fixed +20%/-8% relative-change
    definition, and the model-search endpoint definition - all three were
    superseded, unused, and have since been deleted as dead code (see
    version control history for the prior implementations).

    No change was made to: the NEGI equation, the energy-cost
    formulation, the XGBoost model or its hyperparameter tuning, nested
    GroupKFold CV, the confirmatory holdout, calibration, residual
    diagnostics, Moran's I, the uncertainty-band methodology, robustness
    criteria, endpoint diagnostics, the QA system, or export formats -
    this function only changes how Scenario 2's (NDVI, NDBI) trajectory
    is constructed; everything downstream (predict_lst, NEGI, uncertainty,
    diagnostics, plotting) consumes whatever trajectory it is handed and
    is otherwise untouched.
    """
    model, data = model_results.model, model_results.data
    df = data.df_train

    scenarios = np.arange(0.0, 1.0001, cfg.scenario_step)
    assert np.all(np.diff(scenarios) > 0), "Scenario axis must be strictly increasing."

    fixed_values     = fixed_reference_values(df)
    fixed_elevation  = fixed_values["Elevation"]
    fixed_st_emis    = fixed_values["ST_EMIS"]
    fixed_st_emsd    = fixed_values["ST_EMSD"]
    baseline_ndbi    = df["NDBI"].median()
    baseline_ndvi    = df["NDVI"].median()
    ndvi_target      = df["NDVI"].quantile(cfg.ndvi_target_percentile)
    ndvi_ndbi_corr   = df["NDVI"].corr(df["NDBI"])

    log_info(
        f"\nScenario axis: evaluated from observed median baseline NDVI "
        f"({baseline_ndvi:.4f}) to observed NDVI p95 target ({ndvi_target:.4f})."
    )

    baseline_lst, baseline_ood = predict_lst(
        model, baseline_ndvi, baseline_ndbi, fixed_elevation,
        fixed_st_emis, fixed_st_emsd,
        data.feature_min, data.feature_max,
    )
    if baseline_ood:
        raise RuntimeError(
            "The baseline feature vector is outside observed feature bounds."
        )

    ndvi_min_observed, ndvi_max_observed = df["NDVI"].min(), df["NDVI"].max()
    ndbi_min_observed, ndbi_max_observed = df["NDBI"].min(), df["NDBI"].max()

    baseline_ndbi_percentile = float((df["NDBI"] <= baseline_ndbi).mean())
    s2_endpoint = select_scenario2_archetype_endpoint(df, cfg, summary)
    s2_ndvi_endpoint = s2_endpoint["ndvi_endpoint"]
    s2_ndbi_endpoint = s2_endpoint["ndbi_endpoint"]

    log_info(
        f"\nScenario 2 (realistic municipal greening intervention, "
        f"archetype endpoint fixed independently of the model): NDVI "
        f"baseline {baseline_ndvi:.4f} -> endpoint {s2_ndvi_endpoint:.4f}; "
        f"NDBI baseline {baseline_ndbi:.4f} -> endpoint {s2_ndbi_endpoint:.4f} "
        f"(NDBI percentile {baseline_ndbi_percentile:.2f} -> "
        f"{s2_endpoint['endpoint_percentile']:.2f}); elevation held fixed "
        f"at {fixed_elevation:.4f}; ST_EMIS held fixed at {fixed_st_emis:.4f} "
        f"and ST_EMSD held fixed at {fixed_st_emsd:.4f} (model predictors, "
        "not intervention variables)."
    )

    def scenario_ndvi(fraction: float) -> float:
        """Scenario 1 NDVI values along the trajectory (UNCHANGED)."""
        return baseline_ndvi + fraction * (ndvi_target - baseline_ndvi)

    ndvi_traj_s2, ndbi_traj_s2 = build_scenario2_empirical_trajectory(
        df, baseline_ndbi_percentile, s2_endpoint["endpoint_percentile"],
        scenarios, cfg, baseline_ndvi, baseline_ndbi,
    )

    lst_s1, lst_s2, ood_s1, ood_s2 = [], [], [], []
    ndvi_s1_vals, ndvi_s2_vals, ndbi_s2_vals = [], [], []
    for i, fraction in enumerate(scenarios):
        ndvi_val = scenario_ndvi(fraction)
        ndvi_s1_vals.append(ndvi_val)

        v1, o1 = predict_lst(
            model, ndvi_val, baseline_ndbi, fixed_elevation,
            fixed_st_emis, fixed_st_emsd,
            data.feature_min, data.feature_max,
        )
        lst_s1.append(v1)
        ood_s1.append(o1)

        ndvi2_val = float(ndvi_traj_s2[i])
        ndbi2_val = float(ndbi_traj_s2[i])
        ndvi_s2_vals.append(ndvi2_val)
        ndbi_s2_vals.append(ndbi2_val)
        v2, o2 = predict_lst(
            model, ndvi2_val, ndbi2_val, fixed_elevation,
            fixed_st_emis, fixed_st_emsd,
            data.feature_min, data.feature_max,
        )
        lst_s2.append(v2)
        ood_s2.append(o2)

    assert np.isclose(ndvi_s2_vals[0], baseline_ndvi, atol=1e-9), (
        "Scenario 2's 0% NDVI does not match the baseline NDVI - the "
        "baseline-anchor fix in build_scenario2_empirical_trajectory is "
        "not being applied."
    )
    assert np.isclose(ndbi_s2_vals[0], baseline_ndbi, atol=1e-9), (
        "Scenario 2's 0% NDBI does not match the baseline NDBI - the "
        "baseline-anchor fix in build_scenario2_empirical_trajectory is "
        "not being applied."
    )
    assert np.isclose(lst_s2[0], baseline_lst, atol=1e-6), (
        "Scenario 2's predicted LST at 0% scenario intensity does not "
        "match baseline_lst - Scenario 2 is not starting from the true "
        "baseline, so its NEGI values would be measured against the "
        "wrong reference point."
    )

    s2_endpoint_cooling_c = float(baseline_lst - lst_s2[-1])
    s2_endpoint_meaningful = s2_endpoint_cooling_c >= cfg.s2_meaningful_cooling_threshold_c
    summary.log(
        "Scenario construction",
        "Scenario 2 definition",
        "Realistic municipal greening intervention (urban-archetype "
        "endpoint, defined independently of the model): NDBI percentile "
        f"{baseline_ndbi_percentile:.2f} (baseline) -> "
        f"{s2_endpoint['endpoint_percentile']:.2f} (archetype); endpoint is "
        f"the REAL observed cumulative-bin median at that percentile "
        f"(NDVI={s2_ndvi_endpoint:.4f}, NDBI={s2_ndbi_endpoint:.4f}); "
        f"elevation ({fixed_elevation:.4f}), ST_EMIS ({fixed_st_emis:.4f}) and "
        f"ST_EMSD ({fixed_st_emsd:.4f}) unchanged (fixed reference values; "
        "not intervention variables). No synthetic "
        "combination is used for the endpoint or any intermediate point - "
        "see build_scenario2_empirical_trajectory(). POST-HOC only (not "
        f"used to select the endpoint): the model predicts "
        f"{s2_endpoint_cooling_c:.4f} degC of cooling at this archetype, "
        f"which {'meets' if s2_endpoint_meaningful else 'does not meet'} "
        f"the {cfg.s2_meaningful_cooling_threshold_c:.2f} degC descriptive "
        "threshold used elsewhere in this report to characterise "
        "'meaningful' predicted cooling.",
    )
    if not s2_endpoint_meaningful:
        summary.add_warning(
            "Scenario 2: the independently-defined archetype endpoint "
            f"(NDBI percentile {s2_endpoint['endpoint_percentile']:.2f}) "
            f"predicts only {s2_endpoint_cooling_c:.4f} degC of cooling, "
            f"below the {cfg.s2_meaningful_cooling_threshold_c:.2f} degC "
            "descriptive threshold. This is reported rather than "
            "concealed; the endpoint was not changed in response, since "
            "doing so would reintroduce model-guided selection."
        )

    s1 = ScenarioTrajectory(
        "Scenario 1", scenarios,
        np.array(ndvi_s1_vals), np.full(len(scenarios), baseline_ndbi),
        fixed_elevation, fixed_st_emis, fixed_st_emsd,
        np.array(lst_s1), np.array(ood_s1),
    )
    s2 = ScenarioTrajectory(
        "Scenario 2", scenarios,
        np.array(ndvi_s2_vals), np.array(ndbi_s2_vals),
        fixed_elevation, fixed_st_emis, fixed_st_emsd,
        np.array(lst_s2), np.array(ood_s2),
    )

    def _report_ood(traj: ScenarioTrajectory, label: str) -> None:
        if traj.out_of_bounds.any():
            first_idx = np.argmax(traj.out_of_bounds)
            log_info(
                f"\n{label}: {traj.out_of_bounds.sum()}/{len(traj.out_of_bounds)} "
                f"points outside observed feature bounds "
                f"(first at scenario={traj.scenario_pct[first_idx]:.1f}%)."
            )
        else:
            log_info(
                f"\n{label}: The evaluated trajectory remains within the "
                "observed predictor bounds throughout."
            )

    _report_ood(s1, "Scenario 1")
    _report_ood(s2, "Scenario 2")

    baseline = ScenarioBaseline(
        baseline_lst, baseline_ndvi, baseline_ndbi,
        ndvi_target, fixed_elevation, ndvi_ndbi_corr,
        fixed_st_emis, fixed_st_emsd,
    )
    return baseline, s1, s2


def run_scenario2_percentile_sensitivity(
    model_results: ModelResults,
    baseline: ScenarioBaseline,
    s1: ScenarioTrajectory,
    support: SupportDiagnostics,
    cfg: Config,
    summary: SummaryLog,
) -> pd.DataFrame:
    """Engineering addition: evaluate the SAME model-blind Scenario 2
    archetype construction at several alternative NDBI percentiles, purely
    for reporting. This is a sensitivity/robustness check on the choice of
    ``cfg.s2_archetype_ndbi_percentile`` - it never changes, searches, or
    optimises the primary Scenario 2 endpoint, and the primary endpoint is
    computed entirely separately (in ``run_scenario_trajectories``).

    For each percentile in ``cfg.s2_sensitivity_percentiles``:
      1. Build the endpoint exactly as ``select_scenario2_archetype_endpoint``
         does - the nearest real observed pixel to that percentile's
         (NDVI, NDBI) target (never a synthetic combination) - via the
         same ``_empirical_cumulative_bin_stats`` helper.
      2. Build the full baseline -> endpoint path with the unmodified
         ``build_scenario2_empirical_trajectory``.
      3. Evaluate the already-fitted model along that path with the
         unmodified ``predict_lst`` (no refitting, no model changes).
      4. Compute NEGI with the unmodified ``compute_negi_results``.
      5. Compute the endpoint's spatial-refit uncertainty band with the
         unmodified ``compute_uncertainty_bands`` (same spatial-block
         holdout procedure, same ``cfg.n_spatial_refits``, used elsewhere
         for the primary Scenario 2 trajectory).
      6. Classify endpoint robustness with the unmodified, single-source-
         of-truth ``build_endpoint_diagnostics`` / ``evaluate_negi_candidate``
         - the same function used for the primary Scenario 2 endpoint - so
         "robust" means exactly the same thing here as everywhere else in
         this report.

    No scenario-generation, NEGI, uncertainty, or robustness FORMULA is
    redefined here; every one of the six steps above calls an existing,
    unmodified function with a different percentile as input. This
    function only assembles those already-existing outputs into one
    comparison table.
    """
    data  = model_results.data
    model = model_results.model
    df    = data.df_train
    scenarios = s1.scenario_fraction
    baseline_ndbi_percentile = float((df["NDBI"] <= baseline.baseline_ndbi).mean())

    nn_model_k = NearestNeighbors(n_neighbors=cfg.support_knn_k).fit(
        support.scaler.transform(data.X_train)
    )

    rows = []
    for pct in cfg.s2_sensitivity_percentiles:
        ndvi_end, ndbi_end, n_end, _pixel_idx_end, _dist_end = _empirical_cumulative_bin_stats(
            df, float(pct)
        )
        if n_end < cfg.s2_search_min_bin_count:
            log_warning(
                f"\nScenario 2 percentile sensitivity: percentile {pct:.2f} "
                f"has only {n_end} observed pixels at/below it (< "
                f"{cfg.s2_search_min_bin_count}) - too sparse for a stable "
                "empirical endpoint; skipped."
            )
            summary.add_warning(
                f"Scenario 2 percentile sensitivity: percentile {pct:.2f} "
                f"skipped ({n_end} observed pixels, below the "
                f"{cfg.s2_search_min_bin_count}-pixel stability floor)."
            )
            continue

        ndvi_traj, ndbi_traj = build_scenario2_empirical_trajectory(
            df, baseline_ndbi_percentile, float(pct), scenarios, cfg,
            baseline.baseline_ndvi, baseline.baseline_ndbi,
        )

        lst_vals, ood_vals = [], []
        for ndvi_v, ndbi_v in zip(ndvi_traj, ndbi_traj):
            v, o = predict_lst(
                model, float(ndvi_v), float(ndbi_v), baseline.fixed_elevation,
                baseline.fixed_st_emis, baseline.fixed_st_emsd,
                data.feature_min, data.feature_max,
            )
            lst_vals.append(v)
            ood_vals.append(o)

        s2_p = ScenarioTrajectory(
            f"Scenario 2 (p={pct:.2f})", scenarios,
            np.array(ndvi_traj), np.array(ndbi_traj),
            baseline.fixed_elevation, baseline.fixed_st_emis, baseline.fixed_st_emsd,
            np.array(lst_vals), np.array(ood_vals),
        )

        negi_p, _ref_cooling_p = compute_negi_results(baseline, s1, s2_p, cfg)

        mean_knn_p   = _mean_knn_distance(s2_p, support.scaler, nn_model_k, cfg)
        in_support_p = mean_knn_p <= support.support_threshold_k

        unc_p = compute_uncertainty_bands(
            model_results, baseline, s1, s2_p, negi_p, cfg,
            log_convergence_detail=False,
            max_refits=cfg.n_spatial_refits_percentile_sensitivity,
        )

        endpoint_p = build_endpoint_diagnostics(
            f"Scenario 2 (percentile={pct:.2f})", negi_p.negi_s2, negi_p.scenario_pct,
            -1, unc_p.percentiles.get("negi_s2"), in_support_p, mean_knn_p, cfg,
            sparsity_threshold=support.sparsity_threshold_k,
            density_moderate_threshold=support.density_moderate_threshold_k,
            reference_knn_distribution=support.reference_mean_knn_dist,
            actual_n_refits=unc_p.n_refits_used,
        )

        endpoint_deltat = float(baseline.baseline_lst - s2_p.lst[-1])
        endpoint_ci_width = (
            endpoint_p["endpoint_uncertainty_high"] - endpoint_p["endpoint_uncertainty_low"]
        )
        row = {
            "percentile":        float(pct),
            "endpoint_ndvi":     float(s2_p.ndvi[-1]),
            "endpoint_ndbi":     float(s2_p.ndbi[-1]),
            "endpoint_lst":      float(s2_p.lst[-1]),
            "endpoint_deltaT":   endpoint_deltat,
            "endpoint_negi":     float(endpoint_p["endpoint_evaluated_negi"]),
            "maximum_scenario_pct": float(negi_p.scenario_pct[negi_p.s2_optimum_idx]),
            "maximum_negi":         float(negi_p.negi_s2[negi_p.s2_optimum_idx]),
            "endpoint_ci_lower": endpoint_p["endpoint_uncertainty_low"],
            "endpoint_ci_upper": endpoint_p["endpoint_uncertainty_high"],
            "endpoint_ci_width": endpoint_ci_width,
            "endpoint_robust":   endpoint_p["endpoint_robust"],
            "converged_early":   bool(unc_p.converged_early),
            "n_refits_used":     int(unc_p.n_refits_used),
        }
        rows.append(row)

    table = pd.DataFrame(rows)

    if len(table):
        n_total      = len(table)
        n_failed     = int((~table["converged_early"]).sum())
        n_converged  = int(table["converged_early"].sum())
        median_width = float(table["endpoint_ci_width"].median())
        log_info(
            f"\nScenario 2 sensitivity: {n_total} percentile setting(s) evaluated.\n"
            f"  {n_failed} failed convergence before max refits.\n"
            f"  {n_converged} converged.\n"
            f"  Median endpoint CI width = {median_width:.5f}\n"
            "  (Per-percentile convergence detail is available in "
            "development run_mode / debug logs only.)"
        )

    return table


def report_scenario2_percentile_sensitivity(
    table: pd.DataFrame, cfg: Config, summary: SummaryLog,
) -> Path:
    """Export the Scenario 2 percentile-sensitivity table to CSV and log
    it to the summary report. Reporting/export only - every value here is
    read directly from an already-computed `table` (whether it was just
    computed fresh or restored from cache), so this always runs and always
    produces the same summary-log rows regardless of cache status - the
    same compute/report split already used for
    compute_uncertainty_bands / report_uncertainty_bands elsewhere."""
    path = save_csv(table, cfg.data_dir / "scenario2_percentile_sensitivity.csv", cfg=cfg, debug_only=True)
    log_info(
        f"\nScenario 2 percentile sensitivity: {len(table)} percentile(s) "
        f"evaluated ({', '.join(f'{p:.2f}' for p in table['percentile'])}); "
        f"exported to {path}."
    )

    if len(table):
        actual_min = int(table["n_refits_used"].min())
        actual_max = int(table["n_refits_used"].max())
        actual_str = (
            str(actual_min) if actual_min == actual_max
            else f"{actual_min}-{actual_max} (adaptive convergence stopped some points early)"
        )
        log_info(
            "\nScenario 2 percentile sensitivity refit budget:\n"
            f"  Configured refits per percentile (ceiling): {cfg.n_spatial_refits_percentile_sensitivity}\n"
            f"  Actual refits per percentile used:           {actual_str}"
        )
        summary.log(
            "Scenario 2 percentile sensitivity", "Refit budget (configured / actual)",
            f"{cfg.n_spatial_refits_percentile_sensitivity} / {actual_str}",
        )
    for _, row in table.iterrows():
        summary.log(
            "Scenario 2 percentile sensitivity",
            f"Percentile {row['percentile']:.2f}",
            f"NDVI={row['endpoint_ndvi']:.4f}, NDBI={row['endpoint_ndbi']:.4f}, "
            f"predicted deltaT={row['endpoint_deltaT']:.4f} degC, "
            f"NEGI={row['endpoint_negi']:.4f} (95% CI "
            f"[{row['endpoint_ci_lower']:.4f}, {row['endpoint_ci_upper']:.4f}]), "
            f"robust={row['endpoint_robust']}. Computed post-hoc for "
            "reporting only; does not affect the primary Scenario 2 "
            f"endpoint, fixed at percentile {cfg.s2_archetype_ndbi_percentile:.2f}.",
        )
    n_robust = int(table["endpoint_robust"].sum()) if len(table) else 0
    summary.log(
        "Scenario 2 percentile sensitivity", "Percentiles evaluated",
        ", ".join(f"{p * 100:.0f}%" for p in table["percentile"]) if len(table) else "none",
    )
    summary.log(
        "Scenario 2 percentile sensitivity", "Robust endpoints",
        f"{n_robust}/{len(table)}",
    )

    stats = compute_scenario2_percentile_summary_stats(table)
    stats_path = save_csv(
        pd.DataFrame([stats]), cfg.data_dir / "scenario2_percentile_summary.csv",
        cfg=cfg, debug_only=True,
    )
    log_info(
        f"\nScenario 2 percentile-sensitivity summary ({stats['n_percentiles_evaluated']} "
        f"percentiles evaluated): robust endpoint in "
        f"{stats['fraction_robust'] * 100:.1f}% of percentiles; "
        f"CI excludes zero in {stats['fraction_ci_excludes_zero'] * 100:.1f}%; "
        f"positive endpoint NEGI in {stats['fraction_positive_negi'] * 100:.1f}%; "
        f"endpoint NEGI range [{stats['min_endpoint_negi']:.4f}, "
        f"{stats['max_endpoint_negi']:.4f}]; exported to {stats_path}."
    )
    summary.log(
        "Scenario 2 percentile sensitivity", "Summary across widened sweep",
        f"Robust: {stats['fraction_robust'] * 100:.1f}% | "
        f"CI excludes zero: {stats['fraction_ci_excludes_zero'] * 100:.1f}% | "
        f"Positive endpoint NEGI: {stats['fraction_positive_negi'] * 100:.1f}% | "
        f"NEGI range: [{stats['min_endpoint_negi']:.4f}, {stats['max_endpoint_negi']:.4f}].",
    )
    return path


def compute_scenario2_percentile_summary_stats(table: pd.DataFrame) -> dict:
    """Summary statistics across the Scenario 2 percentile-sensitivity
    sweep (integrity-pass item 4). Reporting/aggregation only - computed
    strictly from the already-evaluated per-percentile `table`; introduces
    no new NEGI, uncertainty, or robustness formula and does not feed back
    into the primary Scenario 2 endpoint.
    """
    if len(table) == 0:
        return {
            "n_percentiles_evaluated":   0,
            "fraction_robust":           float("nan"),
            "fraction_ci_excludes_zero": float("nan"),
            "fraction_positive_negi":    float("nan"),
            "min_endpoint_negi":         float("nan"),
            "max_endpoint_negi":         float("nan"),
            "min_refits_used":           float("nan"),
            "max_refits_used":           float("nan"),
        }
    ci_excludes_zero = ~(
        (table["endpoint_ci_lower"] <= 0.0) & (table["endpoint_ci_upper"] >= 0.0)
    )
    return {
        "n_percentiles_evaluated":   int(len(table)),
        "fraction_robust":           float(table["endpoint_robust"].mean()),
        "fraction_ci_excludes_zero": float(ci_excludes_zero.mean()),
        "fraction_positive_negi":    float((table["endpoint_negi"] > 0).mean()),
        "min_endpoint_negi":         float(table["endpoint_negi"].min()),
        "max_endpoint_negi":         float(table["endpoint_negi"].max()),
        "min_refits_used":           int(table["n_refits_used"].min()),
        "max_refits_used":           int(table["n_refits_used"].max()),
    }


def _mean_nearest_neighbor_distance(coords: np.ndarray) -> float:
    """Mean 1-nearest-neighbour Euclidean distance among a set of 2-D
    coordinates. Pure geometric diagnostic - no model, NEGI, or Moran's I
    arithmetic involved; used only by
    ``compute_scenario2_archetype_spatial_diagnostics`` below.
    """
    n = len(coords)
    if n < 2:
        return float("nan")
    k_eff = min(2, n)
    nn = NearestNeighbors(n_neighbors=k_eff, algorithm="kd_tree").fit(coords)
    dist, _ = nn.kneighbors(coords)
    if dist.shape[1] < 2:
        return float("nan")
    return float(np.mean(dist[:, 1]))


def compute_scenario2_archetype_spatial_diagnostics(
    df: pd.DataFrame, cfg: Config, summary: SummaryLog,
) -> dict:
    """Spatial-clustering diagnostic for the primary Scenario 2 archetype
    subset (integrity-pass item 5) - reporting only, does not affect
    scenario construction, NEGI, or any existing Moran's I calculation.

    Recomputes the same, deterministic, model-free
    ``_empirical_cumulative_bin_stats`` call already made inside
    ``select_scenario2_archetype_endpoint`` (same training-only df, same
    ``cfg.s2_archetype_ndbi_percentile`` - see the item 10 holdout-
    independence fix: the caller now passes ``data.df_train``), so the
    pixel identified here is identical to the one used as the primary
    Scenario 2 endpoint; this function only ADDS a geometric diagnostic on
    top, it does not reselect or alter the endpoint.

    Compares the mean nearest-neighbour distance among the cumulative
    archetype subset's pixel coordinates against the same statistic for
    the reference population passed in (the training partition, per the
    item 10 fix), as a simple, existing-utility-only proxy for whether
    the "n observed pixels" reported for the archetype subset overstates
    genuinely independent spatial information (relevant given the
    dataset's residuals are already known to be spatially autocorrelated
    per the existing Moran's I diagnostic, which this function does not
    alter in any way).
    """
    ndvi_end, ndbi_end, n, pixel_idx, dist_to_target = _empirical_cumulative_bin_stats(
        df, cfg.s2_archetype_ndbi_percentile
    )
    coord_cols = _find_spatial_columns(df, cfg)
    if coord_cols is None:
        mean_nn_subset    = float("nan")
        mean_nn_reference = float("nan")
        clustering_ratio  = float("nan")
        effective_sample_comment = (
            "No spatial coordinate columns found (looked for "
            f"{cfg.spatial_x_candidates} / {cfg.spatial_y_candidates}); "
            "spatial clustering of the archetype subset could not be assessed."
        )
    else:
        x_col, y_col = coord_cols
        ndbi_threshold = df["NDBI"].quantile(cfg.s2_archetype_ndbi_percentile)
        subset_mask    = df["NDBI"] <= ndbi_threshold
        subset_coords  = df.loc[subset_mask, [x_col, y_col]].to_numpy(dtype=float)
        full_coords    = df[[x_col, y_col]].to_numpy(dtype=float)
        mean_nn_subset    = _mean_nearest_neighbor_distance(subset_coords)
        mean_nn_reference = _mean_nearest_neighbor_distance(full_coords)
        if mean_nn_reference and mean_nn_reference > EPS and not np.isnan(mean_nn_reference):
            clustering_ratio = mean_nn_subset / mean_nn_reference
        else:
            clustering_ratio = float("nan")
        if np.isnan(clustering_ratio):
            effective_sample_comment = (
                "Clustering ratio undefined (degenerate reference pixel spacing)."
            )
        elif clustering_ratio < 0.5:
            effective_sample_comment = (
                "Archetype subset pixels are markedly more spatially clustered "
                f"than the citywide average (mean-NN-distance ratio={clustering_ratio:.2f}); "
                f"the reported n={n} pixel count likely overstates genuinely "
                "independent spatial information."
            )
        elif clustering_ratio < 0.85:
            effective_sample_comment = (
                "Archetype subset pixels are somewhat more spatially clustered "
                f"than the citywide average (ratio={clustering_ratio:.2f}); some "
                "overstatement of independent spatial information is likely."
            )
        else:
            effective_sample_comment = (
                "Archetype subset pixels are not markedly more spatially "
                f"clustered than the citywide average (ratio={clustering_ratio:.2f})."
            )

    diagnostics = {
        "ndbi_percentile":                      cfg.s2_archetype_ndbi_percentile,
        "subset_size":                          n,
        "archetype_pixel_index":                pixel_idx,
        "distance_to_target":                   dist_to_target,
        "mean_nn_distance_subset":               mean_nn_subset,
        "mean_nn_distance_reference_citywide":   mean_nn_reference,
        "clustering_ratio":                      clustering_ratio,
        "effective_sample_comment":              effective_sample_comment,
    }
    summary.log(
        "Scenario 2 archetype diagnostics", "Spatial clustering", effective_sample_comment,
    )
    return diagnostics


def export_scenario2_archetype_diagnostics(diagnostics: dict, cfg: Config) -> Path:
    """Export the Scenario 2 archetype spatial-clustering diagnostic to
    CSV. Reporting/export only."""
    table = pd.DataFrame([diagnostics])
    path = save_csv(table, cfg.data_dir / "scenario2_archetype_diagnostics.csv", cfg=cfg, debug_only=True)
    log_info(f"\nScenario 2 archetype spatial diagnostics exported to {path}.")
    return path


def tree_artifact_diagnostics(
    pred_arr: np.ndarray, label: str, cfg: Config, summary: SummaryLog
) -> tuple[float, float, int]:
    """Flag abrupt jumps in the tree-based prediction surface that may
    indicate split-boundary artifacts."""
    delta         = np.diff(pred_arr)
    mean_abs_jump = np.mean(np.abs(delta))
    max_jump      = np.max(np.abs(delta))
    jump_count    = int(np.sum(np.abs(delta) > cfg.tree_jump_threshold_c))
    summary.log("Supplementary Diagnostics - Staircase",
                f"{label} mean jump (°C)",  f"{mean_abs_jump:.5f}")
    summary.log("Supplementary Diagnostics - Staircase",
                f"{label} maximum jump (°C)", f"{max_jump:.5f}")
    summary.log("Supplementary Diagnostics - Staircase",
                f"{label} jump count (> {cfg.tree_jump_threshold_c}°C)", str(jump_count))
    return mean_abs_jump, max_jump, jump_count


def local_gradient_diagnostics(
    pred_arr: np.ndarray, scenario_pct: np.ndarray, label: str, summary: SummaryLog
) -> np.ndarray:
    """Compute local finite-difference gradients of predicted LST along
    the scenario trajectory."""
    derivative = safe_divide(np.diff(pred_arr), np.diff(scenario_pct), fill=0.0)
    summary.log("Supplementary Diagnostics - Local Gradient",
                f"{label} median local derivative (°C/scenario%)",
                f"{np.median(derivative):.5f}")
    summary.log("Supplementary Diagnostics - Local Gradient",
                f"{label} max local derivative (°C/scenario%)",
                f"{np.max(derivative):.5f}")
    summary.log("Supplementary Diagnostics - Local Gradient",
                f"{label} min local derivative (°C/scenario%)",
                f"{np.min(derivative):.5f}")
    return derivative


def compute_display_smoothing(
    s1: ScenarioTrajectory, s2: ScenarioTrajectory, cfg: Config
) -> dict:
    """Apply Savitzky-Golay smoothing to a trajectory for display only,
    without altering the underlying predictions."""
    scenario_pct = s1.scenario_pct
    step_pct     = scenario_pct[1] - scenario_pct[0]
    window = int(round(cfg.savgol_smooth_target_window_pct / step_pct))
    if window % 2 == 0:
        window += 1
    window = max(window, 5)
    max_window = (
        len(scenario_pct) if len(scenario_pct) % 2 == 1 else len(scenario_pct) - 1
    )
    window = min(window, max_window)

    lst_s1_smooth  = savgol_filter(s1.lst, window_length=window, polyorder=2)
    lst_s2_smooth  = savgol_filter(s2.lst, window_length=window, polyorder=2)
    spline_s2      = UnivariateSpline(scenario_pct, s2.lst, s=len(scenario_pct) * 0.5)
    lst_s2_spline  = spline_s2(scenario_pct)

    return {
        "lst_s1_smooth":  lst_s1_smooth,
        "lst_s2_smooth":  lst_s2_smooth,
        "lst_s2_spline":  lst_s2_spline,
        "spline_s2_obj":  spline_s2,
        "window":         window,
    }


def _energy_cost(
    scenario_fraction: np.ndarray, w0: float, exponent: float, cfg: Config
) -> np.ndarray:
    return w0 * cfg.desal_energy_intensity * (scenario_fraction ** exponent)


def _energy_norm(
    scenario_fraction: np.ndarray, w0: float, exponent: float, cfg: Config
) -> tuple[np.ndarray, np.ndarray]:
    """Return (raw_energy_cost, normalised_energy_cost).

    The denominator is always the maximum of _energy_cost at
    cfg.reference_w0.  When w0 == cfg.reference_w0, the normalised cost
    reduces to f^exponent / max(f^exponent) - a deliberate
    nondimensionalisation so the index is unit-free and scale-free with
    respect to absolute resource-cost scale at the reference operating point.
    w0 is a resource-cost scaling coefficient (a relative scaling
    parameter), not a directly measured Jeddah water or energy quantity;
    it and desal_energy_intensity affect results only in the sensitivity
    analysis, where w0_val != cfg.reference_w0.
    """
    raw           = _energy_cost(scenario_fraction, w0, exponent, cfg)
    reference_max = _energy_cost(scenario_fraction, cfg.reference_w0, exponent, cfg).max()
    return raw, safe_divide(raw, reference_max)


def compute_negi_results(
    baseline: ScenarioBaseline,
    s1: ScenarioTrajectory,
    s2: ScenarioTrajectory,
    cfg: Config,
) -> tuple["NEGIResults", float]:
    """Compute the NEGI (Normalized Environmental Gain Index) for Scenario 1
    and Scenario 2 trajectories."""
    delta_t_s1 = s1.cooling(baseline.baseline_lst)
    delta_t_s2 = s2.cooling(baseline.baseline_lst)

    warming_fraction_s2  = float((delta_t_s2 < 0).mean())
    cooling_fraction_s2  = float((delta_t_s2 > 0).mean())
    max_cooling_s2       = float(delta_t_s2.max())
    max_warming_s2       = float((-delta_t_s2).max())
    cooling_points       = np.where(delta_t_s2 > 0)[0]
    first_cooling_pct    = (
        float(s2.scenario_pct[cooling_points[0]])
        if len(cooling_points) > 0 else float("nan")
    )

    reference_cooling_c = cfg.reference_cooling_scale_factor * max(
        np.abs(delta_t_s1).max(), np.abs(delta_t_s2).max()
    )
    if reference_cooling_c <= TEMP_ZERO_GUARD:
        raise RuntimeError(
            "Scenario predictions do not differ from baseline; NEGI undefined."
        )

    desal_energy_sqrt,   energy_norm_sqrt   = _energy_norm(
        s1.scenario_fraction, cfg.reference_w0, cfg.sqrt_exponent,   cfg
    )
    desal_energy_linear, energy_norm_linear = _energy_norm(
        s1.scenario_fraction, cfg.reference_w0, cfg.linear_exponent, cfg
    )

    cooling_benefit_s1 = np.maximum(delta_t_s1, 0.0)
    cooling_benefit_s2 = np.maximum(delta_t_s2, 0.0)
    reference_benefit_scale = max(cooling_benefit_s1.max(), cooling_benefit_s2.max(), EPS)

    benefit_s1_norm = safe_divide(cooling_benefit_s1, reference_benefit_scale)
    benefit_s2_norm = safe_divide(cooling_benefit_s2, reference_benefit_scale)

    negi_s1        = cfg.alpha_weight * benefit_s1_norm - cfg.beta_weight * energy_norm_sqrt
    negi_s2        = cfg.alpha_weight * benefit_s2_norm - cfg.beta_weight * energy_norm_sqrt
    negi_s1_linear = cfg.alpha_weight * benefit_s1_norm - cfg.beta_weight * energy_norm_linear
    negi_s2_linear = cfg.alpha_weight * benefit_s2_norm - cfg.beta_weight * energy_norm_linear

    for arr, name in [
        (negi_s1,        "NEGI_s1"),
        (negi_s2,        "NEGI_s2"),
        (negi_s1_linear, "NEGI_s1_linear"),
        (negi_s2_linear, "NEGI_s2_linear"),
    ]:
        assert_finite(arr, name)

    upper_bound = cfg.alpha_weight + 1e-8
    assert np.all(negi_s1 <= upper_bound), \
        "negi_s1 exceeds the theoretical upper bound alpha_weight."
    assert np.all(negi_s2 <= upper_bound), \
        "negi_s2 exceeds the theoretical upper bound alpha_weight."
    assert len(s1.scenario_fraction) == len(s2.scenario_fraction) \
        == len(negi_s1) == len(negi_s2), \
        "Scenario arrays (s1, s2, negi_s1, negi_s2) must all have equal length."

    return NEGIResults(
        scenarios=s1.scenario_fraction,
        baseline_lst=baseline.baseline_lst,
        delta_t_s1=delta_t_s1, delta_t_s2=delta_t_s2,
        reference_benefit_scale=reference_benefit_scale,
        desal_energy_sqrt=desal_energy_sqrt,
        desal_energy_linear=desal_energy_linear,
        energy_norm_sqrt=energy_norm_sqrt,
        energy_norm_linear=energy_norm_linear,
        negi_s1=negi_s1, negi_s2=negi_s2,
        negi_s1_linear=negi_s1_linear, negi_s2_linear=negi_s2_linear,
        warming_fraction_s2=warming_fraction_s2,
        cooling_fraction_s2=cooling_fraction_s2,
        max_cooling_s2=max_cooling_s2, max_warming_s2=max_warming_s2,
        first_cooling_pct=first_cooling_pct,
    ), reference_cooling_c


def report_negi_scenario_diagnostics(
    negi: NEGIResults, reference_cooling_c: float, cfg: Config, summary: SummaryLog
) -> None:
    """Log and record NEGI scenario summary statistics and boundary-maximum checks."""
    first_str = (
        f"{negi.first_cooling_pct:.2f}%"
        if np.isfinite(negi.first_cooling_pct) else "none (never cools)"
    )
    log_info("\nScenario 2 thermal diagnostics (unclipped):")
    log_info(f"  Fraction warming      : {negi.warming_fraction_s2 * 100:.2f}%")
    log_info(f"  Fraction cooling      : {negi.cooling_fraction_s2 * 100:.2f}%")
    log_info(f"  Maximum cooling       : {negi.max_cooling_s2:.4f} °C")
    log_info(f"  Maximum warming       : {negi.max_warming_s2:.4f} °C")
    log_info(f"  First cooling point   : {first_str}")
    log_info("  NEGI cooling transform: max(\u0394T, 0)")

    summary.log(
        "Scenario diagnostics", "Interpretive note (unclipped thermal response)",
        f"The coordinated vegetation-built-up (Scenario 2) trajectory exhibits a "
        f"nonlinear predicted thermal response. Predicted LST exceeded the "
        f"baseline across approximately {negi.warming_fraction_s2 * 100:.1f}% of "
        f"the evaluated trajectory, while positive cooling emerged only in the "
        f"upper portion of the intervention range (first cooling at "
        + (f"{negi.first_cooling_pct:.1f}%" if np.isfinite(negi.first_cooling_pct) else "no point")
        + "). This pattern reflects the response of the empirical surrogate along "
        "the specified joint NDVI-NDBI trajectory and should not be interpreted "
        "as evidence that vegetation causes warming. This unclipped thermal "
        "diagnostic (fraction warming/cooling, extrema, first cooling point) is "
        "retained separately from the clipped cooling-benefit transformation, "
        "max(\u0394T, 0), used by the NEGI formulation itself.",
    )
    summary.log(
        "Scenario diagnostics", "Interpretive note (Scenario 1 sign convention)",
        interpretation_text("scenario1_sign_convention"),
    )

    summary.log("Scenario diagnostics", "S2 fraction warming (unclipped)",
                f"{negi.warming_fraction_s2 * 100:.2f}%")
    summary.log("Scenario diagnostics", "S2 fraction cooling (unclipped)",
                f"{negi.cooling_fraction_s2 * 100:.2f}%")
    summary.log("Scenario diagnostics", "S2 maximum cooling (unclipped, °C)",
                f"{negi.max_cooling_s2:.4f}")
    summary.log("Scenario diagnostics", "S2 maximum warming (unclipped, °C)",
                f"{negi.max_warming_s2:.4f}")
    summary.log(
        "Scenario diagnostics", "S2 first cooling point (scenario %)",
        f"{negi.first_cooling_pct:.2f}" if np.isfinite(negi.first_cooling_pct) else "none",
    )

    vlog_info(cfg, f"\nREFERENCE_COOLING_C = {reference_cooling_c:.4f} \u00b0C")
    vlog_info(cfg, "  Use: generalized/sensitivity formulation")
    vlog_info(cfg, f"\nREFERENCE_BENEFIT_SCALE = {negi.reference_benefit_scale:.4f} \u00b0C")
    vlog_info(cfg, "  Use: primary within-study NEGI cooling normalization, "
                   "held fixed across every spatial refit (see item 9)")

    summary.log(
        "Scenario diagnostics", "REFERENCE_COOLING_C derivation",
        f"{reference_cooling_c:.4f} \u00b0C "
        f"({cfg.reference_cooling_scale_factor}x observed max |\u0394T| = "
        f"{max(np.abs(negi.delta_t_s1).max(), np.abs(negi.delta_t_s2).max()):.4f} \u00b0C); "
        "used in the generalized/sensitivity formulation "
        "(run_sensitivity_analysis) only.",
    )
    summary.log(
        "Scenario diagnostics", "REFERENCE_BENEFIT_SCALE derivation",
        f"{negi.reference_benefit_scale:.4f} \u00b0C (maximum positive predicted "
        "cooling across the evaluated scenarios of the primary, full-training "
        "fit); normalizes the within-study cooling-benefit component of the "
        "primary NEGI formulation only. This value is held FIXED across every "
        "spatial-block refit used for uncertainty estimation, so NEGI values "
        "reported across refits remain on a common scale; the reported "
        "uncertainty interval is therefore conditional on this fixed "
        "reference scale, not a per-refit-renormalized quantity. "
        "REFERENCE_COOLING_C and REFERENCE_BENEFIT_SCALE are not interchangeable.",
    )

    log_info(f"\n{interpretation_text('scenario1_terminology')}")
    log_info(f"\n{interpretation_text('non_causal_safeguard')}")
    summary.log("Scenario diagnostics", "Scenario 1 terminology / interpretation",
                interpretation_text("scenario1_terminology"))
    summary.log("Scenario diagnostics", "Non-causal interpretation safeguard",
                interpretation_text("non_causal_safeguard"))


def check_interior_optimum(
    negi_arr: np.ndarray, scenarios: np.ndarray, label: str, cfg: Config = CFG
) -> None:
    """Report where the numerical maximum of a NEGI profile lies.

    If the maximum is at the first or last evaluated scenario point, emit
    the prescribed boundary message so readers understand it is not a
    robust interior optimum.  If it is in the interior, note that it may
    still be sensitive to tree partitions.
    """
    idx      = int(np.argmax(negi_arr))
    n_at_max = int(np.sum(np.isclose(negi_arr, negi_arr[idx])))
    at_lower = idx == 0
    at_upper = idx == len(negi_arr) - 1

    if at_lower or at_upper:
        edge = "lower" if at_lower else "upper"
        msg = (
            f"  {label}: maximum observed at the {edge} edge of the evaluated "
            f"range ({scenarios[idx] * 100:.1f}%).  "
            "This maximum should not be interpreted as a robust interior optimum."
        )
    else:
        msg = (
            f"  {label}: maximum observed at scenario = "
            f"{scenarios[idx] * 100:.1f}%; may be sensitive to tree partitions."
        )
    log_debug(msg)
    if cfg.debug:
        log_debug(
            f"  {label}: distinct scenario values at the maximum = "
            f"{n_at_max} (tie count at argmax)."
        )


def check_trajectory_stability(
    negi_arr: np.ndarray, scenarios: np.ndarray, label: str,
    cfg: Config = CFG, summary: Optional["SummaryLog"] = None,
) -> None:
    """Interpretation-only diagnostic (item 6): compare the maximum evaluated
    NEGI against its immediate trajectory neighbours to flag local
    instability. Does not modify the NEGI curve, the argmax calculation, or
    any downstream numerical result - it only decides whether a warning is
    printed/logged.
    """
    idx = int(np.argmax(negi_arr))
    n   = len(negi_arr)
    steps = cfg.trajectory_stability_neighbor_steps
    neighbor_idxs = [
        i for i in range(idx - steps, idx + steps + 1)
        if 0 <= i < n and i != idx
    ]
    if not neighbor_idxs:
        return

    max_val = float(negi_arr[idx])
    neighbor_vals = negi_arr[neighbor_idxs]
    biggest_jump = float(np.max(np.abs(neighbor_vals - max_val)))
    sign_flip = bool(
        not np.isclose(max_val, 0.0)
        and np.any(np.sign(neighbor_vals) != np.sign(max_val))
    )
    unstable = sign_flip or (biggest_jump > cfg.trajectory_stability_jump_threshold)

    if unstable:
        msg = (
            f"  {label}: the maximum evaluated NEGI occurs within a locally "
            f"unstable portion of the trajectory (largest neighbouring change "
            f"= {biggest_jump:.4f} NEGI units within "
            f"\u00b1{steps} evaluated step(s)"
            + (", including a sign change" if sign_flip else "")
            + ") and may be sensitive to tree partitioning or small changes "
            "in intervention intensity."
        )
        log_debug(msg)
        if summary is not None:
            summary.add_warning(f"{label}: " + msg.strip())


def _boundary_maximum_diagnostics(
    negi_arr: np.ndarray, scenarios: np.ndarray, cfg: Config,
) -> dict:
    """Compact-reporting helper: recomputes (never redefines) the same
    argmax / boundary / neighbour-jump / sign-flip quantities already
    produced by check_interior_optimum and check_trajectory_stability, for
    console display purposes only.
    """
    idx = int(np.argmax(negi_arr))
    n   = len(negi_arr)
    at_edge   = idx == 0 or idx == n - 1
    near_edge = is_near_trajectory_boundary(
        idx, n, cfg.scenario_boundary_tolerance_pct, positions=scenarios,
    )

    steps = cfg.trajectory_stability_neighbor_steps
    neighbor_idxs = [
        i for i in range(idx - steps, idx + steps + 1)
        if 0 <= i < n and i != idx
    ]
    if neighbor_idxs:
        max_val = float(negi_arr[idx])
        neighbor_vals = negi_arr[neighbor_idxs]
        largest_change = float(np.max(np.abs(neighbor_vals - max_val)))
        sign_flip = bool(
            not np.isclose(max_val, 0.0)
            and np.any(np.sign(neighbor_vals) != np.sign(max_val))
        )
    else:
        largest_change = None
        sign_flip = False

    return {
        "idx": idx, "pct": float(scenarios[idx] * 100.0),
        "at_edge": at_edge, "near_edge": near_edge,
        "largest_change": largest_change, "sign_flip": sign_flip,
    }


def report_boundary_maximum_assessment(negi: "NEGIResults", cfg: Config = CFG) -> None:
    """Compact console summary of the boundary-maximum assessment (STEP 9).
    Reads the same argmax / boundary / neighbour-jump / sign-flip values as
    check_interior_optimum and check_trajectory_stability - no calculation
    is changed here."""
    log_info("\nBoundary-maximum assessment:")
    for label, negi_arr in (("Scenario 1", negi.negi_s1), ("Scenario 2", negi.negi_s2)):
        d = _boundary_maximum_diagnostics(negi_arr, negi.scenarios, cfg)
        log_info(f"  {label}:")
        log_info(f"    {'Maximum evaluated NEGI':<22} : {negi_arr[d['idx']]:.4f}")
        log_info(f"    {'Scenario position':<22} : {d['pct']:.1f}%")
        if d["at_edge"]:
            log_info(f"    {'Boundary':<22} : YES")
        elif d["near_edge"]:
            log_info(f"    {'Near scenario-trajectory boundary':<22} : YES")
        else:
            log_info(f"    {'Boundary':<22} : NO")
        if d["largest_change"] is not None:
            log_info(f"    {'Largest local change':<22} : {d['largest_change']:.4f}")
        if d["sign_flip"]:
            log_info(f"    {'Local sign change':<22} : YES")


def _compute_nn1_distances(
    traj: ScenarioTrajectory,
    scaler: StandardScaler,
    nn_model_1: NearestNeighbors,
    label: str,
    cfg: Config,
    summary: SummaryLog,
) -> tuple[np.ndarray, float, np.ndarray]:
    points_std    = scaler.transform(traj.model_frame())
    nn_dist, _    = nn_model_1.kneighbors(points_std, n_neighbors=1)
    nn_dist       = nn_dist.ravel()
    p95           = np.percentile(nn_dist, 95)
    beyond_p95    = nn_dist > p95

    vlog_info(cfg, f"\n[{label}] Trajectory-level 1-NN diagnostic "
                   f"(distance to nearest observed point, standardised units; "
                   f"reporting-only, not the formal feature-support classification):")
    vlog_info(cfg, f"  Maximum NN distance : {nn_dist.max():.4f}")
    vlog_info(cfg, f"  Median NN distance  : {np.median(nn_dist):.4f}")
    vlog_info(cfg, f"  95th percentile     : {p95:.4f}")
    if beyond_p95.any():
        idx_flagged = np.where(beyond_p95)[0]
        pct_flagged = ", ".join(
            f"{traj.scenario_pct[i]:.1f}%" for i in idx_flagged[:10]
        )
        more = f" (+{len(idx_flagged) - 10} more)" if len(idx_flagged) > 10 else ""
        vlog_info(cfg,
                  f"  Points beyond 95th percentile ({len(idx_flagged)}): "
                  f"scenario = {pct_flagged}{more}")
    else:
        vlog_info(cfg, "  No points beyond the 95th percentile.")

    summary.log("Support diagnostics", f"{label} maximum NN distance",     f"{nn_dist.max():.4f}")
    summary.log("Support diagnostics", f"{label} median NN distance",      f"{np.median(nn_dist):.4f}")
    summary.log("Support diagnostics", f"{label} 95th percentile NN dist", f"{p95:.4f}")
    summary.log("Support diagnostics", f"{label} points beyond 95th pct",  str(int(beyond_p95.sum())))
    return nn_dist, p95, beyond_p95


def _mean_knn_distance(
    traj: ScenarioTrajectory,
    scaler: StandardScaler,
    nn_model_k: NearestNeighbors,
    cfg: Config,
) -> np.ndarray:
    points_std = scaler.transform(traj.model_frame())
    dist, _    = nn_model_k.kneighbors(points_std, n_neighbors=cfg.support_knn_k)
    return dist.mean(axis=1)


def _compute_extrapolation_report(
    traj: ScenarioTrajectory,
    mean_knn_dist: np.ndarray,
    label: str,
    data: DatasetBundle,
    support_threshold_k: float,
    cfg: Config,
    summary: SummaryLog,
    sparsity_threshold_k: Optional[float] = None,
    sparsity_percentile: Optional[float] = None,
) -> bool:
    frame             = traj.feature_frame()
    bounds_ok = {
        feat: bool(frame[feat].between(data.feature_min[feat],
                                        data.feature_max[feat]).all())
        for feat in FEATURES
    }
    ndvi_ok           = bounds_ok["NDVI"]
    ndbi_ok           = bounds_ok["NDBI"]
    elevation_ok      = bounds_ok["Elevation"]
    nn_ok             = (mean_knn_dist <= support_threshold_k).all()
    fully_supported   = (all(bounds_ok.values()) and nn_ok
                         and not traj.out_of_bounds.any())

    max_nn      = float(mean_knn_dist.max())
    median_nn   = float(np.median(mean_knn_dist))
    n_sparse    = 0
    frac_sparse = 0.0
    if fully_supported and sparsity_threshold_k is not None:
        in_support_mask = mean_knn_dist <= support_threshold_k
        sparse_mask      = sparse_region_mask(mean_knn_dist, in_support_mask, sparsity_threshold_k)
        n_sparse         = int(sparse_mask.sum()) if sparse_mask is not None else 0
        frac_sparse      = (n_sparse / len(mean_knn_dist)) if len(mean_knn_dist) else 0.0

    if fully_supported:
        if n_sparse > 0:
            vlog_info(
                cfg,
                f"[{label}] Empirical feature-support diagnostic: "
                f"PASS \u2014 within formal feature-space support, with locally "
                f"sparse regions ({n_sparse}/{len(mean_knn_dist)} points, "
                f"{frac_sparse * 100:.1f}%, pass the formal kNN threshold but "
                f"exceed the empirical {sparsity_percentile:.0f}th-percentile "
                f"reference NN-distance threshold of {sparsity_threshold_k:.4f}, "
                "derived from the observed/model-reference feature space)",
            )
        else:
            vlog_info(cfg, f"[{label}] Empirical feature-support diagnostic: PASS \u2014 within formal feature-space support.")
    else:
        reasons = []
        if not ndvi_ok:      reasons.append("NDVI outside observed range")
        if not ndbi_ok:      reasons.append("NDBI outside observed range")
        if not elevation_ok: reasons.append("elevation outside observed range")
        for _feat in ("ST_EMIS", "ST_EMSD"):
            if not bounds_ok[_feat]:
                reasons.append(f"{_feat} outside observed range")
        if not nn_ok:        reasons.append("nearest-neighbour distance exceeds support threshold")
        vlog_info(cfg, f"[{label}] Empirical feature-support diagnostic: FAIL ({'; '.join(reasons)})")

    vlog_info(cfg, f"  [{label}] Maximum scenario NN distance      : {max_nn:.4f}")
    vlog_info(cfg, f"  [{label}] Median scenario NN distance       : {median_nn:.4f}")
    if sparsity_threshold_k is not None:
        vlog_info(cfg, f"  [{label}] Empirical reference percentile    : {sparsity_percentile:.1f}")
        vlog_info(cfg, f"  [{label}] Empirical NN-distance threshold   : {sparsity_threshold_k:.4f}")
        vlog_info(cfg, f"  [{label}] Points exceeding empirical threshold: {n_sparse}/{len(mean_knn_dist)} ({frac_sparse * 100:.1f}%)")

    summary.log("Support diagnostics", f"{label} fully within formal support",
                str(fully_supported))
    summary.log("Support diagnostics", f"{label} maximum scenario NN distance", f"{max_nn:.4f}")
    summary.log("Support diagnostics", f"{label} median scenario NN distance",  f"{median_nn:.4f}")
    if sparsity_threshold_k is not None:
        summary.log("Support diagnostics", f"{label} empirical sparsity percentile", f"{sparsity_percentile:.1f}")
        summary.log("Support diagnostics", f"{label} empirical sparsity threshold",  f"{sparsity_threshold_k:.4f}")
        summary.log("Support diagnostics", f"{label} points exceeding empirical threshold", str(n_sparse))
        summary.log("Support diagnostics", f"{label} fraction exceeding empirical threshold", f"{frac_sparse:.4f}")
    return fully_supported


def compute_support_diagnostics(
    model_results: ModelResults,
    s1: ScenarioTrajectory, s2: ScenarioTrajectory,
    cfg: Config, summary: SummaryLog,
) -> SupportDiagnostics:
    """Compute nearest-neighbor feature-space support diagnostics for
    the scenario trajectories.

    Item 10 fix (holdout independence): the reference population used to
    decide whether a scenario point is "in support" - the standardization,
    the 1-NN/k-NN distances, and the empirical kNN-distance reference
    distribution that support_threshold_k/sparsity_threshold_k/
    density_moderate_threshold_k are percentiles of - is now built from
    data.X_train only. Previously this was fit on the full data.X
    (train + confirmatory holdout), even though every downstream
    docstring in this module already documented it as "training
    observations" - the confirmatory holdout must never be allowed to
    help decide whether a counterfactual it will later help evaluate is
    admissible.
    """
    data    = model_results.data
    scaler  = StandardScaler().fit(data.X_train)
    feat_mx = scaler.transform(data.X_train)

    nn_model_1 = NearestNeighbors(n_neighbors=1).fit(feat_mx)
    nn_dist_s1, nn_p95_s1, nn_beyond_p95_s1 = _compute_nn1_distances(
        s1, scaler, nn_model_1, "Scenario 1", cfg, summary
    )
    nn_dist_s2, nn_p95_s2, nn_beyond_p95_s2 = _compute_nn1_distances(
        s2, scaler, nn_model_1, "Scenario 2", cfg, summary
    )

    nn_model_k   = NearestNeighbors(n_neighbors=cfg.support_knn_k).fit(feat_mx)
    obs_dist_k, obs_idx_k = nn_model_k.kneighbors(feat_mx, n_neighbors=cfg.support_knn_k + 1)
    n_obs = len(feat_mx)
    self_mask = obs_idx_k == np.arange(n_obs)[:, None]
    self_hits_per_row = self_mask.sum(axis=1)
    empirical_knn_self_excluded = bool(np.all(self_hits_per_row == 1))

    if not empirical_knn_self_excluded:
        n_missing_self = int(np.sum(self_hits_per_row == 0))
        n_multi_self    = int(np.sum(self_hits_per_row > 1))
        vlog_info(
            cfg,
            "[Support] WARNING: self-neighbour exclusion could not assume "
            f"column 0 = self for every row ({n_missing_self} row(s) had no "
            f"self-match within the top-{cfg.support_knn_k + 1}, "
            f"{n_multi_self} row(s) had duplicate self-matches). This "
            "usually indicates duplicate/near-duplicate feature rows. "
            "Falling back to explicit per-row self-index removal; rows "
            "with no self-match keep only the k nearest returned "
            "neighbours rather than silently dropping an arbitrary column.",
        )

    obs_dist_k_clean = np.empty((n_obs, cfg.support_knn_k), dtype=obs_dist_k.dtype)
    for i in range(n_obs):
        row_self_mask = self_mask[i]
        if row_self_mask.any():
            keep = np.ones(obs_dist_k.shape[1], dtype=bool)
            keep[np.argmax(row_self_mask)] = False
            obs_dist_k_clean[i] = obs_dist_k[i, keep][:cfg.support_knn_k]
        else:
            obs_dist_k_clean[i] = obs_dist_k[i, :cfg.support_knn_k]
    obs_dist_k    = obs_dist_k_clean
    obs_mean_knn_dist   = obs_dist_k.mean(axis=1)
    support_threshold_k = np.percentile(obs_mean_knn_dist, cfg.support_percentile)

    sparsity_percentile  = cfg.FEATURE_SPARSITY_PERCENTILE
    sparsity_threshold_k = float(np.percentile(obs_mean_knn_dist, sparsity_percentile))

    density_moderate_percentile  = cfg.DENSITY_MODERATE_PERCENTILE
    density_moderate_threshold_k = float(np.percentile(obs_mean_knn_dist, density_moderate_percentile))
    if not (density_moderate_threshold_k < sparsity_threshold_k):
        vlog_info(
            cfg,
            "[Support] WARNING: DENSITY_MODERATE_PERCENTILE "
            f"({density_moderate_percentile}) does not yield a threshold "
            f"strictly below FEATURE_SPARSITY_PERCENTILE's threshold "
            f"({density_moderate_threshold_k:.4f} vs {sparsity_threshold_k:.4f}); "
            "the MODERATE local-density band may be empty for this dataset.",
        )

    mean_knn_dist_s1 = _mean_knn_distance(s1, scaler, nn_model_k, cfg)
    mean_knn_dist_s2 = _mean_knn_distance(s2, scaler, nn_model_k, cfg)
    extrap_s1_ok = _compute_extrapolation_report(
        s1, mean_knn_dist_s1, "Scenario 1", data, support_threshold_k, cfg, summary,
        sparsity_threshold_k=sparsity_threshold_k, sparsity_percentile=sparsity_percentile,
    )
    extrap_s2_ok = _compute_extrapolation_report(
        s2, mean_knn_dist_s2, "Scenario 2", data, support_threshold_k, cfg, summary,
        sparsity_threshold_k=sparsity_threshold_k, sparsity_percentile=sparsity_percentile,
    )
    in_support_s1 = mean_knn_dist_s1 <= support_threshold_k
    in_support_s2 = mean_knn_dist_s2 <= support_threshold_k

    return SupportDiagnostics(
        scaler=scaler,
        nn_dist_s1=nn_dist_s1, nn_p95_s1=nn_p95_s1, nn_beyond_p95_s1=nn_beyond_p95_s1,
        nn_dist_s2=nn_dist_s2, nn_p95_s2=nn_p95_s2, nn_beyond_p95_s2=nn_beyond_p95_s2,
        support_threshold_k=support_threshold_k,
        mean_knn_dist_s1=mean_knn_dist_s1, mean_knn_dist_s2=mean_knn_dist_s2,
        in_support_s1=in_support_s1, in_support_s2=in_support_s2,
        extrap_s1_ok=extrap_s1_ok, extrap_s2_ok=extrap_s2_ok,
        sparsity_percentile=sparsity_percentile, sparsity_threshold_k=sparsity_threshold_k,
        density_moderate_percentile=density_moderate_percentile,
        density_moderate_threshold_k=density_moderate_threshold_k,
        empirical_knn_self_excluded=empirical_knn_self_excluded,
        reference_mean_knn_dist=obs_mean_knn_dist,
    )


def compute_joint_multivariate_support(
    support: SupportDiagnostics,
    s1: ScenarioTrajectory, s2: ScenarioTrajectory,
    cfg: Config, summary: SummaryLog,
) -> dict:
    """Joint multivariate feature-space support diagnostic (Task 1).

    This is a SEPARATE report on top of the existing marginal feature-space
    support check (`compute_support_diagnostics` / "Feature support = PASS
    (marginal feature-space support)") - it does not replace or alter that
    diagnostic in any way.

    Reuses the EXISTING kNN infrastructure rather than building a new
    model: `support.mean_knn_dist_s1` / `support.mean_knn_dist_s2` are
    already each scenario point's mean distance to its `cfg.support_knn_k`
    nearest TRAINING observations in standardized joint (NDVI, NDBI,
    Elevation, ST_EMIS, ST_EMSD) space - i.e. they are already a genuinely JOINT,
    multivariate distance, not a per-feature marginal one.
    `support.reference_mean_knn_dist` is the corresponding distribution of
    that same statistic computed for every training observation against
    the rest of the training set - the empirical reference this function
    compares each scenario point against.

    For each scenario point, this computes what percentile of the
    TRAINING reference distribution its own joint kNN distance falls at
    (via the empirical CDF of `reference_mean_knn_dist`). Direction, made
    explicit: a HIGHER percentile means the point's neighbours in the
    training data are, on average, farther away than most training points'
    own neighbours are - i.e. LARGER joint kNN distance = SPARSER joint
    support. A point at or below `cfg.joint_support_percentile` (default:
    the existing `cfg.support_percentile`, 95th) is classified as jointly
    supported.

    Being inside each feature's separate marginal [min, max] range does
    NOT by itself establish this - a point can be within every marginal
    range individually while still being far from any real training
    observation in the joint space (the classic "corner of the box"
    failure mode of marginal-only checks). This diagnostic is the joint
    check that catches that case; it does not alter the scenario
    trajectory in any way and is for interpretation/robustness reporting
    only.
    """
    threshold_pct = float(cfg.joint_support_percentile)
    reference = support.reference_mean_knn_dist

    def _percentiles_of(distances: np.ndarray) -> np.ndarray:
        sorted_ref = np.sort(reference)
        ranks = np.searchsorted(sorted_ref, distances, side="right")
        return 100.0 * ranks / len(sorted_ref)

    results = {}
    for label, traj, mean_knn in (
        ("Scenario 1", s1, support.mean_knn_dist_s1),
        ("Scenario 2", s2, support.mean_knn_dist_s2),
    ):
        pct_of_ref   = _percentiles_of(mean_knn)
        beyond       = pct_of_ref > threshold_pct
        n_total      = len(mean_knn)
        n_beyond     = int(beyond.sum())
        frac_within  = 1.0 - (n_beyond / n_total if n_total else 0.0)
        worst_idx    = int(np.argmax(pct_of_ref))
        endpoint_pct = float(pct_of_ref[-1])
        worst_pct    = float(pct_of_ref[worst_idx])

        if frac_within >= 1.0:
            joint_status = "PASS"
        elif frac_within >= 0.90:
            joint_status = "PARTIAL"
        else:
            joint_status = "FAIL"

        vlog_info(
            cfg,
            f"\n[{label}] Joint multivariate ({', '.join(FEATURES)}) support "
            f"diagnostic (separate from, and in addition to, the marginal "
            f"feature-space support check above):"
        )
        vlog_info(cfg, f"  Joint multivariate support        = {joint_status}")
        vlog_info(cfg, f"  Points within {threshold_pct:.0f}th-percentile joint-support threshold: "
                        f"{n_total - n_beyond}/{n_total} ({frac_within * 100:.1f}%)")
        vlog_info(cfg, f"  Worst trajectory point percentile  = {worst_pct:.1f} "
                        f"(scenario={traj.scenario_pct[worst_idx]:.1f}%; higher = sparser training-data "
                        f"neighbourhood, i.e. less joint support)")
        vlog_info(cfg, f"  Endpoint percentile                = {endpoint_pct:.1f}")

        summary.log("Joint multivariate support", f"{label} joint support status", joint_status)
        summary.log("Joint multivariate support", f"{label} pct points within threshold", f"{frac_within * 100:.1f}")
        summary.log("Joint multivariate support", f"{label} worst trajectory percentile", f"{worst_pct:.1f}")
        summary.log("Joint multivariate support", f"{label} endpoint percentile", f"{endpoint_pct:.1f}")

        conclusion = (
            f"{label}: joint multivariate ({', '.join(FEATURES)}) support = "
            f"{joint_status} ({n_total - n_beyond}/{n_total} points, "
            f"{frac_within * 100:.1f}%, within the {threshold_pct:.0f}th-percentile "
            f"threshold; worst trajectory point at percentile {worst_pct:.1f}, "
            f"endpoint at {endpoint_pct:.1f})."
        )

        results[label] = {
            "joint_support_status": joint_status,
            "pct_points_within_threshold": frac_within * 100.0,
            "n_points_beyond_threshold": n_beyond,
            "n_points_total": n_total,
            "worst_percentile": worst_pct,
            "worst_scenario_pct": float(traj.scenario_pct[worst_idx]),
            "endpoint_percentile": endpoint_pct,
            "threshold_percentile_used": threshold_pct,
            "conclusion": conclusion,
        }

    vlog_info(
        cfg,
        "\nNote: joint multivariate support is an interpretation/robustness "
        "diagnostic only. It does not alter, filter, or clip either "
        "scenario trajectory, and a single isolated sparse point does not "
        "by itself invalidate the trajectory - see the reported fraction "
        "and location of any sparse points above.",
    )

    log_headline(
        "Joint multivariate support: "
        f"Scenario 1 = {results['Scenario 1']['joint_support_status']}, "
        f"Scenario 2 = {results['Scenario 2']['joint_support_status']} "
        "(full detail in robustness_summary.csv)."
    )
    return results


def report_joint_multivariate_support(
    joint_support: dict, cfg: Config,
) -> pd.DataFrame:
    """Assemble the joint multivariate support results (Task 1) into a
    long-format table suitable for inclusion in the consolidated
    robustness_summary.csv (Task 3) - reporting/export only, computes
    nothing new."""
    rows = []
    for label, d in joint_support.items():
        rows.append({
            "diagnostic": "joint_multivariate_support",
            "scenario": label,
            "joint_support_status": d["joint_support_status"],
            "pct_points_within_threshold": d["pct_points_within_threshold"],
            "worst_percentile": d["worst_percentile"],
            "worst_scenario_pct": d["worst_scenario_pct"],
            "endpoint_percentile": d["endpoint_percentile"],
            "threshold_percentile_used": d["threshold_percentile_used"],
            "conclusion": d["conclusion"],
        })
    return pd.DataFrame(rows)


def report_data_support_check(
    s2: ScenarioTrajectory, negi: NEGIResults,
    support: SupportDiagnostics, cfg: Config, summary: SummaryLog,
) -> None:
    """Log and record the feature-space support (extrapolation) check results."""
    in_support = support.in_support_s2
    s2_ood     = s2.out_of_bounds
    knn_dist   = support.mean_knn_dist_s2

    sparse_mask   = sparse_region_mask(knn_dist, in_support, support.sparsity_threshold_k)
    n_sparse      = int(sparse_mask.sum()) if sparse_mask is not None else 0
    frac_sparse   = (n_sparse / len(sparse_mask)) if sparse_mask is not None and len(sparse_mask) else 0.0

    log_info("\nData support diagnostics:")
    log_info(f"  Feature bounds check : {'PASS' if not s2_ood.any() else 'FAIL'}")
    if in_support.all():
        knn_status = (
            f"PASS \u2014 within formal feature-space support, with locally "
            f"sparse regions ({n_sparse}/{len(in_support)} points, "
            f"{frac_sparse * 100:.1f}%, pass the formal kNN support threshold "
            f"but exceed the empirical {support.sparsity_percentile:.0f}th-"
            f"percentile reference NN-distance threshold of "
            f"{support.sparsity_threshold_k:.4f})"
            if n_sparse > 0 else "PASS \u2014 within formal feature-space support."
        )
    else:
        knn_status = f"FAIL - {(~in_support).sum()} points outside"
    log_info(f"  kNN support check    : {knn_status}")
    log_info(f"  Maximum scenario NN distance       : {float(knn_dist.max()):.4f}")
    log_info(f"  Median scenario NN distance        : {float(np.median(knn_dist)):.4f}")
    log_info(f"  Empirical reference percentile     : {support.sparsity_percentile:.1f}")
    log_info(f"  Empirical NN-distance threshold    : {support.sparsity_threshold_k:.4f}")
    log_info(f"  Points exceeding empirical threshold: {n_sparse}/{len(knn_dist)} ({frac_sparse * 100:.1f}%)")

    if (~in_support).any():
        first_out_idx = np.argmax(~in_support)
        log_info(f"  Dense region exit at : scenario = {s2.scenario_pct[first_out_idx]:.1f}%")

    max_idx = negi.s2_optimum_idx
    log_info(
        f"  Support at maximum   : bounds="
        f"{'PASS' if not s2_ood[max_idx] else 'FAIL'}, "
        f"kNN_dist={knn_dist[max_idx]:.4f}, "
        f"kNN_support={'PASS' if in_support[max_idx] else 'FAIL'}, "
        f"sparse={'YES' if (sparse_mask is not None and sparse_mask[max_idx]) else 'NO'}"
    )

    trustworthy         = in_support & (~s2_ood)
    negi_s2_trustworthy = np.where(trustworthy, negi.negi_s2, np.nan)
    if np.all(np.isnan(negi_s2_trustworthy)):
        log_info("  No Scenario 2 points pass both support checks.")
    else:
        best_idx = int(np.nanargmax(negi_s2_trustworthy))
        log_info(
            f"  Numerical maximum (both checks): "
            f"scenario = {s2.scenario_pct[best_idx]:.1f}% "
            f"(NEGI = {negi.negi_s2[best_idx]:.3f}); "
            f"pre-support-filter maximum position: {s2.scenario_pct[negi.s2_optimum_idx]:.1f}%"
        )
        log_info(f"  Points passing both checks: {trustworthy.sum()}/{len(trustworthy)}")

    summary.log("Support diagnostics", "Feature bounds check",
                "PASS" if not s2_ood.any() else "FAIL")
    summary.log("Support diagnostics", "kNN support check", knn_status)
    summary.log("Support diagnostics", "Scenario 2 maximum NN distance (trajectory)", f"{float(knn_dist.max()):.4f}")
    summary.log("Support diagnostics", "Scenario 2 median NN distance (trajectory)",  f"{float(np.median(knn_dist)):.4f}")
    summary.log("Support diagnostics", "Empirical sparsity reference percentile",     f"{support.sparsity_percentile:.1f}")
    summary.log("Support diagnostics", "Empirical sparsity NN-distance threshold",    f"{support.sparsity_threshold_k:.4f}")
    summary.log("Support diagnostics", "Scenario 2 points exceeding empirical threshold", str(n_sparse))
    summary.log("Support diagnostics", "Scenario 2 fraction exceeding empirical threshold", f"{frac_sparse:.4f}")
    summary.log(
        "Support diagnostics", "Support at maximum evaluated NEGI",
        f"bounds={'PASS' if not s2_ood[max_idx] else 'FAIL'}, "
        f"kNN_dist={knn_dist[max_idx]:.4f}, "
        f"kNN_support={'PASS' if in_support[max_idx] else 'FAIL'}, "
        f"sparse={'YES' if (sparse_mask is not None and sparse_mask[max_idx]) else 'NO'}",
    )
    summary.log(
        "Support diagnostics", "NEGI numerical maximum in locally sparse region",
        "YES" if (sparse_mask is not None and sparse_mask[max_idx]) else "NO",
    )


def compute_uncertainty_bands(
    model_results: ModelResults,
    baseline: ScenarioBaseline,
    s1: ScenarioTrajectory, s2: ScenarioTrajectory,
    negi: NEGIResults, cfg: Config,
    log_convergence_detail: bool = True,
    max_refits: Optional[int] = None,
) -> UncertaintyResults:
    """Compute NEGI and LST uncertainty bands from repeated spatial-block
    holdout refits.

    `log_convergence_detail`: purely a console-verbosity
    switch. The adaptive convergence loop, its checkpoints, and the
    returned `converged_early` / `n_refits_used` / `convergence_checkpoints`
    fields are ALWAYS computed identically regardless of this flag - it
    only controls whether the full per-checkpoint "Adaptive convergence
    summary" text is emitted to the console/log via report_development()
    for THIS call. Callers that invoke this function many times in a loop
    (e.g. run_scenario2_percentile_sensitivity, once per percentile) should
    pass False and instead print one condensed summary across all calls
    once the loop finishes, using the checkpoint/convergence fields on each
    returned UncertaintyResults. The primary Scenario 1/2 call keeps the
    default (True) so the full detail is still available for that one,
    most-important run.

    `max_refits`: the refit-count ceiling this call uses, overriding
    `cfg.n_spatial_refits` when given (defaults to `cfg.n_spatial_refits`
    when left as None). This is the single computational-budget knob for
    every caller - a primary publication run, a cheaper repeated sweep
    (e.g. `cfg.n_spatial_refits_percentile_sensitivity`), a debug run, or
    a unit test can each pass whatever ceiling suits them through this one
    parameter, rather than each needing its own bespoke config field wired
    through a separate code path.

    Adaptive convergence (data-driven refit-count stopping rule): when
    `cfg.adaptive_uncertainty_convergence` is True (the default), refits
    are fit INCREMENTALLY in batches of
    `cfg.uncertainty_convergence_check_interval`, up to the
    `cfg.n_spatial_refits` ceiling. After every batch, the Scenario-2
    trajectory ENDPOINT's 95% CI width is recomputed from every refit
    collected so far and recorded as a checkpoint. Fitting stops early
    only once BOTH of the following hold (CRUX FIX - these two checks
    used to be independent and could disagree; see the inline comments
    immediately above this loop for why):
      1. the most recent `cfg.uncertainty_convergence_required_stable_checkpoints`
         checkpoints agree with each other to within
         `cfg.adaptive_convergence_tolerance` (relative); AND
      2. no single checkpoint-to-checkpoint change over the wider
         lookback window (twice as many checkpoints) exceeds
         `cfg.uncertainty_convergence_tolerance` - the same, stricter
         full-history criterion `compute_uncertainty_convergence_diagnostic`
         reports on afterward.
    If either check fails, fitting continues up to the `n_spatial_refits`
    ceiling. The actual number of refits used for publication is
    therefore a property of the data, not a hard-coded constant, AND is
    now guaranteed consistent with the "Publication convergence" status
    reported later - that status can no longer read "NOT YET STABILIZED"
    for a band this loop already decided to stop collecting more data
    for.

    Splits are drawn once, up front, from a single seeded
    GroupShuffleSplit(random_state=cfg.uncertainty_refit_seed), so an
    early-stopped run's refits are always an exact PREFIX of what a
    longer/non-adaptive run would have fit - never a different resample.
    Setting `cfg.adaptive_uncertainty_convergence=False` restores the
    previous fixed-count-only behaviour exactly.
    """
    data      = model_results.data
    scenarios = s1.scenario_fraction

    max_refits = int(cfg.n_spatial_refits) if max_refits is None else int(max_refits)
    splits = list(GroupShuffleSplit(
        n_splits=max_refits,
        test_size=cfg.holdout_test_size,
        random_state=cfg.uncertainty_refit_seed,
    ).split(data.X, data.y, groups=data.groups))

    all_groups = data.groups.to_numpy()

    model_stochasticity_varied = False

    _fixed_ref = baseline.fixed_feature_values()
    feat_s1 = build_feature_frame(s1.ndvi, s1.ndbi, _fixed_ref)
    feat_s2 = build_feature_frame(s2.ndvi, s2.ndbi, _fixed_ref)

    df_full = data.df
    ndbi_range = np.linspace(
        df_full["NDBI"].quantile(cfg.ndbi_sweep_quantile_low),
        df_full["NDBI"].quantile(cfg.ndbi_sweep_quantile_high),
        cfg.ndbi_sweep_points,
    )
    ndbi_sweep_frame = build_feature_frame(
        np.full(cfg.ndbi_sweep_points, df_full["NDVI"].median()),
        ndbi_range,
        fixed_reference_values(df_full),
    )

    best_params = model_results.best_params

    def _fit_one_spatial_refit(train_idx_fold):
        Xf, yf = data.X.iloc[train_idx_fold], data.y.iloc[train_idx_fold]
        mf = XGBRegressor(
            objective="reg:squarederror",
            random_state=cfg.random_seed,
            monotone_constraints=cfg.monotone_constraints,
            tree_method=cfg.xgb_tree_method,
            n_jobs=1, **best_params,
        )
        mf.fit(Xf, yf)

        baseline_df_f = build_feature_frame(
            [baseline.baseline_ndvi], [baseline.baseline_ndbi], _fixed_ref,
        )
        baseline_lst_f = float(mf.predict(baseline_df_f)[0])
        lst_s1_f = mf.predict(feat_s1)
        lst_s2_f = mf.predict(feat_s2)

        cooling_s1_f = baseline_lst_f - lst_s1_f
        cooling_s2_f = baseline_lst_f - lst_s2_f
        benefit_s1_f = safe_divide(np.maximum(cooling_s1_f, 0.0), negi.reference_benefit_scale)
        benefit_s2_f = safe_divide(np.maximum(cooling_s2_f, 0.0), negi.reference_benefit_scale)
        negi_s1_f    = (cfg.alpha_weight * benefit_s1_f
                        - cfg.beta_weight * negi.energy_norm_sqrt)
        negi_s2_f    = (cfg.alpha_weight * benefit_s2_f
                        - cfg.beta_weight * negi.energy_norm_sqrt)

        return (
            negi_s1_f, negi_s2_f, lst_s1_f, lst_s2_f,
            cooling_s1_f, cooling_s2_f, mf.predict(ndbi_sweep_frame),
        )

    def _run_refit_batch(batch_splits, n_jobs: int):
        return Parallel(n_jobs=n_jobs)(
            delayed(_fit_one_spatial_refit)(train_idx_fold)
            for train_idx_fold, _ in batch_splits
        )

    def _run_refit_batch_with_retry(batch_splits):
        """Run one Parallel batch of refits, retrying with fewer concurrent
        workers if loky reports a dead/killed worker (see
        `spatial_refit_n_jobs` docstring above for why this is safe: each
        refit is independent, so a retry with a smaller pool reproduces the
        exact same per-refit results, just with lower peak memory)."""
        n_jobs = cfg.spatial_refit_n_jobs
        attempt = 0
        while True:
            try:
                result = _run_refit_batch(batch_splits, n_jobs)
                gc.collect()
                return result
            except Exception as exc:
                attempt += 1
                gc.collect()
                if attempt > cfg.spatial_refit_max_batch_retries:
                    raise
                current = n_jobs if n_jobs and n_jobs > 0 else (os.cpu_count() or 4)
                reduced = max(1, current // 2)
                log_info(
                    f"    [spatial-refit] batch of {len(batch_splits)} refits hit "
                    f"{type(exc).__name__} ({exc}); retrying with n_jobs={reduced} "
                    f"(was {n_jobs}), attempt {attempt}/{cfg.spatial_refit_max_batch_retries}. "
                    f"Already-completed refits from earlier batches are unaffected."
                )
                n_jobs = reduced

    def _ci_stats_at(negi_s2_folds_arr: np.ndarray, idx: int, prefix: str) -> dict:
        vals   = negi_s2_folds_arr[:, idx]
        lower  = float(np.percentile(vals, 2.5))
        upper  = float(np.percentile(vals, 97.5))
        median = float(np.percentile(vals, 50.0))
        return {
            f"{prefix}_mean_negi":   float(vals.mean()),
            f"{prefix}_median_negi": median,
            f"{prefix}_ci_lower":    lower,
            f"{prefix}_ci_upper":    upper,
            f"{prefix}_ci_width":    upper - lower,
        }

    def _endpoint_ci_stats(negi_s2_folds_arr: np.ndarray) -> dict:
        return _ci_stats_at(negi_s2_folds_arr, -1, "endpoint")

    def _maximum_ci_stats(negi_s2_folds_arr: np.ndarray) -> dict:
        return _ci_stats_at(negi_s2_folds_arr, negi.s2_optimum_idx, "maximum")

    negi_s1_acc, negi_s2_acc, lst_s1_acc, lst_s2_acc = [], [], [], []
    cooling_s1_acc, cooling_s2_acc, ndbi_sweep_acc = [], [], []
    partition_hashes: list[str] = []
    convergence_checkpoints: list[dict] = []
    converged_early = False

    check_interval = max(1, int(cfg.uncertainty_convergence_check_interval))
    min_refits = max(1, int(cfg.uncertainty_convergence_min_refits))
    required_stable = max(1, int(cfg.uncertainty_convergence_required_stable_checkpoints))
    tol = float(cfg.adaptive_convergence_tolerance)

    lookback = max(required_stable, 2 * required_stable)
    full_history_tol = float(cfg.uncertainty_convergence_tolerance)

    _CONVERGENCE_QUANTITIES = ("endpoint_median_negi", "endpoint_ci_lower", "endpoint_ci_upper")
    _CONVERGENCE_QUANTITIES_MAXIMUM = ("maximum_median_negi", "maximum_ci_lower", "maximum_ci_upper")

    def _rel_range_stable(values: list, tolerance: float) -> bool:
        """True if `values` agree to within `tolerance` (relative range).

        Falls back to an absolute-range comparison, scaled by
        `cfg.trajectory_stability_jump_threshold` (reused here purely as
        an existing "NEGI units small enough to call negligible" floor -
        not a redefinition of that field's original meaning), whenever
        the mean of `values` is small relative to that floor. This is the
        same numerical concern recommendation 6 raised about coefficient
        of variation: `endpoint_ci_lower` in particular routinely
        straddles zero (e.g. [-0.118, 0.019] in the reviewer's own
        numbers), where a plain relative range is undefined/explosive
        rather than meaningful. `np.isclose(mean_v, 0.0)`'s default
        tolerance is far too tight to catch that case (it only fires for
        means within ~1e-8 of exactly zero), so the comparison is made
        against the jump-threshold floor instead.
        """
        if len(values) < 2:
            return True
        mean_v = float(np.mean(values))
        rng    = max(values) - min(values)
        floor  = float(cfg.trajectory_stability_jump_threshold)
        if abs(mean_v) < floor:
            return rng < (tolerance * floor)
        return (max(values) - min(values)) / abs(mean_v) < tolerance

    idx = 0
    is_first_batch = True
    while idx < max_refits:
        if not cfg.adaptive_uncertainty_convergence:
            step = max_refits
        elif is_first_batch:
            step = min_refits
        else:
            step = check_interval
        batch_end = min(idx + step, max_refits)
        batch_splits = splits[idx:batch_end]

        fold_results = _run_refit_batch_with_retry(batch_splits)
        for train_idx_fold, test_idx_fold in batch_splits:
            held_out_groups = frozenset(all_groups[test_idx_fold])
            h = hashlib.sha256(
                ",".join(sorted(str(g) for g in held_out_groups)).encode("utf-8")
            ).hexdigest()[:16]
            partition_hashes.append(h)
        for (n1, n2, l1, l2, c1, c2, nsweep) in fold_results:
            negi_s1_acc.append(n1); negi_s2_acc.append(n2)
            lst_s1_acc.append(l1);  lst_s2_acc.append(l2)
            cooling_s1_acc.append(c1); cooling_s2_acc.append(c2)
            ndbi_sweep_acc.append(nsweep)

        idx = batch_end
        is_first_batch = False
        n_so_far = len(negi_s2_acc)

        if cfg.adaptive_uncertainty_convergence:
            _acc_arr = np.array(negi_s2_acc)
            stats = _endpoint_ci_stats(_acc_arr)
            stats_max = _maximum_ci_stats(_acc_arr)
            prev = convergence_checkpoints[-1] if convergence_checkpoints else None

            prev_width = prev["endpoint_ci_width"] if prev is not None else None
            abs_change = None if prev_width is None else stats["endpoint_ci_width"] - prev_width
            pct_change = (
                None if prev_width is None or prev_width == 0
                else abs_change / prev_width
            )
            _floor = float(cfg.trajectory_stability_jump_threshold)
            pct_change_by_quantity = {
                q: (
                    None if prev is None
                    else (stats[q] - prev[q]) / max(abs(prev[q]), _floor)
                )
                for q in _CONVERGENCE_QUANTITIES
            }
            pct_change_by_quantity_maximum = {
                q: (
                    None if prev is None
                    else (stats_max[q] - prev[q]) / max(abs(prev[q]), _floor)
                )
                for q in _CONVERGENCE_QUANTITIES_MAXIMUM
            }
            convergence_checkpoints.append({
                "refit_count": n_so_far,
                **stats,
                **stats_max,
                "abs_change": abs_change,
                "pct_change": pct_change,
                "pct_change_by_quantity": pct_change_by_quantity,
                "pct_change_by_quantity_maximum": pct_change_by_quantity_maximum,
            })

            local_window_stable = False
            if len(convergence_checkpoints) >= required_stable:
                recent = convergence_checkpoints[-required_stable:]
                local_window_stable = all(
                    _rel_range_stable([c[q] for c in recent], tol)
                    for q in _CONVERGENCE_QUANTITIES
                )

            extended_window_stable = False
            if len(convergence_checkpoints) >= lookback:
                window = convergence_checkpoints[-lookback:]

                def _quantity_extended_stable(q):
                    changes = [
                        c["pct_change_by_quantity"][q] for c in window
                        if c["pct_change_by_quantity"][q] is not None
                    ]
                    return (
                        len(changes) == lookback - 1
                        and all(abs(pc) < full_history_tol for pc in changes)
                    )

                extended_window_stable = all(
                    _quantity_extended_stable(q) for q in _CONVERGENCE_QUANTITIES
                )

            if local_window_stable and extended_window_stable and idx < max_refits:
                converged_early = True
                break

    if cfg.adaptive_uncertainty_convergence and convergence_checkpoints:
        checkpoint_lines = "\n".join(
            f"  {c['refit_count']:>4d} refits | CI width = {c['endpoint_ci_width']:.5f} | "
            + (
                "(first checkpoint)" if c["abs_change"] is None
                else f"\u0394 = {c['abs_change']:+.5f} ({c['pct_change'] * 100:+.2f}%)"
            )
            for c in convergence_checkpoints
        )
        final_width = convergence_checkpoints[-1]["endpoint_ci_width"]
        n_checkpoints = len(convergence_checkpoints)
        if converged_early:
            result_line = (
                f"Result: STOPPED EARLY at {len(negi_s2_acc)}/{max_refits} refits - "
                f"endpoint median, lower-CI, and upper-CI each stable across "
                f"the last {required_stable} checkpoints (relative range < "
                f"{tol * 100:.2f}% tolerance) AND across the wider last "
                f"{lookback} checkpoints (no single change in any of the "
                f"three > {full_history_tol * 100:.2f}%, the same criterion "
                "Publication convergence checks)."
            )
            condensed = (
                "Adaptive uncertainty convergence:\n"
                f"  {n_checkpoints} checkpoint(s) evaluated\n"
                f"  Final CI width = {final_width:.5f}\n"
                f"  Convergence criterion reached at {len(negi_s2_acc)}/{max_refits} refits "
                f"(median/lower-CI/upper-CI each < {tol * 100:.2f}% relative range "
                f"across last {required_stable} checkpoints AND < "
                f"{full_history_tol * 100:.2f}% change across last {lookback} checkpoints)."
            )
            if log_convergence_detail:
                report_development(cfg, "\n" + condensed)
        else:
            result_line = (
                f"Result: MAX REFITS REACHED ({len(negi_s2_acc)}/{max_refits}) "
                f"without meeting BOTH the {tol * 100:.2f}% local stability "
                f"criterion (median/lower-CI/upper-CI) across {required_stable} "
                f"consecutive checkpoints AND the {full_history_tol * 100:.2f}% "
                f"extended stability criterion across the last {lookback} checkpoints."
            )
            condensed = (
                "Adaptive uncertainty convergence:\n"
                f"  {n_checkpoints} checkpoint(s) evaluated\n"
                f"  Final CI width = {final_width:.5f}\n"
                f"  Convergence criterion not reached before {max_refits} refits "
                f"(configured tolerance = {tol * 100:.2f}%)."
            )
            if log_convergence_detail:
                report_development(cfg, "\n" + condensed, level="warning")
        log_debug(f"\nAdaptive convergence detail:\n{checkpoint_lines}\n{result_line}")

    negi_s1_folds    = np.array(negi_s1_acc)
    negi_s2_folds    = np.array(negi_s2_acc)
    lst_s1_folds     = np.array(lst_s1_acc)
    lst_s2_folds     = np.array(lst_s2_acc)
    cooling_s1_folds = np.array(cooling_s1_acc)
    cooling_s2_folds = np.array(cooling_s2_acc)
    ndbi_sweep_folds = np.array(ndbi_sweep_acc)
    n_refits_used    = int(negi_s2_folds.shape[0])

    n_unique_partitions = len(set(partition_hashes))
    if n_unique_partitions < len(partition_hashes):
        log_warning(
            f"\nSpatial-refit uncertainty: only {n_unique_partitions}/"
            f"{len(partition_hashes)} refit partitions are unique "
            "(duplicate held-out group sets detected). The uncertainty "
            "band's effective sample size is smaller than n_refits_used."
        )
    else:
        log_info(
            f"\nSpatial-refit uncertainty: confirmed {n_unique_partitions}/"
            f"{len(partition_hashes)} refit partitions are unique."
        )

    negi_s1_mean = negi_s1_folds.mean(axis=0)
    negi_s1_std  = negi_s1_folds.std(axis=0)
    negi_s2_mean = negi_s2_folds.mean(axis=0)
    negi_s2_std  = negi_s2_folds.std(axis=0)

    def _pct_summary(fold_array: np.ndarray) -> dict:
        return {p: np.percentile(fold_array, p, axis=0)
                for p in cfg.uncertainty_percentiles}

    percentiles = {
        "negi_s1":    _pct_summary(negi_s1_folds),
        "negi_s2":    _pct_summary(negi_s2_folds),
        "lst_s1":     _pct_summary(lst_s1_folds),
        "lst_s2":     _pct_summary(lst_s2_folds),
        "cooling_s1": _pct_summary(cooling_s1_folds),
        "cooling_s2": _pct_summary(cooling_s2_folds),
    }
    p = percentiles
    table = pd.DataFrame({
        "Scenario (%)":      scenarios * 100,
        "NEGI_S1_median":    p["negi_s1"][50],
        "NEGI_S1_Q25":       p["negi_s1"][25],
        "NEGI_S1_Q75":       p["negi_s1"][75],
        "NEGI_S1_P2.5":      p["negi_s1"][2.5],
        "NEGI_S1_P97.5":     p["negi_s1"][97.5],
        "NEGI_S2_median":    p["negi_s2"][50],
        "NEGI_S2_Q25":       p["negi_s2"][25],
        "NEGI_S2_Q75":       p["negi_s2"][75],
        "NEGI_S2_P2.5":      p["negi_s2"][2.5],
        "NEGI_S2_P97.5":     p["negi_s2"][97.5],
        "NEGI_S1_mean":      negi_s1_mean,
        "NEGI_S1_std":       negi_s1_std,
        "NEGI_S2_mean":      negi_s2_mean,
        "NEGI_S2_std":       negi_s2_std,
        "LST_S1_median":     p["lst_s1"][50],
        "LST_S1_Q25":        p["lst_s1"][25],
        "LST_S1_Q75":        p["lst_s1"][75],
        "LST_S2_median":     p["lst_s2"][50],
        "LST_S2_Q25":        p["lst_s2"][25],
        "LST_S2_Q75":        p["lst_s2"][75],
        "Cooling_S1_median": p["cooling_s1"][50],
        "Cooling_S1_Q25":    p["cooling_s1"][25],
        "Cooling_S1_Q75":    p["cooling_s1"][75],
        "Cooling_S2_median": p["cooling_s2"][50],
        "Cooling_S2_Q25":    p["cooling_s2"][25],
        "Cooling_S2_Q75":    p["cooling_s2"][75],
    })

    return UncertaintyResults(
        negi_s1_folds=negi_s1_folds, negi_s2_folds=negi_s2_folds,
        lst_s1_folds=lst_s1_folds,   lst_s2_folds=lst_s2_folds,
        cooling_s1_folds=cooling_s1_folds, cooling_s2_folds=cooling_s2_folds,
        negi_s1_mean=negi_s1_mean, negi_s1_std=negi_s1_std,
        negi_s2_mean=negi_s2_mean, negi_s2_std=negi_s2_std,
        percentiles=percentiles, table=table,
        ndbi_range=ndbi_range, ndbi_sweep_folds=ndbi_sweep_folds,
        partition_hashes=partition_hashes, n_unique_partitions=n_unique_partitions,
        model_stochasticity_varied=model_stochasticity_varied,
        n_refits_used=n_refits_used, converged_early=converged_early,
        convergence_checkpoints=convergence_checkpoints,
    )


def run_spatial_block_size_sensitivity(
    model_results: ModelResults,
    baseline: ScenarioBaseline,
    s1: ScenarioTrajectory, s2: ScenarioTrajectory,
    negi: NEGIResults, cfg: Config, summary: SummaryLog,
) -> pd.DataFrame:
    """Spatial block-size sensitivity for the Scenario 2 endpoint's
    uncertainty conclusion (Task 2).

    Reuses the EXISTING spatial-refit uncertainty procedure
    (`compute_uncertainty_bands`) UNCHANGED - no new bootstrap method is
    introduced. Only the spatial grouping passed to it varies, via
    `regrid_spatial_blocks` (a genuine re-grid of the original lat/lon at
    `cfg.spatial_block_cell_size_deg * merge_factor` - see its docstring;
    this replaces the previous ordinal merge of `block_id` labels, which
    did not track geographic adjacency). The PRIMARY block size/grouping
    and `Config.n_spatial_refits` are never altered by this function;
    `merge_factor=1` always reproduces the primary result exactly, and
    this function does not select or report a new "best" block size - it
    is a robustness diagnostic only.
    """
    data  = model_results.data
    rows  = []
    for merge_factor in cfg.spatial_block_merge_factors:
        new_groups = regrid_spatial_blocks(data.df, cfg, float(merge_factor)).loc[data.groups.index]
        n_groups   = int(new_groups.nunique())

        if n_groups < cfg.spatial_block_min_groups:
            log_warning(
                f"\nSpatial block-size sensitivity: merge_factor={merge_factor} "
                f"yields only {n_groups} spatial groups (< "
                f"{cfg.spatial_block_min_groups}) - too few for meaningful "
                "spatial resampling; skipped."
            )
            rows.append({
                "merge_factor": int(merge_factor),
                "n_spatial_groups": n_groups,
                "endpoint_negi": np.nan,
                "endpoint_ci_lower": np.nan,
                "endpoint_ci_upper": np.nan,
                "endpoint_ci_width": np.nan,
                "includes_zero": np.nan,
                "robustness_classification": "SKIPPED",
                "n_refits_used": np.nan,
                "converged_early": np.nan,
                "skip_reason": f"only {n_groups} spatial groups (< {cfg.spatial_block_min_groups} required)",
            })
            continue

        data_variant  = dataclasses.replace(data, groups=new_groups)
        results_variant = dataclasses.replace(model_results, data=data_variant)

        unc = compute_uncertainty_bands(
            results_variant, baseline, s1, s2, negi, cfg,
            log_convergence_detail=False,
            max_refits=cfg.n_spatial_refits,
        )
        stats = _endpoint_uncertainty_stats(unc, -1)
        includes_zero = bool(stats["endpoint_ci_lower"] <= 0.0 <= stats["endpoint_ci_upper"])
        robustness = "NOT ROBUST (CI crosses zero)" if includes_zero else "ROBUST (CI excludes zero)"

        rows.append({
            "merge_factor": int(merge_factor),
            "n_spatial_groups": n_groups,
            "endpoint_negi": stats["endpoint_mean_negi"],
            "endpoint_ci_lower": stats["endpoint_ci_lower"],
            "endpoint_ci_upper": stats["endpoint_ci_upper"],
            "endpoint_ci_width": stats["endpoint_ci_width"],
            "includes_zero": includes_zero,
            "robustness_classification": robustness,
            "n_refits_used": int(unc.n_refits_used),
            "converged_early": bool(unc.converged_early),
            "skip_reason": "",
        })

    table = pd.DataFrame(rows)

    evaluated = table[table["robustness_classification"] != "SKIPPED"]
    if len(evaluated) >= 2:
        all_agree_zero_status = evaluated["includes_zero"].nunique() == 1
        conclusion = (
            "Endpoint robustness IS NOT sensitive to spatial block-size "
            "choice (the CI-crosses-zero conclusion is the same across all "
            "evaluated block-size settings)."
            if all_agree_zero_status else
            "Endpoint robustness IS sensitive to spatial block-size choice "
            "(the CI-crosses-zero conclusion differs across the evaluated "
            "block-size settings)."
        )
    else:
        conclusion = (
            "Fewer than two block-size settings could be evaluated (too "
            "many were skipped for having too few spatial groups); no "
            "robustness conclusion can be drawn from this sweep."
        )
    log_headline(
        f"Spatial block-size sensitivity: {len(evaluated)}/{len(table)} "
        f"merge factor(s) evaluated. {conclusion}"
    )
    table["conclusion"] = conclusion
    summary.log("Spatial block-size sensitivity", "Conclusion", conclusion)
    summary.log(
        "Spatial block-size sensitivity",
        "Note",
        "Block-size sensitivity re-grids the original lat/lon coordinates at "
        f"cfg.spatial_block_cell_size_deg ({cfg.spatial_block_cell_size_deg} deg) "
        "times each merge factor (a genuine geographic coarsening) and reuses the "
        "unmodified spatial-refit uncertainty procedure; it never selects "
        "or substitutes a new primary block size. Heteroscedasticity "
        "(Breusch-Pagan) and residual spatial dependence (Moran's I) do "
        "not, by themselves, disqualify the tree-based model (which does "
        "not require i.i.d. Gaussian residuals the way OLS does); they are "
        "relevant primarily to how the spatial-refit uncertainty band "
        "should be interpreted, which is exactly what this block-size "
        "sensitivity sweep is an additional robustness check on.",
    )
    return table


def _endpoint_uncertainty_stats(unc: "UncertaintyResults", end_idx: int, prefix: str = "endpoint") -> dict:
    """Read the four required quantities, AT THE GIVEN INDEX, off an
    already-computed UncertaintyResults object. Pure read - never
    recomputes anything.

    `prefix` names the point being read (e.g. "endpoint" for index -1, or
    "maximum" for `negi.s2_optimum_idx`) so the same reader can serve the
    endpoint-specific and maximum-specific convergence diagnostics without
    conflating one point's quantities with the other's."""
    p = unc.percentiles["negi_s2"]
    lower = float(p[2.5][end_idx])
    upper = float(p[97.5][end_idx])
    return {
        f"{prefix}_mean_negi":   float(unc.negi_s2_mean[end_idx]),
        f"{prefix}_median_negi": float(p[50.0][end_idx]),
        f"{prefix}_ci_lower":    lower,
        f"{prefix}_ci_upper":    upper,
        f"{prefix}_ci_width":    upper - lower,
    }


def audit_uncertainty_refit_consistency(
    cfg: Config,
    uncertainty: "UncertaintyResults",
) -> dict:
    """Final consistency audit of the uncertainty subsystem (read-only).

    Verifies that the refit count actually embedded in the
    ``UncertaintyResults`` object in use (whether that object was just
    computed fresh or loaded from
    ``.negi_cache/uncertainty_bands__*.pkl``) is consistent with
    ``Config.n_spatial_refits``.

    Since the adaptive-convergence fix, ``n_spatial_refits`` is a MAXIMUM
    ceiling, not a fixed target (see the config comment above
    ``n_spatial_refits_publication``): ``compute_uncertainty_bands()`` may
    stop early, once the endpoint CI width has converged, and record that
    via ``UncertaintyResults.converged_early`` /
    ``UncertaintyResults.n_refits_used``. So ``actual_refits < configured``
    is EXPECTED and valid whenever ``converged_early`` is True - it is not
    itself evidence of a stale cache. What would still indicate a stale or
    corrupted cache entry is:
      - ``actual_refits > configured_refits`` (impossible for an object
        actually produced under the current ceiling), or
      - ``actual_refits < configured_refits`` while ``converged_early`` is
        False (an incomplete run that neither reached the ceiling nor
        converged - i.e. was truncated/hand-edited/corrupted on disk).

    Deliberately does NOT delete the cache or recompute on a genuine
    mismatch: for research software, a run should either be a faithful
    reproduction of what's on disk or an explicit, visible new computation
    - never a silent substitution of one for the other. So this raises
    immediately and lets the user decide whether to delete the stale cache
    file or intentionally rerun the uncertainty stage; it never touches
    the cache, NEGI values, confidence intervals, or any other computation
    itself.

    Returns ``{"configured_refits", "actual_refits", "converged_early",
    "cache_status"}`` - reached only when the object is consistent, so
    ``cache_status`` is always ``"VALID"`` on a normal return.
    """
    configured_refits = int(cfg.n_spatial_refits)
    actual_refits = int(uncertainty.negi_s2_folds.shape[0])
    converged_early = bool(uncertainty.converged_early)

    if actual_refits > configured_refits:
        raise RuntimeError(
            f"Cached uncertainty object was generated with {actual_refits} "
            f"refits, which exceeds the current configuration's ceiling of "
            f"{configured_refits} refits (Config.n_spatial_refits). "
            "Delete the stale cache entry under .negi_cache/ (stage "
            "'uncertainty_bands') or otherwise force a rerun of the "
            "uncertainty stage before publishing - the pipeline will not "
            "silently recompute or discard the mismatch for you."
        )

    if actual_refits < configured_refits and not converged_early:
        raise RuntimeError(
            f"Cached uncertainty object has only {actual_refits} refits "
            f"(short of the {configured_refits}-refit ceiling in "
            "Config.n_spatial_refits) but is not marked as having "
            "converged early (UncertaintyResults.converged_early is "
            "False). This looks like a truncated, hand-edited, or "
            "otherwise corrupted cache entry rather than a legitimate "
            "adaptive-convergence stop. Delete the stale cache entry "
            "under .negi_cache/ (stage 'uncertainty_bands') or otherwise "
            "force a rerun of the uncertainty stage before publishing - "
            "the pipeline will not silently recompute or discard the "
            "mismatch for you."
        )

    return {
        "configured_refits": configured_refits,
        "actual_refits": actual_refits,
        "converged_early": converged_early,
        "cache_status": "VALID",
    }


def build_uncertainty_convergence_criteria(
    audit: dict, convergence_status: Optional[str],
    convergence_skip_reason: Optional[str], cfg: Config,
) -> dict:
    """Single source of truth for BOTH convergence criteria.

    CRUX FIX: as of this pass, the live adaptive-stopping loop in
    compute_uncertainty_bands() requires BOTH its short local-window
    check (`cfg.adaptive_convergence_tolerance` over the most recent
    `cfg.uncertainty_convergence_required_stable_checkpoints`
    checkpoints) AND an extended lookback-window check using the
    stricter `cfg.uncertainty_convergence_tolerance` before it will stop
    early. The two criteria reported below should therefore agree for
    any run computed after this fix. They can still legitimately
    disagree only for:
      * runs loaded from a cache written before this fix, or with
        `adaptive_uncertainty_convergence` disabled entirely (fixed-count
        mode, no adaptive loop at all); or
      * the (by construction, now rare) case where the ceiling
        `cfg.n_spatial_refits` is hit before the extended window ever
        has enough checkpoints to evaluate.
    A CRITERIA DISAGREE result on a fresh, non-cached, adaptive run is a
    signal something upstream (e.g. a config change bypassing this
    unification) needs investigating, not an expected/normal outcome.

    Read-only over values already computed elsewhere - no new computation,
    no change to the published uncertainty bands or NEGI values. Used by
    both `print_uncertainty_consistency_summary` (console) and the
    `uncertainty_convergence_status.csv` export, so the two can never
    drift out of sync with each other.
    """
    adaptive_stop_passed = bool(audit["converged_early"])
    if adaptive_stop_passed:
        stopped_because = (
            "adaptive checkpoint convergence reached "
            f"({cfg.uncertainty_convergence_required_stable_checkpoints} "
            "consecutive stable checkpoints)"
        )
    else:
        stopped_because = (
            f"reached configured ceiling (Config.n_spatial_refits = "
            f"{audit['configured_refits']}) without stabilizing"
        )

    if convergence_status in ("STABILIZED", "NOT YET STABILIZED"):
        publication_convergence = convergence_status
    elif convergence_skip_reason == "cached run (no checkpoint history was recorded)":
        publication_convergence = "SKIPPED (legacy cache)"
    elif convergence_skip_reason:
        publication_convergence = f"SKIPPED ({convergence_skip_reason})"
    else:
        publication_convergence = "N/A"

    return {
        "configured_refits":        audit["configured_refits"],
        "completed_refits":         audit["actual_refits"],
        "adaptive_stop":            "PASS" if adaptive_stop_passed else "FAIL",
        "stopped_because":          stopped_because,
        "publication_convergence":  publication_convergence,
    }


def print_uncertainty_consistency_summary(
    audit: dict, convergence_status: Optional[str],
    convergence_skip_reason: Optional[str] = None,
    tolerance_pct: Optional[float] = None,
    n_refits: Optional[int] = None,
    cfg: Optional[Config] = None,
) -> dict:
    """Print the explicit consistency summary block (reporting-only) and
    return the same criteria dict used to build it, so the caller can
    export it (uncertainty_convergence_status.csv) without recomputing.

    Only ever called after `audit_uncertainty_refit_consistency` has
    returned normally (i.e. configured/actual refits already agree), so
    this always ends with the "internally consistent" confirmation line.

    Adaptive stopping and publication convergence are printed as two
    separate, explicitly labelled lines rather than folded into one
    paragraph - see build_uncertainty_convergence_criteria() for why the
    two can disagree. The full quantitative detail (tolerance, refit
    count) is still logged, just at debug level, so nothing is lost from
    the console cleanup.

    The adaptive-stopping label is deliberately never a bare "PASS": read
    alone, "PASS" invites the misreading "convergence was confirmed",
    when it only means the narrower checkpoint-agreement rule fired. A
    qualifier is appended every time, and an explicit CRITERIA DISAGREE
    line is printed whenever adaptive stopping passed but publication
    convergence did not - the exact case a skimming reader is most likely
    to misread as "converged".
    """
    criteria = build_uncertainty_convergence_criteria(
        audit, convergence_status, convergence_skip_reason, cfg,
    )

    if convergence_status == "NOT YET STABILIZED" and tolerance_pct is not None and n_refits is not None:
        log_debug(
            f"Publication convergence detail: CI width has not yet "
            f"stabilized under the configured {tolerance_pct:.0f}% "
            f"criterion after {n_refits} spatial refits."
        )

    adaptive_stop_display = (
        f"{criteria['adaptive_stop']} (algorithmic stopping rule only - "
        "does NOT confirm CI stabilization; see Publication convergence below)"
        if criteria["adaptive_stop"] == "PASS"
        else "Not achieved before configured ceiling"
    )

    disagree_block = ""
    if criteria["adaptive_stop"] == "PASS" and criteria["publication_convergence"] == "NOT YET STABILIZED":
        disagree_block = (
            "\n"
            "CRITERIA DISAGREE - do not read \"Adaptive stopping: PASS\" as\n"
            "convergence achieved. Fitting stopped, but the stricter\n"
            "full-history stabilization check has NOT passed.\n"
        )

    log_headline(
        "\nUncertainty consistency\n"
        "-----------------------\n"
        f"Configured refits : {criteria['configured_refits']}\n"
        f"Actual refits     : {criteria['completed_refits']}\n"
        f"Cache status      : {audit['cache_status']}\n"
        "\n"
        "Adaptive stopping:\n"
        f"    {adaptive_stop_display}\n"
        "\n"
        "Publication convergence:\n"
        f"    {criteria['publication_convergence']}\n"
        f"{disagree_block}"
        "\n"
        "Reason:\n"
        f"    Adaptive stopping checks recent checkpoints only "
        f"({criteria['stopped_because']}).\n"
        "    Publication convergence checks stabilization across the "
        "full refit history.\n"
        "\nUncertainty subsystem internally consistent."
    )
    return criteria


def compute_uncertainty_convergence_diagnostic(
    model_results: ModelResults,
    baseline: ScenarioBaseline,
    s1: ScenarioTrajectory, s2: ScenarioTrajectory,
    negi: "NEGIResults", cfg: Config,
    uncertainty: "UncertaintyResults",
    point_label: str = "endpoint",
) -> dict:
    """DIAGNOSTIC ONLY - does not touch the spatial-refit uncertainty
    methodology, NEGI, or any published uncertainty band. Reports how the
    Scenario 2 ENDPOINT's (or, when `point_label="maximum"`, the Scenario 2
    MAXIMUM's) mean/lower-CI/upper-CI/CI-width evolved across the ACTUAL
    incremental refit-count checkpoints recorded while computing the
    published `uncertainty` object.

    Reviewer fix: the endpoint and the maximum are different points on the
    Scenario 2 trajectory, and their bands can converge at different rates
    (or not at all). Calling this function once with the default
    `point_label="endpoint"` reproduces the original endpoint-only
    diagnostic exactly; calling it again with `point_label="maximum"` reads
    the SAME already-recorded checkpoint history at `negi.s2_optimum_idx`
    instead of the trajectory's last index, producing an independent
    "STABILIZED"/"NOT YET STABILIZED" verdict for the maximum. The two
    verdicts must not be used interchangeably - the maximum's own verdict
    is what belongs in any caveat attached to the maximum's reported band;
    the endpoint's belongs with the endpoint's.

    Earlier versions of this function re-fit a separate, independent probe
    at each of a fixed sequence of refit counts (25, 50, 100, 200 refits),
    each capped at its own ceiling - so every probe below the required
    stable-checkpoint count would "reach its own cap without stabilizing"
    and log its own MAX-REFITS warning. That is a staged, fixed-cap
    schedule (run 25 and stop, run 50 and stop, ...), not a single
    sequential convergence run, and it triples model-fitting work that the
    adaptive loop in `compute_uncertainty_bands` had already done once.

    This function no longer re-fits anything. It reads
    `uncertainty.convergence_checkpoints` - the same incremental checkpoint
    history (refit count, endpoint CI width, absolute and relative change)
    the adaptive stopping loop recorded while producing `uncertainty` -
    directly off that already-computed object: zero additional model fits,
    bitwise identical numbers, one coherent trajectory instead of several
    staged probes.

    ``model_results``, ``baseline``, ``s1``, ``s2`` are accepted for
    call-site/backward-compatibility only and are no longer used to
    recompute anything.

    Returns a dict with the per-checkpoint table, the pairwise
    checkpoint-to-checkpoint percent-change transition table, and a
    "STABILIZED"/"NOT YET STABILIZED" convergence verdict using the existing
    `cfg.uncertainty_convergence_tolerance` (None if adaptive convergence was
    disabled for this run, so no checkpoint history exists to diagnose).
    """
    if point_label not in ("endpoint", "maximum"):
        raise ValueError(f"point_label must be 'endpoint' or 'maximum', got {point_label!r}")
    point_idx = (len(negi.negi_s2) - 1) if point_label == "endpoint" else negi.s2_optimum_idx
    end_idx = point_idx
    quantities = tuple(f"{point_label}_{suffix}" for suffix in
                        ("mean_negi", "median_negi", "ci_lower", "ci_upper", "ci_width"))
    verdict_quantities = tuple(f"{point_label}_{suffix}" for suffix in
                                ("median_negi", "ci_lower", "ci_upper"))

    published_n_refits = int(uncertainty.negi_s2_folds.shape[0])
    checkpoints = list(uncertainty.convergence_checkpoints or [])

    checkpoints_missing_fields = bool(checkpoints) and not all(
        q in checkpoints[0] for q in quantities
    )

    if not cfg.adaptive_uncertainty_convergence or not checkpoints or checkpoints_missing_fields:
        if not cfg.adaptive_uncertainty_convergence:
            skip_reason = "adaptive convergence disabled for this run"
        elif checkpoints_missing_fields:
            skip_reason = (
                f"cached checkpoint history predates '{point_label}'-specific "
                "convergence tracking; clear the uncertainty-bands cache entry "
                "and rerun to get a verdict for this point"
            )
        elif _CACHE_STAGE_STATUS.get("uncertainty_bands") == "cache hit":
            skip_reason = "cached run (no checkpoint history was recorded)"
        else:
            skip_reason = "no checkpoint history available"
        convergence_df = pd.DataFrame([{
            "refit_count": published_n_refits,
            "source": f"published (single fixed-count run; {skip_reason})",
            **_endpoint_uncertainty_stats(uncertainty, end_idx, prefix=point_label),
        }])
        transitions_df = pd.DataFrame(columns=["transition"] + [f"{q}_pct_change" for q in quantities])
        verdict = None
        report_development(
            cfg,
            f"\nUncertainty convergence diagnostic: SKIPPED ({skip_reason}) - "
            "no incremental refit-count checkpoint history was captured for "
            "this run, so there is nothing to diagnose.",
        )
    else:
        convergence_df = pd.DataFrame(checkpoints)
        convergence_df.insert(
            1, "source", "published (adaptive checkpoint; not recomputed)"
        )

        transitions = []
        for i in range(len(checkpoints) - 1):
            row_from, row_to = checkpoints[i], checkpoints[i + 1]
            trow = {"transition": f"{row_from['refit_count']}\u2192{row_to['refit_count']}"}
            for q in quantities:
                v_from, v_to = row_from[q], row_to[q]
                trow[f"{q}_pct_change"] = (
                    100.0 * (v_to - v_from) / v_from if v_from != 0 else float("nan")
                )
            transitions.append(trow)
        transitions_df = pd.DataFrame(transitions)

        tol_pct = cfg.uncertainty_convergence_tolerance * 100.0
        lookback_n = max(1, int(cfg.uncertainty_convergence_lookback_checkpoints))
        recent_transitions_df = transitions_df.tail(lookback_n)
        if len(recent_transitions_df) > 0:
            converged = bool(all(
                (recent_transitions_df[f"{q}_pct_change"].abs() < tol_pct).all()
                for q in verdict_quantities
            ))
            verdict = "STABILIZED" if converged else "NOT YET STABILIZED"
        else:
            verdict = None

        report_development(
            cfg,
            f"\nUncertainty convergence diagnostic (reading the "
            f"{len(checkpoints)} checkpoint(s) recorded while computing the "
            f"published uncertainty object - {published_n_refits} refits, "
            f"no recomputation):",
        )
        report_development(
            cfg,
            convergence_df[["refit_count", "source"] + list(quantities)].to_string(index=False),
        )
        if len(transitions_df):
            report_development(
                cfg,
                f"\nCheckpoint-to-checkpoint transitions (% change, tolerance {tol_pct:.0f}%):",
            )
            report_development(cfg, transitions_df.to_string(index=False))

    refit_counts = (
        [int(c["refit_count"]) for c in checkpoints] if checkpoints else [published_n_refits]
    )
    result = {
        "refit_counts":         refit_counts,
        "published_n_refits":   published_n_refits,
        "tolerance_pct":        cfg.uncertainty_convergence_tolerance * 100.0,
        "table":                convergence_df.to_dict(orient="records"),
        "transitions":          transitions_df.to_dict(orient="records"),
        "convergence_status":   verdict,
        "convergence_skip_reason": None if checkpoints else skip_reason,
        "table_df":             convergence_df,
        "transitions_df":       transitions_df,
    }
    return result


def report_convergence_extension_comparison(
    uncertainty: UncertaintyResults, cfg: Config, summary: SummaryLog,
) -> Optional[dict]:
    """Compare the Scenario 2 endpoint conclusion at the OLD publication
    refit ceiling (`cfg.n_spatial_refits_previous_publication_ceiling`)
    against the conclusion at the final adaptive refit count actually
    used, using only the checkpoint history already recorded while
    computing `uncertainty` - zero additional model fits.

    This answers the question the ceiling-raising comments in Config
    exist to justify: does the "CI excludes/includes zero" conclusion
    actually depend on how many refits were used, or was the earlier,
    smaller ceiling already sufficient? Reporting-only; never used to
    drive any computation or alter the published uncertainty band.

    Returns None (and reports why) when there is no checkpoint history to
    compare - adaptive convergence was disabled for this run, or the
    result was loaded from a cache written before checkpoints were
    tracked.
    """
    checkpoints = list(uncertainty.convergence_checkpoints or [])
    if not checkpoints:
        report_development(
            cfg,
            "\nConvergence-extension comparison: SKIPPED (no checkpoint "
            "history available for this run).",
        )
        return None

    old_ceiling = int(cfg.n_spatial_refits_previous_publication_ceiling)
    final_cp = checkpoints[-1]
    old_cp = next(
        (c for c in checkpoints if c["refit_count"] >= old_ceiling), None,
    )

    def _excludes_zero(c: dict) -> bool:
        return c["endpoint_ci_lower"] > 0.0 or c["endpoint_ci_upper"] < 0.0

    if old_cp is None:
        msg = (
            f"\nConvergence-extension comparison: the run stopped at "
            f"{final_cp['refit_count']} refits, before reaching the old "
            f"{old_ceiling}-refit ceiling - the extension was not needed "
            "for this run."
        )
        report_development(cfg, msg)
        result = {
            "old_ceiling_refits": old_ceiling, "old_ceiling_reached": False,
            "final_refits": final_cp["refit_count"],
            "conclusion_changed": False,
        }
    else:
        same_conclusion = _excludes_zero(old_cp) == _excludes_zero(final_cp)
        msg = (
            f"\nConvergence-extension comparison "
            f"(old ceiling = {old_ceiling} refits vs. final = "
            f"{final_cp['refit_count']} refits):\n"
            f"  At {old_cp['refit_count']} refits : CI = "
            f"[{old_cp['endpoint_ci_lower']:.4f}, {old_cp['endpoint_ci_upper']:.4f}] "
            f"({'excludes' if _excludes_zero(old_cp) else 'includes'} zero)\n"
            f"  At {final_cp['refit_count']} refits : CI = "
            f"[{final_cp['endpoint_ci_lower']:.4f}, {final_cp['endpoint_ci_upper']:.4f}] "
            f"({'excludes' if _excludes_zero(final_cp) else 'includes'} zero)\n"
            f"  Endpoint zero-crossing conclusion "
            f"{'is unchanged' if same_conclusion else 'CHANGED'} between "
            "the old and final refit counts."
        )
        report_development(
            cfg, msg, level=("info" if same_conclusion else "warning"),
        )
        result = {
            "old_ceiling_refits": old_ceiling, "old_ceiling_reached": True,
            "old_ceiling_ci_lower": old_cp["endpoint_ci_lower"],
            "old_ceiling_ci_upper": old_cp["endpoint_ci_upper"],
            "final_refits": final_cp["refit_count"],
            "final_ci_lower": final_cp["endpoint_ci_lower"],
            "final_ci_upper": final_cp["endpoint_ci_upper"],
            "conclusion_changed": not same_conclusion,
        }

    summary.log(
        "Convergence-extension comparison", "Conclusion changed vs. old ceiling",
        str(result["conclusion_changed"]),
    )
    return result


def report_uncertainty_bands(
    uncertainty: UncertaintyResults, s1: ScenarioTrajectory,
    cfg: Config, summary: SummaryLog,
) -> None:
    """Log and record summary statistics for the uncertainty bands."""
    save_csv(uncertainty.table, cfg.data_dir / "uncertainty_summary.csv")
    scenario_pct  = s1.scenario_pct
    p             = uncertainty.percentiles
    peak_std_idx  = int(np.argmax(uncertainty.negi_s2_std))
    peak_iqr_idx  = int(np.argmax(p["negi_s2"][75] - p["negi_s2"][25]))
    n_refits      = len(uncertainty.negi_s2_folds)

    lower_pct = 2.5
    upper_pct = 97.5
    uncertainty_type_label = "95% spatial-refit uncertainty band (empirical 2.5th-97.5th percentile)"
    seed_behaviour = (
        f"train/holdout partitions for all {n_refits} refits are drawn from a "
        f"single GroupShuffleSplit call seeded once with "
        f"Config.uncertainty_refit_seed={cfg.uncertainty_refit_seed} (deterministic, "
        "reproducible split generation, not independently re-seeded per refit); "
        f"each refit's XGBoost model uses the fixed Config.random_seed="
        f"{cfg.random_seed}"
    )

    log_info("\nSpatial-refit uncertainty procedure:")
    log_info(f"  Number of spatial refits    : {n_refits}")
    log_info(f"  Uncertainty type            : {uncertainty_type_label}")
    log_info(f"  Lower percentile            : {lower_pct}")
    log_info(f"  Upper percentile            : {upper_pct}")
    log_info(f"  Random-seed behaviour       : {seed_behaviour}")
    log_info(
        f"  Unique partitions           : {uncertainty.n_unique_partitions}/{n_refits}"
    )
    uncertainty_source_label = (
        "spatial-partition variation + model stochasticity"
        if uncertainty.model_stochasticity_varied
        else "spatial-partition variation only (model random seed held fixed "
             "across refits; XGBoost's own training stochasticity is NOT a "
             "component of this band)"
    )
    log_info(f"  Uncertainty source          : {uncertainty_source_label}")

    summary.log("Uncertainty diagnostics", "N spatial refits",              str(n_refits))
    summary.log("Uncertainty diagnostics", "Spatial-refit uncertainty type", uncertainty_type_label)
    summary.log("Uncertainty diagnostics", "Spatial-refit lower percentile", str(lower_pct))
    summary.log("Uncertainty diagnostics", "Spatial-refit upper percentile", str(upper_pct))
    summary.log("Uncertainty diagnostics", "Spatial-refit random-seed behaviour", seed_behaviour)
    summary.log("Uncertainty diagnostics", "Spatial-refit unique partitions",
                f"{uncertainty.n_unique_partitions}/{n_refits}")
    summary.log("Uncertainty diagnostics", "Spatial-refit uncertainty source", uncertainty_source_label)
    if n_refits < 100:
        summary.add_warning(
            f"Spatial-refit uncertainty bands use {n_refits} refits. Percentile "
            "estimates (especially the 2.5th/97.5th) are correspondingly "
            "imprecise; treat this as an approximate empirical band, not a "
            "formal confidence interval. Set Config.spatial_refit_mode="
            "'publication' (200 refits) for the final manuscript numbers."
        )
    summary.log("Uncertainty diagnostics", "Peak S2 NEGI SD (scenario %)",
                f"{scenario_pct[peak_std_idx]:.1f}")
    summary.log("Uncertainty diagnostics", "Peak S2 NEGI SD (value)",
                f"{uncertainty.negi_s2_std[peak_std_idx]:.4f}")
    summary.log("Uncertainty diagnostics", "Peak S2 NEGI IQR (scenario %)",
                f"{scenario_pct[peak_iqr_idx]:.1f}")
    summary.log("Uncertainty diagnostics", "Peak S2 NEGI IQR (Q75-Q25)",
                f"{(p['negi_s2'][75] - p['negi_s2'][25])[peak_iqr_idx]:.4f}")


def compute_negi_zero_crossing_diagnostics(
    negi: "NEGIResults",
    uncertainty: "UncertaintyResults",
    support: "SupportDiagnostics",
    cfg: Config,
) -> pd.DataFrame:
    """Report whether zero lies inside the spatial-refit NEGI uncertainty
    interval at a small set of specific trajectory positions, for each
    scenario:

      - the maximum evaluated NEGI (numerical maximum along the trajectory)
      - the final evaluated trajectory point
      - the first nominally positive NEGI point, if one exists

    This is a descriptive diagnostic only. It does not constitute a formal
    hypothesis test; "the interval includes zero" is reported rather than
    any claim of statistical (in)significance.
    """
    scenario_pct = negi.scenario_pct
    p = uncertainty.percentiles

    def _first_positive_idx(arr: np.ndarray) -> Optional[int]:
        positive = np.where(arr > 0)[0]
        return int(positive[0]) if positive.size > 0 else None

    scenario_specs = [
        ("Scenario 1", negi.negi_s1, negi.s1_optimum_idx, p["negi_s1"],
         None),
        ("Scenario 2", negi.negi_s2, negi.s2_optimum_idx, p["negi_s2"],
         support.in_support_s2),
    ]

    rows = []
    for label, negi_arr, max_idx, pct, support_arr in scenario_specs:
        positions = {
            "maximum_evaluated_negi": max_idx,
            "final_trajectory_point": len(negi_arr) - 1,
        }
        first_pos_idx = _first_positive_idx(negi_arr)
        if first_pos_idx is not None:
            positions["first_positive_negi"] = first_pos_idx

        for position_name, idx in positions.items():
            lower = float(pct[2.5][idx])
            upper = float(pct[97.5][idx])
            includes_zero = bool(lower <= 0.0 <= upper)
            support_pass = bool(support_arr[idx]) if support_arr is not None else None
            is_boundary = is_near_trajectory_boundary(
                idx, len(negi_arr), cfg.scenario_boundary_tolerance_pct,
                positions=scenario_pct,
            )
            rows.append({
                "scenario":            label,
                "scenario_position":   position_name,
                "scenario_pct":        float(scenario_pct[idx]),
                "negi_point_estimate": float(negi_arr[idx]),
                "uncertainty_lower":   lower,
                "uncertainty_upper":   upper,
                "includes_zero":       includes_zero,
                "support_pass":        support_pass,
                "interior_or_boundary": "boundary" if is_boundary else "interior",
            })

    table = pd.DataFrame(rows)
    return table


def report_negi_zero_crossing_diagnostics(
    table: pd.DataFrame, cfg: Config, summary: "SummaryLog",
) -> pd.DataFrame:
    """Log and export the NEGI zero-crossing diagnostic table."""
    save_csv(table, cfg.data_dir / "negi_zero_crossing_summary.csv", cfg=cfg, debug_only=True)
    position_labels = {
        "maximum_evaluated_negi": "maximum evaluated NEGI",
        "final_trajectory_point": "final trajectory point",
        "first_positive_negi":    "first positive NEGI",
    }
    for _, row in table.iterrows():
        position_label = position_labels.get(
            row["scenario_position"], row["scenario_position"].replace("_", " ")
        )
        log_info(f"{row['scenario']} | {position_label}")
        log_info(f"  {'Scenario position':<18}: {row['scenario_pct']:.1f}%")
        log_info(f"  {'NEGI':<18}: {row['negi_point_estimate']:.4f}")
        log_info(
            f"  {'95% spatial-refit uncertainty band':<18}: "
            f"[{row['uncertainty_lower']:.4f}, {row['uncertainty_upper']:.4f}]"
        )
        log_info(f"  {'Includes zero':<18}: {'YES' if row['includes_zero'] else 'NO'}")
        summary.log(
            "NEGI zero-crossing diagnostic",
            f"{row['scenario']} - {row['scenario_position']}",
            f"NEGI={row['negi_point_estimate']:.4f}, "
            f"95%_spatial-refit_uncertainty_band=[{row['uncertainty_lower']:.4f}, {row['uncertainty_upper']:.4f}], "
            f"includes_zero={row['includes_zero']}, "
            f"{row['interior_or_boundary']}"
            + (
                "; statistically compatible with a neutral normalized balance "
                "under spatial-refit uncertainty - not evidence of a robust "
                "positive environmental gain"
                if row["scenario_position"] == "maximum_evaluated_negi" and row["includes_zero"]
                else ""
            ),
        )
    return table


def _distribution_summary(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    q25, q50, q75 = np.percentile(values, [25, 50, 75])
    return {
        "mean":   float(values.mean()),
        "sd":     float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "median": float(q50),
        "q25":    float(q25),
        "q75":    float(q75),
        "iqr":    float(q75 - q25),
        "p2.5":   float(np.percentile(values, 2.5)),
        "p97.5":  float(np.percentile(values, 97.5)),
    }


def summarize_spatial_refit_diagnostics(
    validation: ValidationResults,
    uncertainty: UncertaintyResults,
    s1: ScenarioTrajectory,
    cfg: Config,
) -> pd.DataFrame:
    """Summarise repeated spatial-block refits (median/IQR/percentiles)."""
    scenario_pct      = s1.scenario_pct
    cooling_s2_folds  = uncertainty.cooling_s2_folds
    negi_s2_folds     = uncertainty.negi_s2_folds

    per_fold_max_cooling       = cooling_s2_folds.max(axis=1)
    per_fold_trajectory_max    = negi_s2_folds.max(axis=1)
    per_fold_first_cooling_pct = np.full(len(cooling_s2_folds), np.nan)
    for i, row in enumerate(cooling_s2_folds):
        cp = np.where(row > 0)[0]
        if len(cp) > 0:
            per_fold_first_cooling_pct[i] = scenario_pct[cp[0]]

    finite_fcp = per_fold_first_cooling_pct[np.isfinite(per_fold_first_cooling_pct)]
    rows = {
        "Repeated holdout R2":                  _distribution_summary(validation.repeated_r2_scores),
        "Repeated holdout RMSE (C)":            _distribution_summary(validation.repeated_rmse_scores),
        "Repeated holdout MAE (C)":             _distribution_summary(validation.repeated_mae_scores),
        "Spatial-refit S2 max cooling (C)":     _distribution_summary(per_fold_max_cooling),
        "Spatial-refit S2 NEGI trajectory max": _distribution_summary(per_fold_trajectory_max),
        "Spatial-refit S2 first cooling pct (%)": (
            _distribution_summary(finite_fcp)
            if len(finite_fcp) > 0
            else {k: np.nan for k in ["mean", "sd", "median", "q25", "q75", "iqr", "p2.5", "p97.5"]}
        ),
    }
    table = pd.DataFrame(rows).T
    table.index.name = "Quantity"
    table = table.reset_index()

    table["spatial_refit_count"]            = int(uncertainty.n_refits_used)
    table["spatial_refit_count_ceiling"]    = int(cfg.n_spatial_refits)
    table["spatial_refit_converged_early"]  = bool(uncertainty.converged_early)
    table["spatial_refit_lower_percentile"] = 2.5
    table["spatial_refit_upper_percentile"] = 97.5
    table["spatial_refit_uncertainty_type"] = (
        "95% spatial-refit uncertainty band (empirical 2.5th-97.5th percentile)"
    )
    save_csv(table, cfg.data_dir / "spatial_refit_summary.csv", cfg=cfg, debug_only=True)
    return table


def run_sensitivity_analysis(
    negi: NEGIResults, s1: ScenarioTrajectory, cfg: Config
) -> SensitivityResults:
    """Sweep NEGI cost-function parameters (alpha, beta, w0, exponent)
    for the sensitivity analysis."""
    scenarios     = s1.scenario_fraction
    scenario_pct  = scenarios * 100
    cooling_norm_s2 = safe_divide(np.maximum(negi.delta_t_s2, 0.0), negi.reference_benefit_scale)

    reference_max_by_exponent = {
        exp: _energy_cost(scenarios, cfg.reference_w0, exp, cfg).max()
        for exp in cfg.sensitivity_exponents
    }

    rows = []
    for exponent in cfg.sensitivity_exponents:
        ref_max = reference_max_by_exponent[exponent]
        for alpha_val in cfg.sensitivity_alpha_values:
            for beta_val in cfg.sensitivity_beta_values:
                for w0_val in cfg.sensitivity_w0_values:
                    energy_penalty = _energy_cost(scenarios, w0_val, exponent, cfg)
                    energy_norm_arr = safe_divide(energy_penalty, ref_max)
                    negi_profile    = alpha_val * cooling_norm_s2 - beta_val * energy_norm_arr
                    best = int(np.argmax(negi_profile))
                    rows.append({
                        "Exponent": exponent,
                        "Alpha":    alpha_val,
                        "Beta":     beta_val,
                        "w0":       w0_val,
                        "Numerical_Maximum_Scenario (%)": scenario_pct[best],
                        "Numerical_Maximum_NEGI":         negi_profile[best],
                    })

    for exp, ref_max in reference_max_by_exponent.items():
        check_val = _energy_cost(scenarios, cfg.reference_w0, exp, cfg).max() / ref_max
        assert np.isclose(check_val, 1.0, atol=1e-9), \
            f"Normalisation broken for exponent={exp}"

    ref_max_default = reference_max_by_exponent[cfg.sensitivity_default_exponent]
    e_ref = safe_divide(
        _energy_cost(scenarios, cfg.reference_w0, cfg.sensitivity_default_exponent, cfg),
        ref_max_default,
    )
    for _w0_test in [50.0, 100.0, 300.0]:
        _e_test = safe_divide(
            _energy_cost(scenarios, _w0_test, cfg.sensitivity_default_exponent, cfg),
            ref_max_default,
        )
        assert np.allclose(_e_test, e_ref * (_w0_test / cfg.reference_w0), atol=1e-9), \
            f"w0 linearity guard failed at w0={_w0_test}"

    table = pd.DataFrame(rows)

    plot_table = table[
        isclose_mask(table["Exponent"], cfg.sensitivity_default_exponent)
        & isclose_mask(table["Beta"],   cfg.sensitivity_default_beta)
    ]
    assert len(plot_table) > 0, \
        "plot_table is empty - check sensitivity_default_exponent / sensitivity_default_beta."

    return SensitivityResults(table=table, plot_table=plot_table)


def report_sensitivity_analysis(sensitivity: SensitivityResults, cfg: Config) -> None:
    """Log and record the sensitivity analysis results."""
    save_csv(sensitivity.table, cfg.data_dir / "sensitivity_summary.csv")
    unique_optima = sensitivity.table["Numerical_Maximum_Scenario (%)"].nunique()
    log_info(f"\nDistinct numerical maximum scenario values across parameter grid: {unique_optima}")


@dataclass
class CostRegimeResult:
    """Result of compute_cost_regime_diagnostic() for one cost exponent."""
    exponent:                 float
    reference_w0:             float
    reference_alpha_over_beta: float
    reference_effective_cost: float
    n_grid_cells:             int
    grid_reproduction_mismatches: int
    invariance_violations:    int
    monotone_nonincreasing:   bool
    scan_c:                   np.ndarray
    scan_position_pct:        np.ndarray
    jumps:                    pd.DataFrame
    regime_detected:          bool
    primary_jump:             Optional[dict]
    split_pct:                float
    low_regime_range_pct:     tuple
    high_regime_range_pct:    tuple
    dominant_jump_fraction:   float
    is_sharp:                 bool
    grid_low_fraction:        float
    grid_high_fraction:       float
    is_bimodal_on_grid:       bool
    n_grid_cells_at_baseline: int
    reference_position_pct:   float
    reference_regime:         str
    reference_boundary_w0:    float
    reference_distance_factor: float
    boundary_table:           pd.DataFrame
    verdict:                  str


def _s2_cost_profile_inputs(
    negi: NEGIResults, exponent: float, cfg: Config,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(scenario_pct, normalised cooling benefit b(f), cost shape g(f)) for
    Scenario 2 - built exactly as run_sensitivity_analysis builds them."""
    scenarios = negi.scenarios
    benefit   = safe_divide(np.maximum(negi.delta_t_s2, 0.0), negi.reference_benefit_scale)
    ref_cost  = _energy_cost(scenarios, cfg.reference_w0, exponent, cfg)
    shape     = safe_divide(ref_cost, ref_cost.max())
    return scenarios * 100.0, np.asarray(benefit, dtype=float), np.asarray(shape, dtype=float)


def _analyze_cost_regime_for_exponent(
    negi: NEGIResults, sensitivity: SensitivityResults, exponent: float, cfg: Config,
) -> CostRegimeResult:
    pos_pct, benefit, shape = _s2_cost_profile_inputs(negi, exponent, cfg)
    n_scen = len(pos_pct)

    table = sensitivity.table
    sub = table[isclose_mask(table["Exponent"], exponent)].copy()
    if sub.empty:
        raise ValueError(f"Sensitivity grid has no rows for exponent={exponent}.")
    alpha = sub["Alpha"].to_numpy(dtype=float)
    beta  = sub["Beta"].to_numpy(dtype=float)
    w0    = sub["w0"].to_numpy(dtype=float)
    stored_pos = sub["Numerical_Maximum_Scenario (%)"].to_numpy(dtype=float)
    ratio = alpha / beta
    k_arr = w0 / cfg.reference_w0
    c_arr = k_arr / ratio

    mismatches = 0
    for a_v, b_v, w_v, pos_v in zip(alpha, beta, w0, stored_pos):
        profile = a_v * benefit - b_v * safe_divide(
            _energy_cost(negi.scenarios, w_v, exponent, cfg),
            _energy_cost(negi.scenarios, cfg.reference_w0, exponent, cfg).max(),
        )
        if not np.isclose(pos_pct[int(np.argmax(profile))], pos_v, atol=1e-9):
            mismatches += 1

    c_key = np.round(c_arr, 9)
    invariance_violations = 0
    for key in np.unique(c_key):
        positions = np.unique(np.round(stored_pos[c_key == key], 9))
        if len(positions) > 1:
            invariance_violations += int((c_key == key).sum())

    def _idx_at(c: float) -> int:
        return int(np.argmax(benefit - c * shape))

    c_lo = float(c_arr.min()) / 2.0
    c_hi = float(c_arr.max()) * 2.0
    scan_c = np.geomspace(c_lo, c_hi, int(cfg.cost_regime_boundary_search_points))
    scan_idx = np.array([_idx_at(c) for c in scan_c], dtype=int)
    scan_pos = pos_pct[scan_idx]
    monotone_nonincreasing = bool(np.all(np.diff(scan_pos) <= 1e-9))

    jump_rows: list[dict] = []
    i = 0
    while i < len(scan_c) - 1:
        if scan_idx[i + 1] == scan_idx[i]:
            i += 1
            continue
        cur_c, cur_idx, end_c = float(scan_c[i]), int(scan_idx[i]), float(scan_c[i + 1])
        end_idx = int(scan_idx[i + 1])
        _guard = 0
        while cur_idx != end_idx and _guard <= 2 * n_scen:
            _guard += 1
            lo, hi = cur_c, end_c
            for _ in range(200):
                mid = float(np.sqrt(lo * hi))
                if _idx_at(mid) == cur_idx:
                    lo = mid
                else:
                    hi = mid
                if hi / lo - 1.0 < 1e-13:
                    break
            after_idx = _idx_at(hi)
            jump_rows.append({
                "effective_cost_boundary": float(hi),
                "position_before_pct":     float(pos_pct[cur_idx]),
                "position_after_pct":      float(pos_pct[after_idx]),
                "jump_pct":                float(pos_pct[cur_idx] - pos_pct[after_idx]),
            })
            cur_c, cur_idx = hi, after_idx
        i += 1
    jumps = pd.DataFrame(
        jump_rows,
        columns=["effective_cost_boundary", "position_before_pct",
                 "position_after_pct", "jump_pct"],
    )

    total_movement = float(scan_pos.max() - scan_pos.min())
    primary: Optional[dict] = None
    regime_detected = False
    split_pct = float("nan")
    dominant_fraction = float("nan")
    if len(jumps):
        primary_row = jumps.loc[jumps["jump_pct"].abs().idxmax()]
        primary = {k: float(primary_row[k]) for k in jumps.columns}
        regime_detected = abs(primary["jump_pct"]) >= cfg.cost_regime_min_gap_pct
        split_pct = 0.5 * (primary["position_before_pct"] + primary["position_after_pct"])
        dominant_fraction = (
            abs(primary["jump_pct"]) / total_movement if total_movement > 0 else float("nan")
        )
    is_sharp = bool(
        regime_detected
        and np.isfinite(dominant_fraction)
        and dominant_fraction >= cfg.cost_regime_dominant_jump_fraction
    )

    def _side_range(mask: np.ndarray) -> tuple:
        return (float(scan_pos[mask].min()), float(scan_pos[mask].max())) if mask.any() \
            else (float("nan"), float("nan"))

    if regime_detected:
        low_range  = _side_range(scan_pos <= split_pct)
        high_range = _side_range(scan_pos > split_pct)
        grid_high  = stored_pos > split_pct
        grid_low_fraction  = float((~grid_high).mean())
        grid_high_fraction = float(grid_high.mean())
    else:
        low_range = high_range = (float("nan"), float("nan"))
        grid_low_fraction = grid_high_fraction = float("nan")
    is_bimodal = bool(
        regime_detected
        and grid_low_fraction >= cfg.cost_regime_min_cluster_fraction
        and grid_high_fraction >= cfg.cost_regime_min_cluster_fraction
    )

    ref_ratio = cfg.alpha_weight / cfg.beta_weight
    ref_c     = 1.0 / ref_ratio
    ref_pos   = float(pos_pct[_idx_at(ref_c)])
    if regime_detected:
        ref_regime = "high-intervention" if ref_pos > split_pct else "low-intervention"
        ref_boundary_w0 = cfg.reference_w0 * primary["effective_cost_boundary"] * ref_ratio
        ref_distance = ref_c / primary["effective_cost_boundary"]
    else:
        ref_regime = "single regime (no distinct transition)"
        ref_boundary_w0 = float("nan")
        ref_distance = float("nan")

    ratio_key = np.round(ratio, 9)
    rows = []
    w0_min, w0_max = float(w0.min()), float(w0.max())
    for key in np.unique(ratio_key):
        sel = ratio_key == key
        w_vals = np.unique(w0[sel])
        pos_by_w0 = np.array([stored_pos[sel & np.isclose(w0, w)][0] for w in w_vals])
        r_val = float(ratio[sel][0])
        row = {
            "alpha_over_beta":         r_val,
            "w0_min_evaluated":        w0_min,
            "w0_max_evaluated":        w0_max,
            "optimum_pct_at_w0_min":   float(pos_by_w0[0]),
            "optimum_pct_at_w0_max":   float(pos_by_w0[-1]),
        }
        if regime_detected:
            b_w0 = cfg.reference_w0 * primary["effective_cost_boundary"] * r_val
            in_range = bool(w0_min <= b_w0 <= w0_max)
            regime_by_w0 = pos_by_w0 > split_pct
            flips = np.where(regime_by_w0[:-1] != regime_by_w0[1:])[0]
            if len(flips):
                br_lo, br_hi = float(w_vals[flips[0]]), float(w_vals[flips[0] + 1])
            else:
                br_lo = br_hi = float("nan")
            row.update({
                "boundary_w0_refined":      float(b_w0),
                "boundary_in_evaluated_w0_range": in_range,
                "grid_bracket_w0_lower":    br_lo,
                "grid_bracket_w0_upper":    br_hi,
                "grid_bracket_consistent":  (
                    bool(br_lo <= b_w0 <= br_hi) if len(flips) else (not in_range)
                ),
                "regime_at_w0_min":         "high-intervention" if regime_by_w0[0] else "low-intervention",
                "regime_at_w0_max":         "high-intervention" if regime_by_w0[-1] else "low-intervention",
            })
        else:
            row.update({
                "boundary_w0_refined": float("nan"),
                "boundary_in_evaluated_w0_range": False,
                "grid_bracket_w0_lower": float("nan"),
                "grid_bracket_w0_upper": float("nan"),
                "grid_bracket_consistent": True,
                "regime_at_w0_min": "single regime",
                "regime_at_w0_max": "single regime",
            })
        rows.append(row)
    boundary_table = pd.DataFrame(rows).sort_values("alpha_over_beta").reset_index(drop=True)

    if not regime_detected:
        verdict = (
            "NO DISTINCT REGIMES: the selected Scenario 2 maximum moves "
            "gradually (or by steps smaller than "
            f"{cfg.cost_regime_min_gap_pct:.1f} percentage points) across the evaluated cost range."
        )
    elif is_sharp and is_bimodal:
        verdict = (
            "SHARP BIMODAL TRANSITION: one dominant snap separates a "
            "low-intervention regime from a high-intervention regime, and "
            "both regimes are populated in the evaluated grid."
        )
    elif is_sharp:
        verdict = (
            "SHARP TRANSITION, ONE-SIDED GRID: a dominant snap exists, but "
            "the evaluated grid populates one regime almost exclusively."
        )
    else:
        verdict = (
            "STEPWISE TRANSITION: a regime change of at least "
            f"{cfg.cost_regime_min_gap_pct:.1f} percentage points exists, but no single "
            "jump dominates the movement of the maximum (multiple comparable steps)."
        )

    return CostRegimeResult(
        exponent=float(exponent), reference_w0=float(cfg.reference_w0),
        reference_alpha_over_beta=float(ref_ratio), reference_effective_cost=float(ref_c),
        n_grid_cells=int(len(sub)),
        grid_reproduction_mismatches=int(mismatches),
        invariance_violations=int(invariance_violations),
        monotone_nonincreasing=monotone_nonincreasing,
        scan_c=scan_c, scan_position_pct=scan_pos, jumps=jumps,
        regime_detected=bool(regime_detected), primary_jump=primary,
        split_pct=float(split_pct),
        low_regime_range_pct=low_range, high_regime_range_pct=high_range,
        dominant_jump_fraction=float(dominant_fraction), is_sharp=is_sharp,
        grid_low_fraction=grid_low_fraction, grid_high_fraction=grid_high_fraction,
        is_bimodal_on_grid=is_bimodal,
        n_grid_cells_at_baseline=int(np.isclose(stored_pos, 0.0).sum()),
        reference_position_pct=ref_pos, reference_regime=ref_regime,
        reference_boundary_w0=float(ref_boundary_w0),
        reference_distance_factor=float(ref_distance),
        boundary_table=boundary_table, verdict=verdict,
    )


def compute_cost_regime_diagnostic(
    negi: NEGIResults, sensitivity: SensitivityResults, cfg: Config,
) -> tuple[CostRegimeResult, pd.DataFrame]:
    """Describe where the Scenario 2 maximum switches between the
    low-intervention and high-intervention regime across the existing
    alpha/beta/w0 sensitivity grid.

    Reads `sensitivity.table` (from the unchanged run_sensitivity_analysis)
    and the primary fit's Scenario 2 cooling benefit; recomputes nothing
    about the NEGI formulation. The detailed analysis is for
    `cfg.sensitivity_default_exponent`; a compact per-exponent summary
    (returned as the second value) covers every exponent in the grid.
    """
    primary = _analyze_cost_regime_for_exponent(
        negi, sensitivity, cfg.sensitivity_default_exponent, cfg,
    )
    per_exp_rows = []
    for exponent in cfg.sensitivity_exponents:
        r = primary if np.isclose(exponent, cfg.sensitivity_default_exponent) \
            else _analyze_cost_regime_for_exponent(negi, sensitivity, exponent, cfg)
        per_exp_rows.append({
            "exponent":                 float(exponent),
            "regime_detected":          r.regime_detected,
            "is_sharp":                 r.is_sharp,
            "is_bimodal_on_grid":       r.is_bimodal_on_grid,
            "n_jumps":                  int(len(r.jumps)),
            "boundary_w0_at_reference_alpha_over_beta": r.reference_boundary_w0,
            "split_pct":                r.split_pct,
            "low_regime_min_pct":       r.low_regime_range_pct[0],
            "low_regime_max_pct":       r.low_regime_range_pct[1],
            "high_regime_min_pct":      r.high_regime_range_pct[0],
            "high_regime_max_pct":      r.high_regime_range_pct[1],
            "reference_position_pct":   r.reference_position_pct,
            "reference_regime":         r.reference_regime,
            "grid_reproduction_mismatches": r.grid_reproduction_mismatches,
            "invariance_violations":    r.invariance_violations,
        })
    return primary, pd.DataFrame(per_exp_rows)


def cost_regime_to_dict(r: CostRegimeResult, cfg: Config) -> dict:
    """JSON-safe export of a CostRegimeResult (arrays summarised, tables as records)."""
    def _f(x):
        return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else float(x)
    return {
        "definitions": {
            "w0": (
                "resource-cost scaling coefficient: a relative scaling "
                "parameter of the NEGI cost term (normalised by its value at "
                "reference_w0). Not a directly measured Jeddah water or "
                "energy quantity; not calibrated to real-world data."
            ),
            "alpha_beta": "relative benefit and cost weights of the NEGI formulation",
            "effective_cost_ratio_c": "c = (beta / alpha) * (w0 / reference_w0); the maximum position depends on (alpha, beta, w0) only through c",
            "boundary_curve": "w0_boundary(alpha/beta) = reference_w0 * c_boundary * (alpha/beta)",
        },
        "exponent": r.exponent,
        "reference_w0": r.reference_w0,
        "reference_alpha_over_beta": r.reference_alpha_over_beta,
        "n_grid_cells": r.n_grid_cells,
        "checks": {
            "grid_reproduction_mismatches": r.grid_reproduction_mismatches,
            "invariance_violations": r.invariance_violations,
            "maximum_position_monotone_nonincreasing_in_cost": r.monotone_nonincreasing,
        },
        "regime_detected": r.regime_detected,
        "verdict": r.verdict,
        "is_sharp": r.is_sharp,
        "is_bimodal_on_evaluated_grid": r.is_bimodal_on_grid,
        "dominant_jump_fraction": _f(r.dominant_jump_fraction),
        "primary_transition": None if r.primary_jump is None else {k: _f(v) for k, v in r.primary_jump.items()},
        "regime_split_position_pct": _f(r.split_pct),
        "low_intensity_regime_position_range_pct": [_f(v) for v in r.low_regime_range_pct],
        "high_intensity_regime_position_range_pct": [_f(v) for v in r.high_regime_range_pct],
        "grid_fraction_low_regime": _f(r.grid_low_fraction),
        "grid_fraction_high_regime": _f(r.grid_high_fraction),
        "grid_cells_with_maximum_at_baseline_0pct": r.n_grid_cells_at_baseline,
        "reference_case": {
            "alpha_over_beta": r.reference_alpha_over_beta,
            "w0": r.reference_w0,
            "effective_cost_ratio_c": r.reference_effective_cost,
            "maximum_position_pct": _f(r.reference_position_pct),
            "regime": r.reference_regime,
            "boundary_w0_at_reference_alpha_over_beta": _f(r.reference_boundary_w0),
            "reference_cost_over_boundary_cost": _f(r.reference_distance_factor),
        },
        "all_transitions": r.jumps.to_dict(orient="records"),
        "boundary_by_alpha_over_beta": [
            {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in row.items()}
            for row in r.boundary_table.to_dict(orient="records")
        ],
    }


def report_cost_regime_diagnostic(
    result: CostRegimeResult, by_exponent: pd.DataFrame, cfg: Config, summary: SummaryLog,
) -> dict:
    """Log, export, and record the cost-regime diagnostic. Reporting only."""
    r = result
    save_csv(r.boundary_table, cfg.data_dir / "cost_regime_boundary.csv")
    save_csv(r.jumps, cfg.data_dir / "cost_regime_transitions.csv")
    save_csv(by_exponent, cfg.data_dir / "cost_regime_by_exponent.csv")
    payload = cost_regime_to_dict(r, cfg)
    save_json(payload, cfg.data_dir / "cost_regime_diagnostic.json")

    section = "Cost-regime diagnostic"
    scope_note = interpretation_text("cost_parameter_scope")
    summary.log(section, "Scope of w0", scope_note)
    summary.log(section, "Exponent analysed in detail", f"{r.exponent}")
    summary.log(section, "Verdict", r.verdict)
    summary.log(
        section, "Numerical checks",
        f"grid-reproduction mismatches = {r.grid_reproduction_mismatches}; "
        f"invariance violations = {r.invariance_violations}; "
        f"maximum position non-increasing in cost = {r.monotone_nonincreasing}",
    )
    if r.grid_reproduction_mismatches or r.invariance_violations or not r.monotone_nonincreasing:
        summary.add_warning(
            "Cost-regime diagnostic: a numerical self-check failed "
            f"(grid-reproduction mismatches={r.grid_reproduction_mismatches}, "
            f"invariance violations={r.invariance_violations}, "
            f"monotone={r.monotone_nonincreasing}); the boundary curve below "
            "should not be relied on until this is understood."
        )

    lines = [f"\nCost-regime diagnostic (exponent = {r.exponent}; "
             f"{r.n_grid_cells} alpha x beta x w0 grid cells)", f"  {r.verdict}"]
    if r.regime_detected:
        pj = r.primary_jump
        lines += [
            f"  Primary transition at effective cost c = (beta/alpha)*(w0/w0_ref) = "
            f"{pj['effective_cost_boundary']:.4f}: maximum snaps from "
            f"{pj['position_before_pct']:.2f}% (cheaper side) to "
            f"{pj['position_after_pct']:.2f}% (costlier side).",
            f"  Low-intervention regime : maximum at {r.low_regime_range_pct[0]:.2f}-"
            f"{r.low_regime_range_pct[1]:.2f}% "
            f"({r.grid_low_fraction * 100:.1f}% of grid cells).",
            f"  High-intervention regime: maximum at {r.high_regime_range_pct[0]:.2f}-"
            f"{r.high_regime_range_pct[1]:.2f}% "
            f"({r.grid_high_fraction * 100:.1f}% of grid cells).",
            f"  Boundary curve: w0_boundary = {cfg.reference_w0 * pj['effective_cost_boundary']:.2f} x (alpha/beta) "
            "(resource-cost scaling units, relative to reference_w0).",
            f"  Reference alpha/beta = {r.reference_alpha_over_beta:g}: boundary at "
            f"w0 = {r.reference_boundary_w0:.2f}; the reference w0 = {r.reference_w0:g} sits in the "
            f"{r.reference_regime} regime (maximum {r.reference_position_pct:.2f}%; "
            f"reference cost is {r.reference_distance_factor:.2f}x the boundary cost).",
            f"  Sharp (single dominant snap): {r.is_sharp} "
            f"(dominant jump = {r.dominant_jump_fraction * 100:.0f}% of total movement); "
            f"bimodal on the evaluated grid: {r.is_bimodal_on_grid}; "
            f"transitions found: {len(r.jumps)}.",
        ]
        summary.log(
            section, "Primary transition",
            f"effective cost c = {pj['effective_cost_boundary']:.4f}; maximum "
            f"{pj['position_before_pct']:.2f}% -> {pj['position_after_pct']:.2f}%",
        )
        summary.log(
            section, "Boundary curve",
            f"w0_boundary = {cfg.reference_w0 * pj['effective_cost_boundary']:.2f} x (alpha/beta)",
        )
        summary.log(
            section, "Reference alpha/beta boundary",
            f"alpha/beta = {r.reference_alpha_over_beta:g}: w0 = {r.reference_boundary_w0:.2f}; "
            f"reference w0 = {r.reference_w0:g} is in the {r.reference_regime} regime",
        )
        summary.log(
            section, "Low-intensity regime",
            f"maximum at {r.low_regime_range_pct[0]:.2f}-{r.low_regime_range_pct[1]:.2f}% "
            f"({r.grid_low_fraction * 100:.1f}% of grid cells)",
        )
        summary.log(
            section, "High-intensity regime",
            f"maximum at {r.high_regime_range_pct[0]:.2f}-{r.high_regime_range_pct[1]:.2f}% "
            f"({r.grid_high_fraction * 100:.1f}% of grid cells)",
        )
        summary.log(
            section, "Sharp / bimodal",
            f"sharp={r.is_sharp}; bimodal on evaluated grid={r.is_bimodal_on_grid}; "
            f"dominant jump fraction={r.dominant_jump_fraction:.2f}; transitions={len(r.jumps)}",
        )
        n_out = int((~r.boundary_table["boundary_in_evaluated_w0_range"]).sum())
        if n_out:
            summary.log(
                section, "Boundary outside evaluated w0 range",
                f"{n_out} of {len(r.boundary_table)} alpha/beta values have their boundary "
                f"outside w0 = [{r.boundary_table['w0_min_evaluated'].iloc[0]:g}, "
                f"{r.boundary_table['w0_max_evaluated'].iloc[0]:g}] "
                "(the grid there sits entirely in one regime).",
            )
    else:
        lines.append("  No regime boundary to report for this exponent.")
    lines.append("  " + scope_note)
    log_info("\n".join(lines))

    if r.regime_detected:
        log_headline(
            f"Cost regime: {r.verdict.split(':')[0].title()}; boundary w0 = "
            f"{r.reference_boundary_w0:.0f} at alpha/beta = {r.reference_alpha_over_beta:g} "
            f"(reference w0 = {r.reference_w0:g} in {r.reference_regime} regime)."
        )
    else:
        log_headline("Cost regime: no distinct transition in the evaluated grid.")
    summary.add_conclusion(
        "Cost-regime diagnostic: " + r.verdict + " " + scope_note
    )
    return payload


def _layer(name, kind, low, high, description, **extra) -> dict:
    return {
        "layer": name, "kind": kind,
        "position_low_pct": float(low), "position_high_pct": float(high),
        "width_pct": float(high - low), "description": description, **extra,
    }


def compute_decision_envelope(
    negi: NEGIResults,
    uncertainty: Optional[UncertaintyResults],
    percentile_table: Optional[pd.DataFrame],
    sensitivity: SensitivityResults,
    cost_regime: Optional[CostRegimeResult],
    cfg: Config,
) -> dict:
    """Assemble the decision envelope from already-computed results.
    Introduces no new NEGI, model, or uncertainty calculation."""
    pct = negi.scenario_pct
    ref_pos  = float(pct[negi.s2_optimum_idx])
    ref_negi = float(negi.negi_s2[negi.s2_optimum_idx])
    layers: list[dict] = [_layer(
        "Reference case", "reference", ref_pos, ref_pos,
        "Primary fit; reference alpha, beta, w0; sqrt cost exponent.",
        negi_low=ref_negi, negi_high=ref_negi, n=1,
    )]

    if uncertainty is not None and getattr(uncertainty, "negi_s2_folds", None) is not None \
            and uncertainty.negi_s2_folds.ndim == 2 and uncertainty.negi_s2_folds.shape[1] == len(pct):
        fold_pos = pct[np.argmax(uncertainty.negi_s2_folds, axis=1)]
        p2_5, med, p97_5 = np.percentile(fold_pos, [2.5, 50, 97.5])
        fold_max_negi = np.max(uncertainty.negi_s2_folds, axis=1)
        layers.append(_layer(
            "Spatial-refit range", "structural_model", p2_5, p97_5,
            "Position of the Scenario 2 maximum across spatial-block refits "
            "(2.5th-97.5th percentile of per-refit maxima).",
            n=int(len(fold_pos)), median_pct=float(med),
            full_min_pct=float(fold_pos.min()), full_max_pct=float(fold_pos.max()),
            negi_low=float(np.percentile(fold_max_negi, 2.5)),
            negi_high=float(np.percentile(fold_max_negi, 97.5)),
        ))

    if percentile_table is not None and len(percentile_table) \
            and "maximum_scenario_pct" in percentile_table.columns:
        pos = percentile_table["maximum_scenario_pct"].to_numpy(dtype=float)
        layers.append(_layer(
            "Archetype-percentile range", "structural_scenario", pos.min(), pos.max(),
            "Position of the Scenario 2 maximum across the alternative "
            "archetype NDBI-percentile endpoints evaluated.",
            n=int(len(pos)),
            per_percentile=[
                {"percentile": float(p), "maximum_scenario_pct": float(m)}
                for p, m in zip(percentile_table["percentile"], pos)
            ],
            negi_low=float(percentile_table["maximum_negi"].min()),
            negi_high=float(percentile_table["maximum_negi"].max()),
        ))

    table = sensitivity.table
    at_default = table[isclose_mask(table["Exponent"], cfg.sensitivity_default_exponent)]
    col = "Numerical_Maximum_Scenario (%)"
    cost_pos = at_default[col].to_numpy(dtype=float)
    on_diag = at_default[np.isclose(at_default["Alpha"], at_default["Beta"], atol=1e-6)]
    layers.append(_layer(
        "Cost-weight range", "decision_parameters", cost_pos.min(), cost_pos.max(),
        "Position of the Scenario 2 maximum across the alpha x beta x w0 grid "
        "(alpha, beta: relative weights; w0: resource-cost scaling coefficient).",
        n=int(len(cost_pos)), n_distinct_positions=int(np.unique(np.round(cost_pos, 9)).size),
        alpha_equals_beta_low_pct=float(on_diag[col].min()) if len(on_diag) else None,
        alpha_equals_beta_high_pct=float(on_diag[col].max()) if len(on_diag) else None,
        negi_low=float(at_default["Numerical_Maximum_NEGI"].min()),
        negi_high=float(at_default["Numerical_Maximum_NEGI"].max()),
        all_exponents_low_pct=float(table[col].min()),
        all_exponents_high_pct=float(table[col].max()),
    ))

    by_name = {l["layer"]: l for l in layers}
    structural = [l for l in layers if l["kind"].startswith("structural")]
    cost_layer = by_name["Cost-weight range"]
    regime_ok = bool(cost_regime is not None and cost_regime.regime_detected)
    split = cost_regime.split_pct if regime_ok else float("nan")
    low_regime_max = cost_regime.low_regime_range_pct[1] if regime_ok else float("nan")
    for l in layers:
        l["within_low_intensity_regime"] = (
            bool(l["position_high_pct"] <= split) if regime_ok else None
        )
    structural_all_low = bool(regime_ok and structural and all(l["within_low_intensity_regime"] for l in structural))
    cost_spans_regimes = bool(regime_ok and cost_layer["position_low_pct"] <= split < cost_layer["position_high_pct"])
    structural_widths = [l["width_pct"] for l in structural]
    structural_narrower = bool(structural_widths and max(structural_widths) < cost_layer["width_pct"])
    stable_location = bool(structural_all_low and structural_narrower)

    def _rng(l):
        lo, hi = l["position_low_pct"], l["position_high_pct"]
        return f"{lo:.2f}%" if np.isclose(lo, hi) else f"{lo:.2f}-{hi:.2f}%"

    published: list[str] = [
        f"Reference-case maximum: {ref_pos:.2f}% (NEGI = {ref_negi:.3f}); this is one "
        "point of the decision envelope, not an overall optimum."
    ]
    for l in structural:
        published.append(f"{l['layer']}: maximum at {_rng(l)} ({l['description']})")
    published.append(
        f"Cost-weight range: maximum at {_rng(cost_layer)} across the alpha x beta x w0 grid "
        f"(NEGI at the maximum spans {cost_layer['negi_low']:.3f} to {cost_layer['negi_high']:.3f})."
    )
    if stable_location:
        published.append(
            "Under the structural and model sensitivities tested, the location of "
            f"the low-intensity maximum is comparatively stable: every structural range "
            f"stays on the low-intensity side of the regime split ({split:.2f}%, the midpoint of "
            "the low/high-intensity transition) and is narrower than the range produced by "
            "the cost weights. The exact "
            "optimum and the NEGI magnitude are conditional on resource-cost assumptions "
            "(alpha/beta and the relative resource-cost scaling coefficient w0)"
            + ("; across the cost-weight grid the maximum crosses between the low- and "
               "high-intensity regimes." if cost_spans_regimes else ".")
        )
    else:
        reasons = []
        if not regime_ok:
            reasons.append("no distinct low/high-intensity regime was detected in the cost grid, so regime membership cannot be assessed")
        elif not structural_all_low:
            reasons.append("at least one structural/model range extends beyond the low-intensity regime")
        if structural and not structural_narrower:
            reasons.append("a structural range is not narrower than the cost-weight range")
        if not structural:
            reasons.append("no structural sensitivity results were available")
        published.append(
            "The evidence does not support describing the location of the maximum as "
            "comparatively stable under the structural/model sensitivities tested ("
            + "; ".join(reasons) + "). The exact optimum and NEGI magnitude remain "
            "conditional on resource-cost assumptions (alpha/beta and the relative "
            "resource-cost scaling coefficient w0)."
        )
    published.append(interpretation_text("cost_parameter_scope"))

    return {
        "reference_case": {"position_pct": ref_pos, "negi": ref_negi},
        "layers": layers,
        "assessment": {
            "regime_split_position_pct": None if not regime_ok else float(split),
            "low_intensity_regime_max_pct": None if not regime_ok else float(low_regime_max),
            "structural_layers_all_within_low_regime": structural_all_low if regime_ok else None,
            "cost_weight_range_spans_regimes": cost_spans_regimes if regime_ok else None,
            "structural_ranges_narrower_than_cost_range": structural_narrower,
            "location_comparatively_stable_under_structural_sensitivities": stable_location,
            "ranges_are_reported_separately_and_not_pooled": True,
        },
        "published_language": published,
        "notes": interpretation_text("decision_envelope_scope"),
    }


def report_decision_envelope(envelope: dict, cfg: Config, summary: SummaryLog) -> dict:
    """Log, export, and record the decision envelope. Reporting only."""
    layers_df = pd.DataFrame([
        {k: v for k, v in l.items() if k != "per_percentile"} for l in envelope["layers"]
    ])
    save_csv(layers_df, cfg.data_dir / "decision_envelope.csv")
    save_json(envelope, cfg.data_dir / "decision_envelope.json")

    section = "Decision envelope"
    summary.log(section, "Scope", interpretation_text("decision_envelope_scope"))
    for l in envelope["layers"]:
        lo, hi = l["position_low_pct"], l["position_high_pct"]
        rng = f"{lo:.2f}%" if np.isclose(lo, hi) else f"{lo:.2f}-{hi:.2f}%"
        regime = l.get("within_low_intensity_regime")
        regime_txt = "" if regime is None else f"; within low-intensity regime: {regime}"
        summary.log(section, l["layer"], f"maximum at {rng} [{l['kind']}]{regime_txt}")
    for i, line in enumerate(envelope["published_language"], 1):
        summary.log(section, f"Published-facing statement {i}", line)
    summary.add_conclusion("Decision envelope:\n  " + "\n  ".join(envelope["published_language"]))
    log_info("\nDecision envelope (ranges reported separately; not pooled):")
    for line in envelope["published_language"]:
        log_info("  " + line)
    log_headline(
        "Decision envelope: "
        + "; ".join(
            f"{l['layer'].replace(' range', '').lower()} "
            + (f"{l['position_low_pct']:.2f}%" if np.isclose(l['position_low_pct'], l['position_high_pct'])
               else f"{l['position_low_pct']:.2f}-{l['position_high_pct']:.2f}%")
            for l in envelope["layers"]
        ) + " (not pooled)."
    )
    return envelope


def build_ndbi_quantile_bins(
    df: pd.DataFrame, n_bins: int, bin_names: list[str]
) -> list[tuple]:
    """Build equal-pixel-count NDBI bins (edges at the n_bins-quantiles) with
    human-readable labels, in the same (lo, hi, label) tuple format consumed
    by compute_conditional_ndvi_lst_slopes.

    This is a pure binning-scheme choice (how many equal-count NDBI strata
    to cut the data into) - it does not touch the regression itself. Used
    both for the primary quartile (n_bins=4) stratification and, as a
    robustness check, an alternative equal-count tercile (n_bins=3)
    stratification (see compute_ndbi_binning_robustness).

    NOTE (sign-reversal audit finding): equal-count binning holds NDBI only
    APPROXIMATELY constant within each stratum - substantial continuous NDBI
    variation remains inside a bin (confirmed: within-stratum Pearson
    r(NDVI, NDBI) as strong as -0.74 in the widest strata of this dataset).
    The single-predictor regression fit within each bin by
    compute_conditional_ndvi_lst_slopes is therefore an UNADJUSTED
    (marginal) association, not a stratification that fully controls NDBI.
    See compute_ndvi_ndbi_continuous_adjustment() for the companion
    diagnostic that adjusts for NDBI continuously (plus Elevation) instead.
    """
    if len(bin_names) != n_bins:
        raise ValueError(
            f"build_ndbi_quantile_bins: got {len(bin_names)} bin_names for "
            f"n_bins={n_bins}; these must match 1:1."
        )
    quantile_levels = [i / n_bins for i in range(1, n_bins)]
    edges = df["NDBI"].quantile(quantile_levels).to_numpy()
    lo_bound = df["NDBI"].min()
    hi_bound = df["NDBI"].max() + 1e-9
    cut_points = [lo_bound, *edges, hi_bound]

    bins = []
    for i, name in enumerate(bin_names):
        lo, hi = cut_points[i], cut_points[i + 1]
        if i == 0:
            label = f"{name}\nNDBI < {hi:.3f}"
        elif i == n_bins - 1:
            label = f"{name}\nNDBI > {lo:.3f}"
        else:
            label = f"{name}\n{lo:.3f}-{hi:.3f}"
        bins.append((lo, hi, label))
    return bins


def compute_conditional_ndvi_lst_slopes(
    df: pd.DataFrame, ndbi_bins: list[tuple]
) -> list[ConditionalBinResult]:
    """Compute the NDVI-LST slope within each NDBI stratum (conditional analysis)."""
    results = []
    for ndbi_lo, ndbi_hi, label in ndbi_bins:
        mask   = (df["NDBI"] >= ndbi_lo) & (df["NDBI"] < ndbi_hi)
        subset = df[mask]
        if len(subset) > 10:
            sl, ic, rv, pv, se = scipy_stats.linregress(subset["NDVI"], subset["LST"])
            n      = len(subset)
            t_crit = scipy_stats.t.ppf(0.975, df=n - 2)
            results.append(ConditionalBinResult(
                label=label,
                ndvi=subset["NDVI"].to_numpy(), lst=subset["LST"].to_numpy(),
                n_pixels=n, slope=sl, intercept=ic, slope_se=se,
                slope_ci95=t_crit * se, r_squared=rv ** 2, p_value=pv,
            ))
        else:
            results.append(ConditionalBinResult(
                label=label,
                ndvi=subset["NDVI"].to_numpy(), lst=subset["LST"].to_numpy(),
                n_pixels=len(subset),
                slope=None, intercept=None, slope_se=None,
                slope_ci95=None, r_squared=None, p_value=None,
            ))
    return results


def conditional_slopes_to_dataframe(
    results: list[ConditionalBinResult]
) -> pd.DataFrame:
    """Assemble per-stratum conditional NDVI-LST slope results into a DataFrame."""
    rows = []
    for r in results:
        if r.slope is None:
            continue
        rows.append({
            "NDBI_range":  r.label.replace("\n", " "),
            "n_pixels":    r.n_pixels,
            "slope":       r.slope,
            "slope_SE":    r.slope_se,
            "slope_95CI":  r.slope_ci95,
            "r_squared":   r.r_squared,
            "p_value":     r.p_value,
            "significant": "Y" if r.p_value < 0.05 else "N",
            "direction":   "positive" if r.slope > 0 else "negative",
        })
    return pd.DataFrame(rows)


def spatial_block_bootstrap_ndvi_slope(
    df_train: pd.DataFrame, ndbi_eval_points: dict, cfg: Config,
) -> dict:
    """Spatial-block bootstrap 95% CI for the continuously-adjusted implied
    NDVI-LST slope (d(LST)/d(NDVI) = coef_ndvi + coef_interaction * NDBI),
    evaluated at each NDBI value in ``ndbi_eval_points`` (percentile label
    -> NDBI value, as produced by compute_ndvi_ndbi_continuous_adjustment).

    Mirrors the audit's own spatial-block bootstrap (resample whole
    spatial_block groups with replacement, refit, evaluate at fixed NDBI
    points) and the pipeline's existing spatial_block_bootstrap_metrics()
    pattern used for the primary holdout CIs - same block-resampling
    logic, applied here to the interaction-model coefficients instead of
    a prediction-error metric. NDBI evaluation points are held fixed at
    their full-sample values across all bootstrap draws (only the fitted
    coefficients vary per draw), matching how the audit reports "slope at
    NDBI percentile p" as a single comparable quantity across resamples.

    Uses cfg.bootstrap_iterations / cfg.bootstrap_seed - the same knobs
    (and the same fast_mode cap) as every other bootstrap CI in this
    pipeline - so this diagnostic is exempt from neither fast_mode nor
    the bootstrap draw count used everywhere else.
    """
    ndvi = df_train["NDVI"].to_numpy(dtype=float)
    ndbi = df_train["NDBI"].to_numpy(dtype=float)
    elevation = df_train["Elevation"].to_numpy(dtype=float)
    target = df_train["LST"].to_numpy(dtype=float)
    block_ids = df_train["spatial_block"].to_numpy()

    n = len(df_train)
    unique_blocks = np.unique(block_ids)
    n_unique_blocks = len(unique_blocks)
    block_to_positions = {b: np.where(block_ids == b)[0] for b in unique_blocks}

    n_boot = cfg.bootstrap_iterations
    rng = np.random.default_rng(cfg.bootstrap_seed)

    labels = list(ndbi_eval_points.keys())
    eval_values = np.array([ndbi_eval_points[p] for p in labels], dtype=float)
    samples = np.empty((n_boot, len(labels)), dtype=float)

    for i in range(n_boot):
        sampled_blocks = rng.choice(unique_blocks, size=n_unique_blocks, replace=True)
        positions = np.concatenate([block_to_positions[b] for b in sampled_blocks])
        design_boot = np.column_stack([
            ndvi[positions],
            ndbi[positions],
            elevation[positions],
            ndvi[positions] * ndbi[positions],
        ])
        model_boot = LinearRegression().fit(design_boot, target[positions])
        coef_ndvi_boot, coef_interaction_boot = model_boot.coef_[0], model_boot.coef_[3]
        samples[i, :] = coef_ndvi_boot + coef_interaction_boot * eval_values

    return {
        "labels":          labels,
        "lower":           {p: float(np.percentile(samples[:, j], 2.5)) for j, p in enumerate(labels)},
        "upper":           {p: float(np.percentile(samples[:, j], 97.5)) for j, p in enumerate(labels)},
        "frac_positive":   {p: float(np.mean(samples[:, j] > 0)) for j, p in enumerate(labels)},
        "n_boot":          int(n_boot),
        "n_unique_blocks": int(n_unique_blocks),
    }


def compute_ndvi_ndbi_continuous_adjustment(
    df_train: pd.DataFrame, cfg: Config,
) -> ContinuousAdjustmentResult:
    """Continuously-adjusted companion diagnostic to the NDBI-stratified
    (binned) NDVI-LST slope figures (Figure 6 / Figure S18).

    NDVI-LST sign-reversal audit finding: compute_conditional_ndvi_lst_slopes
    fits a single-predictor (NDVI-only) regression *within* NDBI bins. Because
    equal-count NDBI bins only hold NDBI APPROXIMATELY constant (residual
    within-bin Pearson r(NDVI, NDBI) as strong as -0.74 in the widest
    strata), those per-bin slopes are unadjusted marginal associations that
    can mechanically flip sign purely from leftover within-bin NDBI
    variation - not from a genuine reversal of the NDVI-LST relationship.

    This function instead controls for NDBI (and Elevation) CONTINUOUSLY,
    via a single full-sample interaction model fit once on the same
    data.df_train used everywhere else in the pipeline:

        LST ~ NDVI + NDBI + Elevation + NDVI:NDBI

    The fitted coefficients imply an NDVI slope that varies smoothly with
    NDBI rather than being pooled within wide, imperfectly-homogeneous
    bins:

        d(LST)/d(NDVI) = coef_ndvi + coef_interaction * NDBI

    That implied slope is evaluated at cfg.continuous_adjustment_ndbi_percentiles
    (default p10/p25/p50/p75/p90 of the training NDBI distribution) so it
    can be compared directly, percentile-by-percentile, against the binned
    figure's per-stratum slopes.

    This is purely a diagnostic/reporting computation: it does not touch,
    refit, or otherwise influence the production XGBoost model, the NEGI
    formula, or either scenario trajectory. It is also not a causal
    estimate - it is an observational association with continuous
    (rather than binned) control for NDBI and Elevation, and unmeasured
    confounding may remain. See report_continuous_adjustment() for the
    accompanying caveat text.
    """
    ndvi = df_train["NDVI"].to_numpy(dtype=float)
    ndbi = df_train["NDBI"].to_numpy(dtype=float)
    design = pd.DataFrame({
        "NDVI":         ndvi,
        "NDBI":         ndbi,
        "Elevation":    df_train["Elevation"].to_numpy(dtype=float),
        "NDVI_x_NDBI":  ndvi * ndbi,
    })
    target = df_train["LST"].to_numpy(dtype=float)

    model = LinearRegression().fit(design, target)
    r_squared = float(model.score(design, target))
    coef = dict(zip(design.columns, model.coef_))

    ndbi_percentiles = {
        p: float(df_train["NDBI"].quantile(p / 100.0))
        for p in cfg.continuous_adjustment_ndbi_percentiles
    }
    implied_ndvi_slope = {
        p: float(coef["NDVI"] + coef["NDVI_x_NDBI"] * ndbi_at_p)
        for p, ndbi_at_p in ndbi_percentiles.items()
    }

    boot = spatial_block_bootstrap_ndvi_slope(df_train, ndbi_percentiles, cfg)

    return ContinuousAdjustmentResult(
        n_obs=len(df_train),
        r_squared=r_squared,
        coef_ndvi=float(coef["NDVI"]),
        coef_ndbi=float(coef["NDBI"]),
        coef_elevation=float(coef["Elevation"]),
        coef_interaction=float(coef["NDVI_x_NDBI"]),
        intercept=float(model.intercept_),
        ndbi_percentiles=ndbi_percentiles,
        implied_ndvi_slope=implied_ndvi_slope,
        slope_ci_lower=boot["lower"],
        slope_ci_upper=boot["upper"],
        slope_frac_positive=boot["frac_positive"],
        n_boot=boot["n_boot"],
        n_unique_blocks=boot["n_unique_blocks"],
    )


def continuous_adjustment_to_dataframe(result: ContinuousAdjustmentResult) -> pd.DataFrame:
    """Tabulate the per-percentile implied NDVI-LST slope, its
    spatial-block-bootstrap 95% CI, and whether that CI crosses zero, from
    compute_ndvi_ndbi_continuous_adjustment() for CSV export / plotting."""
    rows = [
        {
            "NDBI_percentile":       f"p{p}",
            "NDBI_value":            result.ndbi_percentiles[p],
            "implied_NDVI_slope":    result.implied_ndvi_slope[p],
            "boot_ci_lower":         result.slope_ci_lower.get(p),
            "boot_ci_upper":         result.slope_ci_upper.get(p),
            "boot_frac_positive":    result.slope_frac_positive.get(p),
            "boot_ci_crosses_zero":  (
                result.slope_ci_lower[p] < 0 < result.slope_ci_upper[p]
                if p in result.slope_ci_lower and p in result.slope_ci_upper
                else None
            ),
        }
        for p in sorted(result.ndbi_percentiles)
    ]
    return pd.DataFrame(rows)


def report_continuous_adjustment(
    result: ContinuousAdjustmentResult, cfg: Config, summary: SummaryLog,
) -> dict:
    """Log and record the continuous NDBI+Elevation adjustment diagnostic,
    and state explicitly whether it resolves the sign pattern seen in the
    binned Figure 6 / Figure S18 stratified slopes.

    Purely descriptive: reports what the already-fitted interaction model
    (compute_ndvi_ndbi_continuous_adjustment) implies. Does not refit
    anything, and does not alter the NEGI/model/scenario pipeline.
    """
    table = continuous_adjustment_to_dataframe(result)
    save_csv(
        table, cfg.data_dir / "ndvi_lst_continuous_adjustment.csv",
        cfg=cfg, debug_only=True,
    )

    slopes = table["implied_NDVI_slope"].to_numpy(dtype=float)
    n_negative = int(np.sum(slopes < 0))
    n_positive = int(np.sum(slopes > 0))
    sign_consistent = (n_negative == len(slopes)) or (n_positive == len(slopes))
    sign_pattern = (
        "consistently negative" if n_negative == len(slopes) else
        "consistently positive" if n_positive == len(slopes) else
        "mixed (sign changes across evaluated NDBI percentiles)"
    )

    any_ci_crosses_zero = bool(table["boot_ci_crosses_zero"].any())
    ci_lines = "\n".join(
        f"    {row['NDBI_percentile']} (NDBI={row['NDBI_value']:.4f}): slope="
        f"{row['implied_NDVI_slope']:.3f}, 95% CI=[{row['boot_ci_lower']:.3f}, "
        f"{row['boot_ci_upper']:.3f}], frac>0={row['boot_frac_positive']:.3f}"
        for _, row in table.iterrows()
    )

    interpretation = (
        f"Continuously adjusting for NDBI and Elevation (LST ~ NDVI + NDBI + "
        f"Elevation + NDVI:NDBI, n={result.n_obs}, R²={result.r_squared:.3f}) "
        f"gives an implied NDVI-LST slope that is {sign_pattern} across the "
        f"evaluated NDBI percentiles (range: {slopes.min():.3f} to "
        f"{slopes.max():.3f} °C per unit NDVI). Spatial-block-bootstrap "
        f"({result.n_boot} draws resampling {result.n_unique_blocks} unique "
        f"training blocks) 95% CIs {'do NOT' if not any_ci_crosses_zero else 'DO'} "
        f"cross zero at any evaluated percentile. "
    )
    if sign_consistent:
        interpretation += (
            "This is a single, non-reversing sign once NDBI is held "
            "continuously constant rather than binned, unlike the "
            "unadjusted within-NDBI-stratum slopes in Figure 6 / Figure S18, "
            "which retain residual continuous NDBI confounding inside each "
            "bin and can appear to change sign as a result. Neither analysis "
            "establishes a causal NDVI effect; unmeasured confounding may "
            "remain."
        )
    else:
        interpretation += (
            "The sign is not fully consistent even under continuous "
            "adjustment, so the apparent sign changes in Figure 6 / Figure "
            "S18 cannot be attributed solely to within-bin NDBI confounding; "
            "see the per-percentile values above. Neither analysis "
            "establishes a causal NDVI effect; unmeasured confounding may "
            "remain."
        )

    log_info(
        "\nContinuous NDBI+Elevation adjustment (companion to Figure 6 / "
        "Figure S18):\n"
        f"  N (train)        : {result.n_obs}\n"
        f"  R²               : {result.r_squared:.4f}\n"
        f"  coef(NDVI)       : {result.coef_ndvi:.4f}\n"
        f"  coef(NDBI)       : {result.coef_ndbi:.4f}\n"
        f"  coef(NDVI:NDBI)  : {result.coef_interaction:.4f}\n"
        f"  Sign pattern     : {sign_pattern}\n"
        f"  Slope range      : {slopes.min():.4f} to {slopes.max():.4f}\n"
        f"  Spatial-block bootstrap ({result.n_boot} draws, "
        f"{result.n_unique_blocks} unique blocks):\n{ci_lines}\n"
        f"  Any 95% CI crosses zero? : {any_ci_crosses_zero}\n"
        f"  -> {interpretation}"
    )
    summary.log("Interpretive Diagnostics",
                "Continuous NDVI-NDBI adjustment: sign pattern", sign_pattern)
    summary.log("Interpretive Diagnostics",
                "Continuous NDVI-NDBI adjustment: slope range",
                f"{slopes.min():.4f} to {slopes.max():.4f}")
    summary.log("Interpretive Diagnostics",
                "Continuous NDVI-NDBI adjustment: any bootstrap CI crosses zero",
                any_ci_crosses_zero)
    summary.log("Interpretive Diagnostics",
                "Continuous NDVI-NDBI adjustment note", interpretation)
    summary.add_conclusion(interpretation)

    return {
        "sign_consistent":       sign_consistent,
        "sign_pattern":          sign_pattern,
        "slope_min":             float(slopes.min()),
        "slope_max":             float(slopes.max()),
        "any_ci_crosses_zero":   any_ci_crosses_zero,
        "n_boot":                result.n_boot,
        "n_unique_blocks":       result.n_unique_blocks,
        "table":                 table,
        "interpretation":        interpretation,
    }


def plot_ndvi_continuous_adjustment(
    result: ContinuousAdjustmentResult, cfg: Config,
) -> Path:
    """Plot the continuously-adjusted implied NDVI-LST slope across NDBI
    percentiles - the properly-adjusted companion diagnostic to the
    unadjusted, binned NDVI-LST slopes in Figure 6 / Figure S18.

    Unlike those figures, NDBI is held continuously constant (via the
    fitted interaction term) rather than binned, so this line has no
    residual within-bin NDBI confounding by construction.
    """
    table = continuous_adjustment_to_dataframe(result)
    fig, ax = plt.subplots(figsize=cfg.default_figsize)
    ax.axhline(0.0, color=cfg.color_neutral, linewidth=1.0, linestyle=":")
    ax.fill_between(
        table["NDBI_value"], table["boot_ci_lower"], table["boot_ci_upper"],
        color=cfg.color_accent, alpha=0.18,
        label=f"{result.n_boot}-draw spatial-block-bootstrap 95% CI",
    )
    ax.plot(
        table["NDBI_value"], table["implied_NDVI_slope"],
        "o-", linewidth=2.2, markersize=8, color=cfg.color_accent,
        label="Implied NDVI slope (continuous NDBI+Elevation adjustment)",
    )
    for _, row in table.iterrows():
        ax.annotate(
            row["NDBI_percentile"],
            xy=(row["NDBI_value"], row["implied_NDVI_slope"]),
            xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8,
        )
    set_axis_labels(ax, "NDBI (evaluation point)", "Implied dLST/dNDVI (°C per unit NDVI)")
    set_title(ax, "Continuously-Adjusted NDVI-LST Slope Across NDBI\n"
                  "(companion diagnostic to Figure 6 / Figure S18)")
    set_legend(ax)
    apply_grid(ax)
    fig.text(
        0.5, -0.04,
        f"NDBI held continuously constant via NDVI:NDBI interaction term; "
        f"band = {result.n_boot}-draw spatial-block bootstrap ({result.n_unique_blocks} "
        "unique training blocks). Not a causal effect; unmeasured confounding may remain.",
        ha="center", fontsize=8, color=cfg.color_neutral,
    )
    return save_fig(
        fig, FIG_NDVI_CONTINUOUS_ADJUSTMENT, cfg,
        caption=(
            "Continuously-adjusted (NDVI + NDBI + Elevation + NDVI:NDBI) "
            "implied NDVI-LST slope across NDBI percentiles, with "
            "spatial-block-bootstrap 95% CIs - the properly-adjusted "
            "companion diagnostic to the unadjusted, within-NDBI-stratum "
            "slopes shown in Figure 6 / Figure S18."
        ),
        section="Interpretive diagnostics",
    )



def _build_varying_coefficient_design(
    ndvi: np.ndarray, ndbi: np.ndarray, elevation: np.ndarray,
    ndbi_df: int, elev_df: int,
    ndbi_spline: Optional[SplineTransformer] = None,
    elev_spline: Optional[SplineTransformer] = None,
    fit_splines: bool = False,
):
    """Build the design matrix for
        LST = f_NDBI(NDBI) + f_Elevation(Elevation) + NDVI * g(NDBI) + error
    with g(NDBI) = b_ndvi + sum_k gamma_k * BSpline_k(NDBI).

    Cubic B-spline bases (quantile knots, linear extrapolation beyond the
    observed range) are used for both smooth terms; ndbi_df / elev_df set
    each basis's degrees of freedom (its knot count), which is the only
    smoothing/bandwidth parameter of this estimator. When fit_splines is
    True, new spline transformers are fit to (ndvi, ndbi, elevation) (the
    training fold); otherwise the caller's already-fitted transformers are
    reused to transform a different fold - this is what makes the
    block-CV and spatial-block-bootstrap loops below leakage-free.
    """
    if fit_splines:
        ndbi_spline = SplineTransformer(
            degree=3, n_knots=max(ndbi_df - 2, 2), knots="quantile",
            include_bias=False, extrapolation="linear",
        ).fit(ndbi.reshape(-1, 1))
        elev_spline = SplineTransformer(
            degree=3, n_knots=max(elev_df - 2, 2), knots="quantile",
            include_bias=False, extrapolation="linear",
        ).fit(elevation.reshape(-1, 1))
    b_ndbi = ndbi_spline.transform(ndbi.reshape(-1, 1))
    b_elev = elev_spline.transform(elevation.reshape(-1, 1))
    k1 = b_ndbi.shape[1]
    interaction = ndvi.reshape(-1, 1) * b_ndbi
    design = np.column_stack([np.ones(len(ndvi)), b_ndbi, b_elev, ndvi, interaction])
    idx_ndvi_main = 1 + k1 + b_elev.shape[1]
    idx_interaction = list(range(idx_ndvi_main + 1, idx_ndvi_main + 1 + k1))
    return design, ndbi_spline, elev_spline, idx_ndvi_main, idx_interaction


def _fit_varying_coefficient(
    ndvi: np.ndarray, ndbi: np.ndarray, elevation: np.ndarray, lst: np.ndarray,
    ndbi_df: int, elev_df: int,
):
    """Fit the varying-coefficient model once and return everything needed
    to evaluate g(NDBI) = b_ndvi + spline_basis(NDBI) @ gamma on any grid."""
    design, ndbi_spline, elev_spline, idx_ndvi_main, idx_interaction = (
        _build_varying_coefficient_design(ndvi, ndbi, elevation, ndbi_df, elev_df, fit_splines=True)
    )
    model = LinearRegression(fit_intercept=False).fit(design, lst)
    coef = model.coef_
    return {
        "ndbi_spline": ndbi_spline,
        "b_ndvi": float(coef[idx_ndvi_main]),
        "gamma": coef[idx_interaction],
        "r_squared": float(model.score(design, lst)),
        "model": model,
    }


def _evaluate_g_of_ndbi(fit: dict, ndbi_grid: np.ndarray) -> np.ndarray:
    basis = fit["ndbi_spline"].transform(np.asarray(ndbi_grid).reshape(-1, 1))
    return fit["b_ndvi"] + basis @ fit["gamma"]


def select_ndbi_spline_df_by_block_cv(
    df_train: pd.DataFrame, cfg: Config,
) -> tuple[int, pd.DataFrame]:
    """Objective bandwidth/smoothing selection for g(NDBI): choose the NDBI
    spline degrees-of-freedom (ndbi_df) that minimises out-of-fold
    prediction RMSE under GroupKFold(spatial_block) cross-validation.

    This is an objective procedure over cfg.varying_coef_ndbi_df_grid - the
    value is never hand-picked to produce a particular curve shape. The
    Elevation spline's df (cfg.varying_coef_elev_df) is held fixed: it is a
    nuisance smooth control, not the quantity Figure 6 reports on, so it is
    not part of the search.
    """
    ndvi = df_train["NDVI"].to_numpy(dtype=float)
    ndbi = df_train["NDBI"].to_numpy(dtype=float)
    elev = df_train["Elevation"].to_numpy(dtype=float)
    lst  = df_train["LST"].to_numpy(dtype=float)
    blocks = df_train["spatial_block"].to_numpy()

    gkf = GroupKFold(n_splits=cfg.varying_coef_cv_folds)
    rows = []
    for ndbi_df in cfg.varying_coef_ndbi_df_grid:
        fold_rmse = []
        for train_idx, test_idx in gkf.split(ndvi, lst, groups=blocks):
            design_tr, sp_n, sp_e, idx_main, idx_inter = _build_varying_coefficient_design(
                ndvi[train_idx], ndbi[train_idx], elev[train_idx],
                ndbi_df, cfg.varying_coef_elev_df, fit_splines=True,
            )
            model = LinearRegression(fit_intercept=False).fit(design_tr, lst[train_idx])
            design_te, _, _, _, _ = _build_varying_coefficient_design(
                ndvi[test_idx], ndbi[test_idx], elev[test_idx],
                ndbi_df, cfg.varying_coef_elev_df,
                ndbi_spline=sp_n, elev_spline=sp_e, fit_splines=False,
            )
            pred = model.predict(design_te)
            fold_rmse.append(float(np.sqrt(np.mean((pred - lst[test_idx]) ** 2))))
        rows.append({
            "ndbi_df": ndbi_df,
            "cv_rmse": float(np.mean(fold_rmse)),
            "cv_rmse_se": float(np.std(fold_rmse) / np.sqrt(cfg.varying_coef_cv_folds)),
        })
    cv_table = pd.DataFrame(rows)
    selected_df = int(cv_table.loc[cv_table["cv_rmse"].idxmin(), "ndbi_df"])
    return selected_df, cv_table


def compute_varying_coefficient_sensitivity(
    df_train: pd.DataFrame, cfg: Config, ndbi_grid: np.ndarray, selected_df: int,
) -> pd.DataFrame:
    """Refit g(NDBI) at neighbouring spline df values (requirement 5) and
    report whether the peak location/value and the (first) zero-crossing
    persist - i.e. whether they are a genuine feature of the adjusted
    association rather than an artifact of one particular smoothing choice.
    """
    ndvi = df_train["NDVI"].to_numpy(dtype=float)
    ndbi = df_train["NDBI"].to_numpy(dtype=float)
    elev = df_train["Elevation"].to_numpy(dtype=float)
    lst  = df_train["LST"].to_numpy(dtype=float)

    rows = []
    for ndbi_df in sorted(set(cfg.varying_coef_sensitivity_df_grid) | {selected_df}):
        fit = _fit_varying_coefficient(ndvi, ndbi, elev, lst, ndbi_df, cfg.varying_coef_elev_df)
        g = _evaluate_g_of_ndbi(fit, ndbi_grid)
        peak_idx = int(np.argmax(g))
        sign_changes = np.where(np.diff(np.sign(g)) != 0)[0]
        if len(sign_changes):
            i = sign_changes[0]
            x0, x1, y0, y1 = ndbi_grid[i], ndbi_grid[i + 1], g[i], g[i + 1]
            zero_crossing = float(x0 - y0 * (x1 - x0) / (y1 - y0))
        else:
            zero_crossing = None
        rows.append({
            "ndbi_df": ndbi_df,
            "is_selected": ndbi_df == selected_df,
            "peak_ndbi": float(ndbi_grid[peak_idx]),
            "peak_g": float(g[peak_idx]),
            "zero_crossing_ndbi": zero_crossing,
        })
    return pd.DataFrame(rows)


def compute_ndvi_ndbi_varying_coefficient_gam(
    df_train: pd.DataFrame, cfg: Config,
) -> VaryingCoefficientResult:
    """Estimator behind the redesigned Figure 6: ONE continuous spline-basis
    varying-coefficient model,

        LST = f_NDBI(NDBI) + f_Elevation(Elevation) + NDVI * g(NDBI) + error

    fit once on data.df_train, replacing the four-independent-regressions
    estimator used by compute_conditional_ndvi_lst_slopes(). g(NDBI) is a
    single fitted curve (no imposed monotonic sign), so there is no
    artificial discontinuity at the NDBI quartile cut points - continuity
    comes from estimating one continuous response surface, not from
    smoothing/interpolating the four old per-bin slopes after the fact.

    Steps, each satisfying one of the stated requirements:
      1-2. NDBI and Elevation are controlled continuously via their own
           B-spline smooths; NDVI's coefficient is allowed to vary
           nonlinearly and continuously with NDBI via the spline-basis
           interaction (see _build_varying_coefficient_design).
      3.   95% CIs for g(NDBI) come from a spatial_block spatial-block bootstrap
           (resample unique blocks with replacement, refit, re-evaluate).
      4.   The NDBI spline's df is chosen by GroupKFold(spatial_block) CV RMSE
           (select_ndbi_spline_df_by_block_cv) - an objective procedure,
           not a hand-picked value.
      5.   compute_varying_coefficient_sensitivity() reports whether the
           peak and zero-crossing persist under neighbouring df choices.
      6.   Local observation/block support is computed on the same grid so
           thin-tail regions can be flagged rather than presented as
           reliable.

    Purely a diagnostic/reporting computation - does not refit or
    otherwise touch the production XGBoost model, NEGI formula, or either
    scenario trajectory. Not a causal estimate: an observational
    association with continuous (rather than binned) control for NDBI and
    Elevation; unmeasured confounding may remain.
    """
    ndvi = df_train["NDVI"].to_numpy(dtype=float)
    ndbi = df_train["NDBI"].to_numpy(dtype=float)
    elev = df_train["Elevation"].to_numpy(dtype=float)
    lst  = df_train["LST"].to_numpy(dtype=float)
    blocks = df_train["spatial_block"].to_numpy()

    selected_df, cv_table = select_ndbi_spline_df_by_block_cv(df_train, cfg)

    fit = _fit_varying_coefficient(ndvi, ndbi, elev, lst, selected_df, cfg.varying_coef_elev_df)

    grid_lo = np.percentile(ndbi, cfg.varying_coef_grid_percentile_lo)
    grid_hi = np.percentile(ndbi, cfg.varying_coef_grid_percentile_hi)
    ndbi_grid = np.linspace(grid_lo, grid_hi, cfg.varying_coef_grid_points)
    g_hat = _evaluate_g_of_ndbi(fit, ndbi_grid)

    half_width = cfg.varying_coef_support_window_halfwidth
    obs_support, block_support = [], []
    for x in ndbi_grid:
        mask = np.abs(ndbi - x) <= half_width
        obs_support.append(int(mask.sum()))
        block_support.append(int(len(np.unique(blocks[mask]))))
    obs_support = np.array(obs_support)
    block_support = np.array(block_support)
    low_support = block_support < cfg.varying_coef_low_support_blocks

    unique_blocks = np.unique(blocks)
    n_unique_blocks = len(unique_blocks)
    block_to_idx = {b: np.where(blocks == b)[0] for b in unique_blocks}
    rng = np.random.default_rng(cfg.random_seed)
    samples = np.empty((cfg.varying_coef_n_boot, len(ndbi_grid)))
    for i in range(cfg.varying_coef_n_boot):
        sampled_blocks = rng.choice(unique_blocks, size=n_unique_blocks, replace=True)
        idx = np.concatenate([block_to_idx[b] for b in sampled_blocks])
        boot_fit = _fit_varying_coefficient(
            ndvi[idx], ndbi[idx], elev[idx], lst[idx], selected_df, cfg.varying_coef_elev_df,
        )
        samples[i, :] = _evaluate_g_of_ndbi(boot_fit, ndbi_grid)
    ci_lower = np.percentile(samples, 2.5, axis=0)
    ci_upper = np.percentile(samples, 97.5, axis=0)

    sign_changes = np.where(np.diff(np.sign(g_hat)) != 0)[0]
    zero_crossings = []
    for i in sign_changes:
        x0, x1, y0, y1 = ndbi_grid[i], ndbi_grid[i + 1], g_hat[i], g_hat[i + 1]
        zc = float(x0 - y0 * (x1 - x0) / (y1 - y0))
        nearby = np.abs(ndbi_grid - zc) <= half_width
        zero_crossings.append({
            "ndbi": zc,
            "ci_excludes_zero_here": bool((ci_lower[i] > 0) or (ci_upper[i] < 0)
                                           or (ci_lower[i + 1] > 0) or (ci_upper[i + 1] < 0)),
            "block_support_nearby": int(block_support[nearby].min()) if nearby.any() else int(block_support[i]),
        })

    sensitivity_table = compute_varying_coefficient_sensitivity(df_train, cfg, ndbi_grid, selected_df)

    return VaryingCoefficientResult(
        n_obs=len(df_train),
        r_squared=fit["r_squared"],
        ndbi_df_selected=selected_df,
        elev_df_fixed=cfg.varying_coef_elev_df,
        cv_table=cv_table,
        ndbi_grid=ndbi_grid,
        g_hat=g_hat,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        obs_support=obs_support,
        block_support=block_support,
        low_support_block_threshold=cfg.varying_coef_low_support_blocks,
        low_support=low_support,
        zero_crossings=zero_crossings,
        n_boot=cfg.varying_coef_n_boot,
        n_unique_blocks=n_unique_blocks,
        sensitivity_table=sensitivity_table,
    )


def varying_coefficient_to_dataframe(result: VaryingCoefficientResult) -> pd.DataFrame:
    """Tabulate the fitted g(NDBI) curve, its spatial-block-bootstrap 95%
    CI, and local effective support for CSV export / plotting."""
    return pd.DataFrame({
        "NDBI":               result.ndbi_grid,
        "adjusted_NDVI_slope": result.g_hat,
        "boot_ci_lower":      result.ci_lower,
        "boot_ci_upper":      result.ci_upper,
        "ci_excludes_zero":   (result.ci_lower > 0) | (result.ci_upper < 0),
        "obs_support":        result.obs_support,
        "block_support":      result.block_support,
        "low_support":        result.low_support,
    })


def report_varying_coefficient_gam(
    result: VaryingCoefficientResult, cfg: Config, summary: SummaryLog,
) -> dict:
    """Log and record the continuous varying-coefficient Figure 6 estimator:
    bandwidth selection, fitted curve, CIs, zero-crossing(s), local support,
    and smoothing-sensitivity persistence. Purely descriptive reporting -
    does not refit anything and does not alter the NEGI/model/scenario
    pipeline (requirement 7)."""
    table = varying_coefficient_to_dataframe(result)
    save_csv(table, cfg.data_dir / "ndvi_adjusted_slope_curve_by_ndbi.csv", cfg=cfg, debug_only=True)
    save_csv(result.cv_table, cfg.data_dir / "ndbi_spline_df_cv_selection.csv", cfg=cfg, debug_only=True)
    save_csv(result.sensitivity_table, cfg.data_dir / "ndvi_adjusted_slope_sensitivity.csv",
             cfg=cfg, debug_only=True)

    zc_lines = "\n".join(
        f"    NDBI ≈ {zc['ndbi']:.4f}  (CI excludes zero at adjacent grid points: "
        f"{zc['ci_excludes_zero_here']}, nearby block support: {zc['block_support_nearby']})"
        for zc in result.zero_crossings
    ) or "    none in the evaluated range"

    persists = (
        result.sensitivity_table["zero_crossing_ndbi"].notna().sum()
        >= max(1, len(result.sensitivity_table) - 1)
    )
    sens_lines = "\n".join(
        f"    ndbi_df={row.ndbi_df:<3d}{'  [selected]' if row.is_selected else '':<12} "
        f"peak at NDBI={row.peak_ndbi:.4f} (g={row.peak_g:.2f}), "
        f"zero-crossing={'NDBI=' + format(row.zero_crossing_ndbi, '.4f') if row.zero_crossing_ndbi is not None else 'none'}"
        for row in result.sensitivity_table.itertuples()
    )

    interpretation = (
        f"Continuous varying-coefficient model (LST ~ f(NDBI) + f(Elevation) + "
        f"NDVI*g(NDBI), n={result.n_obs}, R²={result.r_squared:.3f}, NDBI spline "
        f"df={result.ndbi_df_selected} selected by {cfg.varying_coef_cv_folds}-fold "
        f"GroupKFold(spatial_block) CV RMSE): the adjusted NDVI-LST slope g(NDBI) is "
        f"positive across most of the observed NDBI range, peaks near NDBI≈"
        f"{result.ndbi_grid[int(np.argmax(result.g_hat))]:.3f}, and crosses zero at:\n"
        f"{zc_lines}\n"
        f"This replaces the four disconnected per-bin slopes in the old Figure 6 "
        f"with one continuous fitted curve, so there are no artificial "
        f"discontinuities at the quartile cut points. The peak location and "
        f"zero-crossing "
        f"{'persist' if persists else 'do NOT fully persist'} across neighbouring "
        f"smoothing choices:\n{sens_lines}\n"
        "This is an observational, continuously-adjusted association, not a "
        "causal effect; unmeasured confounding may remain, and the curve should "
        "not be read as reliable in NDBI regions flagged as low-support (see the "
        "block_support column of the exported table)."
    )
    log_info("\nContinuous varying-coefficient g(NDBI) (redesigned Figure 6):\n" + interpretation)
    summary.log("Interpretive Diagnostics", "Varying-coefficient NDBI spline df selected", result.ndbi_df_selected)
    summary.log("Interpretive Diagnostics", "Varying-coefficient zero-crossing(s)", zc_lines)
    summary.log("Interpretive Diagnostics", "Varying-coefficient smoothing sensitivity persists", bool(persists))
    summary.log("Interpretive Diagnostics", "Varying-coefficient interpretation note", interpretation)
    summary.add_conclusion(interpretation)

    return {
        "ndbi_df_selected": result.ndbi_df_selected,
        "zero_crossings":   result.zero_crossings,
        "sensitivity_persists": bool(persists),
        "interpretation":   interpretation,
    }


def check_varying_coefficient_negi_relevance(
    result: VaryingCoefficientResult, s1: "ScenarioTrajectory", s2: "ScenarioTrajectory",
    cfg: Config, summary: SummaryLog,
) -> dict:
    """Explicitly check whether the Figure 6 zero-crossing(s) fall inside
    the NDBI range either NEGI scenario trajectory actually visits.

    A sign reversal in the adjusted NDVI-LST association is only
    load-bearing for the current NEGI conclusions if Scenario 1 (NDBI
    fixed at baseline) or Scenario 2 (empirical greening trajectory) ever
    reach the NDBI region where g(NDBI) changes sign. Scenario 1 holds
    NDBI fixed at its observed median; Scenario 2 moves from that same
    baseline toward a GREENER, LOWER-NDBI archetype endpoint (see
    build_scenario2_empirical_trajectory) - i.e. away from, not toward,
    higher NDBI. This function checks that directly from the already-
    computed trajectories rather than assuming it.
    """
    s1_lo, s1_hi = float(np.min(s1.ndbi)), float(np.max(s1.ndbi))
    s2_lo, s2_hi = float(np.min(s2.ndbi)), float(np.max(s2.ndbi))
    combined_lo, combined_hi = min(s1_lo, s2_lo), max(s1_hi, s2_hi)

    crossings_in_range = [
        zc for zc in result.zero_crossings if combined_lo <= zc["ndbi"] <= combined_hi
    ]
    load_bearing = len(crossings_in_range) > 0

    interpretation = (
        f"Scenario 1 holds NDBI fixed at NDBI={s1_lo:.4f} (the observed baseline). "
        f"Scenario 2's empirical greening trajectory spans NDBI=[{s2_lo:.4f}, {s2_hi:.4f}], "
        f"moving from that same baseline toward a lower-NDBI (greener) archetype endpoint. "
        + (
            "One or more Figure 6 zero-crossings fall inside the NDBI range either "
            "scenario visits, so the adjusted-slope sign reversal IS load-bearing for "
            "the current NEGI conclusions and should be discussed alongside them."
            if load_bearing else
            "None of the Figure 6 zero-crossing(s) "
            f"({', '.join(f'{zc['ndbi']:.3f}' for zc in result.zero_crossings) or 'none found'}) "
            f"fall inside that range ([{combined_lo:.4f}, {combined_hi:.4f}]): both scenario "
            "trajectories stay in the region where the adjusted NDVI-LST slope is positive. "
            "The Figure 6 sign reversal is therefore scientifically real (see the CI and "
            "sensitivity checks above) but NOT load-bearing for the current NEGI "
            "conclusions - neither trajectory enters the region where the association "
            "reverses sign."
        )
    )
    log_info("\nFigure 6 relevance check against NEGI scenario trajectories:\n" + interpretation)
    summary.log("Interpretive Diagnostics", "Varying-coefficient zero-crossing load-bearing for NEGI", load_bearing)
    summary.log("Interpretive Diagnostics", "Varying-coefficient NEGI-relevance note", interpretation)
    summary.add_conclusion(interpretation)
    return {
        "s1_ndbi_range": (s1_lo, s1_hi),
        "s2_ndbi_range": (s2_lo, s2_hi),
        "load_bearing":  load_bearing,
        "interpretation": interpretation,
    }


def plot_ndvi_adjusted_slope_curve(
    result: VaryingCoefficientResult, cfg: Config,
    ndbi_bin_edges: tuple[float, float, float] = (0.054, 0.083, 0.104),
) -> Path:
    """Redesigned Figure 6: continuous adjusted NDVI-LST slope g(NDBI)
    across NDBI, from ONE varying-coefficient model - not four independent
    regressions. x = NDBI, y = adjusted slope (°C per unit NDVI), with a
    spatial-block-bootstrap 95% CI, a zero line, the old Low/Moderate/High/
    Very-high NDBI ranges shown only as reference shading (they no longer
    define separate models), and a support panel so thin-data tails are
    visibly, not just verbally, flagged.
    """
    grid, g_hat = result.ndbi_grid, result.g_hat
    low = result.low_support
    bin_colors = ["#cfe4f2", "#9fcbe8", "#5a9bd4", "#1c5b8c"]
    edges = [grid.min()] + list(ndbi_bin_edges) + [grid.max()]

    fig = plt.figure(figsize=(11, 8.5))
    grid_spec = fig.add_gridspec(2, 1, height_ratios=[3.6, 1], hspace=0.06)
    ax = fig.add_subplot(grid_spec[0])
    ax2 = fig.add_subplot(grid_spec[1], sharex=ax)

    for i in range(4):
        ax.axvspan(edges[i], edges[i + 1], color=bin_colors[i], alpha=0.18, lw=0)
        ax2.axvspan(edges[i], edges[i + 1], color=bin_colors[i], alpha=0.18, lw=0)
    for e in ndbi_bin_edges:
        ax.axvline(e, color=cfg.color_neutral, lw=0.8, ls=":")
        ax2.axvline(e, color=cfg.color_neutral, lw=0.8, ls=":")

    ax.fill_between(grid, result.ci_lower, result.ci_upper, where=~low,
                     color=cfg.color_accent, alpha=0.18,
                     interpolate=True, label=f"{result.n_boot}-draw spatial-block bootstrap 95% CI")
    ax.fill_between(grid, result.ci_lower, result.ci_upper, where=low,
                     color=cfg.color_accent, alpha=0.08, hatch="////", linewidth=0,
                     interpolate=True,
                     label=f"CI where local support < {result.low_support_block_threshold} blocks")
    g_hat_supported = np.where(~low, g_hat, np.nan)
    g_hat_thin = np.where(low, g_hat, np.nan)
    ax.plot(grid, g_hat_supported, color=cfg.color_accent, linewidth=2.4,
            label="Adjusted NDVI-LST slope  g(NDBI)")
    ax.plot(grid, g_hat_thin, color=cfg.color_accent, linewidth=2.4, linestyle="--")
    ax.axhline(0.0, color="black", linewidth=1.1)

    for zc in result.zero_crossings:
        ax.axvline(zc["ndbi"], color="black", lw=1.0, ls="-.", alpha=0.6)
        _zc_note = "" if zc["ci_excludes_zero_here"] else "\n(CI already spans zero here)"
        ax.annotate(f"fitted sign transition\nNDBI ≈ {zc['ndbi']:.3f}{_zc_note}", xy=(zc["ndbi"], 0),
                    xytext=(8, 30), textcoords="offset points", fontsize=8.5,
                    arrowprops=dict(arrowstyle="->", color=cfg.color_neutral, lw=0.8))

    set_axis_labels(ax, "", "Adjusted NDVI-LST slope, g(NDBI)  (°C per unit NDVI)")
    set_title(ax, "Continuously Adjusted NDVI-LST Association Across the NDBI Gradient\n"
                  "One varying-coefficient model: LST = f(NDBI) + f(Elevation) + NDVI·g(NDBI) + ε",
              fontsize=13)
    fig.text(0.5, 0.965,
              "g(NDBI) is one continuous fitted response surface evaluated across NDBI, not four "
              "separate regressions. Shaded bands mark the original quartile NDBI strata for "
              "reference only; dashed segments/hatched CI mark thin local support.",
              ha="center", fontsize=9, style="italic", color=cfg.color_neutral)
    set_legend(ax, loc="upper right", fontsize=8.5)
    apply_grid(ax)
    plt.setp(ax.get_xticklabels(), visible=False)

    ax2.bar(grid, result.block_support, width=(grid[1] - grid[0]) * 0.9,
            color=cfg.color_primary, alpha=0.85, label="unique training blocks (local support)")
    ax2.axhline(result.low_support_block_threshold, color=cfg.color_accent, lw=0.8, ls="--")
    set_axis_labels(ax2, "NDBI", "unique\nblocks")
    ax2.legend(loc="upper right", fontsize=7.5, frameon=True)
    apply_grid(ax2)

    _zc_caveat = ""
    if result.zero_crossings:
        _any_ci_spans_zero = any(not zc["ci_excludes_zero_here"] for zc in result.zero_crossings)
        _zc_caveat = (
            " The fitted curve's sign transition(s) are interpreted as model-estimated "
            "transitions rather than precisely identified thresholds"
            + (", since the bootstrap CI already spans zero near the crossing" if _any_ci_spans_zero else "")
            + "."
        )
    return save_fig(
        fig, FIG_NDVI_CONDITIONAL_BY_NDBI, cfg, use_tight_layout=False,
        caption=(
            "Adjusted NDVI-LST slope g(NDBI) from a single continuous spline-basis "
            "varying-coefficient model (LST ~ f(NDBI) + f(Elevation) + NDVI:spline(NDBI)), "
            f"NDBI spline df={result.ndbi_df_selected} selected by block-grouped CV RMSE, "
            f"with {result.n_boot}-draw spatial-block-bootstrap 95% CIs. Replaces the four "
            "independent per-bin regressions previously used for Figure 6; see Figure S20 "
            "for that unadjusted, purely descriptive comparison."
            + _zc_caveat
        ),
        section="Response curves",
    )


def log_ndvi_causal_caveat(
    fi: "FeatureImportanceResult", corr_ndvi_ndbi: float, cfg: Config, summary: SummaryLog,
    continuous_adjustment: Optional[dict] = None,
) -> None:
    """The pipeline already computes everything needed to
    know that NDVI's permutation importance and its correlation with NDBI
    (urban form) together mean NDVI should not be read as an independently
    identified cooling driver - but previously only printed the bare
    "Permutation importance" table/headline, leaving that connection for a
    reader to draw themselves. This reads the already-computed permutation
    importance table (`fi.table`) and the already-computed NDVI-NDBI
    Pearson r (`corr_ndvi_ndbi`) and, whenever NDVI has non-trivial
    importance AND is non-trivially correlated with NDBI, automatically
    logs the caveat alongside them. No new statistic is computed here -
    only whether to surface the caveat is decided, from numbers that
    already exist elsewhere in the report (the conditional-slope analysis
    in `summarize_conditional_ndvi_slope_signs` already substantiates the
    "explained by urban form" claim empirically, per-NDBI-stratum).

    NDVI-LST sign-reversal audit: `continuous_adjustment`, when supplied,
    is the dict returned by report_continuous_adjustment(). Previously this
    caveat only warned that NDVI's apparent cooling association might not
    survive stratification, without stating the stronger, now-quantified
    finding: once NDBI is held continuously constant (rather than binned),
    the implied NDVI-LST slope is consistently POSITIVE (warming), not
    merely weak or uncertain. When available, that quantified direction is
    stated explicitly instead of only pointing to the stratified table.
    """
    if "NDVI" not in fi.table["Feature"].values:
        return
    ndvi_importance = float(fi.table.loc[fi.table["Feature"] == "NDVI", "Importance"].iloc[0])
    max_importance = float(fi.table["Importance"].max())
    ndvi_has_importance = max_importance > 0 and ndvi_importance / max_importance > 0.05
    ndvi_ndbi_correlated = abs(corr_ndvi_ndbi) > 0.3

    if ndvi_has_importance and ndvi_ndbi_correlated:
        caveat = (
            "NDVI should not be interpreted as an independently identified "
            "cooling driver. NDVI and NDBI (urban form) are correlated "
            f"(r = {corr_ndvi_ndbi:.3f}). The unadjusted within-NDBI-stratum "
            "slopes (see 'Conditional NDVI-NDBI slope sign pattern' below) "
            "retain residual continuous NDBI confounding and should not be "
            "read as partial or causal NDVI effects."
        )
        if continuous_adjustment is not None and continuous_adjustment.get("sign_consistent"):
            caveat += (
                " Continuously adjusting for NDBI and Elevation instead of "
                f"binning it gives a {continuous_adjustment['sign_pattern']} "
                "implied NDVI-LST slope across the observed NDBI range "
                f"({continuous_adjustment['slope_min']:.2f} to "
                f"{continuous_adjustment['slope_max']:.2f} °C per unit NDVI) - "
                "i.e. an isolated NDVI increase is associated with warming, "
                "not cooling, once urban form is held constant. Neither the "
                "unadjusted nor the adjusted analysis establishes a causal "
                "effect; unmeasured confounding may remain."
            )
        else:
            caveat += (
                " A continuous NDBI+Elevation adjustment (rather than "
                "stratification) is needed to assess how much of NDVI's "
                "apparent cooling association survives once urban form is "
                "held constant."
            )
        log_info("\n" + caveat)
        summary.log("Interpretive Diagnostics", "NDVI overinterpretation caveat", caveat)


def summarize_conditional_ndvi_slope_signs(
    conditional_df: pd.DataFrame, summary: SummaryLog
) -> dict:
    """Purely descriptive diagnostic on the conditional NDVI-LST slopes
    already computed per NDBI stratum (see compute_conditional_ndvi_lst_slopes /
    conditional_slopes_to_dataframe, enhanced via enhance_conditional_slopes_table,
    exported as ndvi_conditional_slopes_by_ndbi.csv and drawn in Figure 6). This
    function only reads the already-computed
    per-stratum slopes - it does not refit any regression, alter any bin
    definition, or touch the NEGI/model/scenario/uncertainty/QA logic.

    Reports, and adds to the run's summary log/report:
      * whether the conditional slopes are consistently negative, consistently
        positive, or change sign across NDBI strata,
      * the range (min to max) of the conditional slopes,
      * a cautionary interpretation note that the marginal (unconditional)
        NDVI-LST relationship should be read cautiously when the conditional
        relationships differ, because NDVI and NDBI are correlated.
    """
    if conditional_df is None or len(conditional_df) == 0:
        log_info(
            "\nConditional NDVI-by-NDBI slope sign summary: no NDBI strata had "
            "enough pixels for a conditional slope - summary skipped."
        )
        return {}

    slopes = conditional_df["slope"].to_numpy(dtype=float)
    n_total    = len(slopes)
    n_negative = int(np.sum(slopes < 0))
    n_positive = int(np.sum(slopes > 0))
    sign_consistent = (n_negative == n_total) or (n_positive == n_total)
    sign_pattern = (
        "consistently negative" if n_negative == n_total else
        "consistently positive" if n_positive == n_total else
        "mixed (sign changes across NDBI strata)"
    )
    slope_min, slope_max = float(slopes.min()), float(slopes.max())

    interpretation = (
        f"Unadjusted within-NDBI-stratum NDVI-LST associations are "
        f"{sign_pattern} across the {n_total} evaluated NDBI strata "
        f"(range: {slope_min:.4f} to {slope_max:.4f} °C per unit NDVI). "
        "These are single-predictor (NDVI-only) regressions fit within "
        "each NDBI bin; equal-count binning holds NDBI only "
        "approximately constant, so substantial continuous NDBI "
        "variation can remain inside a bin and bias these slopes. "
    )
    if not sign_consistent:
        interpretation += (
            "The sign changes across NDBI strata seen here should NOT be "
            "read as a conditional, partial, or causal NDVI effect, or as "
            "evidence that NDVI itself causes warming or cooling: broad "
            "NDBI stratification does not fully control continuous NDBI "
            "variation within each bin, and that residual within-bin NDBI "
            "confounding can itself produce an apparent sign reversal. See "
            "compute_ndvi_ndbi_continuous_adjustment() for a diagnostic "
            "that adjusts for NDBI (and Elevation) continuously instead of "
            "by binning."
        )
    else:
        interpretation += (
            "The marginal (unconditional) NDVI-LST relationship should still "
            "be interpreted cautiously to the extent these unadjusted "
            "slopes differ in magnitude across strata, since NDVI and NDBI "
            "are correlated and broad NDBI stratification does not fully "
            "control continuous NDBI variation within each bin."
        )

    log_info(
        "\nUnadjusted within-NDBI-stratum NDVI-LST slope sign summary:\n"
        f"  Strata evaluated : {n_total}\n"
        f"  Sign pattern     : {sign_pattern}\n"
        f"  Slope range      : {slope_min:.4f} to {slope_max:.4f}\n"
        f"  -> {interpretation}"
    )
    summary.log("Interpretive Diagnostics", "Unadjusted within-NDBI-stratum NDVI slope sign pattern", sign_pattern)
    summary.log("Interpretive Diagnostics", "Unadjusted within-NDBI-stratum NDVI slope range",
                f"{slope_min:.4f} to {slope_max:.4f}")
    summary.log("Interpretive Diagnostics", "Unadjusted within-NDBI-stratum NDVI interpretation note", interpretation)
    summary.add_conclusion(interpretation)

    return {
        "n_strata":        n_total,
        "n_negative":      n_negative,
        "n_positive":      n_positive,
        "sign_consistent": sign_consistent,
        "sign_pattern":    sign_pattern,
        "slope_min":       slope_min,
        "slope_max":       slope_max,
        "interpretation":  interpretation,
    }


def compute_ndbi_response_sweep(model_results: ModelResults, cfg: Config) -> dict:
    """Sweep NDBI at fixed NDVI and fixed Elevation/ST_EMIS/ST_EMSD to trace
    the predicted LST response curve."""
    df              = model_results.data.df
    ndbi_median_val = df["NDBI"].median()
    ndbi_range = np.linspace(
        df["NDBI"].quantile(cfg.ndbi_sweep_quantile_low),
        df["NDBI"].quantile(cfg.ndbi_sweep_quantile_high),
        cfg.ndbi_sweep_points,
    )
    sweep_df = build_feature_frame(
        np.full(cfg.ndbi_sweep_points, df["NDVI"].median()),
        ndbi_range,
        fixed_reference_values(df),
    )
    pred_raw    = model_results.model.predict(sweep_df)
    pred_smooth = savgol_filter(pred_raw, 11, 2)
    return {
        "ndbi_range":   ndbi_range,
        "ndbi_median":  ndbi_median_val,
        "pred_raw":     pred_raw,
        "pred_smooth":  pred_smooth,
    }


def compute_response_surface(
    model_results: ModelResults, cfg: Config, grid_size: int = 150
) -> dict:
    """Compute the predicted-LST response surface over the NDVI-NDBI feature grid."""
    df              = model_results.data.df
    ndvi_max_plot   = np.percentile(df["NDVI"], 99)
    ndvi_grid = np.linspace(df["NDVI"].min(), ndvi_max_plot, grid_size)
    ndbi_grid = np.linspace(df["NDBI"].min(), df["NDBI"].max(), grid_size)
    ndvi_mesh, ndbi_mesh = np.meshgrid(ndvi_grid, ndbi_grid)
    grid_df  = build_feature_frame(
        ndvi_mesh.ravel(), ndbi_mesh.ravel(), fixed_reference_values(df),
    )
    lst_mesh = model_results.model.predict(grid_df).reshape(ndvi_mesh.shape)
    return {
        "ndvi_mesh":    ndvi_mesh,
        "ndbi_mesh":    ndbi_mesh,
        "lst_mesh":     lst_mesh,
        "ndvi_max_plot": ndvi_max_plot,
    }


def compute_ndvi_decile_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise LST statistics by NDVI decile."""
    return (
        df.assign(NDVI_bin=pd.qcut(df["NDVI"], 10, duplicates="drop"))
        .groupby("NDVI_bin", observed=True)
        .agg(
            NDVI_median=("NDVI", "median"),
            LST_median=("LST", "median"),
            LST_mean=("LST", "mean"),
            LST_std=("LST", "std"),
            NDBI_median=("NDBI", "median"),
            n_pixels=("LST", "size"),
        )
        .reset_index(drop=True)
    )


def compute_saturation_diagnostic(
    negi: NEGIResults, s2: ScenarioTrajectory, cfg: Config
) -> dict:
    """Diagnose cooling saturation behaviour at high, mid, and near-zero NDVI fractions."""
    max_pos = negi.delta_t_s2.max()
    if max_pos > EPS:
        frac = safe_divide(negi.delta_t_s2, max_pos)
        sat_90_idx = int(np.argmax(frac >= cfg.saturation_high_fraction))
        sat_50_idx = int(np.argmax(frac >= cfg.saturation_mid_fraction))
        near_zero  = cfg.saturation_near_zero_fraction * max_pos
        candidates = np.where(negi.delta_t_s2 <= near_zero)[0]
        collapse_idx = (
            int(candidates[0])
            if len(candidates) > 0 and candidates[0] > sat_90_idx else None
        )
    else:
        sat_50_idx = sat_90_idx = collapse_idx = None
    return {
        "sat_50_idx":    sat_50_idx,
        "sat_90_idx":    sat_90_idx,
        "collapse_idx":  collapse_idx,
    }





def enhance_conditional_slopes_table(
    conditional_df: pd.DataFrame, results: list[ConditionalBinResult], cfg: Config
) -> pd.DataFrame:
    """Add Adjusted R^2, Pearson r, and a variance-explained interpretation
    band to the existing conditional-NDVI-slope table (Figure 6).  The
    underlying per-stratum regressions (results) are unchanged."""
    enhanced = conditional_df.copy()
    adj_r2_col, r_col, band_col, note_col = [], [], [], []
    for _, row in enhanced.iterrows():
        matching = next(
            (r for r in results
             if r.slope is not None and r.label.replace("\n", " ") == row["NDBI_range"]),
            None,
        )
        n = int(row["n_pixels"])
        r2 = float(row["r_squared"])
        adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / max(n - 2, 1)
        r_value = float(np.sign(row["slope"])) * np.sqrt(max(r2, 0.0))

        if r2 < cfg.variance_band_very_weak:
            band = "Very weak (<5%)"
        elif r2 < cfg.variance_band_weak:
            band = "Weak (5-15%)"
        elif r2 < cfg.variance_band_moderate:
            band = "Moderate"
        else:
            band = "Strong"

        note = (
            "Although statistically significant, NDVI explains only a small "
            "proportion of within-stratum LST variability.  Other environmental "
            "variables contribute substantially."
            if r2 < cfg.low_r2_threshold else ""
        )
        adj_r2_col.append(adj_r2); r_col.append(r_value)
        band_col.append(band); note_col.append(note)

    enhanced["Adjusted_R_squared"]       = adj_r2_col
    enhanced["Correlation_coefficient"]  = r_col
    enhanced["Variance_explained_band"]  = band_col
    enhanced["Low_R2_note"]              = note_col
    return enhanced


def report_conditional_slopes_enhanced(
    enhanced_df: pd.DataFrame, cfg: Config, summary: SummaryLog
) -> None:
    """Log and record the conditional NDVI-LST slope diagnostics with confidence intervals."""
    save_csv(enhanced_df, cfg.data_dir / "ndvi_conditional_slopes_by_ndbi.csv", cfg=cfg, debug_only=True)
    for _, row in enhanced_df.iterrows():
        log_info(
            f"\nNDBI stratum {row['NDBI_range']}: slope={row['slope']:.4f}, "
            f"95% CI=±{row['slope_95CI']:.4f}, R²={row['r_squared']:.4f}, "
            f"Adj. R²={row['Adjusted_R_squared']:.4f}, r={row['Correlation_coefficient']:.4f}, "
            f"n={row['n_pixels']}, variance explained: {row['Variance_explained_band']}"
        )
        if row["Low_R2_note"]:
            log_info(f"  -> {row['Low_R2_note']}")
        summary.log(
            "Interpretive Diagnostics",
            f"NDBI {row['NDBI_range']}: variance explained",
            row["Variance_explained_band"],
        )


def compute_ndbi_binning_robustness(
    df_train: pd.DataFrame, cfg: Config, summary: SummaryLog,
    quartile_results: Optional[list[ConditionalBinResult]] = None,
) -> dict:
    """Robustness check on Figure 6: is the unadjusted within-NDBI-stratum
    NDVI-LST slope pattern (sign / magnitude per bin) an artifact of
    choosing quartile bins specifically, or does it hold up under a
    different, equally-defensible equal-count binning (terciles)?

    This does NOT change the primary quartile-based Figure 6 result - it
    reruns the same per-stratum, single-predictor OLS
    (compute_conditional_ndvi_lst_slopes) on an independent 3-bin
    (tercile) stratification of the same data.df_train and reports both
    side by side. If the sign pattern (e.g. "mixed") is stable across both
    binning choices, that only rules out the specific bin edges chosen as
    the cause - it is NOT evidence of a genuine (adjusted/causal) NDVI-LST
    relationship, since both binnings can share the same residual
    continuous NDBI confounding (see
    compute_ndvi_ndbi_continuous_adjustment() for that adjustment). If the
    pattern changes across binnings, that itself is useful information
    about how sensitive the unadjusted analysis is to bin-count choice,
    and is reported as such - the goal is an honest robustness check, not
    a predetermined answer.
    """
    if quartile_results is None:
        quartile_bins = build_ndbi_quantile_bins(
            df_train, 4, ["Low urban", "Moderate urban", "High urban", "Very high urban"]
        )
        quartile_results = compute_conditional_ndvi_lst_slopes(df_train, quartile_bins)
    quartile_df = conditional_slopes_to_dataframe(quartile_results)

    tercile_bins = build_ndbi_quantile_bins(
        df_train, 3, ["Low urban", "Moderate urban", "High urban"]
    )
    tercile_results = compute_conditional_ndvi_lst_slopes(df_train, tercile_bins)
    tercile_df = conditional_slopes_to_dataframe(tercile_results)

    quartile_df = quartile_df.copy()
    tercile_df = tercile_df.copy()
    quartile_df["binning"] = "quartile (4 bins)"
    tercile_df["binning"] = "tercile (3 bins)"

    quartile_sign = _slope_sign_pattern(quartile_df)
    tercile_sign = _slope_sign_pattern(tercile_df)

    comparison_df = pd.concat([quartile_df, tercile_df], ignore_index=True)
    save_csv(
        comparison_df,
        cfg.data_dir / "ndvi_conditional_slopes_binning_robustness.csv",
        cfg=cfg, debug_only=True,
    )

    stable = quartile_sign == tercile_sign
    note = (
        f"Binning-choice robustness check: quartile stratification (4 bins) gives a "
        f"'{quartile_sign}' unadjusted sign pattern; tercile stratification (3 bins) "
        f"gives a '{tercile_sign}' unadjusted sign pattern. "
        + (
            "The sign pattern is STABLE across binning choices - i.e. not an artifact "
            "of picking quartiles specifically. This does NOT establish that the "
            "pattern is a genuine (adjusted or causal) NDVI-LST effect: both binnings "
            "can share the same residual continuous NDBI confounding within bins; see "
            "compute_ndvi_ndbi_continuous_adjustment() for the continuously-adjusted "
            "diagnostic."
            if stable else
            "The sign pattern CHANGES with the binning choice - the unadjusted "
            "within-stratum NDVI-LST association in the intermediate NDBI range is "
            "sensitive to how finely NDBI is stratified, most likely because narrower "
            "NDBI bins leave little within-bin NDVI variation for the regression to "
            "resolve against real noise/confounders."
        )
    )
    log_info("\n" + note)
    summary.log("Interpretive Diagnostics", "NDBI binning robustness (quartile vs tercile)", note)

    return {
        "quartile_results": quartile_results,
        "tercile_results": tercile_results,
        "quartile_df": quartile_df,
        "tercile_df": tercile_df,
        "quartile_sign_pattern": quartile_sign,
        "tercile_sign_pattern": tercile_sign,
        "stable_across_binning": stable,
        "note": note,
    }


def _slope_sign_pattern(conditional_df: pd.DataFrame) -> str:
    """Same sign-pattern classification as summarize_conditional_ndvi_slope_signs,
    factored out so it can be applied to the tercile comparison table without
    duplicating (or double-logging through) the full summary/report side effects."""
    if conditional_df is None or len(conditional_df) == 0:
        return "undefined (no strata with enough pixels)"
    slopes = conditional_df["slope"].to_numpy(dtype=float)
    n_total = len(slopes)
    n_negative = int(np.sum(slopes < 0))
    n_positive = int(np.sum(slopes > 0))
    if n_negative == n_total:
        return "consistently negative"
    if n_positive == n_total:
        return "consistently positive"
    return "mixed (sign changes across NDBI strata)"


def compute_qq_summary(residuals: np.ndarray, cfg: Config) -> dict:
    """Descriptive summary of the existing residual Q-Q plot: how much of
    the residual distribution falls outside the normal reference
    envelopes, and whether deviation is tail-symmetric.  Residuals
    themselves are not altered."""
    arr = np.asarray(residuals, dtype=float)
    n = len(arr)
    standardized = safe_divide(arr - arr.mean(), np.array([arr.std(ddof=1)]), fill=0.0)
    sorted_std = np.sort(standardized)
    theoretical = scipy_stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)
    deviation = sorted_std - theoretical

    low_pct, high_pct = cfg.qq_envelope_percentiles
    z_low  = scipy_stats.norm.ppf(0.5 + low_pct / 200.0)
    z_high = scipy_stats.norm.ppf(0.5 + high_pct / 200.0)
    frac_outside_low  = float(np.mean(np.abs(standardized) > z_low))
    frac_outside_high = float(np.mean(np.abs(standardized) > z_high))

    max_abs_dev = float(np.max(np.abs(deviation)))
    left_dev  = float(np.mean(np.abs(deviation[theoretical < 0])))  if np.any(theoretical < 0)  else float("nan")
    right_dev = float(np.mean(np.abs(deviation[theoretical > 0])))  if np.any(theoretical > 0)  else float("nan")
    tail_asymmetry = (
        right_dev - left_dev if np.isfinite(left_dev) and np.isfinite(right_dev) else float("nan")
    )

    shapiro_p = float(scipy_stats.shapiro(arr)[1]) if n <= 5000 else float("nan")
    if np.isfinite(shapiro_p) and shapiro_p < 0.05:
        interpretation = (
            "Residuals deviate significantly from normality (Shapiro-Wilk "
            "p < 0.05); prediction intervals rely on the bootstrap procedure "
            "rather than a normality assumption."
        )
    else:
        interpretation = (
            "No strong evidence against approximate normality of residuals "
            "at the 0.05 level."
        )

    qq_df = pd.DataFrame({
        "Theoretical_Quantile": theoretical,
        "Observed_Quantile": sorted_std,
        "Deviation": deviation,
    })
    return {
        "frac_outside_low": frac_outside_low, "frac_outside_high": frac_outside_high,
        "low_pct": low_pct, "high_pct": high_pct,
        "max_abs_deviation": max_abs_dev, "tail_asymmetry": tail_asymmetry,
        "shapiro_p": shapiro_p, "interpretation": interpretation,
        "qq_df": qq_df,
    }


def report_qq_summary(residuals: np.ndarray, cfg: Config, summary: SummaryLog) -> dict:
    """Log and record the QQ-plot tail-symmetry diagnostic summary."""
    qq = compute_qq_summary(residuals, cfg)
    save_csv(qq["qq_df"], cfg.data_dir / "qq_summary.csv", cfg=cfg, debug_only=True)
    log_info(
        f"\nQ-Q diagnostic summary:\n"
        f"  Fraction outside {qq['low_pct']:.0f}% envelope : {qq['frac_outside_low']:.4f}\n"
        f"  Fraction outside {qq['high_pct']:.0f}% envelope : {qq['frac_outside_high']:.4f}\n"
        f"  Maximum absolute deviation           : {qq['max_abs_deviation']:.4f}\n"
        f"  Tail asymmetry (right - left)         : {qq['tail_asymmetry']:.4f}\n"
        f"  {qq['interpretation']}"
    )
    summary.log("Interpretive Diagnostics", f"QQ: fraction outside {qq['low_pct']:.0f}% envelope",
                f"{qq['frac_outside_low']:.4f}")
    summary.log("Interpretive Diagnostics", f"QQ: fraction outside {qq['high_pct']:.0f}% envelope",
                f"{qq['frac_outside_high']:.4f}")
    summary.log("Interpretive Diagnostics", "QQ: maximum absolute deviation",
                f"{qq['max_abs_deviation']:.4f}")
    summary.log("Interpretive Diagnostics", "QQ: tail asymmetry (right - left)",
                f"{qq['tail_asymmetry']:.4f}")
    summary.log("Interpretive Diagnostics", "QQ: normality interpretation", qq["interpretation"])
    return qq




def _lowess_smooth(
    x: np.ndarray, y: np.ndarray, frac: float = 0.3, n_points: int = 100
) -> tuple[np.ndarray, np.ndarray]:
    """LOWESS smoother; uses statsmodels when available, otherwise a
    from-scratch tricube-weighted local linear fallback."""
    order  = np.argsort(x)
    x_s, y_s = x[order], y[order]
    try:
        import importlib
        sm_lowess = importlib.import_module(
            "statsmodels.nonparametric.smoothers_lowess"
        ).lowess
        smoothed = sm_lowess(y_s, x_s, frac=frac, return_sorted=True)
        return smoothed[:, 0], smoothed[:, 1]
    except ImportError:
        n   = len(x_s)
        k   = max(int(frac * n), 5)
        eval_x = np.linspace(x_s.min(), x_s.max(), n_points)
        eval_y = np.empty_like(eval_x)
        for i, x0 in enumerate(eval_x):
            dist = np.abs(x_s - x0)
            idx  = np.argpartition(dist, min(k, n - 1))[:k]
            d    = dist[idx]
            bw   = d.max() if d.max() > 0 else 1.0
            w    = (1 - np.clip(d / bw, 0, 1) ** 3) ** 3
            X_d  = np.column_stack([np.ones(len(idx)), x_s[idx]])
            try:
                beta     = np.linalg.lstsq(np.diag(w) @ X_d,
                                            np.diag(w) @ y_s[idx], rcond=None)[0]
                eval_y[i] = beta[0] + beta[1] * x0
            except np.linalg.LinAlgError:
                eval_y[i] = np.average(y_s[idx], weights=w)
        return eval_x, eval_y


def _breusch_pagan_test(
    y_true: np.ndarray, y_pred: np.ndarray, residuals: np.ndarray
) -> dict:
    """Breusch-Pagan test; uses statsmodels when available."""
    try:
        import importlib
        _t_import = time.perf_counter()
        log_info("  [TIMING] _breusch_pagan_test: importing statsmodels.api...")
        sm = importlib.import_module("statsmodels.api")
        het_bp = importlib.import_module(
            "statsmodels.stats.diagnostic"
        ).het_breuschpagan
        log_info(
            f"  [TIMING] _breusch_pagan_test: statsmodels import took "
            f"{time.perf_counter() - _t_import:.2f}s"
        )
        exog = sm.add_constant(y_pred)
        lm_stat, lm_p, _, _ = het_bp(residuals, exog)
        return {
            "statistic": float(lm_stat), "p_value": float(lm_p),
            "method": "statsmodels het_breuschpagan",
        }
    except ImportError:
        bp_aux   = LinearRegression().fit(y_pred.reshape(-1, 1), residuals ** 2)
        bp_fitted = bp_aux.predict(y_pred.reshape(-1, 1))
        ss_reg   = np.sum((bp_fitted - np.mean(residuals ** 2)) ** 2)
        ss_tot   = np.sum((residuals ** 2 - np.mean(residuals ** 2)) ** 2)
        r2_aux   = safe_divide(np.array([ss_reg]), np.array([ss_tot]), fill=0.0)[0]
        stat     = len(residuals) * r2_aux
        p        = 1 - scipy_stats.chi2.cdf(stat, df=1)
        return {
            "statistic": float(stat), "p_value": float(p),
            "method": "auxiliary-regression approximation (statsmodels unavailable)",
        }


def annotation_caption(
    annotations: dict, sep: str = "  |  ", wrap_width: int = 110
) -> str:
    """Build the NEGI comparison figure footer caption string."""
    if not np.isfinite(annotations["first_cooling_pct"]):
        first_cooling = "S2 1st cooling=none"
    elif annotations["first_cooling_pct"] <= 0.0:
        first_cooling = "S2 1st cooling=baseline (no credited cooling)"
    else:
        first_cooling = f"S2 1st cooling={annotations['first_cooling_pct']:.1f}%"
    _s2_robust = annotations.get("max_negi_s2_robust")
    _traj_max_tag = " [NON-ROBUST]" if _s2_robust is False else ""
    parts = [
        f"Baseline={annotations['baseline_lst']:.2f}°C",
        first_cooling,
        f"Max cooling={annotations['max_cooling_c']:.2f}°C",
        f"Trajectory max={annotations['max_negi_s2_value']:.3f} @ "
        f"{annotations['max_negi_s2_pct']:.0f}%{_traj_max_tag}",
        f"Calibration: pred={annotations['calibration_slope']:.3f}*obs"
        f"{annotations['calibration_intercept']:+.3f}",
        f"Support: {annotations['feature_support_status']}",
    ]
    joined = sep.join(parts)
    if not wrap_width:
        return joined
    return "\n".join(
        textwrap.wrap(joined, width=wrap_width,
                      break_long_words=False, break_on_hyphens=False)
    )


def plot_actual_vs_predicted(
    validation: ValidationResults, data: DatasetBundle, cfg: Config
) -> Path:
    """Plot observed vs. predicted LST for the confirmatory holdout."""
    y_test, pred = data.y_test, validation.holdout_pred
    slope, intercept = validation.calibration_slope, validation.calibration_intercept

    fig, ax = plt.subplots(figsize=cfg.default_figsize)
    ax.scatter(y_test, pred, alpha=0.35, s=14, color=cfg.color_primary, edgecolor="none")
    lims = [y_test.min(), y_test.max()]
    ax.plot(lims, lims, color=cfg.color_neutral, linewidth=1.4, label="1:1 line", zorder=3)
    cal_x = np.linspace(y_test.min(), y_test.max(), 100)
    ax.plot(cal_x, slope * cal_x + intercept,
            color=cfg.color_accent, linewidth=1.6, linestyle="--",
            label=f"Calibration (slope = {slope:.3f})")
    ax.set_aspect("equal", adjustable="box")

    stat_box(ax, (
        f"n = {len(y_test)}\n"
        f"CV R² = {validation.nested_cv_r2:.3f} (nested, non-spatial)\n"
        f"Holdout R² = {validation.holdout_r2:.3f} (single holdout)\n"
        f"Spatial holdout R² = {validation.repeated_r2_mean:.3f} "
        f"± {validation.repeated_r2_ci95:.3f} (repeated, spatial-block)\n"
        f"RMSE = {validation.holdout_rmse:.2f} °C\n"
        f"Calibration: slope = {slope:.3f}, intercept = {intercept:.2f} °C"
    ))
    set_title(ax, "Observed vs Predicted LST")
    set_axis_labels(ax, "Observed LST (°C)", "Predicted LST (°C)")
    set_legend(ax, loc="lower right")
    return save_fig(fig, FIG_ACTUAL_VS_PREDICTED, cfg,
                    caption=(
                        "Observed vs predicted LST with 1:1 and calibration lines. "
                        f"MAE = {validation.holdout_mae:.2f} °C; "
                        f"bias = {validation.calibration_bias:.2f} °C; "
                        f"calibration intercept = {intercept:.2f} °C. "
                        "Nested CV was used for model/hyperparameter selection (non-spatial); "
                        "the repeated spatial-block holdout is the primary estimate of "
                        "geographic generalization and is reported separately because spatial "
                        "dependence is non-trivial here (see Methods). "
                        "Calibration diagnostics are reported descriptively only; "
                        "no post-hoc recalibration was applied."
                    ))


def plot_model_comparison(
    comparison_df: pd.DataFrame, validation: ValidationResults, cfg: Config
) -> Path:
    """Plot the benchmark-model R² comparison bar chart."""
    fig, ax = plt.subplots(figsize=cfg.default_figsize)
    bar_colors = [cfg.color_neutral, cfg.color_secondary, cfg.color_primary, cfg.color_accent]
    bars = ax.bar(comparison_df["Model"], comparison_df["R2"],
                  color=bar_colors, width=0.55, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, comparison_df["R2"]):
        ax.annotate(
            f"{val:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4), textcoords="offset points",
            ha="center", fontsize=10, fontweight="bold", color=cfg.color_neutral,
        )
    set_axis_labels(ax, ylabel="R²")
    ax.set_ylim(0, comparison_df["R2"].max() * 1.18)
    set_title(ax, f"Model Comparison\nSpatial holdout R² = {validation.holdout_r2:.3f}")
    plt.xticks(rotation=30, ha="right")
    apply_grid(ax)
    return save_fig(fig, FIG_MODEL_COMPARISON, cfg,
                    caption="Benchmark model comparison by spatial holdout R².",
                    section="Model comparison")


def plot_residuals_histogram(resid: ResidualDiagnostics, cfg: Config) -> Path:
    """Single consolidated histogram figure (Gaussian fit + KDE)."""
    arr          = resid.residuals
    mu, sigma    = np.mean(arr), np.std(arr, ddof=1)
    grid         = np.linspace(arr.min(), arr.max(), 300)
    gaussian_pdf = scipy_stats.norm.pdf(grid, loc=mu, scale=sigma)
    kde_curve    = scipy_stats.gaussian_kde(arr)(grid)

    fig, ax = plt.subplots(figsize=cfg.default_figsize)
    ax.hist(arr, bins=30, color=cfg.color_primary, edgecolor="white",
            alpha=0.85, density=True, label="Residuals")
    ax.plot(grid, gaussian_pdf, color=cfg.color_accent, linewidth=2.0,
            linestyle="--", label="Gaussian fit")
    ax.plot(grid, kde_curve, color=cfg.color_neutral, linewidth=2.0, label="KDE")
    ax.axvline(0, color=cfg.color_accent, linestyle=":", linewidth=1.2)
    set_title(ax, "Residual Distribution")
    set_axis_labels(ax, "Residual (°C)", "Density")
    set_legend(ax)
    apply_grid(ax)

    stats_d = compute_residual_stats(arr)
    figure_footer(fig, (
        f"mean={stats_d['mean']:.3f}  median={stats_d['median']:.3f}  "
        f"std={stats_d['std']:.3f}  skew={stats_d['skewness']:.3f}  "
        f"kurtosis={stats_d['kurtosis']:.3f}  Shapiro p={stats_d['shapiro_p']:.3g}"
    ), cfg)
    return save_fig(fig, FIG_RESIDUALS_HISTOGRAM, cfg,
                    caption="Holdout residual distribution with Gaussian fit and KDE.",
                    section="Supplementary diagnostics")


def plot_residuals_vs_predicted_full(
    resid: ResidualDiagnostics, pred: np.ndarray,
    cfg: Config, summary: SummaryLog, registry=None,
) -> Path:
    """Residuals vs predicted with LOWESS smoother and Breusch-Pagan test."""
    arr          = resid.residuals
    lowess_x, lowess_y = _lowess_smooth(pred, arr)
    bp = _breusch_pagan_test(np.zeros_like(pred), pred, arr)
    summary.log("Residual diagnostics", "Breusch-Pagan statistic",  f"{bp['statistic']:.4f}")
    summary.log("Residual diagnostics", "Breusch-Pagan p-value",    f"{bp['p_value']:.4g}")
    log_info(
        f"\nBreusch-Pagan test ({bp['method']}): "
        f"statistic = {bp['statistic']:.4f}, p = {bp['p_value']:.4g}"
    )

    fig, ax = plt.subplots(figsize=cfg.default_figsize)
    ax.scatter(pred, arr, alpha=0.35, s=14, color=cfg.color_primary,
               edgecolor="none", label="Residuals")
    ax.axhline(0, color=cfg.color_neutral, linewidth=1.2)
    ax.plot(lowess_x, lowess_y, color=cfg.color_accent, linewidth=2.0, label="LOWESS")
    set_title(ax, "Residuals vs Predicted LST")
    set_axis_labels(ax, "Predicted LST (°C)", "Residual (°C)")
    set_legend(ax)
    apply_grid(ax)
    figure_footer(fig, (
        f"Breusch-Pagan: statistic={bp['statistic']:.3f}, "
        f"p={bp['p_value']:.3g} ({bp['method']})"
    ), cfg)
    return save_fig(fig, FIG_RESIDUALS_VS_PREDICTED, cfg,
                    caption="Holdout residuals vs predicted LST with LOWESS smoother.",
                    section="Supplementary diagnostics")


def plot_residuals_qq(resid: ResidualDiagnostics, cfg: Config) -> Path:
    """Plot the QQ diagnostic of holdout residuals against the normal distribution."""
    fig, ax = plt.subplots(figsize=cfg.default_figsize)
    scipy_stats.probplot(resid.residuals, plot=ax)
    set_title(ax, "Q-Q Plot of Residuals")
    apply_grid(ax)
    return save_fig(fig, FIG_RESIDUALS_QQ, cfg,
                    caption="Normal Q-Q plot of holdout residuals.",
                    section="Supplementary diagnostics")


def plot_residual_spatial_map(
    spatial: Optional[dict], cfg: Config, registry=None
) -> Optional[Path]:
    """Spatial map of holdout residuals coloured by TwoSlopeNorm coolwarm."""
    if spatial is None:
        return None
    resid   = spatial["residuals"]
    max_abs = max(np.abs(resid).max(), TEMP_ZERO_GUARD)
    norm    = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)

    fig, ax = plt.subplots(figsize=cfg.default_figsize)
    sc = ax.scatter(spatial["x"], spatial["y"], c=resid, cmap="coolwarm",
                    norm=norm, s=18, edgecolor="none", alpha=0.85)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Residual (Observed - Predicted, °C)")
    set_axis_labels(ax, spatial["x_col"], spatial["y_col"])
    set_title(ax, "Spatial Distribution of Holdout Residuals")

    moran     = spatial["moran"]
    moran_str = (
        f"Global Moran's I = {moran['morans_i']:.3f} (p = {moran['p_value']:.3g})"
        if np.isfinite(moran["morans_i"]) else "Moran's I unavailable"
    )
    if np.isfinite(moran["p_value"]) and moran["p_value"] < 0.05:
        moran_str += (
            " - Residual spatial dependence remains after spatial validation.  "
            "Confidence intervals should be interpreted accordingly."
        )
    figure_footer(fig, moran_str, cfg)

    caption = generate_caption(
        "Spatial distribution of holdout residuals coloured on a coolwarm scale "
        "centred at zero to show whether prediction error is spatially clustered",
        {"Moran's I": moran["morans_i"], "p": moran["p_value"]},
    )
    return save_fig(fig, FIG_RESIDUAL_SPATIAL_MAP, cfg,
                    caption=caption, section="Supplementary diagnostics")


def plot_feature_importance(fi: FeatureImportanceResult, cfg: Config) -> Path:
    """Plot the permutation feature importance bar chart."""
    table = fi.table
    fig, ax = plt.subplots(figsize=cfg.default_figsize)
    bars = ax.bar(
        table["Feature"], table["Importance"],
        yerr=table["Importance_CI95"], capsize=5,
        color=[
            {"NDBI": cfg.color_neutral, "NDVI": cfg.color_primary,
             "Elevation": cfg.color_secondary}.get(f, cfg.color_s2)
            for f in table["Feature"]
        ],
        edgecolor="white", linewidth=0.5, width=0.55,
        error_kw=dict(elinewidth=1.5, ecolor="#555555"),
    )
    for bar, val in zip(bars, table["Importance"]):
        ax.annotate(
            f"{val:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 5), textcoords="offset points",
            ha="center", fontsize=10, fontweight="bold",
        )
    set_axis_labels(ax, "Feature", "Permutation Importance (ΔR²)")
    n_seeds = int(table["N_Repeats"].iloc[0]) if len(table) else 0
    set_title(ax, f"Permutation Feature Importance\n(mean ± 95% CI, t-distribution, over {n_seeds} seeds)")
    ax.margins(y=0.15)
    apply_grid(ax)
    fig.text(
        0.5, -0.04,
        "Values reflect marginal predictive importance; may be influenced by predictor collinearity.",
        ha="center", fontsize=8, color=cfg.color_neutral,
    )
    return save_fig(fig, FIG_FEATURE_IMPORTANCE, cfg,
                    caption=f"Permutation feature importance: mean ± 95% CI across "
                    f"{n_seeds} outer permutation seeds, CI half-width = "
                    f"t(df={n_seeds - 1}) * SE (Student's t, not a large-sample "
                    "normal approximation, given the small number of seeds).")


def plot_ndvi_vs_lst(df: pd.DataFrame, cfg: Config) -> Path:
    """Plot the NDVI-LST scatter relationship."""
    fig, ax = plt.subplots(figsize=cfg.default_figsize)
    ax.scatter(df["NDVI"], df["LST"], alpha=0.18, s=12,
               color=cfg.color_primary, edgecolor="none")
    reg     = LinearRegression().fit(df[["NDVI"]], df["LST"])
    x_range = np.linspace(df["NDVI"].min(), df["NDVI"].max(), 100)
    ax.plot(x_range,
            reg.predict(pd.DataFrame(x_range, columns=["NDVI"])),
            color=cfg.color_accent, linewidth=2, label="OLS Fit")
    set_title(ax, "Observed NDVI vs Land Surface Temperature")
    set_axis_labels(ax, "NDVI", "Observed LST (°C)")
    set_legend(ax)
    apply_grid(ax)
    return save_fig(fig, FIG_NDVI_VS_LST, cfg,
                    caption="Scatter of observed NDVI vs LST with OLS trend.",
                    section="Response curves")


def plot_lst_response_to_ndbi(sweep: dict, cfg: Config) -> Path:
    """Plot the predicted LST response curve across the NDBI sweep."""
    fig, ax = plt.subplots(figsize=cfg.default_figsize)
    ax.plot(sweep["ndbi_range"], sweep["pred_raw"],
            color=cfg.color_secondary, linewidth=1.0, alpha=0.4, label="Raw output")
    ax.plot(sweep["ndbi_range"], sweep["pred_smooth"],
            color=cfg.color_secondary, linewidth=2.2, label="Savgol-smoothed")
    ax.axvline(sweep["ndbi_median"], color=cfg.color_neutral, linestyle="--",
               linewidth=1.2, label=f"Median NDBI ({sweep['ndbi_median']:.3f})")
    set_axis_labels(ax, "NDBI", "Predicted LST (°C)")
    set_title(ax, "Predicted LST Response to NDBI\n(NDVI, Elevation fixed at sample medians)", fontsize=13)
    set_legend(ax)
    apply_grid(ax)
    return save_fig(fig, FIG_LST_RESPONSE_TO_NDBI, cfg,
                    caption="Model response with NDVI and Elevation fixed at their sample "
                    "medians while NDBI is swept across its observed range "
                    "(partial dependence-style relationship at fixed covariate values, "
                    "not an averaged PDP and not a causal or realistic joint-intervention "
                    "trajectory): predicted LST vs NDBI with Savitzky-Golay smoothing.")


def plot_ndbi_response_uncertainty(
    sweep: dict, uncertainty: UncertaintyResults, cfg: Config
) -> Path:
    """Plot the NDBI response curve with its uncertainty band."""
    if uncertainty.ndbi_sweep_folds is None:
        log_warning("No NDBI sweep refit data; skipping response-curve uncertainty figure.")
        return None

    folds  = uncertainty.ndbi_sweep_folds
    nr     = uncertainty.ndbi_range
    median = np.median(folds, axis=0)
    q25    = np.percentile(folds, 25, axis=0)
    q75    = np.percentile(folds, 75, axis=0)
    p2_5   = np.percentile(folds, 2.5, axis=0)
    p97_5  = np.percentile(folds, 97.5, axis=0)

    fig, ax = plt.subplots(figsize=cfg.default_figsize)
    ax.fill_between(nr, p2_5, p97_5, color=cfg.color_secondary, alpha=0.10,
                    label="95% interval (refits)")
    ax.fill_between(nr, q25,  q75,   color=cfg.color_secondary, alpha=0.22,
                    label="Q25-Q75 (refits)")
    ax.plot(nr, median, color=cfg.color_secondary, linewidth=1.6, linestyle="--",
            label=f"Median across {len(folds)} spatial refits")
    ax.plot(sweep["ndbi_range"], sweep["pred_raw"], color=cfg.color_neutral,
            linewidth=2.0, label="Raw output (primary, authoritative)")
    set_axis_labels(ax, "NDBI", "Predicted LST (°C)")
    set_title(ax, "Supplementary Diagnostic: NDBI Response-Curve Uncertainty "
              "(Spatial-Block Refits)")
    set_legend(ax)
    apply_grid(ax)
    return save_fig(fig, FIG_NDBI_RESPONSE_UNCERTAINTY, cfg,
                    caption="NDBI response-curve uncertainty across repeated spatial refits.",
                    section="Supplementary diagnostics")


def plot_ndvi_decile_analysis(
    summary_df: pd.DataFrame,
    corr_ndvi_lst: float, corr_ndbi_lst: float, corr_ndvi_ndbi: float,
    cfg: Config,
) -> Path:
    """Plot the NDVI-decile LST summary alongside correlation diagnostics."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 9))

    axes[0].errorbar(
        summary_df["NDVI_median"], summary_df["LST_median"],
        yerr=summary_df["LST_std"],
        fmt="o-", linewidth=2.5, markersize=8,
        color=cfg.color_secondary, capsize=5, capthick=2,
        label="Median ±1 SD",
    )
    axes[0].fill_between(
        summary_df["NDVI_median"],
        summary_df["LST_median"] - summary_df["LST_std"],
        summary_df["LST_median"] + summary_df["LST_std"],
        alpha=0.12, color=cfg.color_secondary,
    )
    set_axis_labels(axes[0], "NDVI (decile median)", "Observed LST (°C)")
    set_title(axes[0],
              f"LST vs NDVI Deciles  "
              f"(r_{{NDVI-LST}} = {corr_ndvi_lst:.3f}, "
              f"r_{{NDBI-LST}} = {corr_ndbi_lst:.3f})")
    lower = (summary_df["LST_median"] - summary_df["LST_std"]).min() - 0.5
    upper = (summary_df["LST_median"] + summary_df["LST_std"]).max() + 0.5
    axes[0].set_ylim(lower, upper)
    apply_grid(axes[0])
    set_legend(axes[0])

    axes[1].plot(summary_df["NDVI_median"], summary_df["NDBI_median"],
                 "s-", linewidth=2.5, markersize=8,
                 color=cfg.color_primary, label="Median NDBI")
    set_axis_labels(axes[1], "NDVI (decile median)", "NDBI (median)")
    set_title(axes[1], f"NDVI-NDBI Observed Association  (r = {corr_ndvi_ndbi:.3f})")
    apply_grid(axes[1])
    set_legend(axes[1])
    fig.tight_layout()
    return save_fig(fig, FIG_NDVI_DECILE_ANALYSIS, cfg,
                    caption="LST vs NDVI deciles and NDVI-NDBI observed association.",
                    section="Response curves")


def plot_ndvi_conditional_by_ndbi_bins(
    results: list[ConditionalBinResult], df: pd.DataFrame, cfg: Config,
    fig_filename: str = FIG_NDVI_CONDITIONAL_BY_NDBI,
    title: str = "Unadjusted Within-NDBI-Stratum NDVI-LST Association (Marginal, Not Causal)",
    caption: str = (
        "Unadjusted within-NDBI-stratum NDVI-LST association. Each panel is a "
        "single-predictor (NDVI-only) regression fit within an NDBI bin; "
        "equal-count binning holds NDBI only approximately constant, so "
        "residual continuous NDBI variation inside a bin can bias these "
        "slopes. See Figure S18 for the continuously-adjusted (NDVI + NDBI "
        "+ Elevation) companion diagnostic."
    ),
) -> Path:
    """Plot unadjusted (marginal) NDVI-LST slopes within NDBI bins.

    NDVI-LST sign-reversal audit: these are single-predictor regressions
    fit within each bin, not conditional/partial/causal NDVI effects - see
    the title/caption/footer text this function attaches, and
    compute_ndvi_ndbi_continuous_adjustment() for the continuously-adjusted
    companion diagnostic.

    Layout adapts to len(results) (n_bins) instead of assuming exactly 4:
    a 2x2 grid for 4 strata (the primary quartile figure), a 1xN row for
    2-3 strata (e.g. the tercile robustness check), and a near-square grid
    for anything larger. This lets the same function draw both the primary
    Figure 6 (quartile) and the alternative-binning robustness figure
    without duplicating the plotting logic.
    """
    n_bins = len(results)
    palette = ["#93C6E0", "#4A96C8", "#1B6FA8", "#0D3B66", "#082B4A", "#052033"]
    bin_colors = [palette[i % len(palette)] for i in range(n_bins)]
    conditional_ylim = (df["LST"].quantile(0.01) - 1.0, df["LST"].quantile(0.99) + 1.0)

    if n_bins == 4:
        n_rows, n_cols, figsize = 2, 2, (14, 10)
    elif n_bins <= 3:
        n_rows, n_cols, figsize = 1, n_bins, (6.5 * n_bins, 5.5)
    else:
        n_cols = int(np.ceil(np.sqrt(n_bins)))
        n_rows = int(np.ceil(n_bins / n_cols))
        figsize = (7 * n_cols, 5 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes_flat = axes.flatten()
    for ax in axes_flat[n_bins:]:
        ax.set_visible(False)

    for ax, result, bin_color in zip(axes_flat, results, bin_colors):
        ax.scatter(result.ndvi, result.lst, alpha=0.22, s=10,
                   color=bin_color, edgecolor="none")
        if result.slope is not None:
            x_line = np.array([result.ndvi.min(), result.ndvi.max()])
            ax.plot(x_line, result.slope * x_line + result.intercept,
                    color=cfg.color_accent, linewidth=1.5)
            ax.fill_between(
                x_line,
                (result.slope - result.slope_ci95) * x_line + result.intercept,
                (result.slope + result.slope_ci95) * x_line + result.intercept,
                color=cfg.color_accent, alpha=0.06,
            )
            stat_box(ax,
                     f"Slope = {result.slope:.2f} °C/NDVI\n"
                     f"95% CI = ±{result.slope_ci95:.2f}\n"
                     f"R² = {result.r_squared:.3f}")
        set_axis_labels(ax, "NDVI", "Observed LST (°C)")
        set_title(ax, result.label, fontsize=12, color=bin_color)
        apply_grid(ax)
        ax.set_ylim(conditional_ylim)

    title_y = 0.98
    title_caveat_gap_in = 0.25
    caveat_y = title_y - (title_caveat_gap_in / figsize[1])
    fig.suptitle(title, fontsize=13, fontweight="normal", y=title_y)
    fig.text(0.5, caveat_y,
              "Unadjusted (marginal) within-NDBI-stratum associations - not "
              "conditional, partial, or causal effects. May retain residual "
              "continuous NDBI confounding within each bin.",
              ha="center", fontsize=9.5, style="italic", color=cfg.color_neutral)
    legend_elements = [
        Patch(facecolor=c, label=r.label.replace("\n", " "))
        for c, r in zip(bin_colors, results)
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=n_bins,
               fontsize=9.5, frameon=True, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.02), title="Urbanisation bin (NDBI range)")
    fig.subplots_adjust(top=0.90, bottom=0.10, wspace=0.25, hspace=0.40)
    return save_fig(fig, fig_filename, cfg, use_tight_layout=False, caption=caption)


def plot_response_surface(
    surface: dict, df: pd.DataFrame, s2: ScenarioTrajectory, cfg: Config
) -> Path:
    """Plot the NDVI-NDBI predicted-LST response surface."""
    fig, ax = plt.subplots(figsize=(8, 7))
    contour = ax.contourf(surface["ndvi_mesh"], surface["ndbi_mesh"],
                          surface["lst_mesh"], levels=20, cmap="RdYlBu_r")
    plt.colorbar(contour, label="Predicted LST (°C)")
    ax.contour(surface["ndvi_mesh"], surface["ndbi_mesh"], surface["lst_mesh"],
               levels=20, colors="black", linewidths=0.3, alpha=0.4)
    ax.scatter(df["NDVI"], df["NDBI"], s=3, color="gray",
               alpha=0.08, rasterized=True, zorder=1)

    scenario_df = s2.feature_frame()
    ax.plot(scenario_df["NDVI"], scenario_df["NDBI"],
            color="black", linewidth=2.5, label="Scenario 2 trajectory", zorder=3)
    annotate_every = max(1, len(scenario_df) // 10)
    for i in range(0, len(scenario_df), annotate_every):
        ax.annotate(
            f"{scenario_df['Scenario (%)'].iloc[i]:.0f}%",
            (scenario_df["NDVI"].iloc[i], scenario_df["NDBI"].iloc[i]),
            textcoords="offset points", xytext=(6, 6), fontsize=7, color="black",
        )
    ax.axvline(surface["ndvi_max_plot"], color="gray", linestyle="--", linewidth=1.2,
               label="NDVI 99th percentile")
    ax.set_xlim(df["NDVI"].min(), surface["ndvi_max_plot"])
    set_axis_labels(ax, "NDVI", "NDBI")
    set_title(ax, "Predicted LST Response Surface")
    set_legend(ax)
    return save_fig(fig, FIG_RESPONSE_SURFACE, cfg,
                    caption="Predicted LST response surface in NDVI-NDBI space "
                    "with Scenario 2 trajectory.",
                    section="Supplementary diagnostics")


def plot_scenario2_data_support(
    df: pd.DataFrame, s2: ScenarioTrajectory,
    support: SupportDiagnostics, cfg: Config,
) -> Path:
    """Plot Scenario 2 trajectory points against the training-data feature-space support."""
    scenario_df = s2.feature_frame()
    in_support  = support.in_support_s2
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(df["NDVI"], df["NDBI"], s=3, alpha=0.15, color="gray",
               label="Observed pixels")
    sc = ax.scatter(
        scenario_df["NDVI"], scenario_df["NDBI"],
        c=np.where(in_support, scenario_df["Scenario (%)"], np.nan),
        cmap="viridis", s=25, label="Scenario 2 (in support)",
    )
    ax.scatter(scenario_df["NDVI"][~in_support], scenario_df["NDBI"][~in_support],
               color=cfg.color_accent, s=25, marker="x",
               label="Scenario 2 (outside support)")
    ax.plot(scenario_df["NDVI"], scenario_df["NDBI"],
            color=cfg.color_neutral, linewidth=1, alpha=0.5)
    set_axis_labels(ax, "NDVI", "NDBI")
    set_title(ax, "Scenario 2 Trajectory: Feature-Space Data Support")
    set_legend(ax)
    plt.colorbar(sc, label="Scenario (%) [in-support only]")
    return save_fig(fig, FIG_SCENARIO2_DATA_SUPPORT, cfg,
                    caption="Scenario 2 trajectory feature-space support.",
                    section="Supplementary diagnostics")


def plot_negi_smoothing_robustness(
    negi: NEGIResults, smoothing: dict, s2: ScenarioTrajectory,
    cfg: Config, summary: SummaryLog,
) -> tuple[Path, bool, bool, int]:
    """Plot the NEGI scenario curve with and without display smoothing
    to check robustness of the maximum evaluated NEGI (comparative
    assessment, not a claim of a robust interior optimum)."""
    cooling_savgol = negi.baseline_lst - smoothing["lst_s2_smooth"]
    cooling_spline = negi.baseline_lst - smoothing["lst_s2_spline"]
    negi_savgol = (cfg.alpha_weight *
                   safe_divide(np.maximum(cooling_savgol, 0.0), negi.reference_benefit_scale)
                   - cfg.beta_weight * negi.energy_norm_sqrt)
    negi_spline = (cfg.alpha_weight *
                   safe_divide(np.maximum(cooling_spline, 0.0), negi.reference_benefit_scale)
                   - cfg.beta_weight * negi.energy_norm_sqrt)

    peak_idx_raw    = negi.s2_optimum_idx
    peak_idx_savgol = int(np.argmax(negi_savgol))
    smoothing_robust = bool(
        abs(negi.scenarios[peak_idx_raw] - negi.scenarios[peak_idx_savgol]) * 100 <= 3
    )
    peak_is_ood = bool(s2.out_of_bounds[peak_idx_raw])

    summary.log("Supplementary Diagnostics - Smoothing", "Smoothing robust",
                str(smoothing_robust))
    summary.log("Supplementary Diagnostics - Smoothing",
                "Peak location out-of-feature-bounds", str(peak_is_ood))

    fig, ax = plt.subplots(figsize=cfg.default_figsize)
    ax.plot(negi.scenario_pct, negi.negi_s2,
            color=cfg.color_accent, linewidth=2.2, label="Raw (reported)")
    ax.plot(negi.scenario_pct, negi_savgol,
            color=cfg.color_secondary, linewidth=1.5, linestyle="--",
            label="Savgol-smoothed")
    ax.plot(negi.scenario_pct, negi_spline,
            color=cfg.color_primary, linewidth=1.5, linestyle=":",
            label="Spline-smoothed")
    ax.scatter([negi.scenario_pct[peak_idx_raw]], [negi.negi_s2[peak_idx_raw]],
               color=cfg.color_accent, s=80, zorder=5)
    ax.axhline(0, color=cfg.color_neutral, linestyle=":", linewidth=1)
    set_axis_labels(ax, SCENARIO_AXIS_LABEL, "NEGI")
    set_title(ax, "Supplementary Diagnostic: Scenario 2 NEGI, Raw vs Smoothed Variants")
    set_legend(ax, loc="lower left")
    apply_grid(ax)
    path = save_fig(fig, FIG_NEGI_SMOOTHING_ROBUSTNESS, cfg,
                    caption="Scenario 2 NEGI: raw vs Savgol- and spline-smoothed variants.",
                    section="Supplementary diagnostics")
    return path, smoothing_robust, peak_is_ood, peak_idx_raw


def plot_negi_scenario_comparison(
    negi: NEGIResults, annotations: dict, flag_str: str, cfg: Config
) -> Path:
    """Plot the NEGI comparison between Scenario 1 and Scenario 2."""
    opt_idx = negi.s2_optimum_idx
    fig, ax = plt.subplots(figsize=cfg.default_figsize)
    ax.plot(negi.scenario_pct, negi.negi_s1, linewidth=2.2,
            color=cfg.color_s1, linestyle="--",
            label="Scenario 1: Isolated NDVI counterfactual (NDBI fixed)")
    ax.plot(negi.scenario_pct, negi.negi_s2, linewidth=2.2,
            color=cfg.color_s2, linestyle="-",
            label="Scenario 2: Urban transformation (NDBI co-varies)")
    ax.axhline(0, color=cfg.color_neutral, linestyle="-", linewidth=1.1)
    ax.text(1.5, 0.012, "Break-even (NEGI = 0)",
            fontsize=9, color=cfg.color_neutral, va="bottom", style="italic")
    _s2_max_robust = annotations.get("max_negi_s2_robust")
    if _s2_max_robust is False:
        _reasons = []
        if annotations.get("max_negi_s2_ci_includes_zero"):
            _reasons.append("CI crosses zero")
        if annotations.get("max_negi_s2_locally_stable") is False:
            _reasons.append("locally unstable")
        if annotations.get("max_negi_s2_knn_support") is False:
            _reasons.append("fails feature support")
        if annotations.get("max_negi_s2_near_boundary"):
            _reasons.append("boundary-adjacent")
        _reason_text = ", ".join(_reasons) if _reasons else "reason undetermined"
        ax.scatter(
            [negi.scenario_pct[opt_idx]], [negi.negi_s2[opt_idx]],
            marker="X", s=70, color=cfg.color_accent, zorder=5,
            label=f"Numerical maximum (non-robust: {_reason_text})",
        )
        ax.annotate(
            f"Non-robust\n({_reason_text})",
            xy=(negi.scenario_pct[opt_idx], negi.negi_s2[opt_idx]),
            xytext=(8, 10), textcoords="offset points",
            fontsize=7.5, color=cfg.color_accent, style="italic",
            fontweight="bold",
        )
    else:
        ax.scatter(
            [negi.scenario_pct[opt_idx]], [negi.negi_s2[opt_idx]],
            marker="D", s=25, color=cfg.color_neutral, zorder=5,
            label="Highest evaluated trajectory value",
        )
    if np.isfinite(negi.first_cooling_pct):
        if negi.first_cooling_pct <= 0.0:
            _s2_cool_label = "Baseline (no credited cooling)"
        else:
            _s2_cool_label = f"Scenario 2 first cooling ({negi.first_cooling_pct:.0f}%)"
        ax.axvline(negi.first_cooling_pct, color=cfg.color_neutral,
                   linestyle=":", linewidth=1.0, alpha=0.7)
        ax.annotate(
            _s2_cool_label,
            xy=(negi.first_cooling_pct, 0),
            xytext=(negi.first_cooling_pct + 3, ax.get_ylim()[0] * 0.5),
            fontsize=8, color=cfg.color_neutral, style="italic",
        )
    set_axis_labels(ax, SCENARIO_AXIS_LABEL, "NEGI")
    set_title(ax, "NEGI Scenario Comparison")
    fig.text(
        0.5, -0.035,
        "Scenario 1 predicts no cooling at any evaluated intensity (see caption); "
        "its declining NEGI reflects rising cost against a benefit clipped to zero.",
        ha="center", fontsize=7, color=cfg.color_s1, style="italic",
    )
    fig.text(0.5, -0.09, annotation_caption(annotations),
             ha="center", fontsize=7.5, color=cfg.color_neutral, wrap=True)
    set_legend(ax, loc="lower left")
    apply_grid(ax)
    return save_fig(fig, FIG_NEGI_SCENARIO_COMPARISON, cfg,
                    caption="NEGI scenario comparison: isolated NDVI "
                            "counterfactual (NDBI fixed) vs realistic "
                            "municipal greening intervention. "
                            + interpretation_text("scenario1_sign_convention"))


def plot_negi_energy_cost_comparison(negi: NEGIResults, cfg: Config) -> Path:
    """Plot the desalination energy cost comparison between scenarios."""
    fig, ax = plt.subplots(figsize=cfg.default_figsize)
    ax.plot(negi.scenario_pct, negi.negi_s2,
            color=cfg.color_s2, linewidth=2, linestyle="-",
            label="Scenario 2, sqrt cost (primary)")
    ax.plot(negi.scenario_pct, negi.negi_s2_linear,
            color=cfg.color_s2, linewidth=2, linestyle="--",
            label="Scenario 2, linear cost")
    ax.plot(negi.scenario_pct, negi.negi_s1,
            color=cfg.color_s1, linewidth=2, linestyle="-",
            label="Scenario 1, sqrt cost (primary)")
    ax.plot(negi.scenario_pct, negi.negi_s1_linear,
            color=cfg.color_s1, linewidth=2, linestyle="--",
            label="Scenario 1, linear cost")
    ax.axhline(0, color=cfg.color_neutral, linestyle=":", linewidth=1)
    ax.text(1.5, 0.01, "Break-even (NEGI = 0)",
            fontsize=9, color=cfg.color_neutral, va="bottom", style="italic")
    set_axis_labels(ax, SCENARIO_AXIS_LABEL, "NEGI")
    set_title(ax, "Supplementary Diagnostic: NEGI, sqrt vs Linear Energy-Cost Assumption")
    set_legend(ax, loc="best")
    apply_grid(ax)
    return save_fig(fig, FIG_NEGI_ENERGY_COST_COMPARISON, cfg,
                    caption="NEGI comparison: sqrt vs linear energy-cost exponent assumption.",
                    section="Supplementary diagnostics")


def plot_negi_uncertainty_bands_primary(
    uncertainty: UncertaintyResults, s1: ScenarioTrajectory, cfg: Config
) -> tuple:
    """Plot NEGI uncertainty bands (median and IQR) across spatial-block holdout refits."""
    scenario_pct  = s1.scenario_pct
    p             = uncertainty.percentiles
    peak_iqr_idx  = int(np.argmax(p["negi_s2"][75] - p["negi_s2"][25]))
    peak_iqr_pct  = scenario_pct[peak_iqr_idx]

    fig, ax = plt.subplots(figsize=cfg.default_figsize)
    ax.plot(scenario_pct, p["negi_s1"][50], color=cfg.color_s1,
            linewidth=2.2, linestyle="--", label="Scenario 1 (median)")
    ax.fill_between(scenario_pct, p["negi_s1"][25], p["negi_s1"][75],
                    color=cfg.color_s1, alpha=0.15, label="Scenario 1 Q25-Q75")
    ax.plot(scenario_pct, p["negi_s2"][50], color=cfg.color_s2,
            linewidth=2.2, linestyle="-", label="Scenario 2 (median)")
    ax.fill_between(scenario_pct, p["negi_s2"][25], p["negi_s2"][75],
                    color=cfg.color_s2, alpha=0.20, label="Scenario 2 Q25-Q75")
    ax.axhline(0, color=cfg.color_neutral, linestyle="-", linewidth=1.1)
    ax.text(1.5, 0.01, "Break-even (NEGI = 0)",
            fontsize=9, color=cfg.color_neutral, va="bottom", style="italic")

    endpoint_ci_lo = float(p["negi_s2"][2.5][-1])
    endpoint_ci_hi = float(p["negi_s2"][97.5][-1])
    endpoint_ci_crosses_zero = endpoint_ci_lo <= 0.0 <= endpoint_ci_hi
    if endpoint_ci_crosses_zero:
        endpoint_pct = float(scenario_pct[-1])
        ax.annotate(
            "Endpoint CI crosses NEGI = 0\n"
            f"[{endpoint_ci_lo:.3f}, {endpoint_ci_hi:.3f}]",
            xy=(endpoint_pct, p["negi_s2"][50][-1]),
            xytext=(max(endpoint_pct - 30, 5), p["negi_s2"][50][-1] - 0.25),
            fontsize=8.5, color=cfg.color_neutral, style="italic",
            arrowprops=dict(arrowstyle="->", color=cfg.color_neutral, lw=0.9),
            bbox=STAT_BOX_STYLE,
        )
    else:
        ax.annotate(
            f"Peak IQR = {(p['negi_s2'][75]-p['negi_s2'][25])[peak_iqr_idx]:.3f}\n"
            f"at {peak_iqr_pct:.0f}%",
            xy=(peak_iqr_pct, p["negi_s2"][50][peak_iqr_idx]),
            xytext=(max(peak_iqr_pct - 30, 5), p["negi_s2"][50][peak_iqr_idx] - 0.25),
            fontsize=8.5, color=cfg.color_neutral, style="italic",
            arrowprops=dict(arrowstyle="->", color=cfg.color_neutral, lw=0.9),
            bbox=STAT_BOX_STYLE,
        )
    apply_grid(ax)
    return fig, ax, peak_iqr_idx


def plot_negi_scenario2_robustness_full(
    uncertainty: UncertaintyResults, s1: ScenarioTrajectory, cfg: Config
) -> Path:
    """Plot the full Scenario 2 NEGI robustness diagnostic across spatial-block refits."""
    scenario_pct = s1.scenario_pct
    p            = uncertainty.percentiles
    n_refits     = len(uncertainty.negi_s2_folds)

    fig, ax = plt.subplots(figsize=cfg.default_figsize)
    ax.fill_between(scenario_pct, p["negi_s2"][2.5], p["negi_s2"][97.5],
                    color=cfg.color_s2, alpha=0.12, label="Scenario 2 95% interval")
    ax.fill_between(scenario_pct, p["negi_s2"][25],  p["negi_s2"][75],
                    color=cfg.color_s2, alpha=0.30, label="Scenario 2 Q25-Q75")
    ax.plot(scenario_pct, p["negi_s2"][50],
            color=cfg.color_s2, linewidth=2.2, label="Scenario 2 median")
    ax.axhline(0, color=cfg.color_neutral, linestyle="-", linewidth=1.1)
    set_axis_labels(ax, SCENARIO_AXIS_LABEL, "NEGI")
    set_title(ax, f"Scenario 2 NEGI Robustness Across {n_refits} Spatial Refits")
    set_legend(ax, loc="lower left")
    apply_grid(ax)
    return save_fig(fig, FIG_NEGI_SCENARIO2_ROBUSTNESS, cfg,
                    caption="Scenario 2 NEGI robustness: median, IQR, and 95% interval.",
                    section="Supplementary diagnostics")


def plot_negi_exponent_sensitivity(
    negi: NEGIResults, s1: ScenarioTrajectory, cfg: Config
) -> Path:
    """Plot NEGI sensitivity to the cost-function exponent."""
    scenario_pct   = s1.scenario_pct
    cooling_norm_s2 = safe_divide(np.maximum(negi.delta_t_s2, 0.0), negi.reference_benefit_scale)
    cases = [
        {"alpha": 0.6, "beta": 1.6, "w0": 50},
        {"alpha": 1.0, "beta": 1.0, "w0": 200},
        {"alpha": 1.6, "beta": 0.6, "w0": 50},
    ]
    fig, ax = plt.subplots(figsize=cfg.default_figsize)
    for case in cases:
        a, b, w = case["alpha"], case["beta"], case["w0"]
        for exponent in (cfg.sensitivity_exponents[0], cfg.sensitivity_exponents[-1]):
            ep       = _energy_cost(negi.scenarios, w, exponent, cfg)
            ref_max  = _energy_cost(negi.scenarios, cfg.reference_w0, exponent, cfg).max()
            negi_profile = a * cooling_norm_s2 - b * safe_divide(ep, ref_max)
            ax.plot(scenario_pct, negi_profile, linewidth=1.8,
                    label=f"α={a:.1f}, β={b:.1f}, w0={w}, γ={exponent}")
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.text(1.5, 0.01, "Break-even (NEGI = 0)",
            fontsize=9, color=cfg.color_neutral, va="bottom", style="italic")
    set_axis_labels(ax, SCENARIO_AXIS_LABEL, "NEGI")
    set_title(ax, "Supplementary Diagnostic: NEGI Sensitivity - "
              "Cooling Weight, Energy Weight, Energy Scaling")
    set_legend(ax)
    apply_grid(ax)
    return save_fig(fig, FIG_NEGI_EXPONENT_SENSITIVITY, cfg,
                    caption="NEGI profiles under alternative parameter assumptions.",
                    section="Supplementary diagnostics")


def plot_cooling_saturation_diagnostic(
    negi: NEGIResults, s2: ScenarioTrajectory,
    saturation: dict, support: SupportDiagnostics, cfg: Config,
) -> Path:
    """Plot the cooling saturation diagnostic panels."""
    scenario_pct = s2.scenario_pct
    sat_50_idx   = saturation["sat_50_idx"]
    sat_90_idx   = saturation["sat_90_idx"]
    collapse_idx = saturation["collapse_idx"]
    in_support   = support.in_support_s2

    fig, axes = plt.subplots(3, 1, figsize=(8, 11), sharex=True)

    axes[0].plot(scenario_pct, negi.delta_t_s2,
                 color=cfg.color_secondary, linewidth=2)
    if sat_50_idx is not None:
        axes[0].axvline(
            scenario_pct[sat_50_idx], color=cfg.color_neutral, linestyle=":", linewidth=1,
            label=f"50% of max predicted cooling ({scenario_pct[sat_50_idx]:.1f}%)",
        )
        axes[0].axvline(
            scenario_pct[sat_90_idx], color=cfg.color_accent, linestyle="--", linewidth=1,
            label=f"90% of max predicted cooling ({scenario_pct[sat_90_idx]:.1f}%)",
        )
    else:
        axes[0].text(0.02, 0.05, "No positive predicted cooling beyond the baseline",
                     transform=axes[0].transAxes, fontsize=9, bbox=STAT_BOX_STYLE)
    if collapse_idx is not None:
        axes[0].axvline(scenario_pct[collapse_idx], color="black", linestyle="-",
                        linewidth=1.2, label=f"Near-zero ({scenario_pct[collapse_idx]:.1f}%)")
    warming_mask = negi.delta_t_s2 < 0
    axes[0].fill_between(scenario_pct, negi.delta_t_s2, 0, where=warming_mask,
                         color=cfg.color_accent, alpha=0.18,
                         label="Predicted LST above baseline (ΔT < 0)")
    set_axis_labels(axes[0], ylabel="Signed predicted cooling, S2 (°C)")
    set_title(axes[0], "Supplementary Diagnostic: Scenario 2 Predicted Cooling Along Trajectory")
    set_legend(axes[0], loc="best")
    apply_grid(axes[0])

    axes[1].fill_between(scenario_pct, 0, 1, where=in_support,
                         color=cfg.color_primary, alpha=0.25, step="mid",
                         label="In dense training region")
    axes[1].fill_between(scenario_pct, 0, 1, where=~in_support,
                         color=cfg.color_accent, alpha=0.25, step="mid",
                         label="Outside dense training region")
    if collapse_idx is not None:
        axes[1].axvline(scenario_pct[collapse_idx], color="black",
                        linestyle="-", linewidth=1.2)
    set_axis_labels(axes[1], ylabel="Data support")
    axes[1].set_yticks([])
    set_legend(axes[1], loc="best")

    axes[2].plot(scenario_pct, s2.ndbi, color=cfg.color_primary, linewidth=2)
    if collapse_idx is not None:
        axes[2].axvline(scenario_pct[collapse_idx], color="black",
                        linestyle="-", linewidth=1.2)
    set_axis_labels(axes[2], SCENARIO_AXIS_LABEL, "Scenario 2 NDBI")
    set_title(axes[2], "Empirical NDBI Trajectory")
    apply_grid(axes[2])
    return save_fig(fig, FIG_COOLING_SATURATION, cfg,
                    caption="Scenario 2 cooling saturation diagnostic.",
                    section="Supplementary diagnostics")


def plot_uncertainty_convergence(convergence_result: dict, cfg: Config) -> Optional[Path]:
    """Endpoint CI width vs. number of spatial refits, read from the
    adaptive convergence checkpoint history (see
    `compute_uncertainty_convergence_diagnostic`). Purely a reporting
    figure - never recomputes or alters the published uncertainty bands.

    Returns None (and saves nothing) if there is no checkpoint history to
    plot, e.g. `Config.adaptive_uncertainty_convergence=False`.
    """
    df = convergence_result.get("table_df")
    if (
        df is None or len(df) < 2
        or "refit_count" not in df.columns or "endpoint_ci_width" not in df.columns
    ):
        return None

    fig, ax = plt.subplots(figsize=cfg.default_figsize)
    ax.plot(df["refit_count"], df["endpoint_ci_width"], marker="o",
            color=cfg.color_primary, linewidth=2.0)
    status = convergence_result.get("convergence_status")
    published = convergence_result.get("published_n_refits")
    if published is not None:
        ax.axvline(published, color=cfg.color_accent, linestyle="--",
                    linewidth=1.2, label=f"Published ({published} refits)")
        set_legend(ax)
    set_title(ax, "Endpoint CI Width vs. Spatial Refits (Adaptive Convergence)")
    set_axis_labels(ax, "Number of spatial refits", "Scenario 2 endpoint 95% CI width")
    apply_grid(ax)
    figure_footer(fig, f"Convergence status: {status if status else 'N/A'}", cfg)
    return save_fig(
        fig, FIG_UNCERTAINTY_CONVERGENCE, cfg,
        caption="Scenario 2 endpoint NEGI 95% CI width across the adaptive "
                "spatial-refit convergence checkpoints.",
        section="Supplementary diagnostics",
    )


def plot_sensitivity_maximum_evaluated_negi(sensitivity: SensitivityResults, cfg: Config) -> Path:
    """Plot the maximum evaluated NEGI (numerical maximum along the
    evaluated trajectory) as a function of w0 across alpha values."""
    fig, ax = plt.subplots(figsize=cfg.default_figsize)
    for alpha_val in cfg.sensitivity_alpha_values:
        subset = sensitivity.plot_table[
            isclose_mask(sensitivity.plot_table["Alpha"], alpha_val)
        ]
        ax.plot(subset["w0"], subset["Numerical_Maximum_NEGI"],
                marker="o", linewidth=2, label=f"α = {alpha_val:.1f}")
    set_axis_labels(
        ax,
        f"Resource-cost scaling coefficient w0 (relative scaling parameter)\n"
        f"(relative to REFERENCE_W0 = {cfg.reference_w0:.0f})",
        "Numerical Maximum NEGI",
    )
    set_title(ax,
              f"Supplementary Diagnostic: Sensitivity of Numerical Maximum NEGI\n"
              f"(exponent = {cfg.sensitivity_default_exponent}, "
              f"β = {cfg.sensitivity_default_beta:.1f})")
    apply_grid(ax)
    set_legend(ax)
    return save_fig(fig, FIG_SENSITIVITY_MAXIMUM_EVALUATED_NEGI, cfg,
                    caption="Sensitivity of the highest evaluated NEGI to w0 and alpha parameters.",
                    section="Supplementary diagnostics")


plot_sensitivity_optimal_negi = plot_sensitivity_maximum_evaluated_negi


def plot_scenario2_percentile_sensitivity(
    table: pd.DataFrame, cfg: Config,
) -> Optional[Path]:
    """Publication-quality figure: endpoint NEGI (y) vs. archetype NDBI
    percentile (x) for the Scenario 2 percentile-sensitivity sweep, with
    95% spatial-refit uncertainty bars and robust/non-robust markers.
    Formatting mirrors the other supplementary diagnostics
    (apply_grid / set_axis_labels / set_title / set_legend / save_fig) -
    no new NEGI, uncertainty, or robustness value is computed here; every
    plotted quantity is read directly from `table`."""
    if table is None or len(table) == 0:
        log_warning(
            "\nScenario 2 percentile sensitivity: no percentiles were "
            "evaluated (all skipped as too sparse) - figure not produced."
        )
        return None

    fig, ax = plt.subplots(figsize=cfg.default_figsize)
    x = table["percentile"].to_numpy() * 100.0
    y = table["endpoint_negi"].to_numpy()
    ci_lower = table["endpoint_ci_lower"].to_numpy(dtype=float)
    ci_upper = table["endpoint_ci_upper"].to_numpy(dtype=float)
    robust = table["endpoint_robust"].to_numpy()

    ax.vlines(x, ci_lower, ci_upper, color=cfg.color_neutral,
              linewidth=1.4, zorder=2, label="95% spatial-refit interval")
    ax.hlines(ci_lower, x - 0.4, x + 0.4, color=cfg.color_neutral, linewidth=1.4, zorder=2)
    ax.hlines(ci_upper, x - 0.4, x + 0.4, color=cfg.color_neutral, linewidth=1.4, zorder=2)
    ax.plot(x, y, linestyle="--", linewidth=1.2, color=cfg.color_neutral,
            alpha=0.6, zorder=1)

    robust_mask = robust == True   # noqa: E712
    nonrobust_mask = robust == False  # noqa: E712
    undetermined_mask = ~(robust_mask | nonrobust_mask)

    if robust_mask.any():
        ax.scatter(x[robust_mask], y[robust_mask], marker="o", s=90,
                    color=cfg.color_s2, edgecolor="black", linewidth=0.8,
                    label="Robust endpoint", zorder=3)
    if nonrobust_mask.any():
        ax.scatter(x[nonrobust_mask], y[nonrobust_mask], marker="X", s=90,
                    color="firebrick", edgecolor="black", linewidth=0.8,
                    label="Non-robust endpoint", zorder=3)
    if undetermined_mask.any():
        ax.scatter(x[undetermined_mask], y[undetermined_mask], marker="^", s=90,
                    color="gray", edgecolor="black", linewidth=0.8,
                    label="Robustness undetermined", zorder=3)

    ax.axhline(0.0, color="black", linewidth=0.8, linestyle=":", zorder=0)
    ax.axvline(cfg.s2_archetype_ndbi_percentile * 100.0, color=cfg.color_s2,
               linewidth=1.2, linestyle="-.", alpha=0.7,
               label=f"Primary endpoint ({cfg.s2_archetype_ndbi_percentile * 100:.0f}%)")

    set_axis_labels(ax, "Archetype NDBI percentile (%)", "Endpoint NEGI")
    set_title(
        ax,
        "Supplementary Diagnostic: Scenario 2 Endpoint Sensitivity\n"
        "to Archetype Percentile Choice",
    )
    apply_grid(ax)
    set_legend(ax, loc="best")
    return save_fig(
        fig, FIG_SCENARIO2_PERCENTILE_SENSITIVITY, cfg,
        caption=(
            "Sensitivity of the Scenario 2 endpoint's predicted NEGI (with "
            "95% spatial-refit uncertainty bars) to the choice of archetype "
            "NDBI percentile; robust/non-robust classification uses the "
            "same criteria as the primary Scenario 2 endpoint."
        ),
        section="Supplementary diagnostics",
    )


def plot_decision_envelope(
    envelope: dict, cfg: Config,
) -> Path:
    """Supplementary figure: the Scenario 2 maximum's position under each
    kind of sensitivity, drawn as SEPARATE ranges. No pooled or averaged
    marker is drawn - the point of the figure is that the layers differ in
    kind. Every plotted value is read from `envelope`."""
    layers = envelope["layers"]
    kind_color = {
        "reference":           cfg.color_neutral,
        "structural_model":    cfg.color_primary,
        "structural_scenario": cfg.color_secondary,
        "decision_parameters": cfg.color_accent,
    }
    fig, ax = plt.subplots(figsize=cfg.default_figsize)
    y_positions = np.arange(len(layers))[::-1]
    x_right = max(l["position_high_pct"] for l in layers)
    for y, l in zip(y_positions, layers):
        lo, hi = l["position_low_pct"], l["position_high_pct"]
        color = kind_color.get(l["kind"], cfg.color_neutral)
        if np.isclose(lo, hi):
            ax.scatter([lo], [y], marker="D", s=120, color=color,
                       edgecolor="black", linewidth=0.8, zorder=3)
            label = f"{lo:.2f}%"
        else:
            ax.hlines(y, lo, hi, color=color, linewidth=10, alpha=0.85, zorder=2)
            ax.scatter([lo, hi], [y, y], marker="|", s=260, color="black", zorder=3)
            label = f"{lo:.2f}\u2013{hi:.2f}%"
        ax.annotate(label, (hi, y), xytext=(10, 0), textcoords="offset points",
                    va="center", fontsize=10)

    ref_pos = envelope["reference_case"]["position_pct"]
    ax.axvline(ref_pos, color=cfg.color_neutral, linestyle="--", linewidth=1.1,
               alpha=0.7, label=f"Reference-case maximum ({ref_pos:.2f}%)")
    split = envelope["assessment"].get("regime_split_position_pct")
    x_max = x_right * 1.22 + 1.0
    if split is not None:
        ax.axvspan(split, x_max, color="gray", alpha=0.08,
                   label=f"High-intensity regime (> {split:.2f}%)")
    ax.set_xlim(-0.02 * x_max, x_max)
    ax.set_ylim(-0.6, len(layers) - 0.4)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([l["layer"] for l in layers])
    set_axis_labels(ax, "Position of the Scenario 2 maximum (% intervention)", "")
    set_title(ax, "Supplementary Diagnostic: Decision Envelope\n"
                  "(ranges reported separately; not pooled)")
    apply_grid(ax)
    set_legend(ax, loc="lower right")
    fig.text(
        0.5, -0.04,
        "Structural/model ranges and the cost-weight range differ in kind. w0 is a relative "
        "resource-cost scaling coefficient, not a measured Jeddah water/energy quantity.",
        ha="center", fontsize=8, color=cfg.color_neutral,
    )
    return save_fig(
        fig, FIG_DECISION_ENVELOPE, cfg,
        caption=(
            "Decision envelope: position of the Scenario 2 maximum for the "
            "reference case, spatial-refit resampling, alternative archetype "
            "percentiles, and the alpha/beta/w0 cost-weight grid. Ranges are "
            "shown separately and are not averaged into a single optimum."
        ),
        section="Supplementary diagnostics",
    )


def plot_cost_regime_boundary(result: CostRegimeResult, cfg: Config) -> Path:
    """Supplementary figure for the cost-regime diagnostic: (left) the
    position of the Scenario 2 maximum against the effective cost ratio
    c = (beta/alpha)(w0/reference_w0), showing the snap; (right) the
    resulting boundary w0 as a function of alpha/beta over the evaluated
    grid. Every plotted value is read from `result`."""
    r = result
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(cfg.default_figsize[0] * 1.8, cfg.default_figsize[1]),
    )

    ax_a.step(r.scan_c, r.scan_position_pct, where="post", color=cfg.color_s2, linewidth=2.2)
    ax_a.set_xscale("log")
    if r.regime_detected:
        ax_a.axhline(r.split_pct, color="gray", linestyle=":", linewidth=1)
    for _, jrow in r.jumps.iterrows():
        is_primary = (
            r.primary_jump is not None
            and np.isclose(jrow["effective_cost_boundary"], r.primary_jump["effective_cost_boundary"])
        )
        ax_a.axvline(jrow["effective_cost_boundary"],
                     color=cfg.color_accent if is_primary else "gray",
                     linestyle="--", linewidth=1.4 if is_primary else 0.9,
                     label=(f"Primary transition (c = {jrow['effective_cost_boundary']:.3f})"
                            if is_primary else None))
    ax_a.axvline(r.reference_effective_cost, color=cfg.color_neutral, linestyle="-.",
                 linewidth=1.2, label=f"Reference case (c = {r.reference_effective_cost:.2f})")
    set_axis_labels(ax_a, "Effective cost ratio  c = (\u03b2/\u03b1)\u00b7(w0 / reference w0)  [log scale]",
                    "Position of the Scenario 2 maximum (%)")
    set_title(ax_a, f"Snap of the maximum\n(exponent = {r.exponent})")
    apply_grid(ax_a)
    set_legend(ax_a, loc="best", fontsize=9)

    bt = r.boundary_table
    ratios = bt["alpha_over_beta"].to_numpy(dtype=float)
    w_lo = float(bt["w0_min_evaluated"].iloc[0])
    w_hi = float(bt["w0_max_evaluated"].iloc[0])
    ax_b.axhspan(w_lo, w_hi, color="gray", alpha=0.08, label="Evaluated w0 range")
    if r.regime_detected:
        c_b = r.primary_jump["effective_cost_boundary"]
        r_line = np.geomspace(ratios.min() * 0.8, ratios.max() * 1.25, 200)
        ax_b.plot(r_line, cfg.reference_w0 * c_b * r_line, color=cfg.color_accent,
                  linewidth=2.2, label="Regime boundary  w0 = w0$_{ref}$\u00b7c$_b$\u00b7(\u03b1/\u03b2)")
        ax_b.text(0.03, 0.96, "above the line: low-intervention regime\nbelow the line: high-intervention regime",
                  transform=ax_b.transAxes, va="top", fontsize=9)
    ax_b.scatter([r.reference_alpha_over_beta], [r.reference_w0], marker="*", s=220,
                 color=cfg.color_neutral, edgecolor="white", zorder=4,
                 label=f"Reference case (\u03b1/\u03b2 = {r.reference_alpha_over_beta:g}, w0 = {r.reference_w0:g})")
    ax_b.set_xscale("log")
    ax_b.set_yscale("log")
    ax_b.set_ylim(w_lo * 0.7, w_hi * 1.4)
    set_axis_labels(ax_b, "\u03b1/\u03b2  [log scale]",
                    "w0 (relative resource-cost scaling coefficient)  [log scale]")
    set_title(ax_b, "Regime boundary as a function of \u03b1/\u03b2")
    apply_grid(ax_b)
    set_legend(ax_b, loc="lower right", fontsize=9)
    fig.text(
        0.5, -0.03,
        "w0 is a relative resource-cost scaling coefficient of the NEGI cost term, not a directly "
        "measured Jeddah water/energy quantity; no real-world calibration is implied.",
        ha="center", fontsize=8, color=cfg.color_neutral,
    )
    return save_fig(
        fig, FIG_COST_REGIME_BOUNDARY, cfg,
        caption=(
            "Cost-regime diagnostic on the existing alpha/beta/w0 sensitivity grid: "
            "the Scenario 2 maximum switches between a low- and a high-intervention "
            "regime at a single effective cost ratio c, which maps to a boundary "
            "w0 proportional to alpha/beta. w0 is a relative resource-cost scaling "
            "coefficient, not a measured quantity."
        ),
        section="Supplementary diagnostics",
    )


def main(cfg: Config = CFG) -> None:
    """Run the full NEGI modelling, validation, scenario, and reporting pipeline end to end."""
    run_start = time.monotonic()

    configure_logging(cfg)
    _assert_console_verbosity_invariant(cfg)
    log_headline("JEDDAH LST \u2014 NEGI FRAMEWORK")

    repro_info = print_reproducibility_info(cfg)
    setup_directories(cfg)
    apply_plot_style(cfg)
    summary = SummaryLog()
    PREDICTION_CACHE.clear()

    report_step("[STEP 1] Loading dataset...")
    df   = load_dataset(cfg)
    log_info(f"  Loaded {len(df):,} rows from {cfg.data_path}")
    data = build_dataset_bundle(df, cfg)
    log_info(f"  Train: {len(data.X_train):,} rows | Test: {len(data.X_test):,} rows")
    log_headline(f"Dataset: {len(df):,} observations | Seed: {cfg.random_seed}")
    log_headline("\n[1] Data")
    log_headline(f"Train: {len(data.X_train):,} | Holdout: {len(data.X_test):,}")

    corr_ndvi_lst  = df["NDVI"].corr(df["LST"])
    corr_ndbi_lst  = df["NDBI"].corr(df["LST"])
    corr_ndvi_ndbi = df["NDVI"].corr(df["NDBI"])
    summary.log("EDA", "NDVI-LST Pearson r",  f"{corr_ndvi_lst:.4f}")
    summary.log("EDA", "NDBI-LST Pearson r",  f"{corr_ndbi_lst:.4f}")
    summary.log("EDA", "NDVI-NDBI Pearson r", f"{corr_ndvi_ndbi:.4f}")

    report_step("[STEP 2] Fitting XGBoost model (RandomizedSearchCV)...")
    model_results = run_or_cached_stage(
        cfg, "xgb_tuning", cfg.run_model_tuning,
        _cache_key(
            "xgb_tuning", cfg,
            random_seed=cfg.random_seed,
            monotone_constraints=cfg.monotone_constraints,
            param_distributions=cfg.param_distributions,
            n_random_search_iter=cfg.n_random_search_iter,
            n_group_kfold_splits=cfg.n_group_kfold_splits,
            holdout_test_size=cfg.holdout_test_size,
        ),
        lambda: fit_model(data, cfg, summary),
        step_label="[STEP 2] XGBoost tuning",
    )
    if len(model_results.data.df) != len(data.df):
        log_warning(
            "  [cache] stale 'xgb_tuning' cache entry: cached DatasetBundle "
            f"has {len(model_results.data.df):,} rows but the freshly loaded "
            f"dataset has {len(data.df):,} rows (cache key did not capture "
            "everything that changed the row count). Recomputing instead of "
            "using the stale cache."
        )
        model_results = fit_model(data, cfg, summary)
        cache_save(
            cfg, "xgb_tuning",
            _cache_key(
                "xgb_tuning", cfg,
                random_seed=cfg.random_seed,
                monotone_constraints=cfg.monotone_constraints,
                param_distributions=cfg.param_distributions,
                n_random_search_iter=cfg.n_random_search_iter,
                n_group_kfold_splits=cfg.n_group_kfold_splits,
                holdout_test_size=cfg.holdout_test_size,
            ),
            model_results,
        )
    log_headline("\n[2] XGBoost tuning")
    log_headline(f"Best inner GroupKFold R²: {model_results.search.best_score_:.3f}")

    report_step("[STEP 3] Evaluating model...")
    report_step("[STEP 3a] Nested GroupKFold CV (honest, leakage-free outer-fold R²)...")
    nested_cv = run_or_cached_stage(
        cfg, "nested_cv", cfg.run_nested_validation,
        _cache_key(
            "nested_cv", cfg,
            random_seed=cfg.random_seed,
            monotone_constraints=cfg.monotone_constraints,
            param_distributions=cfg.param_distributions,
            n_random_search_iter=cfg.n_random_search_iter,
            n_group_kfold_splits=cfg.n_group_kfold_splits,
            holdout_test_size=cfg.holdout_test_size,
        ),
        lambda: run_nested_group_kfold_cv(data, cfg, summary),
        step_label="[STEP 3a] Nested GroupKFold CV",
    )
    validation = evaluate_model(model_results, cfg, summary, nested_cv)
    log_headline("\n[3] Spatial validation")
    log_headline(
        f"Nested GroupKFold R²: {nested_cv.outer_r2_mean:.3f} ± {nested_cv.outer_r2_std:.3f}"
    )
    log_headline(
        f"Repeated spatial holdout R²: {validation.repeated_r2_mean:.3f} "
        f"± {ci95(validation.repeated_r2_scores):.3f}"
    )
    log_headline("Independent holdout:")
    log_headline(f"    R² = {validation.holdout_r2:.3f}")
    log_headline(f"    RMSE = {validation.holdout_rmse:.3f} °C")
    log_headline(f"    MAE = {validation.holdout_mae:.3f} °C")
    log_headline(f"Calibration slope = {validation.calibration_slope:.3f}")
    log_headline(f"Mean bias = {validation.calibration_bias:.3f} °C")

    report_step("[STEP 4] Residual diagnostics...")
    _t0 = time.perf_counter()
    resid = compute_residual_diagnostics(validation, data)
    report_development(cfg, f"[TIMING] compute_residual_diagnostics: {time.perf_counter() - _t0:.2f}s")
    _t0 = time.perf_counter()
    report_residual_diagnostics(resid, cfg, summary,)
    report_development(cfg, f"[TIMING] report_residual_diagnostics: {time.perf_counter() - _t0:.2f}s")
    _t0 = time.perf_counter()
    spatial_resid = compute_residual_spatial_diagnostic(validation, data, cfg, summary)
    report_development(cfg, f"[TIMING] compute_residual_spatial_diagnostic: {time.perf_counter() - _t0:.2f}s")
    log_headline("\n[4] Residual diagnostics")
    hetero_label = classify_heteroscedasticity(resid)
    log_headline(
        f"{hetero_label.capitalize()} detected."
        if hetero_label != "no strong evidence of heteroscedasticity"
        else "No strong evidence of heteroscedasticity."
    )
    if spatial_resid is not None:
        moran_i = spatial_resid["moran"]["morans_i"]
        dependence_note = (
            "residual spatial dependence remains"
            if spatial_resid["spatial_dependence_flag"]
            else "no strong evidence of residual spatial dependence"
        )
        log_headline(f"Residual Moran's I = {moran_i:.3f} \u2014 {dependence_note}.")

    spatial_autocorr_df = compute_spatial_autocorrelation_summary(data, spatial_resid, cfg)
    if spatial_autocorr_df is not None:
        save_csv(spatial_autocorr_df, cfg.data_dir / "spatial_autocorrelation_summary.csv", cfg=cfg, debug_only=True)

        _sa = spatial_autocorr_df.set_index("Series")["Moran's I"]
        _obs_i = float(_sa.get("Raw observed LST", np.nan))
        _lr_i  = float(_sa.get("Linear Regression residuals", np.nan))
        _xgb_i = float(_sa.get("XGBoost residuals", np.nan))
        _pct_reduction = (
            100.0 * (1.0 - _xgb_i / _obs_i)
            if np.isfinite(_obs_i) and _obs_i > 0 and np.isfinite(_xgb_i)
            else float("nan")
        )
        moran_comparison_df = pd.DataFrame([{
            "Observed Moran's I":           _obs_i,
            "Linear residual Moran's I":    _lr_i,
            "XGBoost residual Moran's I":   _xgb_i,
            "Percent reduction":            _pct_reduction,
        }])
        save_csv(moran_comparison_df, cfg.data_dir / "moran_comparison.csv", cfg=cfg, debug_only=True)
        log_info(
            "\nResidual spatial-autocorrelation benchmark "
            "(context for interpreting remaining XGBoost residual autocorrelation):"
        )
        log_info(moran_comparison_df.to_string(index=False))

    calibration_table = compute_calibration_table(
        data.y_test.values, validation.holdout_pred,
        intercept_ci=validation.spatial_block_bootstrap_ci.get("calibration_intercept"),
    )
    save_csv(calibration_table, cfg.data_dir / "calibration_metrics.csv", cfg=cfg, debug_only=True)
    log_info(f"\nCalibration metrics table:\n{calibration_table.to_string(index=False)}")

    holdout_predictions_df = pd.DataFrame({
        "Observed LST":  np.asarray(data.y_test.values, dtype=float),
        "Predicted LST": np.asarray(validation.holdout_pred, dtype=float),
        "Residual":      np.asarray(data.y_test.values, dtype=float)
                          - np.asarray(validation.holdout_pred, dtype=float),
    })
    save_csv(holdout_predictions_df, cfg.data_dir / "holdout_predictions.csv")

    model_performance_summary_df = pd.concat(
        [
            pd.DataFrame([{
                "nested_cv_r2":               validation.nested_cv_r2,
                "repeated_spatial_holdout_r2_mean": validation.repeated_r2_mean,
                "repeated_spatial_holdout_r2_ci95":  validation.repeated_r2_ci95,
                "confirmatory_holdout_r2":    validation.holdout_r2,
                "confirmatory_holdout_rmse":  validation.holdout_rmse,
                "confirmatory_holdout_mae":   validation.holdout_mae,
            }]),
            calibration_table.reset_index(drop=True),
        ],
        axis=1,
    )
    save_csv(model_performance_summary_df, cfg.data_dir / "model_performance_summary.csv")

    report_step("[STEP 5] Benchmark model comparison...")
    comparison_df = run_or_cached_stage(
        cfg, "benchmarks", cfg.run_benchmarks,
        _cache_key(
            "benchmarks", cfg,
            random_seed=cfg.random_seed,
            n_benchmark_search_iter=cfg.n_benchmark_search_iter,
            n_benchmark_search_iter_gb=cfg.n_benchmark_search_iter_gb,
            n_group_kfold_splits=cfg.n_group_kfold_splits,
            rf_param_distributions=cfg.rf_param_distributions,
            gb_param_distributions=cfg.gb_param_distributions,
            holdout_test_size=cfg.holdout_test_size,
            xgb_best_params=model_results.best_params,
            nested_xgb_r2=nested_cv.outer_r2_mean,
        ),
        lambda: compare_models(model_results, cfg, summary, nested_cv),
        step_label="[STEP 5] Benchmark model comparison",
    )
    if _CACHE_STAGE_STATUS.get("benchmarks") == "cache hit":
        _replay_benchmark_reporting(comparison_df, cfg, summary)
    log_headline("\n[5] Benchmark comparison completed "
                 f"(metrics: {cfg.data_dir / 'benchmark_comparison.csv'}).")

    report_step("[STEP 6] Permutation feature importance...")
    fi = run_or_cached_stage(
        cfg, "permutation_importance", cfg.run_permutation_importance,
        _cache_key(
            "permutation_importance", cfg,
            random_seed=cfg.random_seed,
            n_permutation_outer_seeds=cfg.n_permutation_outer_seeds,
            n_permutation_inner_repeats=cfg.n_permutation_inner_repeats,
            xgb_best_params=model_results.best_params,
            feature_columns=list(model_results.data.X.columns),
        ),
        lambda: compute_feature_importance(model_results, cfg, summary),
        step_label="[STEP 6] Permutation feature importance",
    )
    log_headline("\n[6] Feature importance")
    log_headline(" > ".join(fi.table["Feature"].tolist()))
    log_headline(
        f"Permutation importance; NDVI\u2013NDBI r = {corr_ndvi_ndbi:.3f}."
    )

    report_step("[STEP 7] Computing scenario trajectories...")
    baseline, s1, s2 = run_scenario_trajectories(model_results, cfg, summary)
    summary.log(
        "Scenario construction", "Scenario 2 archetype percentile",
        f"{cfg.s2_archetype_ndbi_percentile * 100:.0f}%",
    )

    s2_archetype_diagnostics = compute_scenario2_archetype_spatial_diagnostics(
        model_results.data.df_train, cfg, summary,
    )
    export_scenario2_archetype_diagnostics(s2_archetype_diagnostics, cfg)
    summary.log(
        "Scenario construction", "Scenario 2 sensitivity",
        ", ".join(f"{p * 100:.0f}%" for p in cfg.s2_sensitivity_percentiles),
    )

    report_step("[STEP 8] Tree artifact / staircase diagnostics...")
    tree_artifact_diagnostics(s1.lst, "Scenario 1", cfg, summary)
    tree_artifact_diagnostics(s2.lst, "Scenario 2", cfg, summary)
    local_gradient_diagnostics(s1.lst, s1.scenario_pct, "Scenario 1", summary)
    local_gradient_diagnostics(s2.lst, s2.scenario_pct, "Scenario 2", summary)
    smoothing = compute_display_smoothing(s1, s2, cfg)

    report_step("[STEP 9] Computing NEGI results...")
    negi, reference_cooling_c = compute_negi_results(baseline, s1, s2, cfg)
    report_negi_scenario_diagnostics(negi, reference_cooling_c, cfg, summary)
    check_interior_optimum(negi.negi_s1, negi.scenarios, "Scenario 1", cfg)
    check_interior_optimum(negi.negi_s2, negi.scenarios, "Scenario 2", cfg)
    check_trajectory_stability(negi.negi_s1, negi.scenarios, "Scenario 1", cfg, summary)
    check_trajectory_stability(negi.negi_s2, negi.scenarios, "Scenario 2", cfg, summary)
    report_boundary_maximum_assessment(negi, cfg)

    report_step("[STEP 10] Feature-space support diagnostics...")
    support = compute_support_diagnostics(model_results, s1, s2, cfg, summary)
    report_data_support_check(s2, negi, support, cfg, summary)
    log_headline("\n[7] Scenario analysis")
    ndvi_all = np.concatenate([s1.ndvi, s2.ndvi])
    log_headline(f"NDVI range: {ndvi_all.min():.4f} \u2192 {ndvi_all.max():.4f}")
    log_headline(
        "Scenario 1: " + ("within formal feature-space support" if support.extrap_s1_ok
                           else "outside formal feature-space support")
    )
    log_headline(
        "Scenario 2: " + ("within formal feature-space support" if support.extrap_s2_ok
                           else "outside formal feature-space support")
    )

    report_step("[STEP 10b] Joint multivariate feature-space support diagnostic...")
    joint_support = compute_joint_multivariate_support(support, s1, s2, cfg, summary)
    joint_support_df = report_joint_multivariate_support(joint_support, cfg)

    report_step("[STEP 11] Computing uncertainty bands (spatial refits)...")
    uncertainty = run_or_cached_stage(
        cfg, "uncertainty_bands", cfg.run_uncertainty,
        _cache_key(
            "uncertainty_bands", cfg,
            random_seed=cfg.random_seed,
            monotone_constraints=cfg.monotone_constraints,
            n_spatial_refits=cfg.n_spatial_refits,
            uncertainty_refit_seed=cfg.uncertainty_refit_seed,
            holdout_test_size=cfg.holdout_test_size,
            xgb_best_params=model_results.best_params,
            ndbi_sweep_points=cfg.ndbi_sweep_points,
            ndbi_sweep_quantile_low=cfg.ndbi_sweep_quantile_low,
            ndbi_sweep_quantile_high=cfg.ndbi_sweep_quantile_high,
            scenario_step=cfg.scenario_step,
            adaptive_uncertainty_convergence=cfg.adaptive_uncertainty_convergence,
            uncertainty_convergence_check_interval=cfg.uncertainty_convergence_check_interval,
            uncertainty_convergence_min_refits=cfg.uncertainty_convergence_min_refits,
            uncertainty_convergence_required_stable_checkpoints=cfg.uncertainty_convergence_required_stable_checkpoints,
            adaptive_convergence_tolerance=cfg.adaptive_convergence_tolerance,
        ),
        lambda: compute_uncertainty_bands(
            model_results, baseline, s1, s2, negi, cfg,
            max_refits=cfg.n_spatial_refits,
        ),
        step_label="[STEP 11] Uncertainty bands (spatial refits)",
    )

    _uncertainty_audit = audit_uncertainty_refit_consistency(cfg, uncertainty)

    report_uncertainty_bands(uncertainty, s1, cfg, summary)
    refit_summary_df = summarize_spatial_refit_diagnostics(validation, uncertainty, s1, cfg)

    report_step("[STEP 11b] Spatial block-size sensitivity...")
    block_size_sensitivity_df = run_spatial_block_size_sensitivity(
        model_results, baseline, s1, s2, negi, cfg, summary,
    )

    robustness_summary_df = pd.concat(
        [
            joint_support_df.assign(section="joint_multivariate_support"),
            block_size_sensitivity_df.assign(section="spatial_block_size_sensitivity"),
        ],
        ignore_index=True, sort=False,
    )
    save_csv(robustness_summary_df, cfg.data_dir / "robustness_summary.csv", cfg=cfg, debug_only=True)

    if cfg.run_uncertainty_convergence_diagnostic:
        report_step("[STEP 11c] Uncertainty convergence diagnostic (refit-count sweep)...")
        convergence_result = compute_uncertainty_convergence_diagnostic(
            model_results, baseline, s1, s2, negi, cfg, uncertainty,
            point_label="endpoint",
        )
        convergence_result_maximum = compute_uncertainty_convergence_diagnostic(
            model_results, baseline, s1, s2, negi, cfg, uncertainty,
            point_label="maximum",
        )
        save_csv(convergence_result["table_df"], cfg.data_dir / "uncertainty_convergence.csv", cfg=cfg, debug_only=True)
        plot_uncertainty_convergence(convergence_result, cfg)
        save_json(
            {
                "refit_counts":       convergence_result["refit_counts"],
                "published_n_refits": convergence_result["published_n_refits"],
                "tolerance_pct":      convergence_result["tolerance_pct"],
                "table":              convergence_result["table"],
                "transitions":        convergence_result["transitions"],
                "convergence_status": convergence_result["convergence_status"],
                "convergence_status_maximum": convergence_result_maximum["convergence_status"],
            },
            cfg.data_dir / "uncertainty_convergence.json",
        )
        summary.log(
            "Uncertainty convergence diagnostic",
            "Convergence status (endpoint)",
            str(convergence_result["convergence_status"]),
        )
        summary.log(
            "Uncertainty convergence diagnostic",
            "Convergence status (Scenario 2 maximum)",
            str(convergence_result_maximum["convergence_status"]),
        )
        convergence_status_for_report = convergence_result["convergence_status"]
        convergence_status_for_maximum_report = convergence_result_maximum["convergence_status"]
        convergence_skip_reason_for_report = convergence_result.get("convergence_skip_reason")
        report_convergence_extension_comparison(uncertainty, cfg, summary)
    else:
        convergence_result = None
        convergence_result_maximum = None
        convergence_status_for_report = None
        convergence_status_for_maximum_report = None
        convergence_skip_reason_for_report = "not requested (Config.run_uncertainty_convergence_diagnostic=False)"

    _convergence_criteria = print_uncertainty_consistency_summary(
        _uncertainty_audit, convergence_status_for_report, convergence_skip_reason_for_report,
        tolerance_pct=(convergence_result["tolerance_pct"] if convergence_result is not None else None),
        n_refits=(convergence_result["published_n_refits"] if convergence_result is not None else None),
        cfg=cfg,
    )
    save_csv(
        pd.DataFrame([_convergence_criteria]),
        cfg.data_dir / "uncertainty_convergence_status.csv",
        cfg=cfg,
    )

    report_step("[STEP 11d] NEGI uncertainty zero-crossing diagnostics...")
    zero_crossing_df = compute_negi_zero_crossing_diagnostics(negi, uncertainty, support, cfg)
    report_negi_zero_crossing_diagnostics(zero_crossing_df, cfg, summary)

    maximum_diagnostics_s1 = build_maximum_diagnostics(
        "Scenario 1", negi.negi_s1, negi.scenario_pct, negi.s1_optimum_idx,
        uncertainty.percentiles.get("negi_s1"), None, None, cfg,
        actual_n_refits=uncertainty.n_refits_used,
    )
    maximum_diagnostics_s2 = build_maximum_diagnostics(
        "Scenario 2", negi.negi_s2, negi.scenario_pct, negi.s2_optimum_idx,
        uncertainty.percentiles.get("negi_s2"), support.in_support_s2,
        support.mean_knn_dist_s2, cfg,
        sparsity_threshold=support.sparsity_threshold_k,
            density_moderate_threshold=support.density_moderate_threshold_k,
        reference_knn_distribution=support.reference_mean_knn_dist,
        actual_n_refits=uncertainty.n_refits_used,
    )
    maximum_diagnostics_df = pd.DataFrame([maximum_diagnostics_s1, maximum_diagnostics_s2])
    save_csv(maximum_diagnostics_df, cfg.data_dir / "maximum_diagnostics.csv", cfg=cfg, debug_only=True)
    for d in (maximum_diagnostics_s1, maximum_diagnostics_s2):
        summary.log(
            "Maximum diagnostics", f"{d['label']} robustness classification",
            f"maximum_evaluated_negi={d['maximum_evaluated_negi']:.4f} @ "
            f"{d['maximum_scenario_position']:.1f}%; "
            f"near_boundary={d['maximum_near_boundary']}; "
            f"uncertainty_type={d['maximum_uncertainty_type']}; "
            f"ci_includes_zero={d['maximum_ci_includes_zero']}; "
            f"local_stability={d['maximum_local_stability']}; "
            f"knn_support={d['maximum_knn_support']}; "
            f"sparse_region={d['maximum_sparse_region']}; "
            f"robust={d['maximum_robust']}; "
            f"supports_interior_optimum={d['maximum_supports_interior_optimum']}",
        )

    classification_blocks = "\n\n".join(
        format_maximum_classification(
            d, joint_support_status=joint_support.get(d["label"], {}).get("joint_support_status"),
            publication_convergence_status=convergence_status_for_maximum_report,
        )
        for d in (maximum_diagnostics_s1, maximum_diagnostics_s2)
    )
    log_info("\nFinal machine-readable maximum classification:\n" + classification_blocks)
    summary.add_conclusion(
        "Scenario 2 maximum classification:\n"
        + format_maximum_classification(
            maximum_diagnostics_s2,
            joint_support_status=joint_support.get("Scenario 2", {}).get("joint_support_status"),
            publication_convergence_status=convergence_status_for_maximum_report,
        )
    )
    classification_path = cfg.data_dir / "maximum_classification.txt"
    with open(classification_path, "w", encoding="utf-8") as f:
        f.write(classification_blocks + "\n")
    confirm_saved(classification_path)

    log_headline("\n[8] NEGI assessment")
    for d in (maximum_diagnostics_s1, maximum_diagnostics_s2):
        for line in format_maximum_headline(
            d, joint_support_status=joint_support.get(d["label"], {}).get("joint_support_status"),
            publication_convergence_status=convergence_status_for_maximum_report,
        ):
            log_headline(line)

    report_step("[STEP 12] Sensitivity analysis...")
    sensitivity = run_sensitivity_analysis(negi, s1, cfg)
    report_sensitivity_analysis(sensitivity, cfg)

    report_step("[STEP 12a] Cost-regime diagnostic...")
    cost_regime, cost_regime_by_exponent = compute_cost_regime_diagnostic(negi, sensitivity, cfg)
    cost_regime_payload = report_cost_regime_diagnostic(
        cost_regime, cost_regime_by_exponent, cfg, summary,
    )

    report_step("[STEP 12b] Scenario 2 percentile sensitivity...")
    s2_percentile_sensitivity = run_or_cached_stage(
        cfg, "s2_percentile_sensitivity", cfg.run_s2_percentile_sensitivity,
        _cache_key(
            "s2_percentile_sensitivity", cfg,
            random_seed=cfg.random_seed,
            monotone_constraints=cfg.monotone_constraints,
            n_spatial_refits_percentile_sensitivity=cfg.n_spatial_refits_percentile_sensitivity,
            uncertainty_refit_seed=cfg.uncertainty_refit_seed,
            holdout_test_size=cfg.holdout_test_size,
            xgb_best_params=model_results.best_params,
            s2_sensitivity_percentiles=cfg.s2_sensitivity_percentiles,
            s2_archetype_ndbi_percentile=cfg.s2_archetype_ndbi_percentile,
            s2_search_min_bin_count=cfg.s2_search_min_bin_count,
            s2_trajectory_n_bins=cfg.s2_trajectory_n_bins,
            scenario_step=cfg.scenario_step,
        ),
        lambda: run_scenario2_percentile_sensitivity(
            model_results, baseline, s1, support, cfg, summary,
        ),
        step_label="[STEP 12b] Scenario 2 percentile sensitivity",
    )
    report_scenario2_percentile_sensitivity(s2_percentile_sensitivity, cfg, summary)

    report_step("[STEP 12c] Decision envelope...")
    decision_envelope = report_decision_envelope(
        compute_decision_envelope(
            negi, uncertainty, s2_percentile_sensitivity, sensitivity, cost_regime, cfg,
        ),
        cfg, summary,
    )

    log_headline("\n[9] Robustness")
    _s2_robust_word = (
        "is robust" if maximum_diagnostics_s2["maximum_robust"]
        else "remains non-robust"
    )
    log_headline(
        f"Scenario 2 maximum {_s2_robust_word} under uncertainty and sensitivity checks."
    )

    ndbi_sweep = compute_ndbi_response_sweep(model_results, cfg)
    surface    = compute_response_surface(model_results, cfg)

    ndbi_bins = build_ndbi_quantile_bins(
        data.df_train, 4, ["Low urban", "Moderate urban", "High urban", "Very high urban"]
    )
    conditional_results = compute_conditional_ndvi_lst_slopes(data.df_train, ndbi_bins)
    conditional_df = conditional_slopes_to_dataframe(conditional_results)
    conditional_df = enhance_conditional_slopes_table(conditional_df, conditional_results, cfg)
    report_conditional_slopes_enhanced(conditional_df, cfg, summary)
    conditional_sign_summary = summarize_conditional_ndvi_slope_signs(conditional_df, summary)

    binning_robustness = compute_ndbi_binning_robustness(
        data.df_train, cfg, summary, quartile_results=conditional_results,
    )

    continuous_adjustment = compute_ndvi_ndbi_continuous_adjustment(data.df_train, cfg)
    continuous_adjustment_summary = report_continuous_adjustment(continuous_adjustment, cfg, summary)
    plot_ndvi_continuous_adjustment(continuous_adjustment, cfg)

    varying_coefficient = compute_ndvi_ndbi_varying_coefficient_gam(data.df_train, cfg)
    varying_coefficient_summary = report_varying_coefficient_gam(varying_coefficient, cfg, summary)

    log_ndvi_causal_caveat(
        fi, corr_ndvi_ndbi, cfg, summary,
        continuous_adjustment=continuous_adjustment_summary,
    )

    ndvi_decile_df = compute_ndvi_decile_summary(data.df_train)
    saturation     = compute_saturation_diagnostic(negi, s2, cfg)

    feature_support_status = (
        "within bounds" if not s2.out_of_bounds.any() else "partial extrapolation"
    )
    _s2_sparse_mask = sparse_region_mask(
        support.mean_knn_dist_s2, support.in_support_s2, support.sparsity_threshold_k
    )
    _s2_n_sparse = int(_s2_sparse_mask.sum()) if _s2_sparse_mask is not None else 0
    _s2_frac_sparse = (
        _s2_n_sparse / len(_s2_sparse_mask) if _s2_sparse_mask is not None and len(_s2_sparse_mask) else 0.0
    )
    annotations = {
        "baseline_lst":       baseline.baseline_lst,
        "first_cooling_pct":  negi.first_cooling_pct,
        "max_cooling_c":      negi.max_cooling_s2,
        "max_negi_s2_value":  negi.negi_s2[negi.s2_optimum_idx],
        "max_negi_s2_pct":    negi.scenario_pct[negi.s2_optimum_idx],
        "max_negi_s2_robust": maximum_diagnostics_s2["maximum_robust"],
        "max_negi_s2_ci_includes_zero": maximum_diagnostics_s2["maximum_ci_includes_zero"],
        "max_negi_s2_near_boundary": (
            maximum_diagnostics_s2["maximum_near_boundary"]
            or maximum_diagnostics_s2["maximum_at_boundary"]
        ),
        "max_negi_s2_locally_stable": maximum_diagnostics_s2["maximum_local_stability"],
        "max_negi_s2_knn_support": maximum_diagnostics_s2["maximum_knn_support"],
        "calibration_slope":  validation.calibration_slope,
        "calibration_intercept": validation.calibration_intercept,
        "feature_support_status": feature_support_status,
    }
    flag_str = (
        "WARNING: partial extrapolation" if not support.extrap_s2_ok
        else "OK: within training support"
    )

    _endpoint_idx_pre = -1
    endpoint_diagnostics_s2 = build_endpoint_diagnostics(
        "Scenario 2", negi.negi_s2, negi.scenario_pct, _endpoint_idx_pre,
        uncertainty.percentiles.get("negi_s2"), support.in_support_s2,
        support.mean_knn_dist_s2, cfg,
        sparsity_threshold=support.sparsity_threshold_k,
            density_moderate_threshold=support.density_moderate_threshold_k,
        reference_knn_distribution=support.reference_mean_knn_dist,
        actual_n_refits=uncertainty.n_refits_used,
    )
    endpoint_diagnostics_s1 = build_endpoint_diagnostics(
        "Scenario 1", negi.negi_s1, negi.scenario_pct, _endpoint_idx_pre,
        uncertainty.percentiles.get("negi_s1"), None, None, cfg,
        actual_n_refits=uncertainty.n_refits_used,
    )

    scenario_summary_df = pd.DataFrame([{
        "baseline_lst_c":           baseline.baseline_lst,
        "s1_max_negi":              negi.negi_s1[negi.s1_optimum_idx],
        "s1_max_negi_pct":          negi.scenario_pct[negi.s1_optimum_idx],
        "s2_max_negi":              negi.negi_s2[negi.s2_optimum_idx],
        "s2_max_negi_pct":          negi.scenario_pct[negi.s2_optimum_idx],
        "s2_max_cooling_c":         negi.max_cooling_s2,
        "s2_max_warming_c":         negi.max_warming_s2,
        "s2_first_cooling_pct":     negi.first_cooling_pct,
        "s2_warming_fraction":      negi.warming_fraction_s2,
        "s2_cooling_fraction":      negi.cooling_fraction_s2,
        "calibration_slope":        validation.calibration_slope,
        "feature_support_status":   feature_support_status,
        "feature_support_sparse_region_flag": maximum_diagnostics_s2["maximum_sparse_region"],
        "feature_support_empirical_percentile": support.sparsity_percentile,
        "feature_support_empirical_threshold":  support.sparsity_threshold_k,
        "feature_support_max_nn_distance":       float(support.mean_knn_dist_s2.max()),
        "feature_support_median_nn_distance":    float(np.median(support.mean_knn_dist_s2)),
        "feature_support_sparse_count":          _s2_n_sparse,
        "feature_support_sparse_fraction":       _s2_frac_sparse,
        "spatial_refit_count":            int(uncertainty.n_refits_used),
        "spatial_refit_count_ceiling":    int(cfg.n_spatial_refits),
        "spatial_refit_converged_early":  bool(uncertainty.converged_early),
        "spatial_refit_lower_percentile": 2.5,
        "spatial_refit_upper_percentile": 97.5,
        "spatial_refit_uncertainty_type": (
            "95% spatial-refit uncertainty band (empirical 2.5th-97.5th percentile)"
        ),
        "s1_maximum_near_boundary":  maximum_diagnostics_s1["maximum_near_boundary"],
        "s1_maximum_uncertainty_type": maximum_diagnostics_s1["maximum_uncertainty_type"],
        "s1_maximum_ci_includes_zero": maximum_diagnostics_s1["maximum_ci_includes_zero"],
        "s1_maximum_robust":         maximum_diagnostics_s1["maximum_robust"],
        "s1_maximum_supports_interior_optimum": maximum_diagnostics_s1["maximum_supports_interior_optimum"],
        "s1_maximum_local_stability": maximum_diagnostics_s1["maximum_local_stability"],
        "s2_maximum_near_boundary":  maximum_diagnostics_s2["maximum_near_boundary"],
        "s2_maximum_uncertainty_low": maximum_diagnostics_s2["maximum_uncertainty_low"],
        "s2_maximum_uncertainty_high": maximum_diagnostics_s2["maximum_uncertainty_high"],
        "s2_maximum_uncertainty_type": maximum_diagnostics_s2["maximum_uncertainty_type"],
        "s2_maximum_q25":            maximum_diagnostics_s2["maximum_q25"],
        "s2_maximum_q75":            maximum_diagnostics_s2["maximum_q75"],
        "s2_maximum_ci_includes_zero": maximum_diagnostics_s2["maximum_ci_includes_zero"],
        "s2_maximum_local_stability": maximum_diagnostics_s2["maximum_local_stability"],
        "s2_maximum_knn_distance":   maximum_diagnostics_s2["maximum_knn_distance"],
        "s2_maximum_robust":         maximum_diagnostics_s2["maximum_robust"],
        "s2_maximum_supports_interior_optimum": maximum_diagnostics_s2["maximum_supports_interior_optimum"],
        "s1_endpoint_evaluated_negi":     endpoint_diagnostics_s1["endpoint_evaluated_negi"],
        "s1_endpoint_ci_includes_zero":   endpoint_diagnostics_s1["endpoint_ci_includes_zero"],
        "s1_endpoint_local_stability":    endpoint_diagnostics_s1["endpoint_local_stability"],
        "s1_endpoint_robust":             endpoint_diagnostics_s1["endpoint_robust"],
        "s2_endpoint_evaluated_negi":     endpoint_diagnostics_s2["endpoint_evaluated_negi"],
        "s2_endpoint_uncertainty_low":    endpoint_diagnostics_s2["endpoint_uncertainty_low"],
        "s2_endpoint_uncertainty_high":   endpoint_diagnostics_s2["endpoint_uncertainty_high"],
        "s2_endpoint_ci_includes_zero":   endpoint_diagnostics_s2["endpoint_ci_includes_zero"],
        "s2_endpoint_knn_support":        endpoint_diagnostics_s2["endpoint_knn_support"],
        "s2_endpoint_sparse_region":      endpoint_diagnostics_s2["endpoint_sparse_region"],
        "s2_endpoint_local_stability":    endpoint_diagnostics_s2["endpoint_local_stability"],
        "s2_endpoint_local_change":       endpoint_diagnostics_s2["endpoint_local_change"],
        "s2_endpoint_local_sign_change":  endpoint_diagnostics_s2["endpoint_local_sign_change"],
        "s2_endpoint_robust":             endpoint_diagnostics_s2["endpoint_robust"],
    }])
    save_csv(scenario_summary_df, cfg.data_dir / "scenario_summary.csv", cfg=cfg, debug_only=True)

    scenario_results_s2 = compute_scenario_results(
        s2, negi.negi_s2, negi.energy_norm_sqrt, negi.desal_energy_sqrt, baseline.baseline_lst,
    )
    scenario_points_df = scenario_results_s2.to_dataframe()
    for _feat, _val in baseline.fixed_feature_values().items():
        scenario_points_df[f"{_feat} (fixed)"] = _val
    scenario_points_df["Within Support"]            = support.in_support_s2
    scenario_points_df["Nearest Neighbor Distance"] = support.mean_knn_dist_s2
    save_csv(scenario_points_df, cfg.data_dir / "scenario2_trajectory.csv")

    scenario_results_s1 = compute_scenario_results(
        s1, negi.negi_s1, negi.energy_norm_sqrt, negi.desal_energy_sqrt, baseline.baseline_lst,
    )
    scenario1_trajectory_df = scenario_results_s1.to_dataframe()
    for _feat, _val in baseline.fixed_feature_values().items():
        scenario1_trajectory_df[f"{_feat} (fixed)"] = _val
    save_csv(scenario1_trajectory_df, cfg.data_dir / "scenario1_trajectory.csv")

    negi_results_df = pd.concat(
        [
            scenario1_trajectory_df.assign(Scenario="Scenario 1"),
            scenario_points_df.drop(
                columns=["Within Support", "Nearest Neighbor Distance"]
            ).assign(Scenario="Scenario 2"),
        ],
        ignore_index=True,
    )
    negi_results_df = negi_results_df[
        ["Scenario"] + [c for c in negi_results_df.columns if c != "Scenario"]
    ]
    save_csv(negi_results_df, cfg.data_dir / "negi_results.csv")

    _endpoint_idx = -1
    _endpoint_uncertainty_lo = float(uncertainty.percentiles["negi_s2"][2.5][_endpoint_idx])
    _endpoint_uncertainty_hi = float(uncertainty.percentiles["negi_s2"][97.5][_endpoint_idx])
    _endpoint_in_support = bool(support.in_support_s2[_endpoint_idx])

    endpoint_diagnostics_df = pd.DataFrame([endpoint_diagnostics_s2])
    save_csv(endpoint_diagnostics_df, cfg.data_dir / "endpoint_diagnostics.csv", cfg=cfg, debug_only=True)

    endpoint_classification_block = format_endpoint_classification(
        endpoint_diagnostics_s2,
        joint_support_status=joint_support.get("Scenario 2", {}).get("joint_support_status"),
        publication_convergence_status=convergence_status_for_report,
    )
    log_info("\nEndpoint robustness assessment:\n" + endpoint_classification_block)
    summary.log(
        "Endpoint robustness assessment", "Scenario 2 endpoint classification",
        f"endpoint_evaluated_negi={endpoint_diagnostics_s2['endpoint_evaluated_negi']:.4f} @ "
        f"{endpoint_diagnostics_s2['endpoint_scenario_position']:.1f}%; "
        f"ci_includes_zero={endpoint_diagnostics_s2['endpoint_ci_includes_zero']}; "
        f"local_stability={endpoint_diagnostics_s2['endpoint_local_stability']}; "
        f"knn_support={endpoint_diagnostics_s2['endpoint_knn_support']}; "
        f"robust={endpoint_diagnostics_s2['endpoint_robust']}",
    )
    summary.add_conclusion(
        "Scenario 2 endpoint classification:\n" + endpoint_classification_block
    )
    log_headline("\nEndpoint robustness assessment")
    for line in format_endpoint_headline(
        endpoint_diagnostics_s2,
        joint_support_status=joint_support.get("Scenario 2", {}).get("joint_support_status"),
        publication_convergence_status=convergence_status_for_report,
    ):
        log_headline(line)

    endpoint_diagnostics = {
        "endpoint_scenario_position_pct": float(s2.scenario_pct[_endpoint_idx]),
        "endpoint_ndvi":                  float(s2.ndvi[_endpoint_idx]),
        "endpoint_predicted_lst_c":       float(s2.lst[_endpoint_idx]),
        "endpoint_negi":                   float(negi.negi_s2[_endpoint_idx]),
        "endpoint_uncertainty_interval": {
            "lower_p2_5":  _endpoint_uncertainty_lo,
            "upper_p97_5": _endpoint_uncertainty_hi,
        },
        "endpoint_feature_support": {
            "within_formal_support": _endpoint_in_support,
            "nearest_neighbor_distance": float(support.mean_knn_dist_s2[_endpoint_idx]),
        },
        "endpoint_interpretation": (
            interpret_support("within bounds" if _endpoint_in_support else "partial extrapolation")
        ),
        "endpoint_robustness_assessment": endpoint_diagnostics_s2,
        "endpoint_robustness_interpretation": classify_endpoint_interpretation(endpoint_diagnostics_s2),
    }
    save_json(endpoint_diagnostics, cfg.data_dir / "scenario_endpoint_diagnostics.json")

    _imp = fi.table.set_index("Feature")["Importance"]
    _top = str(fi.table["Feature"].iloc[0])
    _others = ", ".join(
        f"{feat} ({_imp[feat]:.3f})" for feat in fi.table["Feature"].iloc[1:]
    )
    summary.add_conclusion(
        f"{_top} is the highest-ranked predictor of LST in Jeddah "
        f"(permutation importance = {_imp[_top]:.3f}); other predictors: {_others}. "
        "Rankings are marginal and may be affected by predictor collinearity."
    )
    summary.add_conclusion(interpretation_text("predictor_scope"))
    summary.add_conclusion(
        f"Spatial holdout R² = {validation.holdout_r2:.3f}, "
        f"RMSE = {validation.holdout_rmse:.2f} °C.  "
        f"Maximum predicted cooling ({negi.max_cooling_s2:.2f} °C) "
        f"is below the model RMSE; cooling-magnitude estimates should be "
        "interpreted with caution."
    )
    s1_at_boundary = is_near_trajectory_boundary(
        negi.s1_optimum_idx, len(negi.negi_s1), cfg.scenario_boundary_tolerance_pct,
        positions=negi.scenario_pct,
    )
    s2_at_boundary = is_near_trajectory_boundary(
        negi.s2_optimum_idx, len(negi.negi_s2), cfg.scenario_boundary_tolerance_pct,
        positions=negi.scenario_pct,
    )
    if s1_at_boundary:
        s1_max_pct  = float(negi.scenario_pct[negi.s1_optimum_idx])
        s1_max_negi = float(negi.negi_s1[negi.s1_optimum_idx])
        summary.add_conclusion(
            f"The highest evaluated NEGI for Scenario 1 occurs at the baseline "
            f"({s1_max_pct:.0f}% intervention; NEGI = {s1_max_negi:.3f}).  "
            "No interior optimum is identifiable under the isolated NDVI "
            "counterfactual (NDBI fixed) "
            "at the reference parameter values; the baseline value should not "
            "be read as a supported intervention optimum.  "
            + interpretation_text("scenario1_sign_convention")
        )
        summary.add_warning(
            "Scenario 1: " + interpretation_text("boundary_maximum") +
            "  Results are sensitive to cost-function parameters (w0, alpha, beta); "
            "see sensitivity analysis."
        )
    if s2_at_boundary:
        summary.add_warning("Scenario 2: " + interpretation_text("boundary_maximum"))

    report_step("[STEP 13] Generating figures...")

    log_info("  [13.1] Validation figures...")
    plot_actual_vs_predicted(validation, data, cfg)
    plot_model_comparison(comparison_df, validation, cfg)

    log_info("  [13.2] Calibration diagnostics...")
    FIGURE_REGISTRY.add(
        filename="calibration_metrics.csv",
        caption=generate_caption(
            "Calibration metrics table (calibration slope, intercept, mean bias, "
            "Pearson r) computed without recalibrating predictions.  "
            "RMSE and MAE are reported in the Confirmatory Holdout section.",
        ),
        description="Single-table calibration diagnostics export.",
        section="Calibration",
    )

    log_info("  [13.3] Residual diagnostics figures...")
    plot_residuals_histogram(resid, cfg)
    plot_residuals_qq(resid, cfg)
    report_qq_summary(resid.residuals, cfg, summary)
    plot_residuals_vs_predicted_full(
        resid, validation.holdout_pred, cfg, summary, registry=FIGURE_REGISTRY
    )
    plot_residual_spatial_map(spatial_resid, cfg, registry=FIGURE_REGISTRY)

    log_info("  [13.4] Feature importance...")
    plot_feature_importance(fi, cfg)

    log_info("  [13.5] Response curves...")
    plot_ndvi_vs_lst(df, cfg)
    plot_lst_response_to_ndbi(ndbi_sweep, cfg)
    plot_ndbi_response_uncertainty(ndbi_sweep, uncertainty, cfg)
    plot_ndvi_decile_analysis(
        ndvi_decile_df, corr_ndvi_lst, corr_ndbi_lst, corr_ndvi_ndbi, cfg
    )
    plot_ndvi_adjusted_slope_curve(varying_coefficient, cfg)
    plot_ndvi_conditional_by_ndbi_bins(
        conditional_results, df, cfg,
        fig_filename=FIG_NDVI_UNADJUSTED_STRATUM_SUPPLEMENT,
        title="Unadjusted Within-NDBI-Stratum Associations",
        caption=(
            "Supplementary, purely descriptive figure: four independent "
            "single-predictor (NDVI-only) regressions fit within each NDBI "
            "quartile bin. These are NOT conditional, partial, or causal "
            "effects, and are NOT the primary Figure 6 - see Figure 6 for "
            "the continuous varying-coefficient adjusted association."
        ),
    )

    log_info("  [13.6] Response surface...")
    plot_response_surface(surface, df, s2, cfg)

    log_info("  [13.7] Scenario support...")
    plot_scenario2_data_support(df, s2, support, cfg)

    log_info("  [13.8] Scenario comparative assessment...")
    _, smoothing_robust, peak_is_ood, peak_idx_raw = plot_negi_smoothing_robustness(
        negi, smoothing, s2, cfg, summary
    )
    plot_negi_exponent_sensitivity(negi, s1, cfg)
    plot_cooling_saturation_diagnostic(negi, s2, saturation, support, cfg)

    log_info("  [13.9] NEGI figures...")
    summary.log("NEGI scenario comparison", "Feature support flag", flag_str)
    plot_negi_scenario_comparison(negi, annotations, flag_str, cfg)
    plot_negi_energy_cost_comparison(negi, cfg)

    log_info("  [13.10] Uncertainty bands...")
    fig_unc, ax_unc, peak_iqr_idx = plot_negi_uncertainty_bands_primary(
        uncertainty, s1, cfg
    )
    set_axis_labels(ax_unc, SCENARIO_AXIS_LABEL, "NEGI")
    set_title(ax_unc, "NEGI Uncertainty Bands (Spatial-Block Holdout Refits)")
    set_legend(ax_unc, loc="lower left")
    fig_unc.text(
        0.5, -0.02,
        (
            f"Median ± IQR across {uncertainty.n_refits_used} spatial-block refits.  "
            f"Peak IQR at scenario = {s1.scenario_pct[peak_iqr_idx]:.0f}%.  "
            f"Calibration: pred = {validation.calibration_slope:.3f}*obs"
            f"{validation.calibration_intercept:+.3f}."
        ),
        ha="center", fontsize=7.5, color=cfg.color_neutral,
    )
    save_fig(fig_unc, FIG_NEGI_UNCERTAINTY_BANDS, cfg,
             caption="NEGI uncertainty bands across spatial-block holdout refits.")
    plot_negi_scenario2_robustness_full(uncertainty, s1, cfg)

    log_info("  [13.11] Sensitivity analysis...")
    plot_sensitivity_maximum_evaluated_negi(sensitivity, cfg)

    log_info("  [13.11b] Scenario 2 percentile sensitivity...")
    plot_scenario2_percentile_sensitivity(s2_percentile_sensitivity, cfg)
    plot_cost_regime_boundary(cost_regime, cfg)
    plot_decision_envelope(decision_envelope, cfg)

    log_info("  [13.12] Supplementary diagnostics complete.")

    report_step("[STEP 14] Exporting summary log...")
    summary.log("NEGI scenario comparison", "Smoothing robust (Savgol)", str(smoothing_robust))
    summary.log("NEGI scenario comparison", "Peak location OOD",         str(peak_is_ood))

    prediction_cache_stats = PREDICTION_CACHE.stats() if PREDICTION_CACHE is not None else None
    if prediction_cache_stats is not None:
        report_development(
            cfg,
            f"Prediction cache: {prediction_cache_stats['hits']} hits, "
            f"{prediction_cache_stats['misses']} misses "
            f"({prediction_cache_stats['hit_rate'] * 100:.1f}% hit rate).",
        )

    report_step("[STEP 14b] Running automated QA checks...")
    log_headline("\n[10] QA (engineering/data-consistency checks - see [7]-[9] above for scientific robustness)")
    _publication_summary_path = cfg.data_dir / "publication_summary.json"
    save_json(
        build_publication_summary(
            cfg, repro_info, validation, calibration_table, resid,
            spatial_resid, comparison_df, fi, scenario_summary_df,
            maximum_diagnostics_df, endpoint_diagnostics_df,
        ),
        _publication_summary_path,
    )
    qa_summary = run_qa_checks(
        cfg, validation, fi, repro_info,
        scenario_summary_df=scenario_summary_df,
        scenario_points_df=scenario_points_df,
        summary=summary,
        maximum_diagnostics_df=maximum_diagnostics_df,
        maximum_diagnostics_s2=maximum_diagnostics_s2,
        zero_crossing_df=zero_crossing_df,
        negi=negi,
        support=support,
        maximum_diagnostics_s1=maximum_diagnostics_s1,
        calibration_table_df=calibration_table,
        data=data,
        uncertainty=uncertainty,
        comparison_df=comparison_df,
        spatial_diagnostic=spatial_resid,
        joint_support_df=joint_support_df,
        block_size_sensitivity_df=block_size_sensitivity_df,
        endpoint_diagnostics_df=endpoint_diagnostics_df,
        endpoint_diagnostics_s1=endpoint_diagnostics_s1,
        endpoint_diagnostics_s2=endpoint_diagnostics_s2,
        publication_summary_path=_publication_summary_path,
        publication_convergence_status=convergence_status_for_report,
    )

    summary.export(
        cfg.data_dir / "full_summary_report.csv",
        cfg.data_dir / "pipeline_summary_log.txt",
        cfg=cfg,
        legacy_md_path=cfg.data_dir / "full_summary_report.md",
    )

    runtime_seconds = time.monotonic() - run_start
    log_step_finish()
    write_reproducibility_report(
        cfg, repro_info,
        best_params=model_results.best_params,
        validation=validation,
        runtime_seconds=runtime_seconds,
        convergence_status=convergence_status_for_report,
    )
    build_and_export_run_manifest(
        cfg, repro_info, runtime_seconds=runtime_seconds,
        validation=validation, qa_summary=qa_summary,
        scenario_summary_df=scenario_summary_df,
        n_train=len(data.X_train), n_holdout=len(data.X_test),
        convergence_criteria=_convergence_criteria,
        decision_envelope=decision_envelope, cost_regime=cost_regime_payload,
    )
    export_cache_status_json(cfg, prediction_cache_stats=prediction_cache_stats)
    export_runtime_summary_json(cfg, runtime_seconds=runtime_seconds)

    save_json(
        build_publication_summary(
            cfg, repro_info, validation, calibration_table, resid,
            spatial_resid, comparison_df, fi, scenario_summary_df,
            maximum_diagnostics_df, endpoint_diagnostics_df,
            qa_summary=qa_summary, runtime_seconds=runtime_seconds,
        ),
        _publication_summary_path,
    )

    engineering_qa = run_engineering_qa_checks(cfg)
    save_json(engineering_qa, cfg.data_dir / "engineering_qa.json")

    log_headline("Figures and detailed diagnostics exported successfully.")
    log_headline("\nPIPELINE COMPLETE")
    log_headline(f"  Figures  -> {cfg.results_dir}")
    log_headline(f"  Data     -> {cfg.data_dir}")
    log_headline(f"  Runtime  -> {runtime_seconds:.1f} s")


if __name__ == "__main__":
    main()