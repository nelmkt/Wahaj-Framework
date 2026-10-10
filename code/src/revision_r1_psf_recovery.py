"""Point-spread extension of the synthetic recovery test (supervisor review M6).

A synthetic 30 m landscape with a known per-pixel effect (beta) of greening and NO physical spillover is sensed
through a Gaussian point-spread function (full width at half maximum about 150 m, i.e. 1.5 x the 100 m sampling
distance of the Landsat 8 thermal band, Holmes et al. 2024), aggregated to 90 m cells, and analysed with the
estimators of Eqs. (3) and (6): the own-only zero-intercept slope and the joint model with external ring counts.
Because there is no physical spillover, any ring coefficient is optical blurring by the sensor.

Writes NEW tables_revision_r1/r1_psf_recovery.csv.  usage: python revision_r1_psf_recovery.py <package_root>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt, gaussian_filter

ROOT = Path(sys.argv[1]).resolve()
OUT = ROOT / "tables_revision_r1" / "r1_psf_recovery.csv"
if OUT.exists():
    raise SystemExit(f"Refusing to overwrite {OUT}")

BETA = -1.2
NOISE = 0.5
NC = 200
P_TREAT = 0.03
RINGS = ((0, 30), (30, 90), (90, 180), (180, 300))
N_REP = 20


def one(seed, fwhm_m, patch):
    """patch: greened land is placed as square patches of patch x patch cells (1 = isolated cells)."""
    rng = np.random.default_rng(seed)
    npx = NC * 3
    green = np.zeros((npx, npx), bool)
    treated = np.zeros((NC, NC), bool)
    while treated.mean() < P_TREAT:
        i, j = rng.integers(0, NC - patch, size=2)
        treated[i:i + patch, j:j + patch] = True
    dose = np.zeros((NC, NC), int)
    for i, j in zip(*np.nonzero(treated)):
        d = rng.integers(1, 10)
        k = rng.choice(9, size=d, replace=False)
        green[3 * i + k // 3, 3 * j + k % 3] = True
        dose[i, j] = d
    field = 9 * BETA * green.astype(float)
    if fwhm_m:
        field = gaussian_filter(field, sigma=fwhm_m / 2.3548 / 30.0, mode="constant")
    dy = field.reshape(NC, 3, NC, 3).mean(axis=(1, 3)) + rng.normal(0, NOISE, (NC, NC))
    dist_px = distance_transform_edt(~green) * 30.0
    dmin = dist_px.reshape(NC, 3, NC, 3).min(axis=(1, 3))
    control = (~treated) & (dmin > 300)
    tau = dy[treated] - dy[control].mean()
    d = dose[treated].astype(float)
    b_own = np.sum(d * tau) / np.sum(d * d)
    X = []
    for i, j in zip(*np.nonzero(treated)):
        r0, r1, c0, c1 = max(3 * i - 11, 0), min(3 * i + 14, npx), max(3 * j - 11, 0), min(3 * j + 14, npx)
        own = np.zeros((r1 - r0, c1 - c0), bool)
        own[3 * i - r0:3 * i - r0 + 3, 3 * j - c0:3 * j - c0 + 3] = True
        dd = distance_transform_edt(~own) * 30.0
        g = green[r0:r1, c0:c1] & ~own
        X.append([np.sum(g & (dd > lo) & (dd <= hi)) for lo, hi in RINGS])
    A = np.c_[d, np.asarray(X, float)]
    theta = np.linalg.lstsq(A, tau, rcond=None)[0]
    gamma = np.linalg.lstsq(d[:, None], np.asarray(X, float), rcond=None)[0][0]
    return dict(fwhm_m=fwhm_m, patch_cells=patch, n_treated=int(treated.sum()), n_controls=int(control.sum()),
                own_only_slope=b_own, joint_own=theta[0], external_part=float(np.dot(theta[1:], gamma)),
                **{f"ring_{lo}_{hi}m": t for (lo, hi), t in zip(RINGS, theta[1:])})


rows = [one(1000 + r, f, k) for k in (1, 3, 5) for f in (0.0, 150.0) for r in range(N_REP)]
raw = pd.DataFrame(rows)
summary = raw.groupby(["patch_cells", "fwhm_m"]).agg(["mean", "std"])
summary.columns = [f"{a}_{b}" for a, b in summary.columns]
summary = summary.reset_index()
summary.insert(1, "true_beta_K_per_px", BETA)
summary.insert(2, "replicates", N_REP)
summary["own_only_recovered_share"] = summary["own_only_slope_mean"] / BETA
summary["joint_own_recovered_share"] = summary["joint_own_mean"] / BETA
with open(OUT, "x", encoding="utf-8", newline="") as fh:
    summary.to_csv(fh, index=False)
print(summary.T.to_string())
