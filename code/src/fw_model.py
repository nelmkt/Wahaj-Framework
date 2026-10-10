"""Machine-learning model of summer land surface temperature across the city (XGBoost).

The model learns, from a representative 10% sample of the study area in summer 2024–25, how LST varies with the
surface (NDVI, NDBI), the ground's fixed emissivity, elevation, distance to the coast and position. Cells that greened
are left out of training. Nearby observations and shared controls still matter for dependence and transfer.

What the model is asked: for a cell, how much cooler is its LST with its surface as it is than with its surface set to
another state (NDVI and NDBI only; everything else about the cell held fixed). A prediction is used only where the
changed cell still looks like real cells in the training data (feature-space support).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

from fw_config import Config

POST = {"ndvi": "ndvi_post", "ndbi": "ndbi_post", "emis": "emis_post"}


def features(df: pd.DataFrame, cfg: Config, ndvi=None, ndbi=None) -> pd.DataFrame:
    """Model inputs for each cell in the post period; ndvi/ndbi replace the surface when given."""
    X = pd.DataFrame(index=df.index)
    for f in cfg.ml_features:
        X[f] = df[POST.get(f, f)].to_numpy(float)
    if ndvi is not None:
        X["ndvi"] = np.asarray(ndvi, float) if np.ndim(ndvi) else float(ndvi)
    if ndbi is not None:
        X["ndbi"] = np.asarray(ndbi, float) if np.ndim(ndbi) else float(ndbi)
    return X


def _xgb(cfg: Config, seed: int) -> XGBRegressor:
    return XGBRegressor(objective="reg:squarederror", random_state=seed, n_jobs=cfg.jobs, **cfg.xgb_params)


def training_set(df: pd.DataFrame, exclude_near_m: float | None = None) -> pd.DataFrame:
    """The representative sample without any greened cell (and, for the stricter check, without any cell within
    exclude_near_m of greened land). This exclusion is not whole-region independence."""
    keep = df["in_sample"] & (df["group"] != "greened")
    if exclude_near_m is not None:
        keep &= df["dist_greened_m"] > exclude_near_m
    return df[keep]


def spatial_cv(train: pd.DataFrame, cfg: Config) -> dict:
    """Absolute-LST skill holding out whole blocks. Adjacent blocks can still share boundaries."""
    X, y = features(train, cfg), train["lst_post"].to_numpy(float)
    oof = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits=cfg.cv_folds).split(X, y, groups=train["block"]):
        m = _xgb(cfg, cfg.seed).fit(X.iloc[tr], y[tr])
        oof[te] = m.predict(X.iloc[te])
    res = y - oof
    return {"r2": float(1 - np.sum(res ** 2) / np.sum((y - y.mean()) ** 2)), "rmse_C": float(np.sqrt(np.mean(res ** 2))),
            "mae_C": float(np.mean(np.abs(res))), "n": int(len(y)), "folds": cfg.cv_folds,
            "oof": pd.DataFrame({"observed": y, "predicted": oof})}


class Support:
    """Is a (changed) cell still like real cells? Mean distance to the k nearest training cells in standardised
    feature space, against the same distance among training cells (own row excluded)."""

    def __init__(self, Xtr: pd.DataFrame, cfg: Config):
        A = Xtr.to_numpy(float)
        self.mu, sd = A.mean(0), A.std(0)
        self.sd = np.where(sd > 0, sd, 1.0)
        self.k = cfg.support_k
        self.tree = cKDTree((A - self.mu) / self.sd)
        d, _ = self.tree.query((A - self.mu) / self.sd, k=self.k + 1)
        self.threshold = float(np.percentile(d[:, 1:].mean(1), cfg.support_percentile))

        self.index = pd.Index(Xtr.index)

    def inside(self, X: pd.DataFrame) -> np.ndarray:
        """A cell's own training row never counts as its neighbour, so the question is always whether OTHER real
        cells look like it."""
        d, i = self.tree.query((X.to_numpy(float) - self.mu) / self.sd, k=self.k + 1)
        own = self.index.get_indexer(X.index)
        keep = i != own[:, None]
        use = keep & (np.cumsum(keep, axis=1) <= self.k)
        return (d * use).sum(1) / use.sum(1) <= self.threshold


class Model:
    def __init__(self, df: pd.DataFrame, cfg: Config, exclude_near_m: float | None = None, refit: bool = True):
        self.cfg = cfg
        self.train = training_set(df, exclude_near_m)
        Xtr = features(self.train, cfg)
        self.fit = _xgb(cfg, cfg.seed).fit(Xtr, self.train["lst_post"].to_numpy(float))
        self.support = Support(Xtr, cfg)
        from fw_matching import Bootstrap
        self.bootstrap = Bootstrap(df["block"], cfg.n_refits, cfg.seed)
        self.panel_index = df.index.copy()
        train_positions = df.index.get_indexer(self.train.index)
        self.refits = []
        if not refit:
            return
        for r in range(cfg.n_refits):
            counts = self.bootstrap.weights(r)[train_positions].astype(int)
            idx = np.repeat(np.arange(len(self.train)), counts)
            if not len(idx):
                raise ValueError("Spatial draw contains no training cells; revise blocks.")
            self.refits.append(_xgb(cfg, cfg.seed).fit(Xtr.iloc[idx], self.train["lst_post"].to_numpy(float)[idx]))
            if (r + 1) % 25 == 0:
                print(f"  joint model refits {r+1}/{cfg.n_refits}", flush=True)

    def change(self, df: pd.DataFrame, to_ndvi, to_ndbi, from_ndvi=None, from_ndbi=None, models=None) -> np.ndarray:
        """Predicted LST(to) − LST(from) per cell for each model (rows: models; negative = cooler)."""
        X1 = features(df, self.cfg, ndvi=to_ndvi, ndbi=to_ndbi)
        X0 = features(df, self.cfg, ndvi=from_ndvi, ndbi=from_ndbi)
        models = models if models is not None else [self.fit] + self.refits
        return np.vstack([m.predict(X1) - m.predict(X0) for m in models])

    def supported(self, df: pd.DataFrame, ndvi=None, ndbi=None) -> np.ndarray:
        return self.support.inside(features(df, self.cfg, ndvi=ndvi, ndbi=ndbi))

    def importance(self) -> pd.DataFrame:
        g = self.fit.get_booster().get_score(importance_type="gain")
        tot = sum(g.values()) or 1.0
        return pd.DataFrame({"feature": list(self.cfg.ml_features),
                             "gain_share": [g.get(f, 0.0) / tot for f in self.cfg.ml_features]})
