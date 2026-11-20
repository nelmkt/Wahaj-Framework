"""Benchmark of regressors for absolute LST on the v11 panel (read-only on the package; writes one NEW table).

Same data, predictors and validation as the framework's own cross-validation (code/src/fw_pipeline.py:64):
  training set  the representative sample without greened cells and without cells within 300 m of greening (Model B)
  predictors    Config.ml_features (seven)
  validation    GroupKFold on Config.block_deg blocks, Config.cv_folds folds; every model sees the same folds

The XGBoost row must reproduce tables/model_cv.csv; the script stops if it does not.
Settings of the comparison models are fixed and untuned, as XGBoost's are. This is not an optimized comparison.

usage: python benchmark_models_v11.py <package_root>      (run with PYTHONDONTWRITEBYTECODE=1)
"""
import sys
import time
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "code" / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

import fw_model as model  # noqa: E402
import fw_panel as panel  # noqa: E402
from fw_config import Config  # noqa: E402

OUT = ROOT / "tables/revision_v11" / "model_benchmark_v11.csv"
if OUT.exists():
    raise SystemExit(f"Refusing to overwrite {OUT}")

cfg = Config(panel_path=ROOT / "code" / "gee" / "panel.csv", out_dir=ROOT)
df, _ = panel.load(cfg)
train = model.training_set(df, cfg.strict_exclusion_m)
X, y = model.features(train, cfg), train["lst_post"].to_numpy(float)
folds = list(GroupKFold(n_splits=cfg.cv_folds).split(X, y, groups=train["block"]))

MODELS = [
    ("XGBoost", "n_estimators=600, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, "
                "min_child_weight=5, reg_lambda=1.0 (code/src/fw_config.py:54-55)",
     lambda: model._xgb(cfg, cfg.seed)),
    ("Gradient boosting (scikit-learn)", "n_estimators=300, max_depth=3, learning_rate=0.1, subsample=0.8",
     lambda: GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.1, subsample=0.8,
                                       random_state=cfg.seed)),
    ("Random forest", "n_estimators=300, min_samples_leaf=5",
     lambda: RandomForestRegressor(n_estimators=300, min_samples_leaf=5, n_jobs=cfg.jobs, random_state=cfg.seed)),
    ("Linear regression", "ordinary least squares", lambda: LinearRegression()),
]

rows = []
for name, settings, make in MODELS:
    t0 = time.time()
    oof = np.full(len(y), np.nan)
    for tr, te in folds:
        oof[te] = make().fit(X.iloc[tr], y[tr]).predict(X.iloc[te])
    res = y - oof
    rows.append(dict(model=name, r2=float(1 - np.sum(res ** 2) / np.sum((y - y.mean()) ** 2)),
                     rmse_C=float(np.sqrt(np.mean(res ** 2))), mae_C=float(np.mean(np.abs(res))),
                     n=int(len(y)), folds=cfg.cv_folds, n_blocks=int(train["block"].nunique()),
                     features=" ".join(cfg.ml_features), settings=settings, seconds=round(time.time() - t0, 1)))
    print(rows[-1]["model"], round(rows[-1]["r2"], 4), round(rows[-1]["rmse_C"], 4), round(rows[-1]["mae_C"], 4), flush=True)

saved = pd.read_csv(ROOT / "tables" / "model_cv.csv").iloc[0]
x = rows[0]
if x["n"] != int(saved["n"]) or abs(x["r2"] - saved["r2"]) > 1e-6 or abs(x["rmse_C"] - saved["rmse_C"]) > 1e-6:
    raise SystemExit(f"XGBoost row does not reproduce tables/model_cv.csv: {x['r2']} vs {saved['r2']}, n {x['n']} vs {saved['n']}")
out = pd.DataFrame(rows)
out["xgboost_matches_model_cv"] = True
out.to_csv(OUT, index=False, mode="x")
print("wrote", OUT)
