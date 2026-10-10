"""Joint spatial effect diagnostics. Supported-subset inference is conditional on original-fit screening."""
from __future__ import annotations
import numpy as np
import pandas as pd
from fw_config import SETTINGS
from fw_matching import _point, att
from fw_panel import coast_band, _grid_id
from fw_model import features, _xgb, Support


def tolerance_status(lo, hi, margin, n, blocks, valid, requested, cfg):
    if margin is None:
        return "tolerance_not_set"
    if n < cfg.min_validation_cells or blocks < cfg.min_validation_blocks:
        return "insufficient_supported_replication"
    if valid < cfg.min_validation_refits or valid < .95 * requested:
        return "insufficient_joint_refits"
    if not np.isfinite([lo, hi]).all():
        return "uncertainty_unavailable"
    return "within_tolerance" if lo >= -margin and hi <= margin else "not_demonstrated"


def compare(df, M, rs, mask, cfg, per_pixel=False):
    mask = mask & (rs["weights"] > 0)
    cells = df[mask]
    n, blocks = len(cells), cells["block"].nunique()
    empty = dict(measured_C=np.nan, measured_lo_C=np.nan, measured_hi_C=np.nan,
                 model_C=np.nan, model_lo_C=np.nan, model_hi_C=np.nan,
                 difference_C=np.nan, difference_lo_C=np.nan, difference_hi_C=np.nan,
                 n_cells=n, n_blocks=blocks, n_joint_valid=0, agrees=False, status="no_matched_cells")
    if not n:
        return empty
    ch = M.change(cells, cells.ndvi_post, cells.ndbi_post, cells.ndvi_pre, cells.ndbi_pre)
    codes, unique = pd.factorize(rs["key"])
    control = rs["controls"].to_numpy(bool)
    eligible = np.bincount(codes, control.astype(float), minlength=len(unique)) >= cfg.min_controls
    pos = df.index.get_indexer(cells.index)
    dose = df.n_greened_px.to_numpy(float)
    y = df.d_lst.to_numpy(float)
    treated = mask.to_numpy(bool)
    def one(weights, predictions):
        nc = np.bincount(codes, weights * control, minlength=len(unique))
        valid = nc[codes[pos]] > 0
        w = weights[pos] * valid
        if not np.any(w > 0):
            return np.nan, np.nan
        measured = _point(y, treated, control, codes, eligible, weights, dose if per_pixel else None)
        pred = np.sum(w*dose[pos]*predictions)/np.sum(w*dose[pos]**2) if per_pixel else np.average(predictions, weights=w)
        return measured, float(pred)
    measured, predicted = one(np.ones(len(df)), ch[0])
    draws = np.array([one(M.bootstrap.weights(r), ch[r+1]) for r in range(len(M.refits))])
    ok = np.isfinite(draws).all(axis=1)
    q = np.percentile(draws[ok], [2.5,97.5], axis=0) if ok.any() else np.full((2,2),np.nan)
    delta = draws[:,1]-draws[:,0]
    dlo,dhi = np.percentile(delta[ok],[2.5,97.5]) if ok.any() else (np.nan,np.nan)
    status = tolerance_status(dlo,dhi,cfg.validation_margin_C,n,blocks,int(ok.sum()),len(draws),cfg)
    return dict(measured_C=measured,measured_lo_C=q[0,0],measured_hi_C=q[1,0],
                model_C=predicted,model_lo_C=q[0,1],model_hi_C=q[1,1],difference_C=predicted-measured,
                difference_lo_C=dlo,difference_hi_C=dhi,n_cells=n,n_blocks=blocks,n_joint_valid=int(ok.sum()),
                agrees=status=="within_tolerance",status=status)


def run(df, M, R, cfg):
    if not M.panel_index.equals(df.index):
        raise ValueError("Joint resampling requires identical panel row order.")
    settings, doses, coasts, supported, supported_doses = [],[],[],[],[]
    for setting in SETTINGS:
        rs=R[setting]
        matched=rs["treated"] & (rs["weights"]>0)
        cells=df[matched]
        sup=pd.Series(False,index=df.index)
        if len(cells):
            sup.loc[cells.index]=M.supported(cells)&M.supported(cells,ndvi=cells.ndvi_pre,ndbi=cells.ndbi_pre)
        share=float(sup.sum()/len(cells)) if len(cells) else 0.
        settings.append(dict(setting=setting,supported_share=share,**compare(df,M,rs,matched,cfg)))
        supported.append(dict(setting=setting,supported_share=share,**compare(df,M,rs,matched&sup,cfg)))
        for lo,hi in cfg.dose_bins:
            label=f"{lo}–{hi}" if lo!=hi else str(lo)
            mask=matched&df.n_greened_px.between(lo,hi)
            if mask.sum()<cfg.min_cells_reported: continue
            doses.append(dict(setting=setting,pixels=label,**compare(df,M,rs,mask,cfg)))
            supported_doses.append(dict(setting=setting,pixels=label,**compare(df,M,rs,mask&sup,cfg)))
        band=coast_band(df.coast_km,cfg).to_numpy()
        for _,r in rs["by_coast"].iterrows():
            v=compare(df,M,rs,matched&(band==r.coast_band),cfg,per_pixel=True)
            coasts.append(dict(setting=setting,coast_band=r.coast_band,n_cells=v['n_cells'],
                               measured_px_C=v['measured_C'],measured_px_lo_C=v['measured_lo_C'],
                               measured_px_hi_C=v['measured_hi_C'],model_px_C=v['model_C'],
                               consistent=v['agrees'],status=v['status']))
    from fw_followup import support_diagnostics
    support_counts, support_coasts = support_diagnostics(df,M,R,cfg)
    return dict(by_setting=pd.DataFrame(settings),by_dose=pd.DataFrame(doses),by_coast=pd.DataFrame(coasts),
                support_counts=support_counts,support_coasts=support_coasts,
                supported=pd.DataFrame(supported),supported_by_dose=pd.DataFrame(supported_doses),
                validated={r['setting']:r['agrees'] for r in supported})


def region_holdouts(df,M,R,cfg):
    """Hold out each complete ~10km region plus a 300m bounding buffer.
    Regional point discrepancies are diagnostics, not equivalence tests or site-level validation.
    """
    rows=[]
    if not cfg.region_holdouts: return pd.DataFrame(rows)
    regions=_grid_id(df,cfg.region_deg)
    matched=pd.Series(False,index=df.index)
    for s in SETTINGS: matched |= R[s]['treated'] & (R[s]['weights']>0)
    for region in sorted(regions[matched].unique()):
        ix,iy=map(int,region.split('_'))
        lat0=iy*cfg.region_deg; lon0=ix*cfg.region_deg
        dlat=cfg.strict_exclusion_m/110570
        dlon=cfg.strict_exclusion_m/(111320*np.cos(np.radians(lat0)))
        tr=M.train
        exclude=tr.lon.between(lon0-dlon,lon0+cfg.region_deg+dlon)&tr.lat.between(lat0-dlat,lat0+cfg.region_deg+dlat)
        train=tr[~exclude]
        if len(train)<=cfg.support_k: continue
        fit=_xgb(cfg,cfg.seed).fit(features(train,cfg),train.lst_post)
        support=Support(features(train,cfg),cfg)
        for setting in SETTINGS:
            mask=matched&(regions==region)&(df.setting==setting)
            cells=df[mask]
            if len(cells)<cfg.min_cells_reported: continue
            x0=features(cells,cfg,ndvi=cells.ndvi_pre,ndbi=cells.ndbi_pre); x1=features(cells,cfg)
            pred=fit.predict(x1)-fit.predict(x0)
            measured=att(df,mask,R[setting]['controls'],'d_lst',R[setting]['key'],cfg,None)['estimate_C']
            rows.append(dict(setting=setting,region=region,n_cells=len(cells),measured_C=measured,
                             model_C=float(pred.mean()),difference_C=float(pred.mean()-measured),
                             supported_share=float((support.inside(x0)&support.inside(x1)).mean()),
                             training_cells=len(train),interval_status='point diagnostic; no regional equivalence claim'))
    return pd.DataFrame(rows)

