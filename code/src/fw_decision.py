"""Scenario accounting, conditional prediction diagnostics and joint ranking sensitivity."""
from __future__ import annotations
import itertools
import numpy as np
import pandas as pd
from fw_config import SETTINGS
from fw_matching import Bootstrap
from fw_model import Support, features
from fw_panel import PIXEL_HA
from fw_allocation import allocate


def bounds(label):
    p=label.split("–")
    return int(p[0]),int(p[-1])


def ratio_summary(draws):
    valid=np.isfinite(draws)
    cool=valid & (draws>0)
    return dict(valid_draws=int(valid.sum()),nonpositive_benefit_share=float(np.mean(draws[valid]<=0)) if valid.any() else np.nan,
                low=float(np.percentile(1/draws[cool],2.5)) if cool.any() and not np.any(draws[valid]<=0) else np.nan,
                high=float(np.percentile(1/draws[cool],97.5)) if cool.any() and not np.any(draws[valid]<=0) else np.nan)


def area_draws(df,rs,mask,boot,cfg):
    """Mean planted area over exactly the same eligible/resampled cells as each matched dose effect."""
    codes,unique=pd.factorize(rs["key"])
    ctrl=rs["controls"].to_numpy(bool)
    eligible=np.bincount(codes,ctrl.astype(float),minlength=len(unique))>=cfg.min_controls
    treated=mask.to_numpy(bool)&eligible[codes]
    dose=df.n_greened_px.to_numpy(float)
    areas=[]
    for r in range(len(boot.draws)):
        w=boot.weights(r)
        nc=np.bincount(codes,w*ctrl,minlength=len(unique))
        ok=treated&(nc[codes]>0)
        areas.append(PIXEL_HA*np.average(dose[ok],weights=w[ok]) if w[ok].sum()>0 else np.nan)
    return np.array(areas)


def candidates(df,M,R,V,cfg):
    """Exploratory transitions only. No automatic municipal recommendations or allocation inputs."""
    output=[]
    for setting in SETTINGS:
        if not V["validated"].get(setting,False): continue
        rs=R[setting]
        tests=V["supported_by_dose"]
        if not len(tests): continue
        for _,test in tests[tests.setting==setting].iterrows():
            if not test.agrees: continue
            lo,hi=bounds(test.pixels)
            treated=df[rs["treated"]&(rs["weights"]>0)&df.n_greened_px.between(lo,hi)]
            supported=M.supported(treated)&M.supported(treated,ndvi=treated.ndvi_pre,ndbi=treated.ndbi_pre)
            treated=treated[supported]
            if len(treated)<=cfg.support_k: continue
            cols=["ndvi_pre","ndbi_pre","ndvi_post","ndbi_post","n_greened_px"]
            scale=treated[cols].std().replace(0,1)
            distance=(((treated[cols]-treated[cols].median())/scale)**2).sum(axis=1)
            target=treated.loc[distance.idxmin()]
            cand=df[(df.group=="control")&(df.setting==setting)&(df.d_own_built.abs()<cfg.stable_surface_max)]
            if not len(cand): continue
            ndvi=cand.ndvi_post+(target.ndvi_post-target.ndvi_pre)
            ndbi=cand.ndbi_post+(target.ndbi_post-target.ndbi_pre)
            training_ok=M.supported(cand)&M.supported(cand,ndvi=ndvi,ndbi=ndbi)
            tested_pre=Support(features(treated,cfg,ndvi=treated.ndvi_pre,ndbi=treated.ndbi_pre),cfg)
            tested_post=Support(features(treated,cfg),cfg)
            tested_ok=tested_pre.inside(features(cand,cfg))&tested_post.inside(features(cand,cfg,ndvi=ndvi,ndbi=ndbi))
            ok=training_ok&tested_ok&ndvi.between(-1,1).to_numpy()&ndbi.between(-1,1).to_numpy()
            selected=cand[ok]
            if not len(selected): continue
            change=M.change(selected,ndvi[ok],ndbi[ok])
            q=np.percentile(change[1:],[2.5,97.5],axis=0)
            part=selected[["lon","lat"]].copy()
            part["setting"]=setting; part["dose"]=test.pixels
            part["target_observed_cell_index"]=str(target.name)
            part["pixels"]=int(target.n_greened_px)
            part["cooling_C"]=change[0]; part["lo_C"]=q[0]; part["hi_C"]=q[1]
            part["status"]=np.where(q[1]<0,"exploratory_cooling_only","no_robust_cooling")
            output.append(part)
    columns=["lon","lat","setting","dose","target_observed_cell_index","pixels","cooling_C","lo_C","hi_C","status"]
    return pd.concat(output,ignore_index=True) if output else pd.DataFrame(columns=columns)


def run(df,M,R,V,L,cfg):
    W=L["water"].set_index("level").m3_per_ha_yr
    boot=Bootstrap(df.block,cfg.n_boot,cfg.seed)
    summary,scenarios,joint=[],[],[]
    for setting in SETTINGS:
        rs=R[setting]
        for _,r in rs["dose"].iterrows():
            if not np.isfinite(r.estimate_C): continue
            lo,hi=bounds(r.pixels)
            mask=rs["treated"]&(rs["weights"]>0)&df.n_greened_px.between(lo,hi)
            area=PIXEL_HA*df.loc[mask,"n_greened_px"].mean()
            cooling=rs["reps"][f"dose {lo}-{hi}"]
            areas=area_draws(df,rs,mask,boot,cfg)
            with np.errstate(divide='ignore',invalid='ignore'):
                benefit_per_m3=-cooling/(areas*W["central"])
            rsumm=ratio_summary(benefit_per_m3)
            central=W["central"]*area/-r.estimate_C if r.estimate_C<0 else np.nan
            ident=f"{setting}: {r.pixels}"
            result=dict(option_id=ident,setting=setting,basis=f"measured: {r.pixels} of 9 pixels greened",
                        cells=int(r.n_treated_matched),pixels_mean=area/PIXEL_HA,cooling_C=r.estimate_C,
                        range_low_C=r.lo_C,range_high_C=r.hi_C,range_kind="95% spatial interval",
                        water_m3=W["central"]*area,m3_per_degree=central,
                        m3_per_degree_low=rsumm["low"],m3_per_degree_high=rsumm["high"],
                        scenario_low=rsumm["low"]*W["low"]/W["central"],
                        scenario_high=rsumm["high"]*W["high"]/W["central"],
                        nonpositive_benefit_share=rsumm["nonpositive_benefit_share"],
                        interpretation="hypothetical irrigation accounting; observed water regime unverified")
            for source,intensities in cfg.kwh_per_m3.items():
                result[f"MWh_per_degree [{source}]"]=central*intensities[1]/1000
                for wi,depth in W.items():
                    for ei,intensity in zip(("low","central","high"),intensities):
                        scenarios.append(dict(option_id=ident,water_case=wi,source=source,energy_case=ei,
                            m3_per_degree=central*depth/W["central"],
                            m3_per_degree_low=rsumm["low"]*depth/W["central"],
                            m3_per_degree_high=rsumm["high"]*depth/W["central"],
                            MWh_per_degree=central*depth/W["central"]*intensity/1000,
                            MWh_per_degree_low=rsumm["low"]*depth/W["central"]*intensity/1000,
                            MWh_per_degree_high=rsumm["high"]*depth/W["central"]*intensity/1000))
            summary.append(result)
            joint.append(benefit_per_m3)
    S=pd.DataFrame(summary)
    rank=[]; pairwise=[]
    if joint:
        values=np.array(joint).T
        complete=np.isfinite(values).all(axis=1)&(values>0).all(axis=1)
        if complete.any():
            maxima=values[complete].max(axis=1,keepdims=True)
            wins=np.isclose(values[complete],maxima,rtol=1e-12,atol=0)
            shares=(wins/wins.sum(axis=1,keepdims=True)).mean(axis=0)
        else: shares=np.full(len(S),np.nan)
        for i,r in S.iterrows():
            rank.append(dict(option_id=r.option_id,share_best_complete_draws=shares[i],
                complete_draws=int(complete.sum()),total_draws=len(values),
                note="joint bootstrap frequencies conditional on shared irrigation assumptions"))
        for i,j in itertools.combinations(range(len(S)),2):
            a,b=S.iloc[i],S.iloc[j]
            ok=np.isfinite(values[:,i])&np.isfinite(values[:,j])&(values[:,i]>0)&(values[:,j]>0)
            pairwise.append(dict(option_A=a.option_id,option_B=b.option_id,
                water_intensity_A_over_B_at_point_tie=b.m3_per_degree/a.m3_per_degree,
                share_A_more_efficient=float((values[ok,i]>values[ok,j]).mean()) if ok.any() else np.nan,
                valid_joint_draws=int(ok.sum())))
    bycoast=[]
    vc=V.get("strict_by_coast",pd.DataFrame())
    for setting in SETTINGS:
        for _,r in R[setting]["by_coast"].iterrows():
            pred=vc[(vc.setting==setting)&(vc.coast_band==r.coast_band)] if len(vc) else vc
            bycoast.append(dict(setting=setting,coast_band=r.coast_band,cells=r.n_treated_matched,
                 cooling_px_C=r.estimate_C,lo_C=r.lo_C,hi_C=r.hi_C,
                 m3_per_degree=PIXEL_HA*W["central"]/-r.estimate_C if r.estimate_C<0 else np.nan,
                 model_px_C=float(pred.iloc[0].model_px_C) if len(pred) else np.nan,
                 model_consistent=bool(pred.iloc[0].consistent) if len(pred) else False))
    return dict(summary=S,scenarios=pd.DataFrame(scenarios),ranking=pd.DataFrame(rank),
                pairwise=pd.DataFrame(pairwise),by_coast=pd.DataFrame(bycoast),
                candidates=candidates(df,M,R,V,cfg),allocation=allocate(cfg),targets={},
                support_coasts=V.get("support_coasts",pd.DataFrame()))

