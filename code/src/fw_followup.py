"""Exploratory paired contrasts, tighter matching, and panel DR-DiD sensitivities.
No specification here is retrospectively prespecified. DR targets cell ATT, not a dose slope.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
import fw_matching as matching, fw_panel as panel, fw_model as model
from fw_config import SETTINGS

BASE=["lst_pre","ndvi_pre","ndbi_pre","emis_pre","coast_km","elev","ghsl_2015","ghsl_nb_2015"]
BALANCE=BASE+["d_nb_built","lon","lat"]
CALIPER=["lst_pre","ndvi_pre","elev"]

def interval(draws):
    a=np.asarray(draws); valid=np.isfinite(a)
    q=np.percentile(a[valid],[2.5,97.5]) if valid.any() else [np.nan,np.nan]
    return dict(lo_C=q[0],hi_C=q[1],n_valid=int(valid.sum()),n_draws=len(a))

def setup(df,cfg):
    key=panel.strata(df,cfg); result={}
    for s in SETTINGS:
        t=(df.group=="greened")&(df.setting==s)
        c=(df.group=="control")&(df.setting==s)&(df.d_own_built.abs()<cfg.stable_surface_max)
        w=matching.control_weights(df,t,c,key,cfg)
        result[s]=dict(treated=t,controls=c,key=key,weights=w)
    return result

def paired_contrasts(df,R,cfg,boot):
    from fw_decision import area_draws
    mix=[]; coastal=[]
    for s in SETTINGS:
        rs=R[s]; t=rs["treated"]&(rs["weights"]>0); c=rs["controls"]; key=rs["key"]
        full=t&(df.n_greened_px==9)
        if full.sum()>=cfg.min_cells_reported:
            f=matching.att(df,full,c,"d_lst",key,cfg,boot,keep_reps=True)
            for lo,hi in cfg.dose_bins[:-1]:
                mask=t&df.n_greened_px.between(lo,hi)
                if mask.sum()<cfg.min_cells_reported:continue
                e=matching.att(df,mask,c,"d_lst",key,cfg,boot,keep_reps=True)
                count=df.loc[mask,"n_greened_px"].mean()
                counts=area_draws(df,rs,mask,boot,cfg)/panel.PIXEL_HA
                benchmark=f["estimate_C"]*count/9
                bdraw=f["_reps"]*counts/9
                delta=e["_reps"]-bdraw
                mix.append(dict(setting=s,pixels=f"{lo}–{hi}",n_cells=int(mask.sum()),mean_pixels=count,
                    measured_C=e["estimate_C"],benchmark_C=benchmark,difference_C=e["estimate_C"]-benchmark,
                    **interval(delta),benchmark_lo_C=interval(bdraw)["lo_C"],benchmark_hi_C=interval(bdraw)["hi_C"]))
        near=t&df.coast_km.between(5,10,inclusive="left"); far=t&(df.coast_km>=10)
        if near.sum() and far.sum():
            a=matching.att(df,near,c,"d_lst",key,cfg,boot,dose="n_greened_px",keep_reps=True)
            b=matching.att(df,far,c,"d_lst",key,cfg,boot,dose="n_greened_px",keep_reps=True)
            coastal.append(dict(setting=s,contrast="at least 10 km minus 5-10 km",n_near=int(near.sum()),
                n_far=int(far.sum()),difference_C=b["estimate_C"]-a["estimate_C"],
                **interval(b["_reps"]-a["_reps"])))
    return pd.DataFrame(mix),pd.DataFrame(coastal)

def support_diagnostics(df,M,R,cfg):
    rows=[]; coasts=[]
    for s in SETTINGS:
        rs=R[s]; t=rs["treated"]&(rs["weights"]>0)
        cells=df[t]; sup=pd.Series(False,index=df.index)
        sup.loc[cells.index]=M.supported(cells)&M.supported(cells,ndvi=cells.ndvi_pre,ndbi=cells.ndbi_pre)
        for lo,hi in cfg.dose_bins:
            mask=t&df.n_greened_px.between(lo,hi); n=int(mask.sum()); ns=int((mask&sup).sum())
            rows.append(dict(setting=s,pixels=f"{lo}–{hi}" if lo!=hi else str(lo),n_matched=n,n_supported=ns,
                support_share=ns/n if n else np.nan,n_supported_blocks=int(df.loc[mask&sup,"block"].nunique())))
        for band in panel.coast_band([0],cfg).cat.categories:
            bandmask=t&(panel.coast_band(df.coast_km,cfg).to_numpy()==band)
            for name,mask in [("supported",bandmask&sup),("unsupported",bandmask&~sup)]:
                cells=df[mask]; n=len(cells)
                measured=matching.att(df,mask,rs["controls"],"d_lst",rs["key"],cfg,None,dose="n_greened_px")["estimate_C"] if n else np.nan
                if n:
                    ch=M.change(cells,cells.ndvi_post,cells.ndbi_post,cells.ndvi_pre,cells.ndbi_pre,models=[M.fit])[0]
                    dose=cells.n_greened_px.to_numpy(float)
                    pred=float(np.dot(ch,dose)/np.dot(dose,dose))
                else: pred=np.nan
                coasts.append(dict(setting=s,coast_band=band,subset=name,n_cells=n,n_all=int(bandmask.sum()),
                    measured_px_C=measured,model_px_C=pred,difference_px_C=pred-measured,
                    status="point diagnostic only; support is not accuracy"))
    return pd.DataFrame(rows),pd.DataFrame(coasts)

def balance_rows(df,t_weights,c_weights,sd,setting,spec):
    rows=[]
    for k in BALANCE:
        mt=np.average(df[k],weights=t_weights) if t_weights.sum() else np.nan
        mc=np.average(df[k],weights=c_weights) if c_weights.sum() else np.nan
        rows.append(dict(setting=setting,specification=spec,covariate=k,treated_mean=mt,control_mean=mc,
            standardized_difference=(mt-mc)/sd[k] if sd[k]>0 else np.nan))
    return rows

def caliper_matrix(df,t,c,key,sd,width,cfg):
    ti=np.flatnonzero(t); ci=np.flatnonzero(c); rr=[]; cc=[]
    for j,i in enumerate(ti):
        pool=ci[key.iloc[ci].to_numpy()==key.iloc[i]]
        if len(pool):
            diff=np.abs(df.iloc[pool][CALIPER].to_numpy()-df.iloc[i][CALIPER].to_numpy(float))
            ok=(diff<=width*sd[CALIPER].to_numpy()).all(axis=1)
            ok&=np.abs(df.iloc[pool].coast_km.to_numpy()-df.iloc[i].coast_km)<=.5
            pool=pool[ok]
        if len(pool)>=cfg.min_controls:
            rr.extend([j]*len(pool));cc.extend(pool)
    A=csr_matrix((np.ones(len(rr)),(rr,cc)),shape=(len(ti),len(df)))
    keep=np.asarray(A.sum(axis=1)).ravel()>0
    return ti[keep],A[keep]

def caliper_effect(y,dose,ti,A,w):
    den=np.asarray(A@w).ravel(); ok=(den>0)&(w[ti]>0)
    if not ok.any():return np.nan,np.nan
    dif=y[ti[ok]]-np.asarray(A@(w*y)).ravel()[ok]/den[ok]
    tw=w[ti[ok]]; dz=dose[ti[ok]]
    return np.average(dif,weights=tw),np.dot(tw*dz,dif)/np.dot(tw,dz*dz)

def dr_panel(y,D,X):
    """Normalized panel DR score, logistic MLE + untreated-change OLS.
    Influence includes both nuisance fits. No hidden trimming or penalization.
    Reference: Sant'Anna & Zhao (2020), eq. 3.1.
    """
    n=len(y); D=np.asarray(D,float)
    if np.linalg.matrix_rank(X)!=X.shape[1]: raise ValueError("Rank-deficient DR design")
    def fg(beta):
        eta=X@beta
        return np.mean(np.logaddexp(0,eta)-D*eta), X.T@(expit(eta)-D)/n
    fit=minimize(fg,np.zeros(X.shape[1]),jac=True,method="BFGS",options={"gtol":1e-8,"maxiter":1000})
    if np.max(np.abs(fit.jac))>1e-6:raise ValueError("Propensity model failed convergence")
    p=expit(X@fit.x)
    if np.any(p>=1-1e-8):raise ValueError("Near-complete propensity separation; DR result withheld")
    c=D==0; beta=np.linalg.lstsq(X[c],y[c],rcond=None)[0]; resid=y-X@beta
    tw=D/D.mean(); raw=(1-D)*p/(1-p); cw=raw/raw.mean()
    muT=np.mean(tw*resid);muC=np.mean(cw*resid)
    A=X.T@((1-D)[:,None]*X)/n
    B=X.T@((p*(1-p))[:,None]*X)/n
    if max(np.linalg.cond(A),np.linalg.cond(B))>1e12:raise ValueError("Ill-conditioned nuisance model")
    ifbeta=((1-D)*resid)[:,None]*X@np.linalg.inv(A)
    ifgamma=(D-p)[:,None]*X@np.linalg.inv(B)
    gb=np.mean((cw-tw)[:,None]*X,axis=0)
    gg=-np.mean((cw*(resid-muC))[:,None]*X,axis=0)
    influence=tw*(resid-muT)-cw*(resid-muC)+ifbeta@gb+ifgamma@gg
    outcome=np.mean(D*(y-X@beta))/D.mean()
    orinf=tw*(resid-outcome)-ifbeta@np.mean(tw[:,None]*X,axis=0)
    return dict(estimate=muT-muC,or_estimate=outcome,influence=influence,or_influence=orinf,
        control_weights=raw,max_control_ps=float(p[c].max()),
        control_weight_ess=float(raw.sum()**2/np.dot(raw,raw)),max_control_weight_share=float(raw.max()/raw.sum()),
        gradient_max=float(np.max(np.abs(fit.jac))))

def dr_design(d):
    vals=d[BASE+["lon","lat"]].to_numpy(float)
    sd=vals.std(axis=0);Z=(vals-vals.mean(axis=0))/np.where(sd>0,sd,1)
    extra=np.column_stack([Z[:,0]**2,Z[:,1]**2])
    regions=pd.get_dummies(panel._grid_id(d,.1),drop_first=True,dtype=float).to_numpy()
    X=np.column_stack([np.ones(len(d)),Z,extra,regions])
    return X[:,np.r_[True,np.std(X[:,1:],axis=0)>1e-10]]

def matching_sensitivity(df,R,cfg,boot):
    results=[]; balances=[]; all_bias=[]
    y=df.d_lst.to_numpy(); dose=df.n_greened_px.to_numpy(float); one=np.ones(len(df))
    for s in SETTINGS:
        rs=R[s];t=rs["treated"]&(rs["weights"]>0);c=rs["controls"];key=rs["key"]
        sd=((df.loc[rs["treated"],BALANCE].var()+df.loc[c,BALANCE].var())/2)**.5
        def add(name,mask,ctrl,k):
            e=matching.att(df,mask,ctrl,"d_lst",k,cfg,boot)
            px=matching.att(df,mask,ctrl,"d_lst",k,cfg,boot,dose="n_greened_px")
            w=matching.control_weights(df,mask,ctrl,k,cfg)
            results.append(dict(setting=s,specification=name,estimand="mean cell ATT",estimate_C=e["estimate_C"],
                lo_C=e["lo_C"],hi_C=e["hi_C"],n_retained=e["n_treated_matched"],n_controls=e["n_controls_matched"],
                per_pixel_C=px["estimate_C"],n_valid=e["n_boot_valid"],inference="full spatial resampling"))
            balances.extend(balance_rows(df,w*mask,w*ctrl,sd,s,name))
        add("main",rs["treated"],c,key)
        add("baseline-only strata",rs["treated"],c,panel.strata(df,cfg,baseline_only=True))
        for width in [.5,.2]:
            name=f"calipers {width:g} SD plus 0.5 km coast"
            ti,A=caliper_matrix(df,t,c,key,sd,width,cfg)
            point,px=caliper_effect(y,dose,ti,A,one)
            draws=np.array([caliper_effect(y,dose,ti,A,boot.weights(i))[0] for i in range(cfg.n_boot)])
            den=np.asarray(A.sum(axis=1)).ravel()
            cw=np.asarray(A.T@(1/den)).ravel() if len(ti) else np.zeros(len(df))
            tw=np.zeros(len(df));tw[ti]=1
            results.append(dict(setting=s,specification=name,estimand="retained-cell ATT",estimate_C=point,**interval(draws),
                n_retained=len(ti),n_controls=int((cw>0).sum()),per_pixel_C=px,
                inference="spatial resampling; fixed baseline caliper graph"))
            balances.extend(balance_rows(df,tw,cw,sd,s,name))
            mask=pd.Series(tw>0,index=df.index); add(f"main on {width:g} SD retained cells",mask,c,key)
        keep=(rs["weights"]>0)&(rs["treated"]|c)
        d=df[keep];pos=np.flatnonzero(keep);D=(d.group=="greened").to_numpy(float)
        try:
            dr=dr_panel(d.d_lst.to_numpy(float),D,dr_design(d))
            for mode in ["DR","outcome regression"]:
                name=mode+" on main matched population"
                est=dr["estimate" if mode=="DR" else "or_estimate"]
                inf=dr["influence" if mode=="DR" else "or_influence"]
                draws=np.array([est+np.dot(boot.weights(i)[pos]-1,inf)/len(d) for i in range(cfg.n_boot)])
                results.append(dict(setting=s,specification=name,estimand="mean cell ATT",estimate_C=est,**interval(draws),
                    n_retained=int(D.sum()),n_controls=int((1-D).sum()),per_pixel_C=np.nan,
                    inference="spatial block influence approximation including nuisance estimation",
                    max_control_ps=dr["max_control_ps"] if mode=="DR" else np.nan,
                    control_weight_ess=dr["control_weight_ess"] if mode=="DR" else np.nan,
                    max_control_weight_share=dr["max_control_weight_share"] if mode=="DR" else np.nan))
                tw=np.zeros(len(df));tw[pos]=D;cw=np.zeros(len(df))
                cw[pos]=dr["control_weights"] if mode=="DR" else (1-D)
                balances.extend(balance_rows(df,tw,cw,sd,s,name))
        except ValueError as e:
            results.append(dict(setting=s,specification="DR/outcome regression",estimate_C=np.nan,n_retained=int(D.sum()),
                inference="withheld: "+str(e)))
        print("  matching sensitivities complete: "+s,flush=True)
    frame=pd.DataFrame(results)
    for _,r in frame.iterrows():
        for bias in np.arange(-6,2.01,.5):
            all_bias.append(dict(setting=r.setting,specification=r.specification,
                assumed_untreated_differential_change_C=bias,adjusted_effect_C=r.estimate_C-bias,
                adjusted_lo_C=r.get("lo_C",np.nan)-bias,adjusted_hi_C=r.get("hi_C",np.nan)-bias,
                note="additive bias scenario, not an identified confounding bound"))
    return frame,pd.DataFrame(balances),pd.DataFrame(all_bias)

