"""Refresh reviewer diagnostics from saved full results, without repeating 400 model refits."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from fw_config import Config,SETTINGS
import fw_panel as panel, fw_matching as matching, fw_model as model, fw_figures as figures, fw_followup as followup
from fw_pipeline import _jsonable
from fw_report import write_report
from diagnose_populations import compute as population_diagnostics

def restore(x):
    if isinstance(x,list):
        return pd.DataFrame(x) if x and isinstance(x[0],dict) else [restore(v) for v in x]
    if isinstance(x,dict):return {k:restore(v) for k,v in x.items()}
    return x

def main():
    cfg=Config();out=cfg.out_dir; tables=out/"tables"
    raw=json.loads((tables/"results.json").read_text(encoding="utf-8"))
    import hashlib
    oldreport=(out/"manuscript"/"REPORT.md").read_text(encoding="utf-8")
    assert hashlib.sha256(cfg.panel_path.read_bytes()).hexdigest() in oldreport
    for key in raw["config"]:
        if key in ["panel_path","out_dir","sio_path","allocation_path"]:
            continue
        assert json.dumps(raw["config"][key])==json.dumps(getattr(cfg,key)),key
    df,_=panel.load(cfg); R=followup.setup(df,cfg)
    boot=matching.Bootstrap(df.block,cfg.n_boot,cfg.seed)
    print("Paired mixing and coastal contrasts",flush=True)
    mix,coast=followup.paired_contrasts(df,R,cfg,boot)
    print("Primary model B fit for support decomposition (saved joint intervals retained)",flush=True)
    M=model.Model(df,cfg,exclude_near_m=cfg.strict_exclusion_m,refit=False)
    counts,support_coast=followup.support_diagnostics(df,M,R,cfg)
    print("Matching calipers, outcome regression, and DR-DiD",flush=True)
    with threadpool_limits(limits=1):
        estimates,balance,bias=followup.matching_sensitivity(df,R,cfg,boot)
    retention,coast_blocks=population_diagnostics(df,cfg)
    retention.to_csv(tables/"population_retention.csv",index=False)
    coast_blocks.to_csv(tables/"coastal_block_counts.csv",index=False)
    wide=balance.pivot(index=["setting","specification"],columns="covariate",values="standardized_difference").add_prefix("smd_").reset_index()
    comparison=estimates.merge(wide,on=["setting","specification"],how="left")
    results=dict(specification_comparison=comparison,area_mixing=mix,coastal_difference=coast,matching_specifications=estimates,
                 specification_balance=balance,bias_scenarios=bias,population_retention=retention,coastal_block_counts=coast_blocks)
    tipping=estimates[["setting","specification","estimate_C","lo_C","hi_C","n_retained"]].copy()
    tipping["untreated_differential_change_to_zero_C"]=tipping.estimate_C
    tipping["untreated_differential_change_to_ci_touch_zero_C"]=np.where(tipping.hi_C<0,tipping.hi_C,np.where(tipping.lo_C>0,tipping.lo_C,0.))
    tipping.loc[tipping.estimate_C.isna(),"untreated_differential_change_to_ci_touch_zero_C"]=np.nan
    tipping["interpretation"]="Signed additive untreated treated-minus-control change over comparison period; not a plausibility bound"
    results["bias_tipping"]=tipping
    for name,d in results.items():
        if name not in ("population_retention","coastal_block_counts"):
            d.to_csv(tables/f"followup_{name}.csv",index=False)
    counts.to_csv(tables/"test_support_counts.csv",index=False)
    support_coast.to_csv(tables/"test_support_coasts.csv",index=False)
    res=restore(raw)
    res["test"]["support_counts"]=counts
    res["test"]["support_coasts"]=support_coast
    res["decision"]["support_coasts"]=support_coast
    res["followup"]=results
    for s in SETTINGS:
        for k,e in res["measured"][s]["estimates"].items():
            res["measured"][s]["estimates"][k]={a:np.nan if b is None else b for a,b in e.items()}
    figures.fig_cooling(res["measured"],cfg,out/"figures/png")
    figures.fig_decision(res["decision"],res["test"],out/"figures/png")
    write_report(res,cfg,out/"manuscript"/"REPORT.md")
    (tables/"results.json").write_text(json.dumps(_jsonable(res),indent=1,ensure_ascii=False,allow_nan=False),encoding="utf-8")
    print("Refreshed report, figures, support tables and paired/matching diagnostics",flush=True)
    print(estimates.to_string(index=False),flush=True)

if __name__=="__main__":main()

