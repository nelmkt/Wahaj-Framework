"""Regression tests for inference and resource-allocation failure modes."""
import dataclasses
import numpy as np
import pandas as pd
import pytest
from fw_config import Config
from fw_validate import tolerance_status
from fw_allocation import allocate
from fw_decision import ratio_summary
from fw_panel import strata
from fw_matching import Bootstrap


def test_equivalence_does_not_reward_wide_intervals():
    cfg=Config()
    args=(40,8,200,200,cfg)
    assert tolerance_status(-2,2,.5,*args)=="not_demonstrated"
    assert tolerance_status(-.2,.2,.5,*args)=="within_tolerance"
    assert tolerance_status(-.2,.2,None,*args)=="tolerance_not_set"
    assert tolerance_status(-.2,.2,.5,40,8,150,200,cfg)=="insufficient_joint_refits"


def test_ratio_sign_uncertainty_is_not_silently_discarded():
    r=ratio_summary(np.array([1.,2.,0.,-1.,np.nan]))
    assert np.isnan(r["low"]) and np.isnan(r["high"])
    assert r["nonpositive_benefit_share"]==.5


def test_bootstrap_draw_prefix_and_alignment():
    blocks=pd.Series(["a","a","c","b","b","c"])
    a=Bootstrap(blocks,2000,42)
    b=Bootstrap(blocks,200,42)
    np.testing.assert_array_equal(a.draws[:200],b.draws)
    np.testing.assert_array_equal(a.weights(15),b.weights(15))


def option(oid,site,cool,water,tse,cost):
    return dict(option_id=oid,site_id=site,intervention="test",cooling_C=cool,benefit_weight=1,
                water_m3=water,wastewater_m3=tse,energy_mwh=water/100,cost=cost,
                period="one test year",benefit_metric="same synthetic proxy",currency="test units",
                evidence_source="synthetic fixture")


def test_allocator_uses_cap_and_mutually_exclusive_alternatives(tmp_path):
    p=tmp_path/"options.csv"
    pd.DataFrame([option("A-tse","A",100,8,8,3),option("A-shade","A",5,0,0,2),
                  option("B-green","B",6,5,0,3)]).to_csv(p,index=False)
    cfg=dataclasses.replace(Config(),allocation_path=p,water_budget_m3=5,wastewater_cap_m3=0,
                            energy_budget_mwh=1,financial_budget=5)
    r=allocate(cfg)
    assert set(r["selected"].option_id)=={"A-shade","B-green"}
    assert r["benefit"]==11
    assert r["resource_totals"]["water_m3"]==5


def test_allocator_refuses_incommensurate_benefits(tmp_path):
    p=tmp_path/"options.csv"
    a,b=option("a","a",1,1,0,1),option("b","b",1,1,0,1)
    b["benefit_metric"]="a different outcome"
    pd.DataFrame([a,b]).to_csv(p,index=False)
    cfg=dataclasses.replace(Config(),allocation_path=p,water_budget_m3=5,wastewater_cap_m3=5,
                            energy_budget_mwh=1,financial_budget=5)
    with pytest.raises(ValueError,match="benefit_metric"):
        allocate(cfg)


def test_no_budget_does_not_mean_unlimited(tmp_path):
    p=tmp_path/"options.csv"
    pd.DataFrame([option("a","a",1,1,0,1)]).to_csv(p,index=False)
    with pytest.raises(ValueError,match="limits"):
        allocate(dataclasses.replace(Config(),allocation_path=p))


def test_invalid_configuration_rejected():
    with pytest.raises(ValueError): Config(n_boot=20,n_refits=30)
    with pytest.raises(ValueError): Config(validation_margin_C=float("nan"))


def test_joint_evaluation_preserves_shared_target_composition():
    from fw_validate import compare
    df=pd.DataFrame(dict(block=np.repeat(["a","b","c"],4),
        d_lst=[-1,0,0,0,-2,0,0,0,-4,0,0,0],n_greened_px=1,
        ndvi_pre=0.,ndbi_pre=0.,ndvi_post=1.,ndbi_post=0.))
    t=pd.Series([True,False,False,False]*3)
    rs=dict(treated=t,controls=~t,key=df.block,weights=pd.Series(1.,index=df.index))
    class Perfect:
        bootstrap=Bootstrap(df.block,200,42)
        refits=[None]*200
        def change(self,cells,*args):
            return np.tile(cells.d_lst.to_numpy(),(201,1))
    r=compare(df,Perfect(),rs,t,Config())
    assert r["measured_hi_C"]>r["measured_lo_C"]
    assert r["difference_C"]==0
    assert r["difference_lo_C"]==0 and r["difference_hi_C"]==0

