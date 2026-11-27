"""Scoring adapter: same official evaluator; use NumPy edge scalars and stored baselines."""
from audit import ROOT,np
from datasets.cases import ResearchCase
from scoring.official import official_evaluator

def scoring_case(raw,teacher):
    n,areas,b2b,p2b,pins,constraints,specs=raw.solver_inputs
    b2b,p2b,pins=(v.numpy() for v in (b2b,p2b,pins))
    m=raw.stored_metrics;official=official_evaluator();ref=raw.reference_positions
    a=float(m[0]) if m is not None and len(m)>=8 and float(m[0])>0 else official.calculate_bbox_area(ref)
    b=float(m[-2]) if m is not None and len(m)>=8 and float(m[-2])>0 else official.calculate_hpwl_b2b(ref,b2b)
    p=float(m[-1]) if m is not None and len(m)>=8 and float(m[-1])>=0 else official.calculate_hpwl_p2b(ref,p2b,pins)
    assert abs(teacher['hpwl_baseline']-(b+p))<1e-10
    assert abs(teacher['bbox_area_baseline']-a)<1e-10
    return ResearchCase(raw.case_id,raw.source,(n,areas,b2b,p2b,pins,constraints,specs),ref,
        {'hpwl_baseline':b+p,'area_baseline':a})
