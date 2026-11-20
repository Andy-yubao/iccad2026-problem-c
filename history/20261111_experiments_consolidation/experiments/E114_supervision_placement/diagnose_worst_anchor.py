"""One saved dev case, one public-constraint repair-entry intervention only."""
import json
import time
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from .io import ROOT,save
from .evaluation_staged.run import repair_inputs,case_from_item
from .evaluation_staged.solver import optimize
from .evaluation_staged.repair.canonical_native import warmup
from datasets.random_reference import RandomReference
from datasets.reference_layouts import public_hard_reference
from scoring.checks import independent_hard_checks
from scoring.research_no_mib import evaluate_case

def main():
    out=ROOT/'output/supervision_q_analysis_20261108/anchor_intervention'
    assert not out.exists()
    save(out/'plan.json',dict(scope='Cause exploration requested 2026-11-08; one known worst case, one intervention; no training or full-dev rerun',index=358,source='frozen staged best48 saved actual draft',change='At postprocessing entry only, use public specified WH for fixed/preplaced modules and public XY for preplaced modules. All free channels stay equal to saved draft. Then call unchanged copied postprocessor/budget.',comparison='Original result already saved; no baseline rerun, no parameter scan or checkpoint selection',test_read=False))
    torch.set_num_threads(1)
    import numba
    numba.set_num_threads(1)
    row=json.loads((ROOT/'experiments/E114_supervision_placement/runs/staged_seed0/evaluation_no_mib_dev500/cases/0358.json').read_text(encoding='utf-8'))
    item=RandomReference('dev_monitor500')[358];inputs=item['solver_inputs'];draft=np.array(row['draft_positions']);modified=draft.copy();c=inputs[5];s=inputs[6];locked=c[:,:2].any(1);pre=c[:,1]>0
    modified[locked,2:]=s[locked,2:];modified[pre,:2]=s[pre,:2]
    with threadpool_limits(limits=1):
        warmup();start=time.perf_counter();final,info=optimize(modified,repair_inputs(inputs));elapsed=time.perf_counter()-start
    scored=evaluate_case(case_from_item(item),final,elapsed)['metrics'];hard=independent_hard_checks(final.tolist(),inputs[1],c,public_hard_reference(inputs[1],c,s));assert hard['is_feasible']
    result=dict(status='complete',index=358,member=row['member'],original_metrics=row['metrics'],diagnostic_metrics=scored,positions=final.tolist(),entry_positions=modified.tolist(),work=info['work'],trace=info['trace'],hard=hard,locked_modules=int(locked.sum()),preplaced_modules=int(pre.sum()),test_read=False,official_scores_unchanged=True,limit='Single exposed difficult development case; supports a local mechanism only. Does not isolate width or establish general improvement.')
    save(out/'result.json',result);print(json.dumps({k:result[k] for k in ['original_metrics','diagnostic_metrics','work','locked_modules','preplaced_modules','limit']},indent=2))

if __name__=='__main__':main()
