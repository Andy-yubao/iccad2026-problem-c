"""One same-budget deterministic diagnostic replay; never replace original results."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[key]='1'
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[4]
sys.path.insert(0,str(ROOT))


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def save(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def rejection_details(x,p):
    x=np.asarray(x);locked=p.fixed|p.preplaced
    lo=x[:,:2];hi=lo+x[:,2:];i,j=np.triu_indices(p.n,1)
    inter=np.minimum(hi[i],hi[j])-np.maximum(lo[i],lo[j]);bad=(inter>1e-6).all(1)
    pairs=np.flatnonzero(bad)
    free=~locked
    return dict(nonnegative_violations=int((lo<0).sum()),negative_min=float(lo.min()),
        dimension_error=float(np.max(np.abs(x[locked,2:]-p.specs[locked,2:]))) if locked.any() else 0,
        preplaced_error=float(np.max(np.abs(x[p.preplaced]-p.specs[p.preplaced]))) if p.preplaced.any() else 0,
        free_area_relative_max=float(np.max(np.abs(x[free,2:].prod(1)-p.areas[free])/p.areas[free])) if free.any() else 0,
        overlap_pairs=int(bad.sum()),overlaps=[dict(i=int(i[k]),j=int(j[k]),depth=inter[k].tolist()) for k in pairs[:30]],
        max_minimum_penetration=float(np.maximum(inter.min(1),0).max()))


def main():
    from datasets.random_reference import RandomReference
    from experiments.E140_frontend_gb4.evaluation import solver
    from experiments.E140_frontend_gb4.evaluation.repair.canonical_native import warmup
    from threadpoolctl import threadpool_limits
    import numba
    numba.set_num_threads(1)
    out=HERE/'diagnostic_replay_dev358';out.mkdir(exist_ok=False)
    plan=dict(reason='Saved traces identify catastrophic dev358 route but do not preserve rejected proposals. One identical-budget replay to classify rejection and fixed-graph incompatibility.',
        case=358,original_draft_only=True,model_inference=False,optimizer_updates=0,
        postprocessing_calls_authorized_for_diagnostic=1,budget_unchanged=True,reference_not_solver_input=True,
        original_results_not_overwritten=True,new_Q_selection=False,test_read=False)
    save(out/'plan.json',plan)
    rowpath=HERE.parent/'evaluation_no_mib_dev500/cases/0358.json';row=read(rowpath)
    item=RandomReference('dev_monitor500')[358];assert item['member']==row['member']
    inputs=tuple(int(v) if i==0 else np.asarray(v).copy() for i,v in enumerate(item['solver_inputs']))
    inputs[5][:,2]=0
    draft=np.asarray(row['draft_positions'])
    original_step,original_project,original_valid,original_optimize=solver.step,solver.project,solver.hard_valid,solver.optimize
    stages=[];checks=[];depth=[0];current=[None]
    def wrap_call(function,kind):
        def call(p,seed,edges,*args,**kwargs):
            index=len(stages);current[0]=index
            proposal,stats=function(p,seed,edges,*args,**kwargs)
            record=dict(index=index,kind=kind,depth=depth[0],stats=stats,has_proposal=proposal is not None,
                hard_tangent=bool(kwargs.get('hard_tangent',False)))
            stages.append(record)
            np.savez_compressed(out/f'proposal_{index:03d}.npz',seed=np.asarray(seed),edges=np.asarray(edges),
                full_edges=np.asarray(kwargs.get('full_edges',edges)),proposal=np.asarray(proposal) if proposal is not None else np.zeros((0,4)),
                specs=p.specs,flags=p.constraints,areas=p.areas)
            return proposal,stats
        return call
    def valid(x,p):
        accepted=original_valid(x,p)
        record=dict(proposal=current[0],depth=depth[0],accepted=bool(accepted),details=rejection_details(x,p))
        checks.append(record)
        if not accepted:np.save(out/f'rejected_{len(checks):03d}.npy',np.asarray(x),allow_pickle=False)
        return accepted
    def optimize(*args,**kwargs):
        depth[0]+=1
        try:return original_optimize(*args,**kwargs)
        finally:depth[0]-=1
    solver.step=wrap_call(original_step,'shape');solver.project=wrap_call(original_project,'projection')
    solver.hard_valid=valid;solver.optimize=optimize
    warmup()
    with threadpool_limits(limits=1):
        final,info=solver.optimize(draft,inputs)
    expected=np.asarray(row['positions']);equal=np.array_equal(final,expected)
    oldstatuses=[(r['phase'],r.get('status'),r.get('legal')) for r in row['trace']]
    newstatuses=[(r['phase'],r.get('status'),r.get('legal')) for r in info['trace']]
    result=dict(final_bitwise_equal=bool(equal),maximum_coordinate_difference=float(np.max(np.abs(final-expected))),
        root_status_legal_sequence_equal=oldstatuses==newstatuses,root_work_equal=info['work']==row['work'],
        stages=stages,acceptance_checks=checks,work=info['work'],trace=info['trace'],
        input_record_sha256=hashlib.sha256(rowpath.read_bytes()).hexdigest(),postprocessing_calls=1,
        new_Q_calls=0,original_result_untouched=True,test_read=False)
    np.save(out/'final.npy',final,allow_pickle=False);save(out/'result.json',result)
    assert equal and oldstatuses==newstatuses and info['work']==row['work'],'Replay drift; do not claim exact reproduction'
    print(json.dumps(dict(final_bitwise_equal=equal,root_work_equal=result['root_work_equal'],
        root_rejections=[v for v in checks if v['depth']==1 and not v['accepted']],
        stages=len(stages)),ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
