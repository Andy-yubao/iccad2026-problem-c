"""Two matched local conic feasibility probes; not a new end-to-end evaluation."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[key]='1'
import json
from pathlib import Path
import sys
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[4];sys.path.insert(0,str(ROOT))
from separation import topology
from area_bound_certificate import propagate


def main():
    from datasets.random_reference import RandomReference
    from experiments.E140_frontend_gb4.evaluation.repair.backend.frozen.packing import prepare_problem
    from experiments.E140_frontend_gb4.evaluation.repair.fast_geometry import step,reduce_edges
    from experiments.E140_frontend_gb4.evaluation.repair.backend.frozen.solver import hard_valid
    from experiments.E140_frontend_gb4.evaluation.repair.canonical_native import canonicalize,warmup
    from experiments.E140_frontend_gb4.evaluation.repair.backend.guard import repair_tiny_negative
    from threadpoolctl import threadpool_limits
    replay=HERE/'diagnostic_replay_dev358';out=replay/'relation_probe';out.mkdir(exist_ok=False)
    plan=dict(case=358,reason='Read-only replay found persistent 100/92 overlap. Test frozen separation-graph compatibility and one reference-compatible disjunction change, no checkpoint/Q selection.',
        local_conic_calls=2,full_postprocessing_calls=0,training=False,test_read=False,
        shared_changes='Set elastic=0 in BOTH probes to ask hard feasibility; identical seed/bounds/objective. Between probes only pair100/103 changes from100 below103 to100 left of103.')
    (out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    snapshot=np.load(replay/'proposal_008.npz');seed=snapshot['seed'];flags=snapshot['flags'];areas=snapshot['areas'];specs=snapshot['specs']
    edges=topology(seed,flags,array=True);changed=edges.copy();mask=(changed[:,0]==100)&(changed[:,1]==103)&(changed[:,2]==1);assert mask.sum()==1;changed[mask,2]=0
    item=RandomReference('dev_monitor500')[358];inputs=tuple(int(v) if i==0 else np.asarray(v).copy() for i,v in enumerate(item['solver_inputs']));inputs[5][:,2]=0
    p=prepare_problem(*inputs);warmup();results={}
    for label,e in [('original_relations',edges),('only_100_left_of_103',changed)]:
        certificate=propagate(e,areas,flags,specs);certificate.pop('caps',None)
        with threadpool_limits(limits=1):proposal,stats=step(p,seed,reduce_edges(e,p.n),full_edges=e,elastic=0.)
        accepted=False
        if proposal is not None:
            clean=canonicalize(proposal,p,tolerance=1e-4);clean,_=repair_tiny_negative(clean,p);accepted=hard_valid(clean,p)
            np.save(out/(label+'.npy'),clean,allow_pickle=False)
        results[label]=dict(certificate=certificate,status=stats['status'],has_proposal=proposal is not None,
            independent_original_hard_valid=bool(accepted),stats=stats)
    # Compare the ORIGINAL initial graphs, without invoking either postprocessor.
    initial={}
    for model,folder in [('E139','E139_delayed_overlap/runs/e119_delayed_overlap_seed0'),('E140','E140_frontend_gb4/runs/e119_front_gb4_seed0')]:
        row=json.loads((ROOT/'experiments'/folder/'evaluation_no_mib_dev500/cases/0358.json').read_text());draft=np.asarray(row['draft_positions']);r=draft[:,2]/draft[:,3]
        wh=np.stack((np.sqrt(areas*r),areas/np.sqrt(areas*r)),-1);wh[p.fixed|p.preplaced]=specs[p.fixed|p.preplaced,2:]
        initial_seed=np.c_[draft[:,:2],wh];cert=propagate(topology(initial_seed,flags,array=True),areas,flags,specs);cert.pop('caps',None)
        initial[model]=cert
    results['initial_graph_certificates']=initial
    (out/'result.json').write_text(json.dumps(results,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:dict(status=v['status'],hard_valid=v['independent_original_hard_valid'],conflict=v['certificate']['conflict'],witness=v['certificate'].get('anchor_witnesses')) for k,v in results.items() if k!='initial_graph_certificates'},ensure_ascii=False,indent=2),flush=True)
    print('Initial graphs', {k:v['conflict'] for k,v in initial.items()},flush=True)


if __name__=='__main__':main()
