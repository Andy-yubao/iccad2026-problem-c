"""Validate the outer deadline on five cases; large guarded runs remain user-started."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='4'
os.environ['NUMBA_NUM_THREADS']='1'
import argparse,time
from pathlib import Path
import numpy as np
from datasets.reference_layouts import ReferenceLayouts,public_hard_reference
from .run import HERE,ROOT,CHECKPOINT,CHECKPOINT_SHA,read,save,sha
from .resident import ResidentPotential

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,default=HERE/'runs/20261024_external_deadline_gate5')
    args=ap.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True);(out/'cases').mkdir(exist_ok=True)
    ids=[428,360,164,443,21]
    assert sha(CHECKPOINT)==CHECKPOINT_SHA
    config={'indices':ids,'reason':'both observed overrun cases plus three hard cases, no wider rerun',
        'budget_seconds':5.,'native_threads':4,'resident_source_sha256':sha(HERE/'resident.py'),
        'solver_source_sha256':sha(HERE/'solver.py'),'checkpoint_sha256':CHECKPOINT_SHA,
        'source_screen100':'20261024_screen100_frozen_v4; soft-deadline results preserved',
        'budget_interpretation':'resident startup/restart separate; input conversion, inference, IPC, independent scoring, selection and worker termination inside5s',
        'status':'deadline adapter gate, not a second100-case quality benchmark'}
    if (out/'config.json').exists():raise FileExistsError('Do not overwrite completed adapter gate')
    save(out/'config.json',config)
    data=ReferenceLayouts('dev_monitor500');resident=ResidentPotential(CHECKPOINT);records=[]
    try:
        resident.ensure_ready()
        for i in ids:
            item=data[i];case=item['case'];case.reference_positions=public_hard_reference(case.solver_inputs[1],case.solver_inputs[5],case.solver_inputs[6])
            x,stats=resident.solve(case.solver_inputs,case=case)
            selected=stats['candidates'][stats['selected_id']]
            assert selected['metrics']['is_feasible']
            row={'index':i,'member':list(item['member']),'full_solve_seconds':stats['seconds'],'selected_metrics':selected['metrics'],'solver':stats}
            save(out/'cases'/f'{i:04d}.json',row);records.append(row)
            print(f"guarded index{i} Q{selected['metrics']['cost']:.6f} {stats['seconds']:.3f}s aborted{stats['external_abort']}",flush=True)
        # Force a stall on a small synthetic public problem to verify actual termination
        # after a checked candidate, rather than merely trusting numerical time limits.
        inputs=(3,np.array([4.,3.,2.]),np.empty((0,3)),np.empty((0,3)),np.empty((0,2)),
                np.array([[0,1,0,0,0],[1,0,0,0,0],[0,0,0,0,0]]),
                np.array([[0.,0.,2.,2.],[-1.,-1.,3.,1.],[-1.,-1.,-1.,-1.]]))
        p=__import__('experiments.E48_quality_potential.backend.frozen.packing',fromlist=['prepare_problem']).prepare_problem(*inputs)
        raw=__import__('experiments.E48_quality_potential.backend.frozen.feasible',fromlist=['row_control']).row_control(p)
        resident.ensure_ready();_,stall=resident.solve(inputs,draft=raw,seconds=.6,force_stall=True)
        assert stall['external_abort'] and stall['seconds']<.65
        save(out/'forced_stall.json',{'aborted':stall['external_abort'],'seconds':stall['seconds'],'valid_incumbent_retained':bool(stall['candidates'])})
    finally:resident.close()
    summary={'count':5,'hard_legal':sum(r['selected_metrics']['is_feasible'] for r in records),
        'max_seconds':max(r['full_solve_seconds'] for r in records),'over5s':sum(r['full_solve_seconds']>5 for r in records),
        'external_abort_count':sum(r['solver']['external_abort'] for r in records),
        'Q_lt1_03':sum(r['selected_metrics']['cost']<1.03 for r in records),'forced_stall_passed':True,
        'scope':'five-case adapter validation only; does not replace or relabel the frozen100 soft-deadline evaluation'}
    save(out/'summary.json',summary);print(summary,flush=True)

if __name__=='__main__':main()
