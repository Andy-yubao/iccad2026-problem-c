"""Five-case sensitivity check prompted by the primary probe's universal failures."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from .orderliness import OUT,ROOT,HERE,REVIEW,read,save,sha,fixed_shape,verify

CASES=('case_0019','case_0524','case_0088','case_0538','case_0743')
ARMS={'bbox_only':dict(span_gain=.10),'wire_only':dict(wire_gain=.10),
      'boundary_only':dict(protect_boundary=False),'larger_move_control':dict(max_move=2.)}


def task(payload):
    key,record=payload
    from datasets.public_member_inputs import public_member_inputs
    row=read(OUT/'repair_local'/f'{key}.json');p=np.asarray(read(ROOT/row['source_draft'])['raw_positions']);inputs=public_member_inputs(record,'local_test',authorize_local_test=True)
    trials=[]
    for arm,kwargs in ARMS.items():
        out,solver=fixed_shape(p,inputs,True,**kwargs);t={'arm':arm,'solver':solver}
        if out is not None:
            snapped=int(np.sum((out[:,:2]<0)&(out[:,:2]>=-1e-7)))
            out[:,:2]=np.where((out[:,:2]<0)&(out[:,:2]>=-1e-7),0.,out[:,:2])
            t.update(verification=verify(p,out,inputs),positions=out.tolist(),tiny_negative_snap_count=snapped)
        trials.append(t)
    save(OUT/'relaxation_local'/f'{key}.json',{'id':key,'score':row['annotation']['score'],'trials':trials})
    return key,[(t['arm'],t['solver']['status'],t.get('verification',{}).get('hard_legal')) for t in trials]


def main():
    (OUT/'relaxation_local').mkdir(exist_ok=True)
    save(OUT/'relaxation_config.json',{'cases':list(CASES),'arms':ARMS,
        'reason':'No primary small-budget full legal witness; separate individual constraints without changing original results',
        'scope':'five predefined rated cases, no grade-based tuning after solver outcomes','source_sha256':sha(__file__),
        'current_solver_source_sha256':sha(HERE/'orderliness.py'),'primary_source_snapshot':'source_snapshot.zip',
        'numerics':'tiny negative coordinates in[-1e-7,0) are snapped to0, then all hard checks repeated',
        'budget_seconds':120,'threads':'4 processes x 1 solver thread','each_MIP_seconds':2})
    config=read(HERE/'runs/20261024_final_test1000_seed20261024/config.json');lookup={r['id']:r for r in read(REVIEW/'manifest.json')['cases']}
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(task,(key,config['records'][lookup[key]['source_index']])) for key in CASES]
        for f in as_completed(futures):print(f.result(),flush=True)


if __name__=='__main__':main()
