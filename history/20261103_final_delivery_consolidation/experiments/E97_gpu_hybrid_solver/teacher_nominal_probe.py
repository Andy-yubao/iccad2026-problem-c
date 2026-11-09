"""Registered same-five teacher-input nominal-decoder feasibility mechanism."""
import json,time,datetime
import numpy as np
from .candidate2 import HERE,ROOT,torch,teacher_rows,attach,seed,IDS,decode85
from .train_io.data import load_rows,collate,LEGACY
from .latency import public_member_inputs,registry,sha
from .mechanism import optimize,SLIM,independent_hard_checks,public_hard_reference,warmup,raw_measures,Objective,prepare_problem
from .train_io.io import save

def main():
    folder=HERE/'runs/20261102_teacher_nominal_probe';folder.mkdir(exist_ok=False);begin=time.perf_counter();seed()
    deadline=datetime.datetime.fromisoformat(json.loads((HERE/'study.json').read_text(encoding='utf-8'))['deadline_utc'])
    labels=teacher_rows('train');rows=attach(load_rows('train',IDS),labels,IDS);entries=registry()['sets']['train']['records'];warmup();results=[]
    for index,row in zip(IDS,rows):
        if (HERE/'runs/STOP').exists() or datetime.datetime.now(datetime.timezone.utc)>=deadline:raise RuntimeError('STOP/deadline')
        g=collate([row],'cpu');state=row['teacher'][None];draft=(decode85(state,g)*g['scale'][:,None,None])[0,:row['graph']['n']].double().numpy()
        inputs=public_member_inputs(entries[index],'train');hard=lambda x:independent_hard_checks(x.tolist(),np.asarray(inputs[1]),np.asarray(inputs[5]),public_hard_reference(inputs[1],inputs[5],inputs[6]))
        final,trace=optimize(draft,inputs,**SLIM);assert hard(final)['is_feasible'];p=prepare_problem(*inputs);obj=Objective(p)
        result=dict(train_id=index,n=len(draft),teacher_nominal_legal=hard(draft)['is_feasible'],final_legal=True,fallback=trace['row_retained'],repair_seconds=trace['seconds'],public_proxy=obj(final),trace=trace['trace']);results.append(result);print(json.dumps(result),flush=True)
    save(folder/'results.json',dict(status='complete',records=results,SLIM=SLIM,seconds=time.perf_counter()-begin,source_sha256=sha(__import__('pathlib').Path(__file__)),test_read=False))

if __name__=='__main__':main()
