"""Finite registered oracle inputs: component diagnostic, never a deployed solver."""
import datetime,json,time,zipfile
from pathlib import Path
import numpy as np
from .candidate5 import HERE,ROOT,torch,teacher_rows,attach,seed,sha
from .nn85.base_model import decode
from .train_io.data import load_rows,collate
from .train_io.io import save,exclusive
from .repair85.fast_geometry import topology
from .solver_relations import optimize
from .mechanism import SLIM,warmup,public_hard_reference,independent_hard_checks
from datasets.expanded_reference import ExpandedReferenceLayouts
from scoring.evaluation import evaluate_case
from scoring.official import official_evaluator

FOLDER=HERE/'runs/20261102_oracle_component_dev500'
ARMS=('oracle_relations','oracle_geometry_and_relations')
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def summary(rows):
    return dict(count=len(rows),weighted_Q=official_evaluator().compute_total_score([x['metrics']['cost'] for x in rows],[x['n'] for x in rows]),legal=sum(x['metrics']['is_feasible'] for x in rows),fallbacks=sum(x['fallback'] for x in rows),mean_repair_seconds=float(np.mean([x['repair_seconds'] for x in rows])),p95_repair_seconds=float(np.percentile([x['repair_seconds'] for x in rows],95)),not_deployable=True)

def main():
    begin=time.perf_counter();seed();study=read(HERE/'study.json');deadline=datetime.datetime.fromisoformat(study['deadline_utc'])
    halt=lambda:(HERE/'runs/STOP').exists() or datetime.datetime.now(datetime.timezone.utc)>=deadline
    if halt():return
    with exclusive(HERE/'runs/oracle_panel.lock'):
        labels=teacher_rows('dev_monitor500');cached=attach(load_rows('dev_monitor500'),labels);dataset=ExpandedReferenceLayouts('dev_monitor500',train_size=16000);rows=[dataset[i] for i in range(500)];warmup()
        FOLDER.mkdir(exist_ok=True);binding=dict(plan_sha256=sha(HERE/'runs/oracle_panel_plan.json'),sources={p.relative_to(ROOT).as_posix():sha(p) for p in [Path(__file__),HERE/'solver_relations.py',HERE/'relation_interface_probe.py',HERE/'candidate5.py']},teacher_manifest_sha256=sha(HERE/'runs/20261101_public_teacher/dev_monitor500/manifest.json'),C1_checkpoint_sha256=sha(HERE/'runs/20261102_candidate1_distill_seed0/checkpoints/best.pt'),members=[list(r['member']) for r in rows],arms=ARMS,repair=SLIM,oracle_only=True,full_service_latency_measured=False,test_read=False)
        if (FOLDER/'frozen.json').exists():assert read(FOLDER/'frozen.json')==json.loads(json.dumps(binding))
        else:save(FOLDER/'frozen.json',binding)
        records={a:[] for a in ARMS}
        for i,(row,cache) in enumerate(zip(rows,cached)):
            if halt():save(FOLDER/'progress.json',dict(status='stopped',completed=i));return
            assert list(row['member'])==list(cache['member']);inputs=row['case'].solver_inputs;n=int(inputs[0]);c,s=np.asarray(inputs[5]),np.asarray(inputs[6])
            row['case'].reference_positions=public_hard_reference(inputs[1],inputs[5],inputs[6])
            host=collate([cache],'cpu');target=decode(cache['teacher'][None],host)[0,:n].numpy().astype(float)*float(host['scale'][0])
            locked=c[:,:2].any(1);target[locked,2:]=s[locked,2:];target[c[:,1]!=0]=s[c[:,1]!=0]
            edges=topology(target,c,array=True)
            old=read(HERE/f'runs/20261102_candidate1_distill_seed0/evaluation/terminal/{i:04d}.json');assert old['member']==list(row['member'])
            drafts={ARMS[0]:np.asarray(old['draft_positions']),ARMS[1]:target}
            # Alternate order to avoid systematically assigning all warm effects to one arm.
            for arm in (ARMS if i%2==0 else ARMS[::-1]):
                path=FOLDER/arm/f'{i:04d}.json'
                if path.exists():record=read(path);assert record['member']==list(row['member'])
                else:
                    final,trace=optimize(drafts[arm],inputs,**SLIM,relations_override=edges)
                    hard=independent_hard_checks(final.tolist(),np.asarray(inputs[1]),c,row['case'].reference_positions);assert hard['is_feasible'] and bool((final[:,:2]>=0).all())
                    result=evaluate_case(row['case'],final,trace['seconds'])
                    record=dict(index=i,member=list(row['member']),n=n,arm=arm,metrics=result['metrics'],repair_seconds=trace['seconds'],fallback=trace['row_retained'],trace=trace['trace'],positions=final.tolist(),oracle_only=True)
                    save(path,record)
                records[arm].append(record)
            if (i+1)%25==0:
                save(FOLDER/'progress.json',dict(status='running',completed=i+1,arms={a:summary(v) for a,v in records.items()},elapsed_seconds=time.perf_counter()-begin,test_read=False));print('oracle diagnostic',i+1,'/500',flush=True)
        result=dict(status='complete',finished=True,arms={a:summary(v) for a,v in records.items()},elapsed_seconds=time.perf_counter()-begin,interpretation='teacher-derived oracle inputs only; no deployable quality or full-latency claim',formal_slots_consumed=0,test_read=False)
        save(FOLDER/'results.json',result);print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as error:save(FOLDER/'error.json',dict(error=repr(error),automatic_retry=False));raise
