"""One frozen model, sequential full requests; test labels are score-only."""
import argparse,time
import numpy as np
import torch
from .common import HERE,RUN,PLAN,ROOT,read,save,sha,seed,check_stop,sources,utc
from .solver import Service
from datasets.expanded_reference import registry
from datasets.worksets import FrozenMemberReader
from datasets.cases import _build_case
from datasets.reference_layouts import public_hard_reference
from scoring.evaluation import evaluate_case
from scoring.official import official_evaluator

def summary(records):
    q=np.array([r['metrics']['cost'] for r in records]);wall=np.array([r['complete_seconds'] for r in records]);n=[r['n'] for r in records]
    return dict(count=len(records),hard_legal=sum(r['metrics']['is_feasible'] for r in records),weighted_Q=official_evaluator().compute_total_score(q.tolist(),n),mean_Q=float(q.mean()),p95_Q=float(np.percentile(q,95)),max_Q=float(q.max()),mean_complete_seconds=float(wall.mean()),p95_complete_seconds=float(np.percentile(wall,95)),max_complete_seconds=float(wall.max()),cases_over_025s=int((wall>.25).sum()),fallbacks=sum(r['fallback'] for r in records))

def main(role):
    seed();check_stop();selection=read(RUN/'frozen_selection.json');model=HERE/'delivery/model.pt'
    assert sha(model)==selection['checkpoint_sha256']
    expected=read(RUN/'sources.json');current=sources()
    # Protect every frozen file; unrelated new shared helpers are not E101 dependencies.
    assert all(current.get(name)==digest for name,digest in expected.items()),'Frozen dependency changed after final freeze'
    for stage in ('L','F'):assert read(RUN/stage/'results.json')['status'] in ('early_stop','budget')
    if role=='local_test':
        assert selection['test_authorization']==PLAN['authorization']
        assert read(RUN/'evaluation/dev_monitor500/results.json')['status']=='complete'
    records_meta=registry()['sets'][role]['records'];expected=2000 if role=='local_test' else 500;assert len(records_meta)==expected
    members=[r['member'] for r in records_meta];parent='local_test' if role=='local_test' else 'dev'
    folder=RUN/'evaluation'/role;folder.mkdir(parents=True,exist_ok=True)
    binding=dict(checkpoint_sha256=sha(model),source_snapshot_sha256=sha(RUN/'source_snapshot.zip'),members=members,role=role,test_read=role=='local_test',frozen_before_test=selection['frozen_utc'],batch_size=1,CPU_threads=1,repair=PLAN['repair'])
    if (folder/'frozen.json').exists():assert read(folder/'frozen.json')==binding
    else:save(folder/'frozen.json',binding)
    # This is the only test-payload read gate in the final package.
    reader=FrozenMemberReader(members,parent,identity=sha(HERE/'plan.json'),allow_local_test=role=='local_test')
    raw=reader.case(0);service=Service(model,raw.solver_inputs);save(folder/'cold_start.json',dict(seconds=service.cold_seconds,excluded_from_resident_request_latency=True))
    records=[];begin=time.perf_counter()
    for i,member in enumerate(members):
        check_stop();path=folder/'cases'/f'{i:04d}.json'
        if path.exists():
            r=read(path);assert r['member']==member and r['checkpoint_sha256']==binding['checkpoint_sha256'];records.append(r);continue
        disk_start=time.perf_counter();raw=reader.case(i);disk_seconds=time.perf_counter()-disk_start
        positions,diagnostic,inputs=service.solve(raw.solver_inputs)
        # Free reference and stored denominators enter only after solve() returns.
        case=_build_case(raw.case_id,raw.source,tuple(inputs[1:6]),raw.reference_positions,raw.stored_metrics)
        case.solver_inputs=inputs;case.reference_positions=public_hard_reference(inputs[1],inputs[5],inputs[6]);scored=evaluate_case(case,positions,diagnostic['complete_seconds'])
        r=dict(index=i,member=member,n=int(inputs[0]),checkpoint_sha256=binding['checkpoint_sha256'],disk_load_seconds=disk_seconds,positions=positions.tolist(),metrics=scored['metrics'],independent_hard_checks=scored['independent_hard_checks'],**diagnostic)
        save(path,r);records.append(r)
        if role=='dev_monitor500' and i<100:
            from visualization.layouts import plot_case,geometry_details
            c=np.asarray(inputs[5]).tolist();variants={k:dict(positions=p) for k,p in [('reference',raw.reference_positions),('placer',r['L_positions']),('corrected',r['draft_positions']),('final',r['positions'])]}
            for v in variants.values():v['soft_details']=geometry_details(v['positions'],c)
            plot_case(dict(case_id=f'dev500:{i}',panel='Final C5 / 64k',constraints=c,variants=variants),folder/'gallery'/f'{i:03d}.png',stages=(('reference','Reference'),('placer','L'),('corrected','F'),('final','Full repair')),show_metrics=False)
        if (i+1)%25==0:
            save(folder/'progress.json',dict(completed=i+1,total=expected,summary=summary(records),elapsed_seconds=time.perf_counter()-begin));print(role,i+1,'/',expected,flush=True)
    value=summary(records);save(folder/'results.json',dict(status='complete',summary=value,elapsed_seconds=time.perf_counter()-begin,test_read=role=='local_test',scope='local final held-out evaluation of the public-MIB-mean-area task; not official contest score; original v2 test exposure recorded in protocol'))
    lines=[f'# Final C5: {role}', '', f'Frozen checkpoint SHA256: `{binding["checkpoint_sha256"]}`', '', '| Metric | Value |','|---|---:|']+[f'| {k} | {v} |' for k,v in value.items()]
    lines+=['','Full request includes public clone/masking/mean-area conversion, features/transfers/L/F/complete original repair/hard checks. Disk, cold start, scoring and gallery are separate. No test-based model selection or further training.']
    (folder/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    if role=='dev_monitor500':
        gallery=folder/'gallery';gallery.mkdir(exist_ok=True);(gallery/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Final C5 same100</title><h1>Reference / L / F / full repair</h1>'+''.join(f'<p>{i}</p><img style="width:100%" loading="lazy" src="{i:03d}.png">' for i in range(100)),encoding='utf-8')
    print(role,value,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('role',choices=['dev_monitor500','local_test']);a=ap.parse_args()
    from threadpoolctl import threadpool_limits
    try:
        with threadpool_limits(1):main(a.role)
    except BaseException as e:save(RUN/'evaluation'/a.role/'error.json',dict(error=repr(e),at_utc=utc(),no_automatic_retry=True));raise
