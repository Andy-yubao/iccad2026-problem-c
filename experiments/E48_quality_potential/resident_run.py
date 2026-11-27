"""Resumable development evaluation through the externally guarded resident API."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='4'
os.environ['NUMBA_NUM_THREADS']='1'
import argparse
from datetime import datetime,timezone
from pathlib import Path
import subprocess
import sys
import time
import zipfile
import numpy as np
from datasets.reference_layouts import ReferenceLayouts,public_hard_reference
from .run import HERE,ROOT,MODEL,CHECKPOINT,CHECKPOINT_SHA,read,save,sha
from .resident import ResidentPotential

HARD_GATE=[428,360,164,443,21]
SCREEN=HERE/'runs/20261024_screen100_frozen_v4'


def selected(row):
    return next(c for c in row['solver']['candidates'] if c['id']==row['solver']['selected_id'])


def summarize(rows):
    if not rows:return {'count':0,'ready_for_screen100':False}
    q=np.array([r['selected_metrics']['cost'] for r in rows]);t=np.array([r['full_solve_seconds'] for r in rows])
    n=np.array([r['n'] for r in rows]);w=np.exp((n-n.max())/12)
    diagnostics=[]
    for r in rows:
        hits=[c for c in r['solver']['candidates'] if c['metrics']['cost']<1.03 and not c['shelf_fallback']]
        diagnostics.append({'index':r['index'],'first_Q_target_seconds':min((c['score_complete_seconds'] for c in hits),default=None),
            'min_found_D_for_Q_target':min((c['distance']['D'] for c in hits if c['distance']['D'] is not None),default=None),
            'selected_D':selected(r)['distance']['D']})
    return {'count':len(rows),'hard_legal':sum(r['selected_metrics']['is_feasible'] for r in rows),
        'Q_lt1_03':int(sum(q<1.03)),'weighted_Q':float(np.average(q,weights=w)),
        'baseline_weighted_Q':float(np.average([r['baseline']['Q'] for r in rows],weights=w)),
        'max_Q':float(q.max()),'p95_Q':float(np.quantile(q,.95)),
        'mean_seconds':float(t.mean()),'max_seconds':float(t.max()),'over5s':int(sum(t>5)),
        'external_abort_count':sum(r['solver']['external_abort'] for r in rows),
        'shelf_selected':sum(r['solver']['shelf_selected'] for r in rows),'per_case':diagnostics,
        'ready_for_screen100':len(rows)==5 and bool((t<=5).all()) and all(r['selected_metrics']['is_feasible'] and not r['solver']['shelf_selected'] for r in rows),
        'scope':'resident adapter evaluation; cannot replace the original soft-deadline100 results',
        'test_read':False,'formal_contest_score':False}


def check_gate(path,sources):
    gate=read(path/'summary.json');config=read(path/'config.json')
    if gate.get('count')!=5 or gate.get('hard_legal')!=5 or gate.get('over5s',gate.get('over5s_count',1))!=0:
        raise ValueError('Resident five-case legality/deadline gate has not passed')
    expected=config.get('source_sha256',{})
    for name,key in (('resident.py','resident_source_sha256'),('solver.py','solver_source_sha256')):
        if key in config:expected={**expected,f'experiments/E48_quality_potential/{name}':config[key]}
    for key,value in expected.items():
        if sources.get(key)!=value:raise ValueError('Source differs from guarded gate: '+key)
    if config['checkpoint_sha256']!=CHECKPOINT_SHA:raise ValueError('Gate checkpoint differs')
    for i in config['indices']:
        if selected(read(path/'cases'/f'{i:04d}.json'))['shelf_fallback']:raise ValueError('Gate selected shelf fallback')
    # The first adapter gate records its two new source hashes; all search
    # dependencies additionally retain their frozen100 identities.
    for key,value in read(SCREEN/'config.json')['source_sha256'].items():
        if sources.get(key)!=value:raise ValueError('Search dependency differs from frozen evaluation: '+key)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stage',choices=('gate','screen'),default='screen')
    ap.add_argument('--output',type=Path)
    ap.add_argument('--gate',type=Path,default=HERE/'runs/20261024_external_deadline_gate5')
    ap.add_argument('--resume',action='store_true');ap.add_argument('--dry-run',action='store_true')
    args=ap.parse_args();out=(args.output or HERE/f'runs/20261024_resident_{args.stage}_v1').resolve()
    if not out.is_relative_to(HERE/'runs'):raise ValueError('Output must stay in E48/runs')
    indices=HARD_GATE if args.stage=='gate' else read(SCREEN/'config.json')['indices']
    files=sorted(p for p in HERE.rglob('*.py') if 'runs' not in p.relative_to(HERE).parts)
    sources={p.relative_to(ROOT).as_posix():sha(p) for p in files}
    if sha(CHECKPOINT)!=CHECKPOINT_SHA:raise ValueError('Frozen checkpoint changed')
    if args.stage=='screen':check_gate(args.gate,sources)
    data=ReferenceLayouts('dev_monitor500')
    config={'stage':args.stage,'indices':indices,'members':[data.records[i] for i in indices],
        'source_sha256':sources,'checkpoint_sha256':CHECKPOINT_SHA,
        'registry_sha256':sha(ROOT/'protocol/reference_workset_v5.json'),'split_sha256':sha(ROOT/'protocol/split_v1.json'),
        'seconds_per_case':5.,'numerical_threads':4,'maximum_invocation_seconds':900.,
        'timing':'input conversion, inference, IPC, all candidates/checks/scoring, selection, termination inside case budget; startup/restart and disk IO separate',
        'panel':'same frozen targeted development100; new run, no reuse across deadline adapters' if args.stage=='screen' else 'two observed overruns plus three hard cases',
        'command':subprocess.list2cmdline(sys.argv),'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'test_read':False,'official_Q_returned_to_search':False}
    if args.dry_run:
        print({'validated_configuration_only':True,'count':len(indices),'output':str(out),'started_solver':False});return
    out.mkdir(parents=True,exist_ok=True);(out/'cases').mkdir(exist_ok=True)
    if (out/'config.json').exists():
        if not args.resume:raise FileExistsError('Use --resume to continue an existing run')
        old=read(out/'config.json')
        for key in ('source_sha256','indices','members','checkpoint_sha256','registry_sha256','split_sha256'):
            if old[key]!=config[key]:raise ValueError('Resume identity changed: '+key)
    else:
        save(out/'config.json',config)
        with zipfile.ZipFile(out/'source_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
            for p in files:z.write(p,p.relative_to(ROOT))
    records=[read(out/'cases'/f'{i:04d}.json') for i in indices if (out/'cases'/f'{i:04d}.json').exists()]
    if len(records)==len(indices):print('Completed run; no inference or solving repeated');return
    baseline={r['index']:r for r in read(MODEL/'terminal/results.json')['records']}
    resident=ResidentPotential(CHECKPOINT);started=time.perf_counter()
    setups=read(out/'setup.json') if (out/'setup.json').exists() else []
    seen_startups=0;done={r['index'] for r in records}
    try:
        for i in indices:
            if i in done:continue
            if (out/'STOP').exists() or (HERE/'runs/STOP').exists() or time.perf_counter()-started>900:break
            resident.ensure_ready()
            if resident.restarts!=seen_startups:
                setups.append({'next_index':i,'startups_this_invocation':resident.restarts,'setup_seconds':resident.last_setup_seconds})
                seen_startups=resident.restarts
            item=data[i];case=item['case'];assert list(item['member'])==baseline[i]['member']
            case.reference_positions=public_hard_reference(case.solver_inputs[1],case.solver_inputs[5],case.solver_inputs[6])
            positions,stats=resident.solve(case.solver_inputs,case=case)
            choice=next(c for c in stats['candidates'] if c['id']==stats['selected_id'])
            assert np.array_equal(positions,np.asarray(choice['positions'])) and choice['metrics']['is_feasible']
            row={'index':i,'member':list(item['member']),'n':case.block_count,'full_solve_seconds':stats['seconds'],
                'selected_metrics':choice['metrics'],'solver':stats,'baseline':{'Q':baseline[i]['metrics']['cost'],'timing_not_matched':True}}
            save(out/'cases'/f'{i:04d}.json',row);records.append(row)
            save(out/'summary.json',summarize(records))
            save(out/'progress.json',{'done':len(records),'total':len(indices),'last_index':i,'updated_utc':datetime.now(timezone.utc).isoformat()})
            print(f"{len(records)}/{len(indices)} index{i} Q{choice['metrics']['cost']:.6f} {stats['seconds']:.3f}s aborted{stats['external_abort']}",flush=True)
    finally:
        resident.close();save(out/'setup.json',setups);save(out/'summary.json',summarize(records))
        save(out/'execution.json',{'complete':len(records)==len(indices),'seconds_this_invocation':time.perf_counter()-started,'count':len(records)})


if __name__=='__main__':main()
