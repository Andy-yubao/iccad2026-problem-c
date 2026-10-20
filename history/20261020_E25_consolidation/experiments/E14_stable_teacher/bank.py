"""New corrected-input teacher bank with cumulative budgets and STOP/resume."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import sys
import time
import uuid
import zipfile
import numpy as np
import torch
from .data import HERE,ROOT,VERSION,reader,derived_case,save,sha,balanced_panel
from .pipeline import Baseline,CHECKPOINT_SHA256
from .stable import POLICY
from .targets import SCHEMA,encode,validate
from .frozen.packing import prepare_problem
from .frozen.solver import hard_valid
from scoring.evaluation import evaluate_case


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--split',choices=['train','dev_monitor'],default='train')
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--max-seconds',type=float,default=86400.)
    ap.add_argument('--resume',action='store_true')
    ap.add_argument('--smoke-count',type=int,choices=range(1,6),help='Implementation check only; never a full training bank')
    args=ap.parse_args();begin=time.perf_counter()
    if args.max_seconds<=0:raise ValueError('Positive cumulative budget required')
    torch.set_num_threads(2);torch.manual_seed(0);np.random.seed(0)
    split='train' if args.split=='train' else 'dev'
    manifest=HERE/f'runs/data_v1/{split}.json'
    data=json.loads(manifest.read_text(encoding='utf-8'))
    members=[e['member'] for e in data['members']]
    contract=json.loads((ROOT/'protocol/derived_workset_v3.json').read_text(encoding='utf-8'))
    if data['version']!=VERSION or members!=contract['sets'][split]['members']:
        raise ValueError('Teacher input manifest differs from the frozen v3 members')
    if args.split=='dev_monitor':
        panels=json.loads((HERE/'runs/data_v1/panels.json').read_text(encoding='utf-8'))
        members=balanced_panel(members,exclude=panels['confirmation500'],tag='dev_monitor')
        assert len(members)==500
        if members!=contract['monitors']['dev_monitor500']:
            raise ValueError('Dev monitoring members differ from the registered monitor')
    if args.smoke_count is not None:
        members=members[:args.smoke_count]
    source=reader(members,split)
    files=sorted(HERE.rglob('*.py'))+[ROOT/'datasets/worksets.py',ROOT/'scoring/evaluation.py']
    identity={'schema':SCHEMA,'task_version':VERSION,'policy':POLICY,'split':args.split,
              'smoke':args.smoke_count is not None,
              'members':members,'manifest_sha256':sha(manifest),
              'checkpoint_sha256':CHECKPOINT_SHA256,'seed':0,'threads':2,'device':'cpu',
              'sources':{p.relative_to(ROOT).as_posix():sha(p) for p in files},
              'max_seconds':args.max_seconds,'local_test_read':False,'reference_free_geometry_in_solver':False,
              'budgets':{'normal_total_target':5.,'additional_rescue_target':16.}}
    if args.resume:
        old=json.loads((args.output/'config.json').read_text(encoding='utf-8'))
        if old['identity']!=identity:raise ValueError('Resume identity/source/budget mismatch')
        state=json.loads((args.output/'results.json').read_text(encoding='utf-8'))
        elapsed=state['elapsed_seconds']
        if (args.output/'progress.json').exists():
            elapsed=max(elapsed,json.loads((args.output/'progress.json').read_text(encoding='utf-8'))['elapsed_seconds'])
        rows=state['records']
        if any('error' in r for r in rows):
            raise ValueError('Previous bank failure requires investigation; refusing to skip a failed member')
        for r in rows:
            if r.get('label_file') and sha(args.output/r['label_file'])!=r['label_sha256']:
                raise ValueError('Completed label integrity mismatch')
    else:
        args.output.mkdir(parents=True,exist_ok=False);(args.output/'labels').mkdir()
        save(args.output/'config.json',{'identity':identity,'command':[sys.executable,*sys.argv],
            'created_utc':datetime.now(timezone.utc).isoformat(),'pid':os.getpid()})
        with zipfile.ZipFile(args.output/'source.zip','w',zipfile.ZIP_DEFLATED) as z:
            for p in files:z.write(p,p.relative_to(ROOT))
        elapsed=0.;rows=[]
    solver=Baseline();status='running';known={r['index'] for r in rows}
    # Crash-safe append records avoid quadratic rewriting of a growing bank.
    journal=args.output/'records.jsonl'
    if args.resume and journal.exists():
        for line in journal.read_text(encoding='utf-8').splitlines():
            r=json.loads(line)
            if r['index'] not in known:
                if r.get('label_file') and sha(args.output/r['label_file'])!=r['label_sha256']:
                    raise ValueError('Journal label integrity mismatch')
                rows.append(r);known.add(r['index'])
        if any('error' in r for r in rows):
            raise ValueError('Journal contains a failed member; refusing to resume past it')
    def persist():
        save(args.output/'progress.json',{'status':status,'pid':os.getpid(),'split':args.split,
            'completed':len(rows),'requested':len(source),'errors':sum('error' in r for r in rows),
            'elapsed_seconds':elapsed+time.perf_counter()-begin,'updated_utc':datetime.now(timezone.utc).isoformat()})
        save(args.output/'results.json',{'status':status,'records':rows,'requested_count':len(source),
             'label_count':sum('label_file' in r for r in rows),'error_count':sum('error' in r for r in rows),
             'elapsed_seconds':elapsed+time.perf_counter()-begin,'schema':SCHEMA,'task_version':VERSION})
    persist()
    try:
        for i in range(len(source)):
            if i in known:continue
            if (args.output/'STOP').exists() or elapsed+time.perf_counter()-begin>=args.max_seconds:
                status='stopped';break
            record={'index':i,'member':members[i]};case_start=time.perf_counter()
            try:
                case,changes=derived_case(source.case(i));p=prepare_problem(*case.solver_inputs)
                x,stats=solver.solve(case.solver_inputs);assert hard_valid(x,p)
                result=evaluate_case(case,x,stats['end_to_end_seconds']);result.pop('positions')
                if not result['metrics']['is_feasible']:raise ValueError('Independent/official hard check rejected teacher')
                if result['metrics']['cost']>2.:
                    record['rejected_metrics']=result['metrics']
                    rejected=args.output/'rejected';rejected.mkdir(exist_ok=True)
                    np.savez_compressed(rejected/f'{i:05d}.npz',positions=x,input_areas=np.asarray(case.solver_inputs[1]))
                    raise ValueError('Quality gate Q>2: preserved rejected candidate; bank stopped without accepting label')
                target=encode(x,p);validate(target,p)
                target['input_areas']=np.asarray(case.solver_inputs[1]).copy()
                target['task_version']=np.asarray(VERSION)
                target['schema']=np.asarray(SCHEMA)
                name=f'labels/{i:05d}.npz';path=args.output/name
                tmp=path.with_suffix('.'+uuid.uuid4().hex+'.tmp')
                with tmp.open('wb') as f:np.savez_compressed(f,**target)
                os.replace(tmp,path)
                ss=stats['stability']
                record.update(case_id=case.case_id,label_file=name,label_sha256=sha(path),
                    metrics=result['metrics'],independent_hard_checks=result['independent_hard_checks'],
                    solver_seconds=stats['end_to_end_seconds'],rescue_used=stats.get('rescue_used',False),
                    stability_triggered=ss['triggered'],stability_changed=not ss['baseline_retained'],
                    certified_ratio_bound=ss['certified_ratio_bound'])
            except Exception as exc:
                record['error']=repr(exc)
            record['seconds']=time.perf_counter()-case_start
            rows.append(record);known.add(i)
            with journal.open('a',encoding='utf-8') as f:
                f.write(json.dumps(record,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
            if 'error' in record:
                status='failed';persist();raise RuntimeError(f'Teacher generation failed at {i}: {record["error"]}')
            if len(rows)%25==0 or len(rows)<=5:persist()
            else:
                save(args.output/'progress.json',{'status':'running','pid':os.getpid(),'split':args.split,
                    'completed':len(rows),'requested':len(source),'errors':0,
                    'elapsed_seconds':elapsed+time.perf_counter()-begin,
                    'updated_utc':datetime.now(timezone.utc).isoformat()})
            print(f'{len(rows)}/{len(source)}',case.case_id,round(record['metrics']['cost'],6),flush=True)
        if len(rows)==len(source):status='completed'
    except KeyboardInterrupt:
        status='interrupted'
    finally:
        persist()


if __name__=='__main__':main()
