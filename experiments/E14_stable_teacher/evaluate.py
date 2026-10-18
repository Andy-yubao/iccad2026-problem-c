"""Paired E10/E14 evaluation, keeping every case including errors."""
import argparse
import json
import subprocess
import sys
import time
import zipfile
import numpy as np
import torch
from .data import HERE,ROOT,reader,derived_case,save,sha
from .pipeline import Baseline,CHECKPOINT_SHA256
from .optimizer import optimize
from .stable import polish,POLICY
from scoring.evaluation import evaluate_case
from scoring.official import official_evaluator


def summary(rows):
    out={}
    for key in ['baseline','stable']:
        records=[r['variants'][key] for r in rows]
        m=[r['metrics'] for r in records];q=np.array([v['cost'] for v in m])
        out[key]={'count':len(rows),'Q_mean':float(q.mean()),
            'Q_weighted':official_evaluator().compute_total_score(q.tolist(),[r['block_count'] for r in rows]),
            'Q_max':float(q.max()),'Q_p95':float(np.quantile(q,.95)),
            'above_1_8':int((q>1.8).sum()),'above_2':int((q>2).sum()),
            'feasible':sum(v['is_feasible'] for v in m),
            'mean_seconds':float(np.mean([v.get('seconds',0.) for v in records])),
            **{f'mean_{k}':float(np.mean([v[k] for v in m if k in v])) for k in
               ['hpwl_gap','area_gap','boundary_violations','grouping_violations','mib_violations']}}
    b=np.array([r['variants']['baseline']['metrics']['cost'] for r in rows])
    s=np.array([r['variants']['stable']['metrics']['cost'] for r in rows])
    out['paired']={'improved':int((s<b-1e-6).sum()),'tied':int((np.abs(s-b)<=1e-6).sum()),
        'worse':int((s>b+1e-6).sum()),'regression_over_1pct':int((s>b*1.01).sum()),
        'normal_regression_over_1pct':int(((b<=1.8)&(s>b*1.01)).sum()),
        'triggered':sum(r['variants']['stable'].get('stats',{}).get('triggered',False) for r in rows),
        'changed':sum(not r['variants']['stable'].get('stats',{}).get('baseline_retained',True) for r in rows)}
    out['acceptance']={'all_legal':out['stable']['feasible']==len(rows),
        'no_Q_above_2':out['stable']['above_2']==0,
        'tail_at_most_1pct':out['stable']['above_1_8']/len(rows)<=.01,
        'mean_better':out['stable']['Q_mean']<out['baseline']['Q_mean']-1e-9,
        'weighted_better':out['stable']['Q_weighted']<out['baseline']['Q_weighted']-1e-9,
        'no_normal_regression_over_1pct':out['paired']['normal_regression_over_1pct']==0}
    out['passed']=all(out['acceptance'].values())
    return out


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--panel',choices=['development100','confirmation500'],required=True)
    ap.add_argument('--output',type=__import__('pathlib').Path,required=True)
    ap.add_argument('--resume',action='store_true')
    ap.add_argument('--baseline-run',type=__import__('pathlib').Path,
                    help='Reuse saved paired E10 solves; never rerun unchanged controls')
    args=ap.parse_args();torch.set_num_threads(2);torch.manual_seed(0);np.random.seed(0)
    panels=json.loads((HERE/'runs/data_v1/panels.json').read_text(encoding='utf-8'))
    members=panels[args.panel];source=reader(members)
    replay=None
    if args.baseline_run:
        replay=json.loads((args.baseline_run/'results.json').read_text(encoding='utf-8'))['cases']
        assert len(replay)==len(members)
    files=sorted(HERE.rglob('*.py'))+[ROOT/'datasets/worksets.py',ROOT/'scoring/evaluation.py']
    hashes={p.relative_to(ROOT).as_posix():sha(p) for p in files}
    identity={'policy':POLICY,'panel':args.panel,'members':members,'seed':0,'threads':2,
              'checkpoint_sha256':CHECKPOINT_SHA256,'source_sha256':hashes,
              'split_sha256':sha(ROOT/'protocol/split_v1.json'),'workset_sha256':sha(ROOT/'protocol/workset_v2.json'),
              'budgets':{'baseline':5.,'extra_rescue':16.},'local_test_read':False,
              'reference_in_solver':False,'comparison':'same baseline solve shared to isolate added rescue; E14 time includes baseline'}
    if replay is not None:
        identity['baseline_replay']={'path':str(args.baseline_run),'results_sha256':sha(args.baseline_run/'results.json')}
    if args.resume:
        old=json.loads((args.output/'config.json').read_text(encoding='utf-8'));assert old['identity']==identity
        report=json.loads((args.output/'results.json').read_text(encoding='utf-8'));rows=report['cases']
    else:
        args.output.mkdir(parents=True,exist_ok=False);rows=[]
        save(args.output/'config.json',{'identity':identity,'command':[sys.executable,*sys.argv],
              'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              'git_status':subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True).strip(),
              'torch':torch.__version__,'numpy':np.__version__})
        with zipfile.ZipFile(args.output/'source.zip','w',zipfile.ZIP_DEFLATED) as z:
            for p in files:z.write(p,p.relative_to(ROOT))
    solver=Baseline();start=time.perf_counter()
    for i in range(len(rows),len(source)):
        if (args.output/'STOP').exists():break
        case,_=derived_case(source.case(i));row={'case_id':case.case_id,'block_count':case.block_count,'variants':{}}
        try:
            begin=time.perf_counter();draft=solver.draft(case.solver_inputs)
            if replay is None:
                b,bs=optimize(draft,case.solver_inputs,seconds=5.)
                base_seconds=time.perf_counter()-begin
            else:
                assert replay[i]['case_id']==case.case_id
                control=replay[i]['variants']['baseline']
                b=np.asarray(control['positions']);bs=control['stats'];base_seconds=control['seconds']
            baseline=evaluate_case(case,b,base_seconds)
            s,ss=polish(draft,b,case.solver_inputs,bs,seconds=16.)
            stable=evaluate_case(case,s,base_seconds+ss['seconds'])
            for key,ev,stats,seconds in [('baseline',baseline,bs,base_seconds),('stable',stable,ss,base_seconds+ss['seconds'])]:
                # Save coordinates once per variant; traces do not duplicate them.
                if key=='baseline':
                    stats['improvements']=[{k:v for k,v in e.items() if k!='positions'} for e in stats.get('improvements',[])]
                row['variants'][key]={'metrics':ev['metrics'],'independent_hard_checks':ev['independent_hard_checks'],
                                      'positions':ev['positions'],'stats':stats,'seconds':seconds}
        except Exception as exc:
            for key in ['baseline','stable']:
                row['variants'].setdefault(key,{'error':repr(exc),'metrics':{'cost':10.,'is_feasible':False}})
        rows.append(row)
        report={'status':'complete' if len(rows)==len(source) else 'running','completed':len(rows),
                'summary':summary(rows),'cases':rows,'invocation_seconds':time.perf_counter()-start}
        save(args.output/'results.json',report)
        print(len(rows),case.case_id,{k:round(v['metrics']['cost'],6) for k,v in row['variants'].items()},
              'pass',report['summary']['passed'],flush=True)
    print(json.dumps(summary(rows),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
