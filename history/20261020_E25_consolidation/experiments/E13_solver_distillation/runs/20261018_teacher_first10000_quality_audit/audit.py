"""Read-only fixed first-10000 teacher/reference paired audit; no solving/training."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import sys,json,time,gzip,hashlib,collections
from datetime import datetime,timezone,timedelta
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT));OUT=Path(__file__).resolve().parent
import numpy as np
import torch
from datasets.worksets import LayoutReader,workset_manifest
from datasets.cases import _build_case
from scoring.evaluation import evaluate_case

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,r):
    p=OUT/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');os.replace(t,p)

def summarize(rows):
    r=np.array([x['reference']['cost'] for x in rows]);t=np.array([x['teacher']['cost'] for x in rows]);gain=r-t
    result={'count':len(rows),'Q_reference_mean':float(r.mean()),'Q_teacher_mean':float(t.mean()),
      'Q_reference_median':float(np.median(r)),'Q_teacher_median':float(np.median(t)),
      'Q_reference_p95':float(np.quantile(r,.95)),'Q_teacher_p95':float(np.quantile(t,.95)),
      'Q_reference_max':float(r.max()),'Q_teacher_max':float(t.max()),
      'mean_absolute_improvement':float(gain.mean()),'mean_Q_reduction_fraction':float(1-t.mean()/r.mean()),
      'paired_relative_reduction_mean':float(np.mean(gain/r)),'paired_relative_reduction_median':float(np.median(gain/r)),
      'improved_count':int((gain>1e-6).sum()),'worse_count':int((gain < -1e-6).sum()),'tied_count':int((np.abs(gain)<=1e-6).sum()),
      'improved_at_least_5pct_count':int((gain/r>=.05).sum()),'improved_at_least_10pct_count':int((gain/r>=.1).sum()),
      'worse_at_least_5pct_count':int((gain/r<=-.05).sum()),
      'reference_hard_legal':sum(x['reference']['is_feasible'] for x in rows),
      'teacher_hard_legal':sum(x['teacher']['is_feasible'] and x['teacher_independent_legal'] for x in rows),
      'rescue_count':sum(x['rescue'] for x in rows),
      'teacher_Q_above_2':int((t>2).sum()),'teacher_Q_above_3':int((t>3).sum()),
      'teacher_metrics_rechecked':sum(x.get('teacher_rescore_checked',False) for x in rows)}
    grouped=collections.defaultdict(list)
    for x in rows:grouped[x['worker']].append(x)
    workers=[]
    for worker,rr in sorted(grouped.items(),key=lambda kv:int(kv[0].split('_')[1])):
        a=np.array([x['reference']['cost'] for x in rr]);b=np.array([x['teacher']['cost'] for x in rr])
        workers.append({'worker':worker,'count':len(rr),'reference_Q_mean':float(a.mean()),'teacher_Q_mean':float(b.mean()),
          'mean_improvement':float((a-b).mean()),'Q_reduction_fraction':float(1-b.mean()/a.mean())})
    result['workers']=workers
    result['improved_worker_count']=sum(w['mean_improvement']>1e-6 for w in workers)
    if len(workers)>1:
        rng=np.random.default_rng(20261018);a=np.array([w['reference_Q_mean'] for w in workers]);b=np.array([w['teacher_Q_mean'] for w in workers]);counts=np.array([w['count'] for w in workers])
        ids=rng.integers(0,len(workers),(10000,len(workers)))
        aa=(a[ids]*counts[ids]).sum(1)/counts[ids].sum(1);bb=(b[ids]*counts[ids]).sum(1)/counts[ids].sum(1)
        result['worker_cluster_bootstrap']={'resamples':10000,'seed':20261018,
          'absolute_improvement_95CI':np.quantile(aa-bb,[.025,.975]).tolist(),
          'relative_mean_Q_reduction_95CI':np.quantile(1-bb/aa,[.025,.975]).tolist(),
          'scope':'Resample workers, retaining every audited layout within a worker; descriptive stability, not hidden-test generalization.'}
    for name in ['bbox_area','hpwl_total','boundary_violations','grouping_violations','mib_violations','total_soft_violations']:
        result[name]={'reference_mean':float(np.mean([x['reference'][name] for x in rows])),
          'teacher_mean':float(np.mean([x['teacher'][name] for x in rows]))}
    result['worst_regressions']=[{k:x[k] for k in ('case_id','worker','Q_delta_teacher_minus_reference','rescue')} for x in sorted(rows,key=lambda x:x['Q_delta_teacher_minus_reference'],reverse=True)[:10]]
    for label,selected in [('rescue',[x for x in rows if x['rescue']]),('normal',[x for x in rows if not x['rescue']])]:
        if selected:result[label]={'count':len(selected),'reference_Q_mean':float(np.mean([x['reference']['cost'] for x in selected])),
           'teacher_Q_mean':float(np.mean([x['teacher']['cost'] for x in selected])),
           'worse_count':sum(x['Q_delta_teacher_minus_reference']>1e-6 for x in selected)}
    return result

def main():
    begin=time.perf_counter();torch.set_num_threads(1)
    bank=ROOT/'experiments/E12_distillation_readiness/runs/teacher_train20000_v1'
    if (OUT/'config.json').exists():raise FileExistsError('Preserve fixed audit; use its saved evidence')
    blob=(bank/'results.json').read_bytes();state=json.loads(blob);config=json.loads((bank/'config.json').read_text(encoding='utf-8'))
    records=state['records'];assert len(records)>=10000
    chosen=records[:10000];assert len({r['case_id'] for r in chosen})==10000
    (OUT/'teacher_results_snapshot.json.gz').write_bytes(gzip.compress(blob,compresslevel=1,mtime=0))
    reader=LayoutReader(workset_manifest(),'train');lookup={m:i for i,m in enumerate(reader.members)}
    assert config['identity']['members']==[list(m) for m in reader.members]
    assert [r['member'] for r in chosen]==[list(m) for m in reader.members[:10000]]
    cfg={'timestamp_Asia_Shanghai':datetime.now(timezone(timedelta(hours=8))).isoformat(),
       'bank_status_at_snapshot':state['status'],'bank_completed_at_snapshot':state['completed_count'],'bank_errors_at_snapshot':state['error_count'],
       'snapshot_sha256':hashlib.sha256(blob).hexdigest(),'bank_config_sha256':sha(bank/'config.json'),
       'audit_source_sha256':sha(__file__),'split_sha256':sha(ROOT/'protocol/split_v1.json'),'workset_sha256':sha(ROOT/'protocol/workset_v2.json'),
       'requested_count':10000,'selection':'First 10000 fixed train workset members, index 0..9999, chosen before evaluating references.',
       'torch_version':str(torch.__version__),'threads':1,'solver_started':False,'training_started':False,'local_test_read':False,
       'quality':'Shared archived official evaluator, runtime=median_runtime=1 for both answers, Q lower is better.',
       'teacher_rescore_indices':[0,2499,4999,7499,9999],'time_cap_seconds':1100}
    save('config.json',cfg);rows=[];checks=set(cfg['teacher_rescore_indices'])
    with (OUT/'comparisons.jsonl').open('w',encoding='utf-8') as f:
        for k,record in enumerate(chosen):
            if time.perf_counter()-begin>1100:break
            raw=reader.case(lookup[tuple(record['member'])]);assert raw.case_id==record['case_id']
            label=bank/record['label_file'];assert label.resolve().is_relative_to(bank.resolve()) and sha(label)==record['label_sha256']
            case=_build_case(raw.case_id,raw.source,raw.solver_inputs[1:6],raw.reference_positions,raw.stored_metrics)
            ref=evaluate_case(case,raw.reference_positions,1.)
            teacher=record['evaluation']['metrics']
            row={'index':k,'case_id':raw.case_id,'worker':record['member'][0].split('/')[0],
              'reference':ref['metrics'],'reference_independent_legal':ref['independent_hard_checks']['is_feasible'],
              'teacher':teacher,'teacher_independent_legal':record['evaluation']['independent_hard_checks']['is_feasible'],
              'Q_delta_teacher_minus_reference':teacher['cost']-ref['metrics']['cost'],
              'rescue':record['teacher_stats'].get('rescue_used',False),'label_sha256_verified':True}
            if k in checks:
                with np.load(label,allow_pickle=False) as z:ev=evaluate_case(case,z['positions'],1.)
                assert abs(ev['metrics']['cost']-teacher['cost'])<1e-10 and ev['metrics']['is_feasible']==teacher['is_feasible']
                row['teacher_rescore_checked']=True
            rows.append(row);f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
            if (k+1)%500==0:
                f.flush();save('progress.json',{'completed':len(rows),'elapsed_seconds':time.perf_counter()-begin})
                print(k+1,'Q ref/teacher:',np.mean([x['reference']['cost'] for x in rows]),np.mean([x['teacher']['cost'] for x in rows]),'elapsed',time.perf_counter()-begin,flush=True)
    result=summarize(rows);result.update(status='completed' if len(rows)==10000 else 'time_cap_partial',elapsed_seconds=time.perf_counter()-begin,
      scope='Teacher labels versus official original answers on first 10000 train members; not student training or unseen evaluation.')
    save('results.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('workers','worst_regressions')},ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
