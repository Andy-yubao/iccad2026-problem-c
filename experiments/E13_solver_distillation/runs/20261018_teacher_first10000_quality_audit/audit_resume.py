"""Continue the SAME first10000 snapshot, preserving partial evidence and cumulative cap."""
import gzip,json,time,hashlib
from datetime import datetime
from audit import ROOT,OUT,np,torch,sha,save,summarize,LayoutReader,workset_manifest,evaluate_case
from fast_case import scoring_case

def main():
    begin=time.perf_counter();torch.set_num_threads(1)
    if (OUT/'resume_config.json').exists():raise FileExistsError('Continuation already started; preserve its evidence')
    cfg=json.loads((OUT/'config.json').read_text(encoding='utf-8'))
    bank=ROOT/'experiments/E12_distillation_readiness/runs/teacher_train20000_v1'
    state=json.loads(gzip.decompress((OUT/'teacher_results_snapshot.json.gz').read_bytes()));records=state['records'][:10000]
    oldfile=OUT/'comparisons.jsonl';blob=oldfile.read_bytes();(OUT/'comparisons_v1_partial.jsonl.gz').write_bytes(gzip.compress(blob,compresslevel=1,mtime=0))
    rows=[]
    for line in blob.splitlines():
        try:rows.append(json.loads(line))
        except json.JSONDecodeError:break
    assert all(r['index']==i and r['case_id']==records[i]['case_id'] for i,r in enumerate(rows))
    previous=json.loads((OUT/'progress.json').read_text(encoding='utf-8'))
    prior=max(previous['elapsed_seconds'],oldfile.stat().st_mtime-datetime.fromisoformat(cfg['timestamp_Asia_Shanghai']).timestamp()+30.)
    reader=LayoutReader(workset_manifest(),'train');lookup={m:i for i,m in enumerate(reader.members)}
    equivalence=[]
    # Five already-scored cases, fixed before looking at their outcomes.
    for index in (0,250,500,750,999):
        old=rows[index];raw=reader.case(index);case=scoring_case(raw,old['teacher'])
        fast=evaluate_case(case,raw.reference_positions,1.)
        for key,value in old['reference'].items():
            if isinstance(value,float):assert abs(value-fast['metrics'][key])<1e-10,(index,key,value,fast['metrics'][key])
            else:assert value==fast['metrics'][key],(index,key)
        equivalence.append(index)
    save('resume_config.json',{'prior_compute_charge_seconds':prior,'continued_from_count':len(rows),
        'same_snapshot_sha256':cfg['snapshot_sha256'],'source_sha256':sha(__file__),'adapter_sha256':sha(OUT/'fast_case.py'),
        'reference_all_metrics_equivalent_indices':equivalence,'cumulative_cap_seconds':1100,
        'reason':'Reuse stored official baseline; NumPy edge scalars avoid repeated scalar tensor dispatch. Official formula/evaluator unchanged.',
        'old_partial_sha256':hashlib.sha256(blob).hexdigest()})
    path=OUT/'comparisons_continued.jsonl'
    with path.open('w',encoding='utf-8') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
        for k,record in enumerate(records[len(rows):],start=len(rows)):
            if prior+time.perf_counter()-begin>1100:break
            raw=reader.case(lookup[tuple(record['member'])]);assert raw.case_id==record['case_id']
            label=bank/record['label_file'];assert sha(label)==record['label_sha256']
            teacher=record['evaluation']['metrics'];case=scoring_case(raw,teacher);ref=evaluate_case(case,raw.reference_positions,1.)
            row={'index':k,'case_id':raw.case_id,'worker':record['member'][0].split('/')[0],
              'reference':ref['metrics'],'reference_independent_legal':ref['independent_hard_checks']['is_feasible'],
              'teacher':teacher,'teacher_independent_legal':record['evaluation']['independent_hard_checks']['is_feasible'],
              'Q_delta_teacher_minus_reference':teacher['cost']-ref['metrics']['cost'],
              'rescue':record['teacher_stats'].get('rescue_used',False),'label_sha256_verified':True,
              'reference_input_adapter':'numpy_connectivity_same_official_evaluator'}
            if k in cfg['teacher_rescore_indices']:
                with np.load(label,allow_pickle=False) as z:ev=evaluate_case(case,z['positions'],1.)
                assert abs(ev['metrics']['cost']-teacher['cost'])<1e-10 and ev['metrics']['is_feasible']==teacher['is_feasible']
                row['teacher_rescore_checked']=True
            rows.append(row);f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
            if (k+1)%500==0:
                f.flush();save('progress_continued.json',{'completed':len(rows),'elapsed_seconds':prior+time.perf_counter()-begin})
                print(k+1,'Q ref/teacher:',np.mean([r['reference']['cost'] for r in rows]),np.mean([r['teacher']['cost'] for r in rows]),'cumulative',prior+time.perf_counter()-begin,flush=True)
    result=summarize(rows);result.update(status='completed' if len(rows)==10000 else 'time_cap_partial',
        elapsed_seconds=prior+time.perf_counter()-begin,comparison_file=path.name,
        scope='Paired teacher/reference quality on fixed first10000 train tasks, not trained-student performance; no solver/training launched.')
    save('results.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('workers','worst_regressions')},ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
