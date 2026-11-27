"""Frozen public policy replay; references and true Q enter only after selection."""
from pathlib import Path
import sys,os,json,math,hashlib,csv
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import numpy as np
from policy import select
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
SOURCE=HERE.parent/'20261024_frozen100_parallel_v1'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    config=read(HERE/'config.json');assert sha(HERE/'policy.py')==config['policy_sha256']
    assert read(SOURCE/'execution.json')['complete']
    assert sha(SOURCE/'config.json')==config['source_config_sha256']
    for i,digest in config['development_case_sha256'].items():assert sha(SOURCE/'cases'/f'{int(i):04d}.json')==digest
    rows=[];public_data=None
    for i in read(SOURCE/'config.json')['indices']:
        r=read(SOURCE/'cases'/f'{i:04d}.json');allc=r['solver']['candidates']
        ns=max(1,allc[0]['metrics']['max_possible_violations'])  # public constraint count, never a reference denominator
        mat=np.array([[c['area'],c['wire']] for c in allc]);scale=np.maximum(mat.max(0),1e-12)
        target=np.array([2*math.exp(c['public_quality_cost']-2*sum(c['soft_violations'])/ns)-1 for c in allc])
        coeff,_,rank,_=np.linalg.lstsq(mat/scale,target,rcond=None);coeff=coeff/scale
        norm_source='recovered only from already-public logged proxy identities'
        if rank<2 or (coeff<=0).any():
            sys.path.insert(0,str(ROOT))
            from datasets.reference_layouts import ReferenceLayouts
            from experiments.E50_anytime_quality.backend.frozen.packing import prepare_problem
            from experiments.E50_anytime_quality.backend.frozen.feasible import Objective
            from experiments.E50_anytime_quality.archive import raw_measures
            if public_data is None:public_data=ReferenceLayouts('dev_monitor500')
            p=prepare_problem(*public_data[i]['case'].solver_inputs)
            coeff=np.array([1/Objective(p).area_scale,1/max(raw_measures(np.asarray(r['draft']),p)[1],1e-12)])
            norm_source='fresh public inputs; degenerate logged proxy system'
        assert np.max(np.abs(mat@coeff-target))<1e-6
        sanitized=[{k:c[k] for k in('id','area','wire','soft_violations','distance')} for c in allc]
        chosen=select(sanitized,1/coeff[0],1/coeff[1],ns)
        actual=next(c for c in allc if c['id']==chosen['id'])  # first use of true Q after choosing
        rows.append({'index':i,'n':r['n'],'split':'development43' if i in config['development_indices'] else 'confirmation57',
            'baseline_Q':r['baseline_Q'],'old_selected_Q':r['selected_metrics']['cost'],'replayed_Q':actual['metrics']['cost'],
            'candidate_id':actual['id'],'selected_D':actual['distance']['D'],'norm_source':norm_source,
            'source_case':str((SOURCE/'cases'/f'{i:04d}.json').relative_to(ROOT)).replace('\\','/')})
    def summary(values):
        q=np.array([r['replayed_Q'] for r in values]);b=np.array([r['baseline_Q'] for r in values]);w=np.exp((np.array([r['n'] for r in values])-120)/12)
        return {'count':len(values),'weighted_Q':float(np.average(q,weights=w)),'baseline_weighted_Q':float(np.average(b,weights=w)),
            'mean_Q':float(q.mean()),'p95_Q':float(np.quantile(q,.95)),'max_Q':float(q.max()),'Q_lt1_03':int(sum(q<1.03)),
            'regressions':[{'index':r['index'],'delta':r['replayed_Q']-r['baseline_Q']} for r in values if r['replayed_Q']>r['baseline_Q']+1e-8],
            'over1_2':[r['index'] for r in values if r['replayed_Q']>1.2]}
    result={'all100':summary(rows),'development43':summary([r for r in rows if r['split']=='development43']),
        'confirmation57':summary([r for r in rows if r['split']=='confirmation57']),
        'scope':'frozen public-only selection replay of saved layouts; not a live new100 or timing validation'}
    (HERE/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    with (HERE/'selected_ids.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
