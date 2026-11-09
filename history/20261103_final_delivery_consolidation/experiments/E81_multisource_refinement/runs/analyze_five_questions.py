"""Saved-result audit for the user's five research questions; no learned forward."""
import json,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.stats import spearmanr,rankdata
from experiments.E81_multisource_refinement.runs.placer_corrector_audit import PATHS,OUT
from experiments.E81_multisource_refinement.losses import WEIGHTS
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def rho(a,b):return float(spearmanr(a,b).statistic)
def exact_n_rank(a,b,ns):
    x=np.zeros(len(a));y=x.copy()
    for n in np.unique(ns):
        ix=ns==n;x[ix]=rankdata(a[ix])-np.mean(rankdata(a[ix]));y[ix]=rankdata(b[ix])-np.mean(rankdata(b[ix]))
    return float(np.corrcoef(x,y)[0,1])
def raw_g(r):return r.get('loss_common_original_G',sum(r.get('loss_'+k,0)*v for k,v in WEIGHTS.items()))
def intersection(p,denom):
    p=np.asarray(p,dtype=float);lo=p[:,:2];hi=lo+p[:,2:]
    inter=np.maximum(np.minimum(hi[:,None],hi[None,:])-np.maximum(lo[:,None],lo[None,:]),0).prod(-1)
    return float(np.triu(inter,1).sum()/denom)

def main():
    feature_path=ROOT/'experiments/E69_case_loss_statistics/runs/20261028_dev2000/features.json'
    features=read(feature_path)['records'];ns=np.array([r['n'] for r in features]);pin=np.array([r['pin_connected_fraction'] for r in features]);density=np.array([r['connection_density'] for r in features])
    sparse=np.array([density[i]<=np.median(density[ns==n]) for i,n in enumerate(ns)])
    groups={'all':ns>0,'low_pin':pin<=.3,'high_pin':pin>=.8,'sparse_same_n':sparse,
      'low_pin_sparse':(pin<=.3)&sparse,'low_pin_dense':(pin<=.3)&~sparse,
      'high_pin_sparse':(pin>=.8)&sparse,'high_pin_dense':(pin>=.8)&~sparse}
    legacy=read(ROOT/'experiments/E69_case_loss_statistics/runs/20261028_dev2000/forwards.json')
    paths=dict(PATHS);paths['E78']=ROOT/'experiments/E78_connectivity_posttrain/runs/20261029_e77_connectivity_seed0'
    raw={n:read(p/'full_dev.json')['records'] for n,p in paths.items() if n!='E42'};raw['E42']=legacy['models']['E42']
    lookup={tuple(r['member']):i for i,r in enumerate(features)}
    output={'scope':'User paused research; saved dev2000/dev500 analysis plus deterministic reference decode, no model inference, optimizer, solver or scoring',
      'test_read':False,'model_forward_calls':0,'training_updates':0,'solver_calls':0,'new_Q_evaluations':0,
      'subgroup_definition':'low pin<=.3; high pin>=.8; sparse density<=median at identical n (ties included)', 'models':{}}
    for name in ('E42','E74','E78','E79','E80','E81','E82'):
        records=raw[name];assert [r['member'] for r in records]==[r['member'] for r in features]
        overlaps=np.array([r['overlap_area_fraction'] for r in records]);g=np.array([raw_g(r) for r in records])
        terminal=read(paths[name]/'terminal/results.json')['records'];ids=np.array([lookup[tuple(r['member'])] for r in terminal]);q=np.array([r['metrics']['cost'] for r in terminal]);panel_n=ns[ids]
        by_group={}
        for label,mask in groups.items():
            ix=mask[ids];selq=q[ix];seln=panel_n[ix]
            d={'count2000':int(mask.sum()),'mean_overlap':float(overlaps[mask].mean()),'mean_G':float(g[mask].mean()),'count500':int(ix.sum()),
              'mean_Q':float(selq.mean()),'weighted_Q':float(np.average(selq,weights=np.exp((seln-seln.max())/12))),
              'p95_Q':float(np.percentile(selq,95)),'max_Q':float(selq.max())}
            by_group[label]=d
        worst=np.argsort(overlaps)[-200:];qworst=np.argsort(q)[-50:]
        concentration={}
        for key,idx in (('worst_overlap10pct',worst),('worst_postQ10pct',ids[qworst])):
            concentration[key]={'count':len(idx),'mean_n':float(ns[idx].mean()),'mean_pin_fraction':float(pin[idx].mean()),
              'low_pin_fraction':float(groups['low_pin'][idx].mean()),'low_pin_sparse_fraction':float(groups['low_pin_sparse'][idx].mean()),
              'sparse_same_n_fraction':float(sparse[idx].mean())}
        associations={}
        for key in ('pin_connected_fraction','connection_density','fixed_fraction','preplaced_fraction','boundary_fraction','area_cv','mib_fraction','group_fraction','ref_area_transform_log_mismatch'):
            x=np.array([r[key] for r in features]);associations[key]={'overlap_rho':rho(x,overlaps),'overlap_exact_n_rank_r':exact_n_rank(x,overlaps,ns),'G_rho':rho(x,g)}
        top=[]
        for j in np.argsort(q)[-10:][::-1]:
            r=terminal[j];i=ids[j];top.append({'index':r['index'],'n':int(ns[i]),'Q':float(q[j]),'pin':float(pin[i]),'density':float(density[i]),
              'sparse_same_n':bool(sparse[i]),'raw_overlap':float(overlaps[i]),'rescue':r.get('rescue_used'),
              'grouping':r['metrics']['grouping_violations'],'boundary':r['metrics']['boundary_violations'],
              'hpwl_gap':r['metrics']['hpwl_gap'],'bbox_gap':r['metrics']['area_gap']})
        output['models'][name]={'groups':by_group,'concentration':concentration,'associations':associations,'worst_Q10':top}
    # Geometry of a perfect fit to the existing reference state under public areas.
    # This decodes labels, never a learned network, and cannot use reference as solver input.
    from experiments.E81_multisource_refinement.data import load_rows,collate
    from experiments.E81_multisource_refinement.batches import decode
    import torch
    torch.set_num_threads(4);rows=load_rows('dev');oracle=[];original=[];mib_bad=[];original_mib_bad=[];reference_area_bad=[]
    with torch.no_grad():
        for first in range(0,len(rows),32):
            selected=rows[first:first+32];g,_=collate(selected,'cpu');decoded=(decode(g['target'],g)*g['scale'][:,None,None]).double().numpy()
            for j,row in enumerate(selected):
                n=row['graph']['n'];inp=row['case'].solver_inputs;denom=np.asarray(inp[1],dtype=float).sum();p=decoded[j,:n]
                ref=(row['graph']['reference']*row['graph']['scale']).double().numpy()
                oracle.append(intersection(p,denom));original.append(intersection(ref,denom))
                c=np.asarray(inp[5]);bad=False;oldbad=False
                reference_area_bad.append(bool(np.any(np.abs(np.log(ref[:,2:].prod(1)/np.asarray(inp[1])))>np.log(1.01))))
                for group in sorted(set(c[:,2])-{0}):
                    ids=np.flatnonzero(c[:,2]==group)
                    if len(ids)>1:
                        if np.max(np.abs(np.log(p[ids,2:]/p[ids[0],2:])))>np.log(1.01):bad=True
                        if np.max(np.abs(np.log(ref[ids,2:]/ref[ids[0],2:])))>np.log(1.01):oldbad=True
                mib_bad.append(bad);original_mib_bad.append(oldbad)
    oracle=np.array(oracle);original=np.array(original);o=np.array([r['total'] for r in legacy['reference_state']])
    output['reference_target_audit']={'count':2000,'mean_original_reference_pair_intersection':float(original.mean()),
      'mean_target_decoded_pair_intersection':float(oracle.mean()),'target_decoded_overlap_gt1e_5_cases':int((oracle>1e-5).sum()),
      'target_decoded_mib_width_or_height_disagreement_over1pct_cases':int(np.sum(mib_bad)),
      'original_reference_mib_width_or_height_disagreement_over1pct_cases':int(np.sum(original_mib_bad)),
      'original_reference_area_vs_current_public_area_disagreement_over1pct_cases':int(np.sum(reference_area_bad)),
      'saved_reference_state_mean_G':float(o.mean()),
      'saved_reference_state_G_vs_E81_G_rho':rho(o,np.array([raw_g(r) for r in raw['E81']])),
      'target_decoded_overlap_vs_E81_overlap_rho':rho(oracle,np.array([r['overlap_area_fraction'] for r in raw['E81']])),
      'limits':'A reference-state diagnostic, not optimum/lower bound or a pure MIB causal effect. Includes public area/locked spec/decode/clamp conventions. MIB discrepancy is shape consistency diagnostic, not official scored validation.'}
    output['source_files']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [feature_path,*[paths[n]/'terminal/results.json' for n in paths]]}
    (OUT/'five_questions.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Reference audit:',json.dumps(output['reference_target_audit'],ensure_ascii=False))
    for name in ('E42','E74','E78','E81','E82'):
        r=output['models'][name];print(name,'groups',json.dumps(r['groups'],ensure_ascii=False));print(name,'tail',r['concentration']);print(name,'rho',r['associations'])

if __name__=='__main__':main()
