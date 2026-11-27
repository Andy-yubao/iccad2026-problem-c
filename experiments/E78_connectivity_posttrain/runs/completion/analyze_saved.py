"""Analyze saved draft/repair pairs; no model forward or solver evaluation."""
import json
from pathlib import Path
import numpy as np
import torch
from experiments.E78_connectivity_posttrain.io import HERE,ROOT,save,sha
from experiments.E78_connectivity_posttrain.data import load_rows,collate
from experiments.E78_connectivity_posttrain.connectivity import public_group_contact,b2b_excess

PATHS={'E78':'E78_connectivity_posttrain/runs/20261029_e77_connectivity_seed0','E74':'E74_collision_32k_posttrain/runs/20261029_e42_topology_seed0',
    'E76':'E76_bounded_refinement/runs/20261029_frozen_e74_corrector_seed0',
    'E77':'E77_compact_refinement/runs/20261029_e76_compact_corrector_seed0'}

def correlation(a,b):
    a=np.asarray(a);b=np.asarray(b)
    return float(np.corrcoef(a,b)[0,1]) if a.std()>1e-12 and b.std()>1e-12 else None

def main():
    torch.set_num_threads(4)
    rows=load_rows('dev_monitor500');records={};result={'models':{},'identities':{},'test_read':False,
        'role':'Saved exposed dev diagnostics only; no inference/solver or checkpoint reselection',
        'E42_raw_geometry_unavailable':'Original500 terminal did not save drafts; do not substitute E63 or rerun E42'}
    for name,path in PATHS.items():
        f=ROOT/'experiments'/path/'terminal/results.json';saved=json.loads(f.read_text(encoding='utf-8'))
        records[name]=saved['records'];assert [r['member'] for r in records[name]]==[list(r['member']) for r in rows]
        result['identities'][name]={'saved_results_sha256':sha(f),'checkpoint_sha256':sha(ROOT/'experiments'/path/'checkpoints/best.pt')}
        values=[]
        for start in range(0,500,32):
            chunk=records[name][start:start+32];g,_=collate(rows[start:start+32],'cpu');n=g['valid'].shape[1]
            arrays={}
            for kind,key in (('draft','draft_positions'),('final','positions')):
                p=torch.zeros(len(chunk),n,4)
                for i,r in enumerate(chunk):p[i,:r['n']]=torch.tensor(r[key])/g['scale'][i]
                with torch.no_grad():
                    group=public_group_contact(p,g);wire,ratio=b2b_excess(p,g)
                arrays[kind]=(group.tolist(),wire.tolist(),ratio.tolist())
            for i,r in enumerate(chunk):
                public=rows[start+i]['case'].solver_inputs
                b2b=np.asarray(public[2]);c=np.asarray(public[5]);p=np.asarray(r['draft_positions']);ref=rows[start+i]['graph']['reference'].numpy()*float(g['scale'][i])
                center=p[:,:2]+p[:,2:]/2;rc=ref[:,:2]+ref[:,2:]/2
                if len(b2b):
                    a,b=b2b[:,:2].astype(int).T;pred=(np.abs(center[a]-center[b]).sum(1)*b2b[:,2]).sum();target=(np.abs(rc[a]-rc[b]).sum(1)*b2b[:,2]).sum()
                    exact_ratio=pred/max(target,1e-8)
                    assert abs(exact_ratio-arrays['draft'][2][i])<2e-5
                else:exact_ratio=1.
                values.append({'index':r['index'],'member':r['member'],'n':r['n'],
                    'draft_group_proxy':arrays['draft'][0][i],'final_group_proxy':arrays['final'][0][i],
                    'draft_b2b_excess':arrays['draft'][1][i],'draft_b2b_ratio':arrays['draft'][2][i],
                    'final_b2b_ratio':arrays['final'][2][i],
                    'raw_grouping':r['raw_metrics'].get('grouping_violations'),
                    'final_grouping':r['metrics']['grouping_violations'],'final_boundary':r['metrics']['boundary_violations'],
                    'Q':r['metrics']['cost'],'raw_intersection':r['draft_diagnostics']['overlap_area_fraction']})
        result['models'][name]={'records':values,'means':{k:float(np.mean([r[k] for r in values])) for k in
            ('draft_group_proxy','final_group_proxy','draft_b2b_excess','draft_b2b_ratio','final_b2b_ratio','raw_grouping','final_grouping','final_boundary')},
            'raw_group_zero_final_positive':sum(r['raw_grouping']==0 and r['final_grouping']>0 for r in values),
            'proxy_final_grouping_correlation':correlation([r['draft_group_proxy'] for r in values],[r['final_grouping'] for r in values])}
    a=result['models']['E77']['records'];b=result['models']['E76']['records']
    result['paired_E77_minus_E76']={'grouping_worse_equal_better':[sum(x['final_grouping']>y['final_grouping'] for x,y in zip(a,b)),sum(x['final_grouping']==y['final_grouping'] for x,y in zip(a,b)),sum(x['final_grouping']<y['final_grouping'] for x,y in zip(a,b))],
        'proxy_change_grouping_change_correlation':correlation([x['draft_group_proxy']-y['draft_group_proxy'] for x,y in zip(a,b)],[x['final_grouping']-y['final_grouping'] for x,y in zip(a,b)])}
    result['tail_cases']={str(i):{name:next(r for r in result['models'][name]['records'] if r['index']==i) for name in PATHS} for i in (164,52,246,217,399)}
    save(HERE/'runs/completion/saved_pair_analysis.json',result)
    print(json.dumps({name:{k:v for k,v in r.items() if k!='records'} for name,r in result['models'].items()},indent=2),flush=True)
    print('paired',result['paired_E77_minus_E76'],flush=True)
    print('tail164',result['tail_cases']['164'],flush=True)

if __name__=='__main__':main()
