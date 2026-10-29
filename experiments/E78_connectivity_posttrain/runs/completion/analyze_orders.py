"""Exact difference-constraint certificates on saved drafts; no solver rerun."""
import json
from collections import Counter
import numpy as np
from experiments.E78_connectivity_posttrain.io import HERE,ROOT,save
from experiments.E78_connectivity_posttrain.data import load_rows
from experiments.E78_connectivity_posttrain.backend.frozen.packing import prepare_problem,_dimensions
from experiments.E78_connectivity_posttrain.fast_geometry import topology
from .analyze_saved import PATHS

def certificate(pos,edges,preplaced,dimensions):
    n=len(pos);center=pos[:,:2]+pos[:,2:]/2;worst={'overfill_local':0.,'axis':None,'path':[],'target':None}
    count=0
    for axis in (0,1):
        selected=edges[edges[:,2]==axis];adj=[[] for _ in range(n)]
        for i,j,_ in selected:adj[i].append(j)
        order=np.lexsort((np.arange(n),center[:,axis]));earliest=np.zeros(n);parent=np.full(n,-1)
        for i in order:
            if preplaced[i]:
                excess=earliest[i]-pos[i,axis]
                if excess>1e-7:
                    count+=1;normalized=excess/np.sqrt(np.prod(pos[i,2:]))
                    if normalized>worst['overfill_local']:
                        chain=[int(i)];at=int(parent[i])
                        while at>=0 and at not in chain:
                            chain.append(at)
                            if preplaced[at]:break
                            at=int(parent[at])
                        worst={'overfill_local':float(normalized),'absolute_excess':float(excess),'axis':axis,'path':chain[::-1],'target':int(i)}
                earliest[i]=pos[i,axis]
            for j in adj[i]:
                proposal=earliest[i]+dimensions[i,axis]
                if proposal>earliest[j]:earliest[j]=proposal;parent[j]=i
    return {'violated_anchor_bounds':count,'worst':worst}

def main():
    rows=load_rows('dev_monitor500');result={'models':{},'no_solver_or_forward_rerun':True,'test_read':False}
    for name,path in PATHS.items():
        records=json.loads((ROOT/'experiments'/path/'terminal/results.json').read_text(encoding='utf-8'))['records'];items=[]
        for row,record in zip(rows,records):
            p=prepare_problem(*row['case'].solver_inputs);raw=np.asarray(record['draft_positions'])
            w,h=_dimensions(p,raw[:,2]/raw[:,3],True);prepared=np.column_stack((raw[:,:2],w,h))
            edges=topology(prepared,p.constraints,array=True)
            relative=np.abs(np.log(prepared[:,2:]/raw[:,2:]))
            minimum=prepared[:,2:]*np.where((p.fixed|p.preplaced)[:,None],1.,.4)
            fixed=certificate(prepared,edges,p.preplaced,prepared[:,2:])
            variable=certificate(prepared,edges,p.preplaced,minimum)
            phases=Counter(x.get('phase') for x in record['trace'])
            status=Counter(x.get('status') for x in record['trace'] if x.get('phase')=='shape')
            items.append({'index':record['index'],'n':record['n'],'Q':record['metrics']['cost'],
                'fixed_shape_certificate':fixed,'necessary_free_shape_certificate':variable,
                'MIB_snap_mean_abs_log_size':float(relative.mean()),'MIB_snap_max_abs_log_size':float(relative.max()),
                'rescue':record['rescue_used'],'row_fallback':record['row_fallback'],
                'shape_candidates':record['shape_candidate_count'],'failed_shape_candidates':record['failed_shape_candidate_count'],
                'phases':dict(phases),'shape_statuses':dict(status),
                'preplaced_count':int(p.preplaced.sum()),'locked_count':int((p.fixed|p.preplaced).sum())})
        result['models'][name]={'records':items,'summary':{
            'fixed_shape_infeasible':sum(r['fixed_shape_certificate']['violated_anchor_bounds']>0 for r in items),
            'necessary_free_shape_infeasible':sum(r['necessary_free_shape_certificate']['violated_anchor_bounds']>0 for r in items),
            'rescue_count':sum(r['rescue'] for r in items),
            'rescue_with_fixed_certificate':sum(r['rescue'] and r['fixed_shape_certificate']['violated_anchor_bounds']>0 for r in items),
            'MIB_snap_max_over_1percent':sum(r['MIB_snap_max_abs_log_size']>.01 for r in items)}}
    result['tail_cases']={str(i):{name:next(r for r in m['records'] if r['index']==i) for name,m in result['models'].items()} for i in (164,52,246)}
    save(HERE/'runs/completion/order_certificates.json',result)
    print(json.dumps({name:m['summary'] for name,m in result['models'].items()},indent=2),flush=True)
    print('tail164',result['tail_cases']['164'],flush=True)

if __name__=='__main__':main()
