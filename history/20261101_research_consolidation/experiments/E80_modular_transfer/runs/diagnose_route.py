"""Analyze saved E74/E79/E80 repair paths, with no inference or solver calls."""
import json,sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import numpy as np
from experiments.E80_modular_transfer.io import HERE,save,sha
from experiments.E80_modular_transfer.data import load_rows
from experiments.E80_modular_transfer.backend.frozen.packing import prepare_problem,_dimensions
from experiments.E80_modular_transfer.fast_geometry import topology

# Independent copy of the old diagnostic, never import another experiment's algorithm.
old=ROOT/'experiments/E78_connectivity_posttrain/runs/diagnosis/analyze_orders.py'
def certificate(pos,edges,preplaced,dimensions):
    n=len(pos);center=pos[:,:2]+pos[:,2:]/2;worst={'overfill_local':0.,'axis':None,'path':[],'target':None};count=0
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
PATHS={'E74':'E74_collision_32k_posttrain/runs/20261029_e42_topology_seed0',
 'E79':'E79_center_refinement/runs/20261029_e78_center_seed0',
 'E80':'E80_modular_transfer/runs/20261029_e42_frozen_e79_transfer'}

def main():
    rows=load_rows('dev_monitor500');models={};identities={}
    for name,path in PATHS.items():
        source=ROOT/'experiments'/path/'terminal/results.json';identities[name]=sha(source)
        data=json.loads(source.read_text(encoding='utf-8'))['records'];items=[]
        for row,r in zip(rows,data):
            assert list(row['member'])==r['member']
            p=prepare_problem(*row['case'].solver_inputs);raw=np.asarray(r['draft_positions'])
            w,h=_dimensions(p,raw[:,2]/raw[:,3],True);prepared=np.column_stack((raw[:,:2],w,h))
            edges=topology(prepared,p.constraints,array=True);final=np.asarray(r['positions'])
            fixed=certificate(prepared,edges,p.preplaced,prepared[:,2:])
            relaxed=certificate(prepared,edges,p.preplaced,prepared[:,2:]*np.where((p.fixed|p.preplaced)[:,None],1.,.4))
            phases=Counter(t.get('phase') for t in r['trace'])
            inter=r['draft_diagnostics']['overlap_area_fraction']
            fp={k:r['metrics'].get(k) for k in ('hpwl_gap','area_gap','grouping_violations','boundary_violations','cost')}
            source_bbox=np.prod((raw[:,:2]+raw[:,2:]).max(0)-raw[:,:2].min(0))
            final_bbox=np.prod((final[:,:2]+final[:,2:]).max(0)-final[:,:2].min(0))
            items.append({'index':r['index'],'member':r['member'],'n':r['n'],'Q':r['metrics']['cost'],
              'raw_intersection':inter,'fixed_shape_certificate':fixed,'relaxed_shape_certificate':relaxed,
              'MIB_max_abs_log_size_jump':float(abs(np.log(prepared[:,2:]/raw[:,2:])).max()),
              'rescue':r['rescue_used'],'phases':dict(phases),'shape_status':dict(Counter(t.get('status') for t in r['trace'] if t.get('phase')=='shape')),
              'shape_candidates':r['shape_candidate_count'],'failed_shape_candidates':r['failed_shape_candidate_count'],
              'repair_bbox_ratio':float(final_bbox/source_bbox),'final':fp})
        models[name]={'records':items,'summary':{
          'fixed_shape_anchor_failures':sum(r['fixed_shape_certificate']['violated_anchor_bounds']>0 for r in items),
          'loose_shape_anchor_failures':sum(r['relaxed_shape_certificate']['violated_anchor_bounds']>0 for r in items),
          'MIB_jump_over1percent':sum(r['MIB_max_abs_log_size_jump']>.01 for r in items),
          'rescue_cases':sum(r['rescue'] for r in items)}}
    tails={str(i):{k:next(r for r in v['records'] if r['index']==i) for k,v in models.items()} for i in (164,443,329,219,21)}
    result={'models':models,'selected_cases':tails,'source_result_sha256':identities,'certificate_source_sha256':sha(old),
     'test_read':False,'forward_calls':0,'solver_calls':0,
     'limits':'Certificates concern selected mode0 order and fixed dimensions; not a statement that all SOCP shapes/orders are infeasible. Relaxed per-axis lower dimensions are only a loose diagnostic.'}
    save(HERE/'runs/route_diagnosis.json',result)
    print(json.dumps({'summaries':{k:v['summary'] for k,v in models.items()},'tails':tails},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
