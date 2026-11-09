"""Saved-result attribution, fixed-shape chain bounds and standard galleries.

No optimizer, inference or training calls. Only five already-exposed public
development inputs are reloaded for constraint identities and diagnostics.
"""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='4'
import csv,json,platform,sys
from datetime import datetime,timezone
import numpy as np
import torch,scipy,clarabel
from datasets.reference_layouts import ReferenceLayouts
from visualization.layouts import plot_case,geometry_details
from .io import HERE,ROOT,save,sha,WEIGHTS
from .confidence import Guidance
from .fast_geometry import topology,reduce_edges
from .backend.frozen.packing import prepare_problem,_dimensions
from .backend.frozen.feasible import Objective
from .witnesses import load_witnesses

TAGS=['20261024_information_gate5','20261024_commit_gate5','20261024_commit_screen100',
      '20261024_global_information_gate5','20261024_global_information_screen100',
      '20261024_reference_information_gate5','20261024_label_sources_screen100',
      '20261024_global_information_confirm400','20261024_timing_accounting_gate5','20261024_shape_structure_gate5','20261024_shape_structure_gate5_v2',
      '20261024_shape_structure_quality2s_gate5','20261024_quality2s_label_sources_screen100']


def chain_bound(edges,wh):
    """Necessary bbox span bound for fixed WH, ignoring extra anchor limits."""
    n=len(wh);bounds=[]
    for axis in (0,1):
        adj=[[] for _ in range(n)];degree=np.zeros(n,int);dist=np.zeros(n)
        for a,b,_ in edges[edges[:,2]==axis]:adj[a].append(b);degree[b]+=1
        todo=list(np.flatnonzero(degree==0));visited=0
        while todo:
            a=todo.pop();visited+=1
            for b in adj[a]:
                dist[b]=max(dist[b],dist[a]+wh[a,axis]);degree[b]-=1
                if degree[b]==0:todo.append(b)
        if visited!=n:raise ValueError('Cyclic relation graph')
        bounds.append(float((dist+wh[:,axis]).max()))
    return np.array(bounds)


def bbox(x):return np.ptp(np.r_[x[:,:2],x[:,:2]+x[:,2:]],axis=0)


def main():
    folder=HERE/'runs/research_summary';folder.mkdir(exist_ok=True)
    all_runs={tag:json.loads((HERE/'runs'/tag/'results_local.json').read_text(encoding='utf-8')) for tag in TAGS}
    assert all(s['finished'] for tag,s in all_runs.items() if tag!='20261024_shape_structure_gate5')
    drafts={};total=0;hard=0;run_evidence=[]
    for tag,s in all_runs.items():
        count=0
        for r in s['records']:
            x=np.array(r['draft'])
            if r['index'] in drafts:assert np.array_equal(drafts[r['index']],x),'Original draft changed across arms/runs'
            else:drafts[r['index']]=x
            for v in r['variants'].values():
                count+=1;total+=1;hard+=int(v['metrics']['is_feasible'] and v['hard_check']['is_feasible'])
        run_evidence.append({'tag':tag,'cases':len(s['records']),'outputs':count,'pipeline_seconds':s['cumulative_pipeline_seconds'],
            'results_sha256':sha(HERE/'runs'/tag/'results_local.json'),'commands':s['commands'],
            'method_status':'center-preservation confound; excluded from causal interpretation' if tag=='20261024_shape_structure_gate5' else 'recorded experiment'})
    screen=all_runs['20261024_global_information_screen100'];confirm=all_runs['20261024_global_information_confirm400']
    frozen_names=('confidence.py','global_information.py','solver.py','fast_geometry.py','projection.py','model.py','features.py','inference.py')
    same_core={name:screen['identity']['sources'][f'experiments/E47_draft_topology_confidence/{name}']==confirm['identity']['sources'][f'experiments/E47_draft_topology_confidence/{name}'] for name in frozen_names}
    assert all(same_core.values())
    assert screen['identity']['config']['checkpoint_sha256']==confirm['identity']['config']['checkpoint_sha256']==sha(WEIGHTS)
    assert screen['identity']['config']['steady']==confirm['identity']['config']['steady']
    assert not set(screen['identity']['config']['panel'])&set(confirm['identity']['config']['panel'])
    combined=screen['records']+confirm['records'];weights=np.exp((np.array([r['n'] for r in combined])-120)/12);decomposition={}
    for policy in ('control','oracle_hybrid'):
        values=[r['variants'][policy]['metrics'] for r in combined]
        h=.5*np.maximum(0,[v['hpwl_gap'] for v in values]);a=.5*np.maximum(0,[v['area_gap'] for v in values])
        soft=np.expm1(2*np.array([v['violations_relative'] for v in values]));interaction=(h+a)*soft
        error=float(np.max(abs(np.array([v['cost']-1 for v in values])-h-a-soft-interaction)))
        assert error<1e-6,'Unexpected score cap or changed decomposition'
        decomposition[policy]={key:float(np.average(value,weights=weights)) for key,value in [('positive_hpwl_half',h),('positive_area_half',a),('soft_exponential',soft),('geometry_soft_interaction',interaction)]}
        decomposition[policy]['max_identity_error']=error
    provenance=json.loads((HERE/'provenance.json').read_text());unchanged={k:sha(ROOT/v['source'])==v['sha256'] for k,v in provenance.items()}
    assert all(unchanged.values()),'An original source experiment changed'
    assert np.allclose(chain_bound(np.array([[0,1,0],[1,2,0]]),np.ones((3,2))),[3,1])
    witnesses=load_witnesses();baseline=load_witnesses('control');data=ReferenceLayouts('dev_monitor500');source_quality={}
    for name,source in [('best_saved_legal',witnesses),('single_historical_baseline',baseline)]:
        differences=np.array([source[r['index']]['Q']-r['variants']['control']['metrics']['cost'] for r in combined])
        source_quality[name]={'better_tie_worse_vs_fresh_control':[int((differences<-1e-6).sum()),int((abs(differences)<=1e-6).sum()),int((differences>1e-6).sum())],
            'weighted_Q_difference':float(np.average(differences,weights=weights)),'worst_Q_difference':float(differences.max()),
            'interpretation':'historical output vs current paired control; not a teacher/student gain or a timing repeat guarantee'}
    worst_confirm=max(confirm['records'],key=lambda r:r['variants']['oracle_hybrid']['metrics']['cost']-r['variants']['control']['metrics']['cost'])['index']
    focus=[443,247,259,368,worst_confirm];public={};diagnostics=[]
    for index in focus:
        row=data[index];p=prepare_problem(*row['case'].solver_inputs);public[index]=np.array(p.constraints)
        draft=drafts[index];raw=topology(draft,p.constraints,array=True);reduced=np.asarray(reduce_edges(raw,p.n),int)
        widths,heights=_dimensions(p,draft[:,2]/draft[:,3],True);initial_wh=np.column_stack((widths,heights))
        for source,w in [('best_saved_legal',witnesses[index]),('single_baseline',baseline[index])]:
            g=Guidance.build(draft,p,'oracle_hybrid',w['positions']);edges=g.static_edges
            changed=np.any(edges!=raw,1);a,b,axis=raw.T;inter=(p.constraints[a,3]!=p.constraints[b,3])|(p.constraints[a,3]==0)
            ra,rb,rx=reduced.T;invalid=w['positions'][rb,rx]-w['positions'][ra,rx]-w['positions'][ra,rx+2]<-1e-7
            bound=chain_bound(edges,initial_wh);wbound=chain_bound(edges,w['positions'][:,2:]);obj=Objective(p)
            diagnostics.append({'index':index,'n':p.n,'source':source,'witness_Q_diagnostic':w['Q'],'witness_policy':w['policy'],
                'original_relations':len(raw),'changed_relations':int(changed.sum()),'changed_between_groups':int((changed&inter).sum()),
                'changed_within_groups':int((changed&~inter).sum()),'raw_reduced_relations':len(reduced),'raw_reduced_incompatible_with_witness':int(invalid.sum()),
                'initial_shape_chain_area_bound_over_draft_bbox':float(bound.prod()/bbox(draft).prod()),
                'initial_shape_chain_area_bound_over_witness_bbox':float(bound.prod()/bbox(w['positions']).prod()),
                'witness_shape_chain_area_bound_over_witness_bbox':float(wbound.prod()/bbox(w['positions']).prod()),
                'mean_abs_log_shape_change':float(np.mean(abs(np.log(initial_wh[:,0]/initial_wh[:,1])-np.log(w['positions'][:,2]/w['positions'][:,3])))),
                'public_native_objective_witness':float(obj(w['positions']))})
    with (folder/'shape_structure_diagnostics.csv').open('w',encoding='utf-8-sig',newline='') as h:
        writer=csv.DictWriter(h,fieldnames=list(diagnostics[0]));writer.writeheader();writer.writerows(diagnostics)
    selections=[(443,'20261024_global_information_gate5',[('draft','Full model draft'),('control','Original postprocessing'),('oracle_hybrid','Global hybrid (privileged)')]),
        (247,'20261024_label_sources_screen100',[('control','Original postprocessing'),('baseline_hybrid','Single baseline structure'),('oracle_hybrid','Better witness structure')]),
        (259,'20261024_commit_screen100',[('control','Original postprocessing'),('oracle_commit','Group contact (privileged)'),('special_global','Global hybrid + contacts')]),
        (368,'20261024_reference_information_gate5',[('draft','Full model draft'),('control','Original postprocessing'),('reference_hybrid','Reference hybrid (privileged)')]),
        (worst_confirm,'20261024_global_information_confirm400',[('draft','Full model draft'),('control','Original postprocessing'),('oracle_hybrid','Frozen hybrid regression')]),
        (worst_confirm,'20261024_shape_structure_gate5_v2',[('oracle_hybrid','Relations only'),('oracle_shape_only','Aspect ratios only'),('oracle_hybrid_shape','Relations + aspect ratios')])]
    images=[];(folder/'images').mkdir(exist_ok=True)
    for index,tag,stages in selections:
        record=next(r for r in all_runs[tag]['records'] if r['index']==index);variants={}
        for key,label in stages:
            if key=='draft':v={'positions':record['draft']}
            elif key=='special_global':v=next(r for r in all_runs['20261024_global_information_screen100']['records'] if r['index']==index)['variants']['oracle_hybrid_contacts']
            else:v=record['variants'][key]
            variants[key]={**v,'soft_details':geometry_details(v['positions'],public[index])}
        suffix='_shape_structure' if tag=='20261024_shape_structure_gate5_v2' else ''
        path=folder/'images'/f'case_{index}{suffix}.png'
        layout=plot_case({'panel':'Development; witness arms are diagnostic only','case_id':f'E47 index {index}',
            'constraints':public[index].tolist(),'variants':variants},path,stages=stages)
        images.append({'index':index,'path':str(path.relative_to(HERE)),'stages':stages,'source_tag':tag,'layout':layout})
    shape_results=[]
    for r in all_runs['20261024_shape_structure_gate5_v2']['records']:
        q={p:v['metrics']['cost'] for p,v in r['variants'].items()}
        shape_results.append({'index':r['index'],'Q':q,'aspect_relation_interaction_Q':q['oracle_hybrid_shape']-q['oracle_hybrid']-q['oracle_shape_only']+q['control'],
            'source_tag':'20261024_shape_structure_gate5_v2','privileged_mechanism_diagnostic_only':True})
    quality_results=[]
    for r in all_runs['20261024_shape_structure_quality2s_gate5']['records']:
        quality_results.append({'index':r['index'],'Q':{p:v['metrics']['cost'] for p,v in r['variants'].items()},
            'complete_seconds_v2':{p:v['complete_seconds'] for p,v in r['variants'].items()},'privileged_mechanism_diagnostic_only':True})
    noncausal=sum(len(r['variants']) for r in all_runs['20261024_shape_structure_gate5']['records'])
    summary={'total_solver_outputs':total,'hard_legal_outputs':hard,'unique_development_cases':len(drafts),'all_full_drafts_bitwise_unchanged':True,
        'noncausal_diagnostic_outputs':noncausal,'historical_source_quality':source_quality,'shape_factorial_results':shape_results,'quality2s_results':quality_results,
        'frozen_global_core_confirmed':same_core,'original_source_files_unchanged':unchanged,'runs':run_evidence,
        'focus_diagnostics':diagnostics,'images':images,'weighted_Q_excess_decomposition':decomposition,'worst_confirmation_case':worst_confirm,'gallery_selection':'four mechanism/source cases plus the largest positive Q difference in frozen400; illustrations, not tuning',
        'research_solver_outputs':total-5-noncausal,'timing_verification_outputs':5,'new_solver_calls_in_audit':0,'trained_models':0,'train_teacher_labels_generated':0,
        'sealed_test_read':False,'environment':{'python':sys.version,'torch':torch.__version__,'scipy':scipy.__version__,
        'clarabel':clarabel.__version__,'platform':platform.platform()},'completed_utc':datetime.now(timezone.utc).isoformat(),
        'limitations':['All Q values are neutral-runtime local MIB-derived development diagnostics.',
            'Auxiliary witness structures are privileged annotations, not model predictions or upper bounds.',
            'Original research latency is v1 recorded solve-stage; shared input preparation and offline witness cost excluded. The five-case v2 accounting repair does not retroactively alter old records.',
            'Fixed-shape chain area is only a lower bound under that shape and graph; not a Q bound or a bound with variable shapes.']}
    save(folder/'evidence.json',summary)
    print(f'Audit complete: {total} outputs, {hard} hard legal; {len(drafts)} unique dev cases; {len(images)} galleries.')
    for r in diagnostics:print(r)


if __name__=='__main__':main()
