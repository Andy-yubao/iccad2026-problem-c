"""Matched public-input E46 experiments with a shared four-hour deadline."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[key]='4'
os.environ['OMP_MAX_ACTIVE_LEVELS']='1';os.environ['E46_THREADS']='4'
import argparse
import json
import sys
import signal
import time
import numpy as np
import torch
from datasets.reference_layouts import ReferenceLayouts,REGISTRY
from .io import HERE,WEIGHTS,WEIGHT_SHA,sha,save,sources,snapshot,exclusive,task_expired
from .solver import optimize
from .model import GeometryModel
from .inference import predict
from .canonical_native import warmup
from .audit_saved import geometry
from .report import report,summarize
from scoring.evaluation import evaluate_case

GATE=[50,368,316,443,164]
STEADY=dict(relaxed=False,mode=0,adaptive=True,fast_budget=False,late_extension=True,plateau=True,robust_seeds=True,
    certified=True,quick_project=True,quadratic_project=True,incumbent_project=True,snap=1e-4)


def numerical_policy(policy,n):
    # These aliases execute exactly the same numerical algorithm. Coupling
    # their output prevents a second wall-budget draw from masquerading as
    # a causal effect of a branch that is inactive for this case.
    if policy=='shape_large_fast':return 'large_area_fast' if n>=90 else 'shape_trust'
    if policy in ('large_area','large_area_fast') and n<90:return 'control'
    if policy=='large_area' and n>=90:return 'native_area'
    return policy


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=('gate','screen','confirm'),required=True)
    parser.add_argument('--tag',required=True);parser.add_argument('--policies',nargs='+',required=True)
    parser.add_argument('--resume',action='store_true');parser.add_argument('--freeze');parser.add_argument('--wall-seconds',type=float,default=1200)
    parser.add_argument('--share-equivalent',action='store_true')
    a=parser.parse_args()
    if any(c in a.tag for c in '/\\') or a.tag in ('.','..'):raise ValueError('Tag must be a run name')
    if task_expired():raise RuntimeError('Four-hour task deadline reached')
    if 'control' not in a.policies or len(a.policies)!=len(set(a.policies)):raise ValueError('Unique paired policies with control required')
    if not 0<a.wall_seconds<=14400:raise ValueError('Bad run budget')
    data=ReferenceLayouts('dev_monitor500');order=sorted(range(500),key=lambda i:(data.records[i]['n'],i));screen=[order[i] for i in range(2,500,5)]
    panel=GATE if a.stage=='gate' else screen if a.stage=='screen' else [i for i in order if i not in screen]
    config={'stage':a.stage,'panel':panel,'members':[data.records[i]['member'] for i in panel],'policies':a.policies,'steady':STEADY,
        'threads':4,'workers':1,'seed':0,'budget_mode':'native','share_equivalent':a.share_equivalent,'pipeline_wall_seconds':a.wall_seconds,'checkpoint_sha256':sha(WEIGHTS),'registry_sha256':sha(REGISTRY),
        'timing':'one shared CPU prediction charged equally plus all per-arm actual optimization; loading,JIT,scoring,IO,diagnostics excluded and pipeline-accounted','test_read':False}
    if config['checkpoint_sha256']!=WEIGHT_SHA:raise ValueError('Frozen weights changed')
    identity={'config':config,'sources':sources()}
    if a.stage!='gate':
        if not a.freeze:raise ValueError('Explicit candidate freeze required')
        frozen=json.loads((HERE/'runs'/a.freeze).read_text(encoding='utf-8'))
        if frozen['sources']!=identity['sources'] or frozen['policies']!=a.policies or frozen['checkpoint_sha256']!=WEIGHT_SHA:raise ValueError('Frozen candidate identity changed')
        if frozen.get('share_equivalent',False)!=config['share_equivalent']:raise ValueError('Frozen comparison coupling changed')
    folder=HERE/'runs'/a.tag;folder.mkdir(exist_ok=True)
    torch.set_num_threads(4);torch.set_num_interop_threads(1);np.random.seed(0);torch.manual_seed(0)
    with exclusive(folder/'.run.lock'):
        path=folder/'results_local.json'
        if path.exists():
            if not a.resume:raise FileExistsError('Use resume')
            s=json.loads(path.read_text(encoding='utf-8'))
            if s['identity']!=identity:raise ValueError('Source/config identity changed')
            if s['finished']:print('Already complete, no solve repeated');return
            if s.get('pending_attempt'):
                s['cumulative_pipeline_seconds']+=s['pending_attempt']['reserved_seconds'];s.setdefault('interrupted_attempt_charges',[]).append(s['pending_attempt']);s['pending_attempt']=None
        else:
            if a.resume:raise FileNotFoundError('No progress')
            snapshot(folder,identity);s={'identity':identity,'records':[],'finished':False,'cumulative_pipeline_seconds':0.,'commands':[],
                'environment':{'python':sys.version,'torch':torch.__version__,'parallel':torch.__config__.parallel_info(),'interop':torch.get_num_interop_threads()},'test_read':False}
        started=time.perf_counter();prior=s['cumulative_pipeline_seconds'];s['commands'].append(sys.argv)
        def persist():s['cumulative_pipeline_seconds']=prior+time.perf_counter()-started;save(path,s)
        def stop(signum,frame):s['stop_reason']='user_interrupt';s['pending_attempt']=None;persist();raise KeyboardInterrupt
        def exhausted():return task_expired() or prior+time.perf_counter()-started>=a.wall_seconds or (HERE/'runs/STOP').exists() or (folder/'STOP').exists()
        signal.signal(signal.SIGINT,stop)
        s['pending_attempt']={'kind':'initialization','reserved_seconds':60};persist();warmup()
        checkpoint=torch.load(WEIGHTS,map_location='cpu',weights_only=False);model=GeometryModel(**checkpoint['model_config']).eval();model.load_state_dict(checkpoint['model'])
        s['pending_attempt']=None;persist()
        for ordinal,index in enumerate(panel):
            if exhausted():s['stop_reason']='budget_or_STOP';persist();report(folder);return
            r=next((r for r in s['records'] if r['index']==index),None);row=data[index];case=row['case']
            if r is None:
                draft,inference=predict(model,case.solver_inputs);r={'index':index,'member':list(row['member']),'n':case.block_count,'draft':draft.tolist(),'inference_seconds':inference,'variants':{}};s['records'].append(r)
            else:draft,inference=np.asarray(r['draft']),r['inference_seconds']
            rotated=a.policies[ordinal%len(a.policies):]+a.policies[:ordinal%len(a.policies)]
            for policy in rotated:
                if policy in r['variants']:continue
                if exhausted():s['stop_reason']='budget_or_STOP';persist();report(folder);return
                dispatch_started=time.perf_counter();effective=numerical_policy(policy,case.block_count) if a.share_equivalent else policy
                dispatch=time.perf_counter()-dispatch_started
                same=next(((name,v) for name,v in r['variants'].items() if v.get('effective_configuration')==effective),None) if a.share_equivalent else None
                if same is not None:
                    name,source=same;v=dict(source);v['metrics']=dict(source['metrics']);v['stats']=dict(source['stats'])
                    v.update(effective_configuration=effective,shared_execution_source=name,execution_dispatch_seconds=dispatch,
                        complete_seconds=inference+source['postprocess_seconds']+dispatch)
                    v['metrics']['runtime_seconds']=v['complete_seconds'];r['variants'][policy]=v;persist()
                    print(f"{ordinal+1}/{len(panel)} index={index} {policy} shared={name} Q={v['metrics']['cost']:.6f} sec={v['complete_seconds']:.4f}",flush=True)
                    continue
                s['pending_attempt']={'kind':'solve','index':index,'policy':policy,'reserved_seconds':20};persist();before=draft.copy();begin=time.perf_counter()
                pos,stats=optimize(draft,case.solver_inputs,global_policy=effective,trace_layouts=a.stage=='gate',**STEADY)
                post=time.perf_counter()-begin;complete=inference+post+dispatch
                if not np.array_equal(draft,before):raise AssertionError('Original draft mutated')
                evaluation=evaluate_case(case,pos,complete)
                r['variants'][policy]={'positions':pos.tolist(),'metrics':evaluation['metrics'],'hard_check':evaluation['independent_hard_checks'],
                    'complete_seconds':complete,'postprocess_seconds':post,'geometry':geometry(draft,pos),'stats':stats,
                    'effective_configuration':effective,'shared_execution_source':None,'execution_dispatch_seconds':dispatch}
                s['pending_attempt']=None;persist()
                print(f"{ordinal+1}/{len(panel)} index={index} {policy} Q={evaluation['metrics']['cost']:.6f} sec={complete:.4f}",flush=True)
        s['finished']=True;s['stop_reason']='panel_complete';persist();out=report(folder)
        if any(v['hard_legal']!=v['count'] for v in out['variants'].values()):raise AssertionError('Hard legality failure')
        print('COMPLETE',flush=True)


if __name__=='__main__':main()
