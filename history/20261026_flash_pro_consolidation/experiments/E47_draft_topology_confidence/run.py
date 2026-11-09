"""Same-draft confidence mechanism gates. Oracle arms are diagnostic only."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[k]='4'
os.environ['OMP_MAX_ACTIVE_LEVELS']='1';os.environ['E47_THREADS']='4'
import argparse,json,sys,time,signal,platform
import numpy as np
import torch
from datasets.reference_layouts import ReferenceLayouts,REGISTRY
from datasets.paths import resolved_data_paths
from scoring.evaluation import evaluate_case
from .io import HERE,WEIGHTS,WEIGHT_SHA,sha,save,sources,snapshot,exclusive,task_expired
from .model import GeometryModel
from .inference import predict
from .confidence import Guidance
from .backend.frozen.packing import prepare_problem
from .solver import optimize
from .canonical_native import warmup
from .report import report
from .diagnostics import geometry
from .witnesses import load_witnesses

GATE=[50,368,316,443,164]
STEADY=dict(relaxed=False,mode=0,adaptive=True,fast_budget=False,late_extension=True,plateau=True,robust_seeds=True,certified=True,quick_project=True,quadratic_project=True,incumbent_project=True,snap=1e-4)
REFERENCE={'reference_full':'oracle_full','reference_hybrid':'oracle_hybrid','reference_shuffle':'shuffle_hybrid'}
BASELINE={'baseline_hybrid':'oracle_hybrid','baseline_full':'oracle_full','baseline_shuffle':'shuffle_hybrid'}
SHAPE={'oracle_shape_only','oracle_hybrid_shape'}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);ap.add_argument('--stage',choices=('gate','screen','confirm'),default='gate')
    ap.add_argument('--policies',nargs='+',default=['control','uniform','geometric','oracle','shuffle','oracle_metric','oracle_contact'])
    ap.add_argument('--indices',type=int,nargs='+',help='At most five already-declared development diagnostic cases; gate only')
    ap.add_argument('--quality-seconds',type=float,choices=(2.,),help='Registered two-second quality diagnostic; measured overruns remain failures')
    ap.add_argument('--resume',action='store_true');ap.add_argument('--wall-seconds',type=float,default=900);a=ap.parse_args()
    if '/' in a.tag or '\\' in a.tag or a.tag in ('.','..'):raise ValueError('Bad tag')
    if task_expired():raise RuntimeError('Task deadline')
    if len(set(a.policies))!=len(a.policies) or 'control' not in a.policies:raise ValueError('Unique policies with control')
    torch.set_num_threads(4);torch.set_num_interop_threads(1);np.random.seed(0);torch.manual_seed(0)
    data=ReferenceLayouts('dev_monitor500');order=sorted(range(500),key=lambda i:(data.records[i]['n'],i));screen=[order[i] for i in range(2,500,5)]
    panel=GATE if a.stage=='gate' else screen if a.stage=='screen' else [i for i in order if i not in set(screen)]
    if a.indices is not None:
        if a.stage!='gate' or not 0<len(a.indices)<=5 or len(set(a.indices))!=len(a.indices) or any(not 0<=i<500 for i in a.indices):raise ValueError('Bad diagnostic indices')
        panel=a.indices
    witnesses=load_witnesses()
    baseline_witnesses=load_witnesses('control') if any(p in BASELINE for p in a.policies) else {}
    config={'stage':a.stage,'policies':a.policies,'panel':panel,'members':[data.records[i]['member'] for i in panel],'steady':STEADY,
        'threads':4,'workers':1,'budget_mode':'native' if a.quality_seconds is None else 'soft_quality_2s_with_measured_overruns',
        'quality_case_seconds':a.quality_seconds,'checkpoint_sha256':sha(WEIGHTS),'registry_sha256':sha(REGISTRY),'wall_seconds':a.wall_seconds,
        'seed':0,'guidance_seed':'20261024+dev_index','resolved_data_paths':resolved_data_paths(),
        'time_accounting':'v2: public problem preparation + CPU features/inference + annotation conversion/guidance + optimizer; offline witness generation/loading excluded; no learned head exists',
        'annotation_sources':{p:('original v5 reference; directional diagnostics, not assumed MIB-legal' if p in REFERENCE else 'single frozen baseline output; no Q selection' if p in BASELINE else 'saved hard-legal development witness' if p not in ('control','uniform','geometric','geometric_commit','draft_static') else 'public inputs only') for p in a.policies},
        'auxiliary_shapes':{p:'privileged witness aspect ratios only; public areas/specifications and original draft lower-left coordinates retained by the same initial-state rule in every arm' for p in a.policies if p in SHAPE},
        'python':sys.version,'torch':torch.__version__,'platform':platform.platform(),
        'privileged_arms':[p for p in a.policies if p not in ('control','uniform','geometric','geometric_commit','draft_static')],
        'oracle_scope':'saved dev witness chosen before current run; diagnostics only, no deployment or upper-bound claim',
        'witnesses':{str(i):{k:v for k,v in witnesses[i].items() if k!='positions'} for i in panel},'test_read':False}
    config['baseline_witnesses']={str(i):{k:v for k,v in baseline_witnesses[i].items() if k!='positions'} for i in panel} if baseline_witnesses else {}
    if config['checkpoint_sha256']!=WEIGHT_SHA:raise ValueError('Weights changed')
    identity={'config':config,'sources':sources()};folder=HERE/'runs'/a.tag;folder.mkdir(exist_ok=True)
    with exclusive(folder/'.run.lock'):
        path=folder/'results_local.json'
        if path.exists():
            if not a.resume:raise FileExistsError('Use resume')
            s=json.loads(path.read_text(encoding='utf-8'))
            if s['identity']!=identity:raise ValueError('Identity changed')
            if s['finished']:print('Complete, no evaluation repeated');return
            if s.get('pending_attempt'):
                s['cumulative_pipeline_seconds']+=s['pending_attempt']['reserved_seconds'];s.setdefault('interrupted_attempt_charges',[]).append(s['pending_attempt']);s['pending_attempt']=None
        else:
            if a.resume:raise FileNotFoundError('No run')
            snapshot(folder,identity);s={'identity':identity,'records':[],'finished':False,'cumulative_pipeline_seconds':0.,'commands':[],'test_read':False,'environment':{'parallel':torch.__config__.parallel_info(),'interop':torch.get_num_interop_threads()}}
        prior=s['cumulative_pipeline_seconds'];started=time.perf_counter();s['commands'].append(sys.argv)
        def persist():s['cumulative_pipeline_seconds']=prior+time.perf_counter()-started;save(path,s)
        def exhausted():return task_expired() or prior+time.perf_counter()-started>=a.wall_seconds or (HERE/'runs/STOP').exists() or (folder/'STOP').exists()
        def interrupt(sig,frame):s['pending_attempt']=None;s['stop_reason']='user_interrupt';persist();raise KeyboardInterrupt
        signal.signal(signal.SIGINT,interrupt);s['pending_attempt']={'kind':'initialize','reserved_seconds':60};persist();warmup()
        checkpoint=torch.load(WEIGHTS,map_location='cpu',weights_only=False);model=GeometryModel(**checkpoint['model_config']).eval();model.load_state_dict(checkpoint['model']);s['pending_attempt']=None;persist()
        for ordinal,index in enumerate(panel):
            if exhausted():s['stop_reason']='budget_or_STOP';persist();report(folder);return
            row=data[index];case=row['case'];public_start=time.perf_counter();p=prepare_problem(*case.solver_inputs)
            public_prepare=time.perf_counter()-public_start;witness=witnesses[index]
            if list(row['member'])!=list(witness['member']):raise ValueError('Witness mismatch')
            r=next((r for r in s['records'] if r['index']==index),None)
            if r is None:
                draft,inference=predict(model,case.solver_inputs);r={'index':index,'n':case.block_count,'member':list(row['member']),'draft':draft.tolist(),'inference_seconds':inference,'public_prepare_seconds':public_prepare,'variants':{}};s['records'].append(r)
            else:draft,inference=np.array(r['draft']),r['inference_seconds']
            policies=a.policies[ordinal%len(a.policies):]+a.policies[:ordinal%len(a.policies)]
            for policy in policies:
                if policy in r['variants']:continue
                if exhausted():s['stop_reason']='budget_or_STOP';persist();report(folder);return
                s['pending_attempt']={'kind':'solve','index':index,'policy':policy,'reserved_seconds':20};persist()
                before=draft.copy();begin=time.perf_counter()
                annotation=np.asarray(row['positions'],float) if policy in REFERENCE else baseline_witnesses[index]['positions'] if policy in BASELINE else witness['positions'] if policy not in ('uniform','geometric','geometric_commit','draft_static') else None
                kind='oracle_hybrid' if policy=='oracle_hybrid_shape' else REFERENCE.get(policy,BASELINE.get(policy,policy))
                guidance=None if policy in ('control','oracle_shape_only') else Guidance.build(draft,p,kind,annotation,seed=20261024+index)
                guidance_seconds=time.perf_counter()-begin
                extra={'initial_shape_ratios':witness['positions'][:,2]/witness['positions'][:,3]} if policy in SHAPE else {}
                settings=STEADY.copy()
                if a.quality_seconds is not None:
                    settings.update(adaptive=False,iterations=64,seconds=max(.01,a.quality_seconds-r['public_prepare_seconds']-inference-guidance_seconds-.05),reserve=.04,late_extension=False)
                positions,stats=optimize(draft,case.solver_inputs,guidance=guidance,**extra,**settings)
                post=time.perf_counter()-begin;complete=r['public_prepare_seconds']+inference+post
                if not np.array_equal(before,draft):raise AssertionError('Draft mutated')
                evaluation=evaluate_case(case,positions,complete)
                r['variants'][policy]={'positions':positions.tolist(),'metrics':evaluation['metrics'],'hard_check':evaluation['independent_hard_checks'],
                    'complete_seconds':complete,'postprocess_seconds':post,'guidance_seconds':guidance_seconds,'geometry':geometry(draft,positions),'stats':stats,
                    'main_optimizer_seconds_requested':settings.get('seconds','native adaptive'),
                    'privileged_diagnostic':policy not in ('control','uniform','geometric','geometric_commit','draft_static')}
                s['pending_attempt']=None;persist();print(f"{ordinal+1}/{len(panel)} {index} {policy} Q={evaluation['metrics']['cost']:.6f} sec={complete:.4f}",flush=True)
        s['finished']=True;s['stop_reason']='panel_complete';persist();out=report(folder)
        assert all(v['count']==v['hard_legal'] for v in out['variants'].values());print('COMPLETE',flush=True)


if __name__=='__main__':main()
