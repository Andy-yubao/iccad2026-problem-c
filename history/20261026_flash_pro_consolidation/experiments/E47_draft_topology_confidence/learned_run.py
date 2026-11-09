"""One frozen-best terminal500, no witness or reference data in guidance."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[key]='4'
os.environ['OMP_MAX_ACTIVE_LEVELS']='1';os.environ['E47_THREADS']='4'
import argparse,json,signal,time,sys
import numpy as np
import torch
from datasets.reference_layouts import ReferenceLayouts,REGISTRY
from datasets.paths import resolved_data_paths
from scoring.evaluation import evaluate_case
from .io import HERE,WEIGHTS,WEIGHT_SHA,sha,save,sources,snapshot,exclusive,task_expired
from .model import GeometryModel
from .head import RelationHead,predict_latent
from .learned_guidance import build
from .backend.frozen.packing import prepare_problem
from .solver import optimize
from .canonical_native import warmup
from .diagnostics import geometry
from .report import report
from .run import STEADY

POLICIES=['control','learned_uniform','learned_geometric','learned','learned_shuffle']


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);ap.add_argument('--head-tag',required=True);ap.add_argument('--resume',action='store_true');a=ap.parse_args()
    if any('/' in tag or '\\' in tag for tag in (a.tag,a.head_tag)):raise ValueError('Bad tag')
    if task_expired():raise RuntimeError('Task deadline')
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(0);np.random.seed(0)
    head_folder=HERE/'runs'/a.head_tag;trained=json.loads((head_folder/'training.json').read_text());calibration=json.loads((head_folder/'calibration.json').read_text())
    if not trained['finished']:raise ValueError('Head training/calibration incomplete')
    head_path=head_folder/'checkpoints/best.pt'
    if sha(head_path)!=calibration['head_best_sha256'] or sha(WEIGHTS)!=WEIGHT_SHA:raise ValueError('Frozen weights changed')
    data=ReferenceLayouts('dev_monitor500');panel=list(range(500));folder=HERE/'runs'/a.tag;folder.mkdir(exist_ok=True);(folder/'cases').mkdir(exist_ok=True)
    config={'stage':'learned_terminal500','policies':POLICIES,'panel':panel,'members':[list(r['member']) for r in data.records],
        'checkpoint_sha256':sha(WEIGHTS),'head_sha256':sha(head_path),'head_best_epoch':trained['state']['best_epoch'],'calibration':calibration,
        'registry_sha256':sha(REGISTRY),'resolved_data_paths':resolved_data_paths(),'threads':4,'workers':1,'seed':0,'quality_case_seconds':2.,
        'time_accounting':'v2 public preparation + CPU frozen model/latent capture + auxiliary model + guidance + optimizer; initialization/evidence IO excluded',
        'semantics':'reference-direction probability probe; not calibrated optimization-gain confidence','privileged_arms':[], 'test_read':False}
    identity={'config':config,'sources':sources()};path=folder/'results_local.json';stop={'value':False}
    signal.signal(signal.SIGINT,lambda sig,frame:stop.__setitem__('value',True))
    with exclusive(folder/'.run.lock'):
        if path.exists():
            s=json.loads(path.read_text())
            if not a.resume or s['identity']!=identity:raise ValueError('Same-identity resume required')
            if s['finished']:print('Complete; no evaluation repeated');return
            if s.get('pending_attempt'):s['cumulative_pipeline_seconds']+=s['pending_attempt']['reserved_seconds'];s['pending_attempt']=None
        elif a.resume:raise FileNotFoundError(path)
        else:
            snapshot(folder,identity);s={'identity':identity,'case_files':[f'cases/{i:03d}.json' for i in panel],'records':[],
                'cumulative_pipeline_seconds':0.,'finished':False,'commands':[],'test_read':False}
        prior=s['cumulative_pipeline_seconds'];started=time.perf_counter();s['commands'].append(sys.argv)
        def persist():s['cumulative_pipeline_seconds']=prior+time.perf_counter()-started;save(path,s)
        def stopping():return stop['value'] or task_expired() or (HERE/'runs/STOP').exists() or (folder/'STOP').exists()
        warmup();checkpoint=torch.load(WEIGHTS,map_location='cpu',weights_only=False);model=GeometryModel(**checkpoint['model_config']).eval();model.load_state_dict(checkpoint['model'])
        head=RelationHead().eval();head.load_state_dict(torch.load(head_path,map_location='cpu',weights_only=False)['model'])
        for ordinal,index in enumerate(panel):
            casepath=folder/f'cases/{index:03d}.json';r=json.loads(casepath.read_text()) if casepath.exists() else None
            if r is not None and set(POLICIES)<=set(r['variants']):continue
            if stopping():s['stop_reason']='budget_or_STOP';persist();report(folder);return
            row=data[index];case=row['case'];begin=time.perf_counter();p=prepare_problem(*case.solver_inputs);frontend=time.perf_counter()-begin
            if r is None:
                draft,features,inference=predict_latent(model,case.solver_inputs)
                # Check raw geometry against earlier frozen-draft runs by hash later.
                begin=time.perf_counter()
                with torch.no_grad():probabilities=torch.sigmoid(head(features)/calibration['temperature'])[0].numpy()
                head_seconds=time.perf_counter()-begin;np.savez_compressed(folder/f'cases/{index:03d}_probabilities.npz',probabilities=probabilities)
                r={'index':index,'n':case.block_count,'member':list(row['member']),'draft':draft.tolist(),'public_prepare_seconds':frontend,
                    'inference_seconds':inference,'head_seconds':head_seconds,'probabilities_file':f'cases/{index:03d}_probabilities.npz','variants':{}}
                save(casepath,r)
            else:
                draft=np.array(r['draft']);inference=r['inference_seconds'];head_seconds=r['head_seconds']
                probabilities=np.load(folder/r['probabilities_file'])['probabilities']
            order=POLICIES[ordinal%len(POLICIES):]+POLICIES[:ordinal%len(POLICIES)]
            for policy in order:
                if policy in r['variants']:continue
                if stopping():s['stop_reason']='budget_or_STOP';persist();report(folder);return
                s['pending_attempt']={'index':index,'policy':policy,'reserved_seconds':10};persist();before=draft.copy()
                begin=time.perf_counter();guidance=None if policy=='control' else build(draft,p,probabilities,policy,20261024+index)
                guidance_seconds=time.perf_counter()-begin;prefix=r['public_prepare_seconds']+inference+(0. if policy=='control' else head_seconds)
                settings=STEADY.copy();settings.update(adaptive=False,iterations=64,seconds=max(.01,2-prefix-guidance_seconds-.05),reserve=.04,late_extension=False)
                positions,stats=optimize(draft,case.solver_inputs,guidance=guidance,**settings);post=time.perf_counter()-begin;complete=prefix+post
                assert np.array_equal(before,draft),'Draft mutated'
                evaluation=evaluate_case(case,positions,complete)
                r['variants'][policy]={'positions':positions.tolist(),'metrics':evaluation['metrics'],'hard_check':evaluation['independent_hard_checks'],
                    'complete_seconds':complete,'postprocess_seconds':post,'guidance_seconds':guidance_seconds,'head_seconds_charged':0. if policy=='control' else head_seconds,
                    'geometry':geometry(draft,positions),'stats':stats,'privileged_diagnostic':False}
                save(casepath,r);s['pending_attempt']=None;persist()
            if (ordinal+1)%10==0:print(f"TERMINAL {ordinal+1}/500",flush=True)
        s['finished']=True;s['stop_reason']='panel_complete';persist();out=report(folder)
        assert all(v['hard_legal']==500 for v in out['variants'].values());print('LEARNED TERMINAL COMPLETE',flush=True)


if __name__=='__main__':main()
