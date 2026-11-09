"""Train relation decisions from fixed input-only replay; no online postprocessor."""
from .resources import configure
RESOURCE=configure()
import argparse
import copy
import datetime as dt
import json
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
import numpy as np
import torch
from datasets.reference_layouts import ReferenceLayouts,REGISTRY
from datasets.paths import resolved_data_paths
from .data import load_rows,collate
from .batches import inputs_only
from .relations import RelationPolicy,objective
from .prepare import load_states,FOLDER
from .schedule import PlateauController,LR_LEVELS
from .io import HERE,ROOT,sha,save,checkpoint,sources,snapshot,exclusive
from .cases import WEIGHTS

def rng_state():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()}

def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch']);torch.cuda.set_rng_state_all(state['cuda'])

def batch(rows,states,device,slots):
    g,_=collate(rows,device);n=g['valid'].shape[1]
    draft=torch.zeros(len(rows),n,3);current=torch.zeros_like(draft)
    for k,(row,slot) in enumerate(zip(rows,slots)):
        x=states[tuple(row['member'])];draft[k,:len(x[0])]=x[0];current[k,:len(x[0])]=x[int(slot)]
    return g,draft.to(device),current.to(device)

@torch.no_grad()
def monitor(model,rows,states,device,stopped=lambda:False):
    model.eval();totals={};percase=[];modules=[]
    for start in range(0,len(rows),32):
        if stopped():return None
        selected=rows[start:start+32];local={}
        for slot in range(3):
            g,draft,current=batch(selected,states,device,[slot]*len(selected))
            values=objective(model(inputs_only(g),draft,current),g)
            for k,v in values.items():local[k]=local.get(k,0)+v.detach().cpu().numpy()/3
        for k,v in local.items():totals[k]=totals.get(k,0.)+float(v.sum())
        percase.extend(local['per_case'].tolist());modules.extend([r['graph']['n'] for r in selected])
    a=np.asarray(percase);n=np.asarray(modules);w=np.exp((n-n.max())/12);tail=float(np.sort(a)[-max(1,int(np.ceil(.05*len(a)))):].mean())
    out={k:v/len(rows) for k,v in totals.items()};out.update(count=len(rows),weighted_relation_proxy=float(np.average(a,weights=w)),cvar95_relation_proxy=tail,
        selection_proxy=.5*float(np.average(a,weights=w))+.25*float(a.mean())+.25*tail,postprocessing_calls=0)
    return out

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--arm',choices=('replay','draft_only'),default='replay')
    ap.add_argument('--seed',type=int,default=0);ap.add_argument('--max-hours',type=float,default=2.);ap.add_argument('--max-epochs',type=int,default=200)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--resume',action='store_true');args=ap.parse_args()
    started=time.perf_counter();deadline=dt.datetime.fromisoformat('2026-10-22T19:39:00+08:00').timestamp();folder=args.output.resolve()
    if not folder.is_relative_to(HERE/'runs'):raise ValueError('Output must remain in E32/runs')
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed);torch.use_deterministic_algorithms(True)
    with exclusive(folder.with_suffix('.lock')):
        identity={'revision':'E32.relation_replay.v1','sources':sources(),'registry_sha256':sha(REGISTRY),'replay_identity_sha256':sha(FOLDER/'identity.json'),
            'replay_manifests':{k:sha(FOLDER/k/'manifest.json') for k in ('train','dev')},'arm':args.arm,'seed':args.seed,'width':64,'batch_size':32,
            'max_seconds':args.max_hours*3600,'max_epochs':args.max_epochs,'lr_levels':list(LR_LEVELS),'device':'cuda',
            'selection':'.5 module-weighted + .25 mean + .25 CVaR95 of mean-over-three-states axis_NLL+.5 balanced_contact_BCE',
            'base_checkpoint_sha256':sha(WEIGHTS),'resources':{k:v for k,v in RESOURCE.items() if k!='pid'},'torch':str(torch.__version__),'test_read':False}
        previous=None;prior=0.
        if args.resume:
            previous=torch.load(folder/'checkpoints/latest.pt',map_location='cpu',weights_only=False)
            if previous['identity']!=identity:raise ValueError('Resume identity changed')
            if previous['finished']:print('Already complete:',previous['status'],flush=True);return
            prior=previous['elapsed_seconds']
            if (folder/'budget.json').exists():prior=max(prior,json.loads((folder/'budget.json').read_text())['elapsed_seconds'])
        else:
            folder.mkdir(parents=True,exist_ok=False)
            save(folder/'config.json',{'identity':identity,'command':[sys.executable,*sys.argv],'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                'git_status':subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True).strip(),'data_paths':resolved_data_paths(),'training_postprocessing':False,'validation_postprocessing':False})
            snapshot(folder/'source_snapshot.zip',identity)
        elapsed=lambda:prior+time.perf_counter()-started
        model=RelationPolicy().cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4);scheduler=PlateauController(optimizer)
        train=load_rows('train');dev=load_rows('dev_monitor500');states_train,_=load_states('train');states_dev,_=load_states('dev')
        members={tuple(r['member']):k for k,r in enumerate(train)};monitor_ids=[members[tuple(r['member'])] for r in ReferenceLayouts('train_monitor500').records]
        progress={'epoch':0,'step':0,'cursor':0,'order':None,'seen':np.zeros(len(train),dtype=bool),'pending_monitor':True,'best':None,'best_epoch':None,'history':[],'batches':[]}
        if previous:
            model.load_state_dict(previous['model']);optimizer.load_state_dict(previous['optimizer']);scheduler.load_state_dict(previous['scheduler']);progress=previous['progress'];restore_rng(previous['rng'])
        requested=[False];old=signal.signal(signal.SIGINT,lambda *_:requested.__setitem__(0,True))
        stopped=lambda:requested[0] or (folder/'STOP').exists()
        exhausted=lambda:elapsed()>=identity['max_seconds'] or time.time()>=deadline
        def store(status,finished=False,best=False):
            save(folder/'budget.json',{'elapsed_seconds':elapsed()})
            value={'model':model.state_dict(),'model_config':{'width':64},'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'rng':rng_state(),
                'progress':progress,'identity':identity,'elapsed_seconds':elapsed(),'status':status,'finished':finished}
            checkpoint(folder/'checkpoints/latest.pt',value)
            if best:checkpoint(folder/'checkpoints/best.pt',value)
            save(folder/'progress.json',{'status':status,'finished':finished,'epoch':progress['epoch'],'step':progress['step'],'cursor':progress['cursor'],
                'best_epoch':progress['best_epoch'],'best_proxy':progress['best'],'seen_current_phase':int(progress['seen'].sum()),'base_pretraining_seen_all_train':True,
                'elapsed_seconds':elapsed(),'lr_drops':scheduler.drops,'lowest_lr_completed_epochs':scheduler.lowest_completed_epochs,'history':progress['history'],'batches':progress['batches'],'test_read':False,'postprocessing_calls':0})
        status='running'
        try:
            if not previous:store('prepared')
            while progress['epoch']<args.max_epochs or progress['pending_monitor']:
                if stopped() or exhausted():status='user_stop' if stopped() else 'budget';break
                if progress['pending_monitor']:
                    val=monitor(model,dev,states_dev,'cuda',lambda:stopped() or exhausted())
                    tr=monitor(model,[train[i] for i in monitor_ids],states_train,'cuda',lambda:stopped() or exhausted()) if val else None
                    if val is None or tr is None:status='user_stop' if stopped() else 'budget';break
                    improved=progress['best'] is None or val['selection_proxy']<progress['best']-1e-5
                    if improved:progress['best']=val['selection_proxy'];progress['best_epoch']=progress['epoch']
                    scheduler.step(val['selection_proxy'],progress['epoch']);seen=progress['seen'][monitor_ids]
                    progress['history'].append({'epoch':progress['epoch'],'step':progress['step'],'dev_monitor':val,'train_monitor':tr,'lr_next':optimizer.param_groups[0]['lr'],
                        'lr_drops':scheduler.drops,'current_phase_train_monitor_seen':int(seen.sum()),'base_pretraining_train_monitor_seen':500,'elapsed_seconds':elapsed()})
                    progress['pending_monitor']=False;store('running',best=improved)
                    print('epoch',progress['epoch'],'proxy',round(val['selection_proxy'],6),'axis',round(val['axis_accuracy'],4),'lr',optimizer.param_groups[0]['lr'],flush=True)
                    if scheduler.should_stop():status='early_stop';break
                    continue
                if progress['order'] is None:progress['order']=np.random.default_rng(args.seed+progress['epoch']).permutation(len(train));progress['cursor']=0
                model.train()
                while progress['cursor']<len(train):
                    if stopped() or exhausted():break
                    ids=progress['order'][progress['cursor']:progress['cursor']+32];rows=[train[i] for i in ids]
                    slots=np.random.randint(0,3,len(rows)) if args.arm=='replay' else np.zeros(len(rows),dtype=int)
                    g,draft,current=batch(rows,states_train,'cuda',slots);optimizer.zero_grad(set_to_none=True)
                    values=objective(model(inputs_only(g),draft,current),g);loss=values['per_case'].mean()
                    if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
                    loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True));optimizer.step()
                    progress['seen'][ids]=True;progress['cursor']+=len(ids);progress['step']+=1
                    if progress['step']%100==0:
                        progress['batches'].append({'step':progress['step'],'epoch':progress['epoch']+1,'loss':float(loss.detach()),'gradient_norm':norm,'seconds':elapsed()});store('running')
                        print('step',progress['step'],'samples',progress['cursor'],'loss',round(float(loss.detach()),5),flush=True)
                if progress['cursor']<len(train):status='user_stop' if stopped() else 'budget';break
                progress['epoch']+=1;progress['order']=None;progress['cursor']=0;progress['pending_monitor']=True;store('monitor_pending')
            if status=='running':status='max_epochs'
            full_complete=False
            if status in ('early_stop','max_epochs') and not exhausted() and not stopped():
                current=copy.deepcopy(model.state_dict());best=torch.load(folder/'checkpoints/best.pt',map_location='cpu',weights_only=False);model.load_state_dict(best['model'])
                full=monitor(model,load_rows('dev'),states_dev,'cuda',lambda:stopped() or exhausted());model.load_state_dict(current)
                if full is not None:save(folder/'full_dev.json',{'count':2000,'metrics':full,'checkpoint_sha256':sha(folder/'checkpoints/best.pt'),'test_read':False});full_complete=True
            store(status,finished=status in ('early_stop','budget','max_epochs'))
            save(folder/'results.json',{'status':status,'epochs':progress['epoch'],'steps':progress['step'],'best_epoch':progress['best_epoch'],'best_proxy':progress['best'],
                'seconds':elapsed(),'full_dev_complete':full_complete,'postprocessing_calls':0,'test_read':False})
            print('Stopped',status,'best',progress['best_epoch'],'seconds',round(elapsed(),1),flush=True)
        except BaseException:store('error');raise
        finally:signal.signal(signal.SIGINT,old)

if __name__=='__main__':main()
