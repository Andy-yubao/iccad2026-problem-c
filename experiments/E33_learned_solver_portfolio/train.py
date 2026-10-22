"""Train a small gate; all gradient/monitor/calibration members stay inside v5 train."""
from .resources import configure
RESOURCE=configure()
import argparse
import copy
import datetime as dt
import hashlib
import json
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
import numpy as np
import torch
from .prepare import load_matrix,FOLDER,ACTION_COUNT
from .routing import Gate,objective
from .schedule import PlateauController,LR_LEVELS
from .io import HERE,ROOT,save,sha,sources,snapshot,checkpoint,exclusive

def inner_split(rows):
    shards=sorted({r['member'][0] for r in rows},key=lambda s:hashlib.sha256(('E33.inner_shard.v1:'+s).encode()).hexdigest())
    held_order=shards[:max(1,int(np.ceil(.2*len(shards))))];held=set(held_order);monitor_shards=set(held_order[:len(held_order)//2]);calibration_shards=held-monitor_shards
    fit=[i for i,r in enumerate(rows) if r['member'][0] not in held];grouped={}
    for i,r in enumerate(rows):
        if r['member'][0] in monitor_shards:grouped.setdefault(r['n'],[]).append(i)
    for values in grouped.values():values.sort(key=lambda i:hashlib.sha256(('E33.inner_monitor.v1:'+json.dumps(rows[i]['member'])).encode()).hexdigest())
    monitor=[]
    for depth in range(max(map(len,grouped.values()))):
        for n in sorted(grouped):
            if depth<len(grouped[n]) and len(monitor)<500:monitor.append(grouped[n][depth])
    monitored=set(monitor);calibration=[i for i,r in enumerate(rows) if r['member'][0] in calibration_shards]
    unused=[i for i,r in enumerate(rows) if r['member'][0] in monitor_shards and i not in monitored]
    fit_groups={}
    for i in fit:fit_groups.setdefault(rows[i]['n'],[]).append(i)
    for values in fit_groups.values():values.sort(key=lambda i:hashlib.sha256(('E33.fit_monitor.v1:'+json.dumps(rows[i]['member'])).encode()).hexdigest())
    fit_monitor=[]
    for depth in range(max(map(len,fit_groups.values()))):
        for n in sorted(fit_groups):
            if depth<len(fit_groups[n]) and len(fit_monitor)<500:fit_monitor.append(fit_groups[n][depth])
    assert len(monitor)==500 and len(calibration)>500
    return {'fit':fit,'fit_monitor':fit_monitor,'monitor':monitor,'calibration':calibration,'unused_monitor_shard_members':unused,'heldout_shards':sorted(held),'base_experts_previously_saw_all_train':True}

def rng_state():return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()}
def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch']);torch.cuda.set_rng_state_all(state['cuda'])

@torch.no_grad()
def monitor(model,x,q,t,n,ids):
    model.eval();values=[];correct=0
    for start in range(0,len(ids),32):
        k=ids[start:start+32];loss,logits,target=objective(model,x[k],q[k],t[k]);values.extend(loss.cpu().tolist());correct+=int((logits.argmax(1)==target).sum())
    values=np.asarray(values);w=np.exp((n[ids]-n[ids].max())/12);tail=np.sort(values)[-max(1,int(np.ceil(.05*len(values)))):].mean()
    return {'loss':float(values.mean()),'weighted_loss':float(np.average(values,weights=w)),'cvar95_loss':float(tail),
        'selection_proxy':.5*float(np.average(values,weights=w))+.25*float(values.mean())+.25*float(tail),'accuracy':correct/len(ids),'count':len(ids),'postprocessing_calls':0}

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--seed',type=int,default=0);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--max-hours',type=float,default=2.);ap.add_argument('--max-epochs',type=int,default=200);ap.add_argument('--resume',action='store_true');args=ap.parse_args()
    started=time.perf_counter();deadline=dt.datetime.fromisoformat('2026-10-22T19:39:00+08:00').timestamp();folder=args.output.resolve()
    if not folder.is_relative_to(HERE/'runs'):raise ValueError('Output outside E33/runs')
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed);torch.use_deterministic_algorithms(True)
    rows=load_matrix()
    if len(rows)!=16000:raise ValueError('Formal gate requires complete paired train16000 labels')
    split=inner_split(rows);fit=np.asarray(split['fit']);mon=np.asarray(split['monitor']);cal=np.asarray(split['calibration'])
    if {rows[i]['member'][0] for i in fit}&{rows[i]['member'][0] for i in np.r_[mon,cal]}:raise ValueError('Gate fit/holdout shard overlap')
    with exclusive(folder.with_suffix('.lock')):
        identity={'revision':'E33.gate.v1','sources':sources(),'label_identity_sha256':sha(FOLDER/'identity.json'),'label_manifest_sha256':sha(FOLDER/'manifest.json'),'inner_split':split,'seed':args.seed,'batch_size':32,
            'max_seconds':args.max_hours*3600,'max_epochs':args.max_epochs,'lr_levels':list(LR_LEVELS),'width':128,'actions':ACTION_COUNT,'dimensions':len(rows[0]['features']),'last_batch':'pad from current epoch prefix to exactly 32; cursor consumes only original remaining members',
            'loss':'CE(weighted-quality/time/risk oracle; regret cap .05) + .2 smoothL1(10log(Qa/Q_E25))','selection':'.5 weighted + .25 mean + .25 CVaR95 of internal-monitor loss',
            'resources':{k:v for k,v in RESOURCE.items() if k!='pid'},'torch':str(torch.__version__),'cuda_device':torch.cuda.get_device_name(0),'outer_dev_used_for_selection':False,'test_read':False}
        previous=None;prior=0.
        if args.resume:
            previous=torch.load(folder/'checkpoints/latest.pt',map_location='cpu',weights_only=False)
            if previous['identity']!=identity:raise ValueError('Gate resume identity changed')
            if previous['finished']:print('Already finished',previous['status'],flush=True);return
            prior=previous['elapsed_seconds']
            if (folder/'budget.json').exists():prior=max(prior,json.loads((folder/'budget.json').read_text())['elapsed_seconds'])
        else:
            folder.mkdir(parents=True,exist_ok=False);save(folder/'config.json',{'identity':identity,'command':[sys.executable,*sys.argv],'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                'git_status':subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True).strip()});snapshot(folder/'source_snapshot.zip',identity)
            save(folder/'inner_members.json',{key:[rows[i]['member'] for i in split[key]] for key in ('fit','monitor','calibration')})
        elapsed=lambda:prior+time.perf_counter()-started
        x=torch.tensor(np.stack([r['features'] for r in rows]),device='cuda');q=torch.tensor(np.stack([r['q'] for r in rows]),dtype=torch.float32,device='cuda');t=torch.tensor(np.stack([r['seconds'] for r in rows]),dtype=torch.float32,device='cuda');n=np.asarray([r['n'] for r in rows])
        model=Gate(identity['dimensions'],actions=ACTION_COUNT).cuda();model.mean.copy_(x[fit].mean(0));model.std.copy_(x[fit].std(0).clamp_min(1e-4))
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4);scheduler=PlateauController(optimizer)
        progress={'epoch':0,'step':0,'cursor':0,'order':None,'pending_monitor':True,'best':None,'best_epoch':None,'seen':np.zeros(len(rows),dtype=bool),'history':[]}
        if previous:
            model.load_state_dict(previous['model']);optimizer.load_state_dict(previous['optimizer']);scheduler.load_state_dict(previous['scheduler']);progress=previous['progress'];restore_rng(previous['rng'])
        stop=[False];old=signal.signal(signal.SIGINT,lambda *_:stop.__setitem__(0,True));stopped=lambda:stop[0] or (folder/'STOP').exists();exhausted=lambda:elapsed()>=identity['max_seconds'] or time.time()>=deadline
        def store(status,finished=False,best=False):
            save(folder/'budget.json',{'elapsed_seconds':elapsed()})
            value={'model':model.state_dict(),'model_config':{'dimensions':identity['dimensions'],'width':128,'actions':ACTION_COUNT},'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'rng':rng_state(),
                'progress':progress,'identity':identity,'elapsed_seconds':elapsed(),'status':status,'finished':finished}
            checkpoint(folder/'checkpoints/latest.pt',value)
            if best:checkpoint(folder/'checkpoints/best.pt',value)
            save(folder/'progress.json',{'status':status,'finished':finished,'epoch':progress['epoch'],'step':progress['step'],'cursor':progress['cursor'],'best_epoch':progress['best_epoch'],
                'best_proxy':progress['best'],'seen':int(progress['seen'].sum()),'elapsed_seconds':elapsed(),'lr_drops':scheduler.drops,'lowest_lr_completed_epochs':scheduler.lowest_completed_epochs,
                'history':progress['history'],'fit_count':len(fit),'monitor_count':len(mon),'calibration_count':len(cal),'test_read':False,'outer_dev_read':False})
        status='running'
        try:
            if not previous:store('prepared')
            while progress['epoch']<args.max_epochs or progress['pending_monitor']:
                if stopped() or exhausted():status='user_stop' if stopped() else 'budget';break
                if progress['pending_monitor']:
                    val=monitor(model,x,q,t,n,mon);train_ids=np.asarray(split['fit_monitor']);tr=monitor(model,x,q,t,n,train_ids)
                    improved=progress['best'] is None or val['selection_proxy']<progress['best']-1e-5
                    if improved:progress['best']=val['selection_proxy'];progress['best_epoch']=progress['epoch']
                    scheduler.step(val['selection_proxy'],progress['epoch']);progress['pending_monitor']=False
                    progress['history'].append({'epoch':progress['epoch'],'step':progress['step'],'internal_monitor':val,'train_monitor':tr,'lr_next':optimizer.param_groups[0]['lr'],
                        'train_monitor_current_phase_seen':int(progress['seen'][train_ids].sum()),'experts_previously_saw_all_train':True,'elapsed_seconds':elapsed()})
                    store('running',best=improved);print('epoch',progress['epoch'],'proxy',round(val['selection_proxy'],5),'accuracy',round(val['accuracy'],3),'lr',optimizer.param_groups[0]['lr'],flush=True)
                    if scheduler.should_stop():status='early_stop';break
                    continue
                if progress['order'] is None:progress['order']=np.random.default_rng(args.seed+progress['epoch']).permutation(fit);progress['cursor']=0
                model.train()
                while progress['cursor']<len(fit):
                    if stopped() or exhausted():break
                    ids=progress['order'][progress['cursor']:progress['cursor']+32]
                    batch_ids=ids if len(ids)==32 else np.r_[ids,progress['order'][:32-len(ids)]]
                    optimizer.zero_grad(set_to_none=True)
                    values,_,_=objective(model,x[batch_ids],q[batch_ids],t[batch_ids]);loss=values.mean()
                    if not torch.isfinite(loss):raise ValueError('Nonfinite gate loss')
                    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step();progress['seen'][ids]=True;progress['cursor']+=len(ids);progress['step']+=1
                    if progress['step']%100==0:store('running')
                if progress['cursor']<len(fit):status='user_stop' if stopped() else 'budget';break
                progress['epoch']+=1;progress['cursor']=0;progress['order']=None;progress['pending_monitor']=True;store('monitor_pending')
            if status=='running':status='max_epochs'
            store(status,finished=status in ('early_stop','max_epochs','budget'));save(folder/'results.json',{'status':status,'epochs':progress['epoch'],'steps':progress['step'],'best_epoch':progress['best_epoch'],
                'best_proxy':progress['best'],'elapsed_seconds':elapsed(),'fit_count':len(fit),'internal_monitor_count':500,'calibration_count':len(cal),'outer_dev_read':False,'test_read':False})
            print('Stopped',status,'best',progress['best_epoch'],flush=True)
        except BaseException:store('error');raise
        finally:signal.signal(signal.SIGINT,old)

if __name__=='__main__':main()
