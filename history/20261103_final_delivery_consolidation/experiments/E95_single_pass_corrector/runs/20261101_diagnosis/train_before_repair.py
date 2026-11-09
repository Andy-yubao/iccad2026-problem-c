"""One frozen L, one F call per update/monitor; complete resumable state."""
import argparse,json,signal,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
from .io import HERE,ROOT,save,sha,sources,snapshot,checkpoint,exclusive
from .dataset_view import REGISTRY,registry
from .data import load_rows,collate,inputs_only
from .model import GeometryModel
from .initialization import load,parent
from .loss import per_case,configuration
from .schedule import Scheduler
from .runtime import CONFIG,OUTPUT,seed,rng,restore_rng,checked_update,historical_ledger

def stopped(output):return (HERE/'runs/STOP').exists() or (output/'STOP').exists()

def verify_preflight():
    check=json.loads((HERE/'runs/preflight/results.json').read_text(encoding='utf-8'))
    if check['status']!='passed' or check['sources']!=sources():
        raise ValueError('Current-source passing five-case preflight required')

def get_batch(rows,ids,device):
    selected=[rows[int(i)] for i in ids];g=inputs_only(collate(selected,device))
    n=g['valid'].shape[1];draft=torch.zeros(len(selected),n,3,device=device)
    for i,row in enumerate(selected):draft[i,:row['graph']['n']]=row['draft'].to(device)
    return g,draft

@torch.no_grad()
def prepare_drafts(model,rows,role,output,identity,device,halt):
    path=output/f'inputs/{role}.pt'
    members=[list(r['member']) for r in rows]
    binding=dict(parent=identity['parent'],registry_sha256=identity['registry_sha256'],sources=identity['sources'],role=role,members=members)
    if path.exists():
        value=torch.load(path,map_location='cpu',weights_only=False)
        if value['identity']!=binding:raise ValueError('Frozen draft identity mismatch')
        states=value['states']
    else:
        states=[];model.eval()
        for start in range(0,len(rows),32):
            if halt():return False
            selected=rows[start:start+32];g=inputs_only(collate(selected,device));draft=model.draft(g).cpu()
            states.extend(draft[i,:r['graph']['n']].clone() for i,r in enumerate(selected))
        checkpoint(path,dict(identity=binding,states=states,source_uses_reference=False))
    if len(states)!=len(rows):raise ValueError('Draft coverage mismatch')
    for r,state in zip(rows,states):r['draft']=state
    return True

@torch.no_grad()
def monitor(model,rows,device,halt=lambda:False,seen=None):
    model.eval();sums={};strata={}
    for start in range(0,len(rows),32):
        if halt():return None
        ids=list(range(start,min(len(rows),start+32)));g,draft=get_batch(rows,ids,device)
        values=per_case(model.corrector(g,draft),g,draft)
        if not bool(torch.isfinite(torch.stack(list(values.values()))).all()):raise RuntimeError('Nonfinite monitor')
        for key,value in values.items():sums[key]=sums.get(key,0.)+float(value.sum())
        if seen is not None:
            for i,value in zip(ids,values['total'].tolist()):
                name='seen' if bool(seen[i]) else 'unseen';part=strata.setdefault(name,dict(count=0,total=0.))
                part['count']+=1;part['total']+=value
    result={k:v/len(rows) for k,v in sums.items()};result['count']=len(rows)
    result['training_exposure']={k:dict(count=v['count'],total=v['total']/v['count']) for k,v in strata.items()}
    return result

def run(args):
    output=args.output.resolve()
    if not output.is_relative_to(HERE/'runs'):raise ValueError('Output outside E95/runs')
    if not 0<args.max_hours<=2 or not 0<args.max_epochs<=200:raise ValueError('Cap is2h/200epochs')
    verify_preflight()
    with exclusive(HERE/'runs/training.lock'):
        begin=time.perf_counter();seed();model=GeometryModel(**CONFIG).to(args.device)
        identity=dict(revision='E95.single_call.v1',sources=sources(),registry_sha256=sha(REGISTRY),parent=parent(),
            model_config=CONFIG,loss=configuration(),batch_size=32,train_count=16000,seed=0,device=args.device,
            max_seconds=args.max_hours*3600,max_epochs=args.max_epochs,precision='FP32',calls=1,test_read=False)
        opt=torch.optim.AdamW(model.corrector.parameters(),lr=.001,weight_decay=.0003);sched=Scheduler()
        progress=dict(epoch=0,cursor=0,step=0,order=None,seen=np.zeros(16000,dtype=bool),pending_monitor=True,
            best_loss=None,best_epoch=None,history=[]);prior=0.;old=None
        if args.resume:
            old=torch.load(output/'checkpoints/latest.pt',map_location='cpu',weights_only=False)
            if old['identity']!=identity:raise ValueError('Resume identity mismatch')
            if old['finished']:print('Already finished');return
            if old['status']=='error':raise ValueError('Error checkpoint requires diagnosis; no automatic retry')
            model.load_state_dict(old['model']);opt.load_state_dict(old['optimizer']);sched.__dict__.update(old['scheduler'])
            progress=old['progress'];prior=old['elapsed_seconds']
            if (output/'budget.json').exists():prior=max(prior,json.loads((output/'budget.json').read_text(encoding='utf-8'))['elapsed_seconds'])
        else:
            output.mkdir(parents=True,exist_ok=False);load(model)
            save(output/'config.json',dict(identity=identity,command=[sys.executable,*sys.argv],
                commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True,cwd=ROOT).strip(),uncommitted_sources_bound_by_snapshot=True))
            save(output/'historical_budget_snapshot.json',historical_ledger());snapshot(output/'source_snapshot.zip',identity)
        request=[False];handler=signal.signal(signal.SIGINT,lambda *_:request.__setitem__(0,True))
        elapsed=lambda:prior+time.perf_counter()-begin
        halt=lambda:request[0] or stopped(output) or elapsed()>=identity['max_seconds']
        last_save=[elapsed()]
        def store(status,finished=False,best=False):
            state=dict(model=model.state_dict(),model_config=CONFIG,optimizer=opt.state_dict(),scheduler=dict(sched.__dict__),
                rng=rng(),progress=progress,identity=identity,elapsed_seconds=elapsed(),status=status,finished=finished)
            checkpoint(output/'checkpoints/latest.pt',state)
            if best:checkpoint(output/'checkpoints/best.pt',state)
            save(output/'budget.json',dict(elapsed_seconds=elapsed()))
            save(output/'progress.json',dict(status=status,finished=finished,epoch=progress['epoch'],cursor=progress['cursor'],
                step=progress['step'],seen_count=int(progress['seen'].sum()),best_epoch=progress['best_epoch'],
                history=progress['history'],scheduler=sched.__dict__,elapsed_seconds=elapsed(),test_read=False))
            last_save[0]=elapsed()
        status='running'
        try:
            train=load_rows('train');dev=load_rows('dev_monitor500')
            lookup={tuple(r['member']):i for i,r in enumerate(train)}
            monitor_ids=[lookup[tuple(r['member'])] for r in registry()['sets']['train_monitor500']['records']]
            if len(train)!=16000 or len(dev)!=500:raise ValueError('Frozen panel sizes changed')
            for role,rows in [('train',train),('dev_monitor500',dev)]:
                if not prepare_drafts(model,rows,role,output,identity,args.device,halt):
                    status='user_stop' if request[0] or stopped(output) else 'budget';break
            if old:restore_rng(old['rng'])
            while status=='running' and (progress['epoch']<args.max_epochs or progress['pending_monitor']):
                if halt():status='user_stop' if request[0] or stopped(output) else 'budget';break
                if progress['pending_monitor']:
                    val=monitor(model,dev,args.device,halt);tr=monitor(model,[train[i] for i in monitor_ids],args.device,halt,progress['seen'][monitor_ids])
                    if val is None or tr is None:status='user_stop' if request[0] or stopped(output) else 'budget';break
                    improved=progress['best_loss'] is None or val['total']<progress['best_loss']
                    if improved:progress['best_loss']=val['total'];progress['best_epoch']=progress['epoch']
                    lr=opt.param_groups[0]['lr'];sched.step(val['total'],progress['epoch'],opt)
                    progress['history'].append(dict(epoch=progress['epoch'],step=progress['step'],train_monitor=tr,dev_monitor=val,lr=lr,lr_next=opt.param_groups[0]['lr']))
                    progress['pending_monitor']=False;store('running',best=improved)
                    print('epoch',progress['epoch'],'one-call devLoss',val['total'],'lr',opt.param_groups[0]['lr'],flush=True)
                    if sched.stop(progress['epoch']):status='early_stop';break
                    if progress['epoch']>=args.max_epochs:status='max_epochs';break
                if progress['order'] is None:progress['order']=np.random.default_rng(progress['epoch']).permutation(16000);progress['cursor']=0
                while progress['cursor']<16000:
                    if halt():break
                    ids=progress['order'][progress['cursor']:progress['cursor']+32];g,draft=get_batch(train,ids,args.device)
                    values,norm=checked_update(model,opt,g,draft)
                    progress['seen'][ids]=True;progress['cursor']+=len(ids);progress['step']+=1
                    if elapsed()-last_save[0]>=30:store('running')
                    if progress['step']%100==0:print('step',progress['step'],'Loss',values['total'],'preclip norm',norm,flush=True)
                if progress['cursor']<16000:status='user_stop' if request[0] or stopped(output) else 'budget';break
                progress['epoch']+=1;progress['cursor']=0;progress['order']=None;progress['pending_monitor']=True
            if status=='running':status='max_epochs'
            finished=status in ('early_stop','budget','max_epochs');store(status,finished)
            save(output/'results.json',dict(status=status,finished=finished,epochs=progress['epoch'],steps=progress['step'],best_epoch=progress['best_epoch'],elapsed_seconds=elapsed(),test_read=False))
            if finished and (output/'checkpoints/best.pt').exists():
                best=torch.load(output/'checkpoints/best.pt',map_location='cpu',weights_only=False)
                checkpoint(output/'corrector_network.pt',dict(model={k[len('corrector.'):]:v for k,v in best['model'].items() if k.startswith('corrector.')},
                    model_config=CONFIG,parent_sha256=sha(output/'checkpoints/best.pt'),interface='public problem + draft; independently exportable F',calls_for_selection=1))
            print('Stopped',status,'best',progress['best_epoch'],flush=True)
        except BaseException:
            store('error');raise
        finally:signal.signal(signal.SIGINT,handler)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUTPUT);p.add_argument('--resume',action='store_true')
    p.add_argument('--device',default='cuda');p.add_argument('--max-hours',type=float,default=2.);p.add_argument('--max-epochs',type=int,default=200)
    run(p.parse_args())

if __name__=='__main__':main()
