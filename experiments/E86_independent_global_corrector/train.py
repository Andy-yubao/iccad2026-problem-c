"""Resumable warm post-training of the independent ONE-readout neural network."""
import argparse,json,random,time,signal,subprocess,sys
from pathlib import Path
import numpy as np
import torch
from .io import HERE,ROOT,save,sha,sources,snapshot,checkpoint,exclusive
from .dataset_view import REGISTRY,registry
from .data import load_rows,collate,inputs_only
from .model import GeometryModel,decode
from .initialization import load,parent
from .loss import losses,per_case,configuration
from .metrics import module_overlap
from .diagnostics import geometry_checks

LEVELS=(1e-3,2.5e-4,1e-4,5e-5)
CONFIG=dict(width=192,layers=12,heads=4,dropout=.05)

def seed(seed=0):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False

def rng():return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())
def restore_rng(x):
    random.setstate(x['python']);np.random.set_state(x['numpy']);torch.set_rng_state(x['torch']);torch.cuda.set_rng_state_all(x['cuda'])

class Scheduler:
    def __init__(self):
        self.best=None;self.bad=0;self.early_bad=0;self.drops=0;self.last_drop=None
    def step(self,value,epoch,opt):
        delta=max(1e-5,abs(self.best)*.001) if self.best is not None else 0.
        if self.best is None or value<self.best-delta:self.best=value;self.bad=0;self.early_bad=0
        else:self.bad+=1;self.early_bad+=1
        if self.bad>=3 and self.drops<3:
            self.drops+=1;self.bad=0;self.last_drop=epoch
            for group in opt.param_groups:group['lr']=LEVELS[self.drops]
    def stop(self,epoch):return self.drops==3 and self.early_bad>=3 and epoch-self.last_drop>=2

@torch.no_grad()
def monitor(model,rows,device,*,records=False,stopped=lambda:False):
    model.eval();sums={};diagnostics=[];legal=0
    for start in range(0,len(rows),32):
        if stopped():return None
        selected=rows[start:start+32];g=collate(selected,device);state=model(inputs_only(g));v=per_case(state,g)
        p=(decode(state,g)*g['scale'][:,None,None]).cpu().double().numpy()
        for key,value in v.items():sums[key]=sums.get(key,0.)+float(value.sum())
        for row,rect in zip(selected,p):
            n=row['graph']['n'];rect=rect[:n];_,hard,geometry=geometry_checks(rect,row['case'].solver_inputs)
            legal+=int(bool(hard['is_feasible']));diagnostics.append(dict(member=list(row['member']),n=n,**module_overlap(rect),**geometry))
    result={k:v/len(rows) for k,v in sums.items()}
    counts=np.array([r['N5'] for r in diagnostics])
    result.update(count=len(rows),N5_mean=float(counts.mean()),N5_max=int(counts.max()),N5_zero_cases=int((counts==0).sum()),
                  raw_hard_legal=legal,overlap_area_fraction=float(np.mean([r['overlap_area_fraction'] for r in diagnostics])))
    if records:result['records']=diagnostics
    return result

def parser():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=HERE/'runs/20261030_global_neural_seed0')
    p.add_argument('--device',default='cuda');p.add_argument('--max-hours',type=float,default=2.)
    p.add_argument('--max-epochs',type=int,default=200);p.add_argument('--resume',action='store_true');return p

def run(args):
    output=args.output.resolve()
    if not output.is_relative_to(HERE/'runs'):raise ValueError('Output outside E86/runs')
    if not 0<args.max_hours<=2 or not 0<args.max_epochs<=200:raise ValueError('Maximum 2h/200 epochs')
    plan=json.loads((HERE/'plan.json').read_text(encoding='utf-8'))
    if not plan.get('formal_objective_registered',False):
        raise ValueError('E86 neural architecture is validated; formal task Loss remains under review. Legacy reference Loss is a diagnostic only.')
    with exclusive(HERE/'runs/training.lock'):
        check=json.loads((HERE/'runs/preflight/results.json').read_text(encoding='utf-8'))
        if check['status']!='passed' or check['sources']!=sources():raise ValueError('Passing current-source five-case preflight required')
        begin=time.perf_counter();seed();model=GeometryModel(**CONFIG).to(args.device)
        identity=dict(revision='E86.independent_global_neural.v1',sources=sources(),registry_sha256=sha(REGISTRY),parent=parent(),
            model_config=CONFIG,batch_size=32,train_count=16000,loss=configuration(),scheduler='epoch0 baseline, no protection; patience3, four tiers, minimum two full low-tier epochs',
            parameters=sum(p.numel() for p in model.parameters()),trainable_parameters=sum(p.numel() for p in model.corrector.parameters()),
            max_seconds=args.max_hours*3600,max_epochs=args.max_epochs,device=args.device,precision='FP32',seed=0,test_read=False)
        optimizer=torch.optim.AdamW(model.corrector.parameters(),lr=.001,weight_decay=.0003);scheduler=Scheduler();prior=0.
        progress=dict(epoch=0,cursor=0,step=0,order=None,seen=np.zeros(16000,dtype=bool),pending_monitor=True,best_loss=None,best_epoch=None,history=[])
        old=None
        if args.resume:
            old=torch.load(output/'checkpoints/latest.pt',map_location='cpu',weights_only=False)
            if old['identity']!=identity:raise ValueError('Resume identity mismatch')
            if old['finished']:print('Already finished');return
            model.load_state_dict(old['model']);optimizer.load_state_dict(old['optimizer']);scheduler.__dict__.update(old['scheduler'])
            progress=old['progress'];prior=old['elapsed_seconds']
            if (output/'budget.json').exists():prior=max(prior,json.loads((output/'budget.json').read_text(encoding='utf-8'))['elapsed_seconds'])
        else:
            output.mkdir(parents=True,exist_ok=False);load(model)
            save(output/'config.json',dict(identity=identity,command=[sys.executable,*sys.argv],commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True,cwd=ROOT).strip(),uncommitted_sources_bound_by_snapshot=True))
            snapshot(output/'source_snapshot.zip',identity)
        train=load_rows('train');dev=load_rows('dev_monitor500');members={tuple(r['member']):i for i,r in enumerate(train)}
        monitor_ids=[members[tuple(r['member'])] for r in registry()['sets']['train_monitor500']['records']];train_monitor=[train[i] for i in monitor_ids]
        if len(train)!=16000 or len(dev)!=500:raise ValueError('Frozen panel count mismatch')
        if old:restore_rng(old['rng'])
        request=[False];handler=signal.signal(signal.SIGINT,lambda *_:request.__setitem__(0,True))
        elapsed=lambda:prior+time.perf_counter()-begin
        stopped=lambda:request[0] or (HERE/'runs/STOP').exists() or (output/'STOP').exists()
        exhausted=lambda:elapsed()>=identity['max_seconds']
        last_save=[elapsed()]
        def store(status,finished=False,best=False):
            save(output/'budget.json',dict(elapsed_seconds=elapsed()))
            state=dict(model=model.state_dict(),model_config=CONFIG,optimizer=optimizer.state_dict(),scheduler=scheduler.__dict__,rng=rng(),progress=progress,identity=identity,elapsed_seconds=elapsed(),status=status,finished=finished)
            checkpoint(output/'checkpoints/latest.pt',state)
            if best:checkpoint(output/'checkpoints/best.pt',state)
            save(output/'progress.json',dict(status=status,finished=finished,epoch=progress['epoch'],cursor=progress['cursor'],step=progress['step'],seen_count=int(progress['seen'].sum()),best_epoch=progress['best_epoch'],history=progress['history'],lr_drops=scheduler.drops,elapsed_seconds=elapsed(),test_read=False))
            last_save[0]=elapsed()
        status='running'
        try:
            while progress['epoch']<args.max_epochs or progress['pending_monitor']:
                if stopped() or exhausted():status='user_stop' if stopped() else 'budget';break
                if progress['pending_monitor']:
                    val=monitor(model,dev,args.device,stopped=lambda:stopped() or exhausted());tr=monitor(model,train_monitor,args.device,stopped=lambda:stopped() or exhausted())
                    if val is None or tr is None:status='user_stop' if stopped() else 'budget';break
                    improved=progress['best_loss'] is None or val['total']<progress['best_loss']-1e-5
                    if improved:progress['best_loss']=val['total'];progress['best_epoch']=progress['epoch']
                    lr=optimizer.param_groups[0]['lr'];scheduler.step(val['total'],progress['epoch'],optimizer)
                    progress['history'].append(dict(epoch=progress['epoch'],step=progress['step'],train_monitor=tr,dev_monitor=val,train_seen_count=int(progress['seen'][monitor_ids].sum()),lr=lr,lr_next=optimizer.param_groups[0]['lr'],lr_drops=scheduler.drops))
                    progress['pending_monitor']=False;store('running',best=improved)
                    print('epoch',progress['epoch'],'devLoss',val['total'],'N5 mean/max',val['N5_mean'],val['N5_max'],'lr',optimizer.param_groups[0]['lr'],flush=True)
                    if scheduler.stop(progress['epoch']):status='early_stop';break
                    if progress['epoch']>=args.max_epochs:status='max_epochs';break
                if progress['order'] is None:progress['order']=np.random.default_rng(progress['epoch']).permutation(len(train));progress['cursor']=0
                model.train()
                while progress['cursor']<len(train):
                    if stopped() or exhausted():break
                    ids=progress['order'][progress['cursor']:progress['cursor']+32];g=collate([train[i] for i in ids],args.device)
                    optimizer.zero_grad(set_to_none=True);values=losses(model(inputs_only(g)),g)
                    if not bool(torch.isfinite(torch.stack(list(values.values()))).all()):raise RuntimeError('Nonfinite Loss before update')
                    values['total'].backward();norm=torch.nn.utils.clip_grad_norm_(model.corrector.parameters(),1.,error_if_nonfinite=True);optimizer.step()
                    progress['seen'][ids]=True;progress['cursor']+=len(ids);progress['step']+=1
                    if elapsed()-last_save[0]>=30:store('running')
                    if progress['step']%100==0:print('step',progress['step'],'epoch',progress['epoch']+1,'Loss',float(values['total']),flush=True)
                if progress['cursor']<len(train):status='user_stop' if stopped() else 'budget';break
                progress['epoch']+=1;progress['cursor']=0;progress['order']=None;progress['pending_monitor']=True
            if status=='running':status='max_epochs'
            finished=status in ('early_stop','budget','max_epochs');store(status,finished=finished)
            save(output/'results.json',dict(status=status,epochs=progress['epoch'],steps=progress['step'],best_epoch=progress['best_epoch'],elapsed_seconds=elapsed(),test_read=False,postprocessed_Q='not evaluated; separate frozen-best stage'))
            if finished:
                best=torch.load(output/'checkpoints/best.pt',map_location='cpu',weights_only=False)
                checkpoint(output/'corrector_network.pt',dict(model={k[len('corrector.'):]:v for k,v in best['model'].items() if k.startswith('corrector.')},model_config=CONFIG,parent_sha256=sha(output/'checkpoints/best.pt'),role='Independent neural network, accepts public problem and draft; no placer or optimizer'))
            print('Stopped',status,'best',progress['best_epoch'],flush=True)
        except BaseException:store('error');raise
        finally:signal.signal(signal.SIGINT,handler)

if __name__=='__main__':run(parser().parse_args())
