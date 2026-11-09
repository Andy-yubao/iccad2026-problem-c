"""One explicitly launched arm; full Loss, one F call and resumable state."""
import argparse,json,signal,subprocess,sys,time
import numpy as np
import torch
from .io import HERE,ROOT,save,sha,sources,snapshot,checkpoint,exclusive
from .runtime import ARMS,model_config,run_dir,seed,rng,restore_rng,verify_preflight,state
from .dataset_view import REGISTRY,registry
from .data import load_rows,inputs_only,collate
from .model import GeometryModel
from .initialization import load,parent
from .loss import per_case,configuration
from .schedule import Scheduler
from .packed import PublicPacker
from .graph_step import CapturedStep

def stopped(output):return (HERE/'runs/STOP').exists() or (output/'STOP').exists()

@torch.no_grad()
def source_drafts(rows,role,output,identity):
    # Reuse an immutable source artifact, not another experiment's algorithm.
    path=ROOT/f'experiments/E95_single_pass_corrector/runs/20261101_single_call_seed0/inputs/{role}.pt'
    value=torch.load(path,map_location='cpu',weights_only=False);source=value['identity']
    if source['parent']!=identity['parent'] or source['registry_sha256']!=identity['registry_sha256']:
        raise ValueError('Frozen L or registry mismatch')
    if source['members']!=[list(r['member']) for r in rows] or len(value['states'])!=len(rows):raise ValueError('Source members differ')
    for name in ('base_model.py','features.py','batches.py','data.py'):
        if source['sources']['experiments/E95_single_pass_corrector/'+name]!=sha(HERE/name):
            raise ValueError('Source input/decoder implementation changed')
    for row,draft in zip(rows,value['states']):row['draft']=draft
    save(output/f'inputs/{role}_reuse.json',dict(path=path.relative_to(ROOT).as_posix(),sha256=sha(path),same_L_and_members=True,test_read=False))

@torch.no_grad()
def monitor(model,rows,packer,halt,seen=None):
    model.eval();sums={};hidden=0.;exposure={}
    for start in range(0,len(rows),32):
        if halt():return None
        selected=rows[start:start+32];g,draft,count=packer.gpu(selected)
        values=per_case(model.corrector(g,draft),g,draft)
        keys=list(values);flat=torch.stack([values[k] for k in keys]).cpu()
        if not bool(torch.isfinite(flat).all()):raise RuntimeError('Nonfinite monitor')
        for key,v in zip(keys,flat):sums[key]=sums.get(key,0.)+float(v.sum())
        hidden=max(hidden,float(model.corrector.last_hidden_rms.max().cpu()))
        if seen is not None:
            for i,v in enumerate(flat[keys.index('total')].tolist()):
                label='seen' if seen[start+i] else 'unseen';part=exposure.setdefault(label,dict(count=0,total=0.))
                part['count']+=1;part['total']+=v
    result={k:v/len(rows) for k,v in sums.items()};result.update(count=len(rows),hidden_rms_max=hidden,
        training_exposure={k:dict(count=v['count'],total=v['total']/v['count']) for k,v in exposure.items()})
    return result

def run(args):
    output=run_dir(args.arm)
    if not args.execute:
        print(json.dumps(dict(prepared_only=True,arm=args.arm,output=str(output),training_started=False),indent=2));return
    if args.device!='cuda':raise ValueError('Formal engine is registered for CUDA only')
    if not 0<args.max_hours<=2 or not 0<args.max_steps<=9872 or not 0<args.max_epochs<=200:raise ValueError('Registered cap exceeded')
    verify_preflight(args.arm)
    with exclusive(HERE/'runs/training.lock'):
        begin=time.perf_counter();seed();config=model_config(args.arm);model=GeometryModel(**config).cuda()
        identity=dict(revision='E96.centroid_bbox05.v2',arm=args.arm,sources=sources(),parent=parent(),
            registry_sha256=sha(REGISTRY),model_config=config,loss=configuration(),seed=0,batch_size=32,
            train_count=16000,precision='FP32',calls=1,engine='fixed public batch32x120x480; efficient SDPA + CUDA graph',
            max_seconds=args.max_hours*3600,max_steps=args.max_steps,max_epochs=args.max_epochs,
            initialization='frozen E90 copied encoder and zero head; fresh AdamW seed0; no E95 corrector or optimizer inherited',
            training_authorization='2026-11-01 user: 好的，现在启动后台训练; centroid bbox0.5 only, registered caps and terminal evaluation',test_read=False)
        opt=torch.optim.AdamW(model.corrector.parameters(),lr=.001,weight_decay=.0003);sched=Scheduler()
        progress=dict(epoch=0,step=0,cursor=0,order=None,seen=np.zeros(16000,dtype=bool),pending_monitor=True,
            best_loss=None,best_epoch=None,history=[])
        prior=0.;saved=None
        if args.resume:
            saved=torch.load(output/'checkpoints/latest.pt',map_location='cpu',weights_only=False)
            if saved['identity']!=identity:raise ValueError('Resume identity mismatch')
            if saved['finished']:print('Already finished');return
            if saved['status']=='error':raise ValueError('Error requires diagnosis, no automatic retry')
            model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);sched.__dict__.update(saved['scheduler'])
            progress=saved['progress'];prior=saved['elapsed_seconds'];restore_rng(saved['rng'])
            if (output/'budget.json').exists():prior=max(prior,json.loads((output/'budget.json').read_text(encoding='utf-8'))['elapsed_seconds'])
        else:
            output.mkdir(parents=True,exist_ok=False);load(model)
            save(output/'config.json',dict(identity=identity,command=[sys.executable,*sys.argv],
                commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),uncommitted_snapshot=True))
            snapshot(output/'source_snapshot.zip',identity)
        requested=[False];handler=signal.signal(signal.SIGINT,lambda *_:requested.__setitem__(0,True))
        elapsed=lambda:prior+time.perf_counter()-begin
        halt=lambda:requested[0] or stopped(output) or elapsed()>=identity['max_seconds']
        reason=lambda:'user_stop' if requested[0] or stopped(output) else 'budget'
        last=[elapsed()]
        def store(status,finished=False,best=False):
            value=dict(model=model.state_dict(),model_config=config,optimizer=opt.state_dict(),scheduler=dict(sched.__dict__),
                rng=rng(),progress=progress,identity=identity,elapsed_seconds=elapsed(),status=status,finished=finished)
            checkpoint(output/'checkpoints/latest.pt',value)
            if best:checkpoint(output/'checkpoints/best.pt',value)
            save(output/'budget.json',dict(elapsed_seconds=elapsed()))
            view=dict(status=status,finished=finished,epoch=progress['epoch'],step=progress['step'],cursor=progress['cursor'],
                seen_count=int(progress['seen'].sum()),best_epoch=progress['best_epoch'],history=progress['history'],
                scheduler=sched.__dict__,elapsed_seconds=elapsed(),test_read=False)
            save(output/'progress.json',view);state(args.arm,dict(status=status,run=output.relative_to(HERE).as_posix(),step=progress['step'],epoch=progress['epoch']))
            last[0]=elapsed()
        status='running'
        try:
            train=load_rows('train');dev=load_rows('dev_monitor500')
            lookup={tuple(r['member']):i for i,r in enumerate(train)}
            monitor_ids=[lookup[tuple(r['member'])] for r in registry()['sets']['train_monitor500']['records']]
            if len(train)!=16000 or len(dev)!=500:raise ValueError('Frozen panels changed')
            for role,rows in [('train',train),('dev_monitor500',dev)]:
                source_drafts(rows,role,output,identity)
                if halt():store(reason(),reason()=='budget');return
            if max(r['graph']['n'] for r in train+dev)>120 or max(len(r['public_ports']['port_xy']) for r in train+dev)>480:
                raise ValueError('Public shape bound exceeded')
            packer=PublicPacker(train[0],p=480);host,draft,_=packer.pack(train[:32])
            engine=CapturedStep(model,opt,{k:v.cuda() for k,v in host.items()},draft.cuda())
            if saved:restore_rng(saved['rng'])
            print('Engine ready',args.arm,'network external tokens',0 if args.arm=='centroid' else 32,flush=True)
            while status=='running':
                if halt():status=reason();break
                if progress['pending_monitor']:
                    val=monitor(model,dev,packer,halt);tr=monitor(model,[train[i] for i in monitor_ids],packer,halt,progress['seen'][monitor_ids])
                    if val is None or tr is None:status=reason();break
                    improved=progress['best_loss'] is None or val['total']<progress['best_loss']
                    if improved:progress['best_loss']=val['total'];progress['best_epoch']=progress['epoch']
                    lr=opt.param_groups[0]['lr'];sched.step(val['total'],progress['epoch'],opt)
                    progress['history'].append(dict(epoch=progress['epoch'],step=progress['step'],train_monitor=tr,dev_monitor=val,lr=lr,lr_next=opt.param_groups[0]['lr']))
                    progress['pending_monitor']=False;store('running',best=improved)
                    print('epoch',progress['epoch'],'devLoss',val['total'],flush=True)
                    if sched.stop(progress['epoch']):status='early_stop';break
                    if progress['epoch']>=args.max_epochs:status='max_epochs';break
                if progress['order'] is None:progress['order']=np.random.default_rng(progress['epoch']).permutation(16000);progress['cursor']=0
                while progress['cursor']<16000:
                    if halt() or progress['step']>=args.max_steps:break
                    ids=progress['order'][progress['cursor']:progress['cursor']+32]
                    g,draft,_=packer.pack([train[int(i)] for i in ids]);model.train();values,norm,hidden=engine.replay(g,draft)
                    progress['seen'][ids]=True;progress['cursor']+=len(ids);progress['step']+=1
                    if elapsed()-last[0]>=30:store('running')
                    if progress['step']%100==0:print('step',progress['step'],'Loss',float(values['total'].cpu()),'hidden',hidden,flush=True)
                if progress['cursor']<16000:
                    status='max_steps' if progress['step']>=args.max_steps else reason();break
                progress['epoch']+=1;progress['cursor']=0;progress['order']=None;progress['pending_monitor']=True
            finished=status in ('early_stop','max_steps','max_epochs','budget');store(status,finished)
            save(output/'results.json',dict(status=status,finished=finished,steps=progress['step'],epochs=progress['epoch'],best_epoch=progress['best_epoch'],elapsed_seconds=elapsed(),test_read=False))
            if finished and (output/'checkpoints/best.pt').exists():
                best=torch.load(output/'checkpoints/best.pt',map_location='cpu',weights_only=False)
                checkpoint(output/'corrector_network.pt',dict(model={k[len('corrector.'):]:v for k,v in best['model'].items() if k.startswith('corrector.')},
                    model_config=config,parent_sha256=sha(output/'checkpoints/best.pt'),interface='raw public problem + draft; port compression internal',primary_calls=1))
        except BaseException:store('error');raise
        finally:signal.signal(signal.SIGINT,handler)

def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=ARMS,default='centroid');p.add_argument('--execute',action='store_true')
    p.add_argument('--resume',action='store_true');p.add_argument('--device',default='cuda');p.add_argument('--max-hours',type=float,default=2.)
    p.add_argument('--max-steps',type=int,default=9872);p.add_argument('--max-epochs',type=int,default=200);run(p.parse_args())

if __name__=='__main__':main()
