"""One frozen L, one F call per update/monitor; complete resumable state."""
import argparse,copy,json,signal,subprocess,sys,time
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
from .runtime import CONFIG,OUTPUT,seed,rng,restore_rng,historical_ledger,prior_family_seconds
from .packed import PublicPacker
from .graph_step import CapturedStep

def stopped(output):return (HERE/'runs/STOP').exists() or (output/'STOP').exists()

def verify_preflight():
    check=json.loads((HERE/'runs/preflight_engine/results.json').read_text(encoding='utf-8'))
    if check['status']!='passed' or check['sources']!=sources():
        raise ValueError('Current-source passing five-case preflight required')

def get_batch(rows,ids,device):
    selected=[rows[int(i)] for i in ids];g=inputs_only(collate(selected,device))
    n=g['valid'].shape[1];draft=torch.zeros(len(selected),n,3,device=device)
    for i,row in enumerate(selected):draft[i,:row['graph']['n']]=row['draft'].to(device)
    return g,draft

@torch.no_grad()
def prepare_drafts(model,rows,role,output,identity,device,halt):
    previous=HERE/f'runs/20261101_single_call_seed0/inputs/{role}.pt'
    if previous.exists():
        value=torch.load(previous,map_location='cpu',weights_only=False)
        source=value['identity'];members=[list(r['member']) for r in rows]
        if len(value['states'])!=len(rows):raise ValueError('Original source draft coverage differs')
        if source['parent']!=identity['parent'] or source['registry_sha256']!=identity['registry_sha256'] or source['members']!=members:
            raise ValueError('Original source draft lineage differs')
        for filename in ('base_model.py','features.py','batches.py','data.py'):
            key=f'experiments/E95_single_pass_corrector/{filename}'
            if source['sources'][key]!=sha(HERE/filename):raise ValueError('Source decoder/input semantics changed')
        for row,state in zip(rows,value['states']):row['draft']=state
        save(output/f'inputs/{role}_reuse.json',dict(path=previous.relative_to(ROOT).as_posix(),sha256=sha(previous),
            same_frozen_placer_and_members=True,corrector_not_in_source_generation=True,test_read=False))
        return True
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
def monitor(model,rows,device,halt=lambda:False,seen=None,packer=None):
    model.eval();sums={};strata={}
    for start in range(0,len(rows),32):
        if halt():return None
        ids=list(range(start,min(len(rows),start+32)))
        if packer is None:g,draft=get_batch(rows,ids,device)
        else:g,draft,_=packer.gpu([rows[i] for i in ids])
        values=per_case(model.corrector(g,draft),g,draft)
        values={k:v[:len(ids)] for k,v in values.items()}
        keys=list(values);flat=torch.stack([values[k] for k in keys]).detach().cpu()
        if not bool(torch.isfinite(flat).all()):raise RuntimeError('Nonfinite monitor')
        for key,value in zip(keys,flat):sums[key]=sums.get(key,0.)+float(value.sum())
        hidden=float(model.corrector.last_hidden_rms[:len(ids)].max().cpu())
        sums['hidden_rms_max']=max(sums.get('hidden_rms_max',0.),hidden)
        if seen is not None:
            for i,value in zip(ids,flat[keys.index('total')].tolist()):
                name='seen' if bool(seen[i]) else 'unseen';part=strata.setdefault(name,dict(count=0,total=0.))
                part['count']+=1;part['total']+=value
    result={k:v/len(rows) for k,v in sums.items() if k!='hidden_rms_max'};result['count']=len(rows)
    result['hidden_rms_max']=sums['hidden_rms_max']
    result['training_exposure']={k:dict(count=v['count'],total=v['total']/v['count']) for k,v in strata.items()}
    return result

def run(args):
    output=args.output.resolve()
    if not output.is_relative_to(HERE/'runs'):raise ValueError('Output outside E95/runs')
    if not 0<args.max_hours<=2 or not 0<args.max_epochs<=200 or not 0<args.max_steps<=100000:raise ValueError('Cap is2h/200epochs')
    verify_preflight()
    with exclusive(HERE/'runs/training.lock'):
        begin=time.perf_counter();seed();model=GeometryModel(**CONFIG).to(args.device)
        identity=dict(revision='E95.direct_bbox.engine_migration.v4',sources=sources(),registry_sha256=sha(REGISTRY),parent=parent(),
            model_config=CONFIG,loss=configuration(),batch_size=32,train_count=16000,seed=0,device=args.device,
            max_seconds=args.max_hours*3600,max_epochs=args.max_epochs,precision='FP32',calls=1,
            engine='FP32 efficient SDPA; fixed32x120x480 CUDA graph F/Loss/backward/clip; guarded original AdamW',
            shape=[32,120,480],normalization='pre-normalize local graph aggregation; all other architecture/Loss unchanged',
            initialization='continue stopped latest4481; preserve model/Adam/RNG/order/cursor/scheduler/best; explicit numerical-engine migration',
            prior_training_charged_seconds=prior_family_seconds(),max_optimizer_steps=args.max_steps,
            migration_sha256=sha(HERE/'engine_migration.json'),test_read=False)
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
            migration=json.loads((HERE/'engine_migration.json').read_text(encoding='utf-8'))
            source=(ROOT/migration['source_run']).resolve()
            if args.migrate_from is None or args.migrate_from.resolve()!=source:
                raise ValueError('Fresh initialization forbidden: explicit registered migration source required')
            if sha(source/'checkpoints/latest.pt')!=migration['latest_sha256'] or sha(source/'checkpoints/best.pt')!=migration['best_sha256']:
                raise ValueError('Migration source checkpoint changed')
            old=torch.load(source/'checkpoints/latest.pt',map_location='cpu',weights_only=False)
            if old['status']!='user_stop' or old['finished'] or old['identity']['loss']!=configuration():
                raise ValueError('Stopped same-objective checkpoint required')
            if old['identity']['registry_sha256']!=identity['registry_sha256'] or old['identity']['parent']!=identity['parent']:
                raise ValueError('Migration input lineage changed')
            model.load_state_dict(old['model']);opt.load_state_dict(old['optimizer']);sched.__dict__.update(old['scheduler'])
            progress=copy.deepcopy(old['progress']);restore_rng(old['rng'])
            inherited=torch.load(source/'checkpoints/best.pt',map_location='cpu',weights_only=False)
            inherited['inherited_source']=dict(path=(source/'checkpoints/best.pt').relative_to(ROOT).as_posix(),sha256=migration['best_sha256'],old_identity=inherited['identity'])
            inherited['identity']=identity;inherited['model_config']=CONFIG;inherited['historical_elapsed_seconds']=inherited['elapsed_seconds'];inherited['elapsed_seconds']=0.
            output.mkdir(parents=True,exist_ok=False)
            checkpoint(output/'checkpoints/best.pt',inherited)
            save(output/'config.json',dict(identity=identity,command=[sys.executable,*sys.argv],
                commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True,cwd=ROOT).strip(),uncommitted_sources_bound_by_snapshot=True))
            save(output/'migration.json',dict(migration,progress_preserved=True,Adam_and_scheduler_preserved=True,starting_step=progress['step'],starting_cursor=progress['cursor']))
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
                step=progress['step'],family_elapsed_seconds=identity['prior_training_charged_seconds']+elapsed(),seen_count=int(progress['seen'].sum()),best_epoch=progress['best_epoch'],
                history=progress['history'],scheduler=sched.__dict__,elapsed_seconds=elapsed(),test_read=False))
            save(HERE/'state.json',dict(status=status,run=output.relative_to(HERE).as_posix(),epoch=progress['epoch'],
                step=progress['step'],best_epoch=progress['best_epoch'],elapsed_seconds=elapsed(),test_read=False))
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
            if status!='running':
                store(status,finished=status=='budget');return
            if max(r['graph']['n'] for r in train+dev)>120 or max(len(r['public_ports']['port_xy']) for r in train+dev)>480:
                raise ValueError('Public data exceeds preflight shape bounds')
            packer=PublicPacker(train[0],p=480)
            if not args.resume:
                model.load_state_dict(inherited['model'])
                baseline=monitor(model,dev,args.device,halt,packer=packer)
                model.load_state_dict(old['model'])
                if baseline is None:
                    restore_rng(old['rng']);store('user_stop' if stopped(output) or request[0] else 'budget');return
                if abs(baseline['total']-progress['best_loss'])>2e-5:
                    raise RuntimeError('Inherited best monitor changed beyond registered tolerance')
                save(output/'inherited_best_revalidation.json',dict(old_value=progress['best_loss'],new_monitor=baseline,source_sha256=migration['best_sha256'],test_read=False))
                progress['best_loss']=baseline['total']
                inherited['progress']=copy.deepcopy(inherited['progress']);inherited['progress']['best_loss']=baseline['total']
                inherited['monitor_revalidation']=baseline;checkpoint(output/'checkpoints/best.pt',inherited)
                print('Inherited best revalidated:',baseline['total'],flush=True)
            host,source,_=packer.pack(train[:32]);sample={k:v.to(args.device) for k,v in host.items()}
            engine=CapturedStep(model,opt,sample,source.to(args.device))
            if old:restore_rng(old['rng'])
            print('Engine ready: FP32 efficient SDPA + CUDA graph32x120x480; peak reserved MiB',torch.cuda.max_memory_reserved()/2**20,flush=True)
            while status=='running' and (progress['epoch']<args.max_epochs or progress['pending_monitor']):
                if halt():status='user_stop' if request[0] or stopped(output) else 'budget';break
                if progress['pending_monitor']:
                    val=monitor(model,dev,args.device,halt,packer=packer);tr=monitor(model,[train[i] for i in monitor_ids],args.device,halt,progress['seen'][monitor_ids],packer)
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
                    if halt() or progress['step']>=args.max_steps:break
                    ids=progress['order'][progress['cursor']:progress['cursor']+32]
                    batch,draft,_=packer.pack([train[int(i)] for i in ids]);model.train()
                    values,norm,hidden=engine.replay(batch,draft)
                    progress['seen'][ids]=True;progress['cursor']+=len(ids);progress['step']+=1
                    if elapsed()-last_save[0]>=30:store('running')
                    if progress['step']%100==0:print('step',progress['step'],'batchLoss',float(values['total'].detach().cpu()),'preclip norm',norm,'hiddenRMS max',hidden,flush=True)
                if progress['cursor']<16000:
                    status='max_steps' if progress['step']>=args.max_steps else ('user_stop' if request[0] or stopped(output) else 'budget');break
                progress['epoch']+=1;progress['cursor']=0;progress['order']=None;progress['pending_monitor']=True
            if status=='running':status='max_epochs'
            finished=status in ('early_stop','budget','max_epochs','max_steps');store(status,finished)
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
    p.add_argument('--migrate-from',type=Path);p.add_argument('--max-steps',type=int,default=100000);p.add_argument('--device',default='cuda');p.add_argument('--max-hours',type=float,default=2.);p.add_argument('--max-epochs',type=int,default=200)
    run(p.parse_args())

if __name__=='__main__':main()
