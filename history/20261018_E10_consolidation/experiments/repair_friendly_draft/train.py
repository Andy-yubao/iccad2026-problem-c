"""E7 matched four-arm training. Long execution is user-run only."""
from __future__ import annotations
import argparse
import os
import random
import signal
import time
from pathlib import Path
import json
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import torch
from datasets.worksets import workset_manifest,LayoutReader,member_pairs
from .batches import sample,collate,inputs_only
from .model import GeometryModel
from .losses import all_components,objective_loss,WEIGHTS
from .occupancy import EXTRA_WEIGHTS,GRID_SIDE
from .records import save,append,replace,provenance,snapshot,run_path
from .panels import read as read_panels

VARIANTS={'attention_geometry':('attention','geometry'),
          'attention_occupancy':('attention','occupancy'),
          'rectangle_geometry':('rectangle','geometry'),
          'rectangle_occupancy':('rectangle','occupancy')}

SCHEDULES={
    'original':{'scheduler':{'factor':.5,'patience':4,'cooldown':1,'relative_min_delta':.002,'min_lr':3e-6},
                'early_stop':{'patience':16,'min_epochs':10,'minimum_lr_drops':2,'checks_after_last_drop':5}},
    'fast':{'scheduler':{'factor':.5,'patience':2,'cooldown':1,'relative_min_delta':.005,'min_lr':3e-6},
            'early_stop':{'patience':10,'min_epochs':10,'minimum_lr_drops':2,'checks_after_last_drop':3}},
}


def make_scheduler(optimizer,policy):
    s=policy['scheduler']
    return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=s['factor'],patience=s['patience'],
        cooldown=s['cooldown'],threshold=s['relative_min_delta'],threshold_mode='rel',min_lr=s['min_lr'])


def early_stop_ready(early,epoch,policy):
    e=policy['early_stop']
    return (early['bad']>=e['patience'] and epoch>=e['min_epochs'] and early['drops']>=e['minimum_lr_drops']
            and early['checks']-early['last_drop']>=e['checks_after_last_drop'])


def monitor_record(value,training,early,step,epoch,lr):
    # Insertion order deliberately puts the decisive values before nested detail.
    return {'phase':'monitor','dev_repair':value['repair'],'best_dev_repair':early['best'],
            'train_repair':training['repair'],'lr':lr,'step':step,'epoch':epoch,
            'dev':value,'train':training}


def seed_all(seed):
    random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


def load_samples(reader,stop=lambda:False):
    samples=[];size=0
    for i in range(len(reader)):
        if stop():raise InterruptedError('Stop during data preparation; no training performed')
        row=sample(*reader[i]);samples.append(row)
        size+=sum(v.numel()*v.element_size() for v in row.values() if torch.is_tensor(v))
        if size>6*1024**3:raise MemoryError('Feature cache exceeds6GiB for one split')
        if (i+1)%5000==0:print(json.dumps({'phase':'prepare','split':reader.name,'samples':i+1,'bytes':size}),flush=True)
    reader.cached_raw=None
    return samples


@torch.no_grad()
def evaluate(model,samples,indices,batch_size,device):
    model.eval();totals={};count=0
    for start in range(0,len(indices),batch_size):
        ids=indices[start:start+batch_size];g=collate(samples,ids,device)
        for k,v in all_components(model(inputs_only(g)),g).items():totals[k]=totals.get(k,0.)+float(v.sum())
        count+=len(ids)
    return {'sample_count':count,**{k:v/count for k,v in totals.items()}}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--variants',nargs='+',choices=tuple(VARIANTS),default=list(VARIANTS))
    p.add_argument('--resume',action='store_true');p.add_argument('--seed',type=int,default=0)
    p.add_argument('--batch-size',type=int,default=32);p.add_argument('--threads',type=int,default=2)
    p.add_argument('--device',default='cuda');p.add_argument('--max-epochs',type=int,default=0)
    p.add_argument('--schedule',choices=tuple(SCHEDULES),default='fast',help='fast:3 stagnant epochs before halving; original preserves the initial policy')
    p.add_argument('--smoke',action='store_true',help='Developer-only: reuse exactly dev5 as fixtures, at most2epochs, never a scientific training result')
    args=p.parse_args()
    if args.batch_size<1 or args.threads<1 or args.max_epochs<0 or len(set(args.variants))!=len(args.variants):p.error('Invalid or duplicate arguments')
    run=run_path(args.run_dir);begin=time.monotonic();run.mkdir(parents=True,exist_ok=args.resume)
    if args.smoke:
        if 'smoke' not in run.name:p.error('Smoke output must be named smoke')
        if args.max_epochs==0:args.max_epochs=2
        if args.max_epochs>2:p.error('Smoke cannot exceed2epochs')
    torch.set_num_threads(args.threads);torch.use_deterministic_algorithms(True)
    if args.device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    m=workset_manifest();panels=read_panels();policy=SCHEDULES[args.schedule]
    config={'variants':args.variants,'seed':args.seed,'batch_size':args.batch_size,'threads':args.threads,
            'device':args.device,'max_epochs':args.max_epochs,'smoke':args.smoke,'workset_sha256':m['manifest_sha256'],
            'workset_counts':m['counts'],'learning_rate':1e-3,'weight_decay':1e-4,
            'model':{'width':96,'layers':4},'geometry_weights':WEIGHTS,'extra_weights':EXTRA_WEIGHTS,
            'grid_side':GRID_SIDE,'warmup_epochs':2,'monitor':'fixed dev_monitor500 repair composite; same for all arms',
            'validation_frequency':'each epoch end; initial step0 baseline',
            'schedule':args.schedule,'scheduler':policy['scheduler'],'early_stop':policy['early_stop'],
            'resource_budget':'No wall-clock cap. max_epochs0 means no epoch cap. Validation early stop remains active.',
            'test_payload_reads':0,'control':'all fresh seed-matched; E6 frozen checkpoint is an external evaluation control',
            **provenance()}
    if args.resume:
        old=json.loads((run/'config.json').read_text(encoding='utf-8'))
        for key in config:
            if key not in ('command','created_utc','git_status','git_head') and config[key]!=old.get(key):
                raise ValueError('Resume configuration/source mismatch:'+key)
    else:save(run/'config.json',config);snapshot(run,config['source_sha256'])
    prior=json.loads((run/'status.json').read_text()).get('cumulative_seconds',0.) if args.resume and (run/'status.json').exists() else 0.
    elapsed=lambda:prior+time.monotonic()-begin
    requested=[False]
    def signal_stop(*unused):requested[0]=True
    signal.signal(signal.SIGINT,signal_stop);signal.signal(signal.SIGTERM,signal_stop)
    stop=lambda:requested[0] or (run/'STOP').exists()
    append(run/'invocations.jsonl',{'resume':args.resume,'pid':os.getpid(),'command':config['command'],'created_utc':config['created_utc']})
    def status(**kw):save(run/'status.json',{'cumulative_seconds':elapsed(),**kw})
    try:
        status(phase='preparing',test_payload_reads=0)
        if args.smoke:
            fixtures=LayoutReader(m,'dev_quality')
            selected=[fixtures.members.index((r['shard'],r['sample_index'])) for r in panels['dev5']]
            train=[sample(*fixtures[i]) for i in selected];dev=train
            ti=vi=list(range(5))
        else:
            tr=LayoutReader(m,'train');dr=LayoutReader(m,'dev')
            train=load_samples(tr,stop);dev=load_samples(dr,stop)
            ti_map={v:i for i,v in enumerate(tr.members)};vi_map={v:i for i,v in enumerate(dr.members)}
            ti=[ti_map[v] for v in member_pairs(m,'train_monitor')];vi=[vi_map[v] for v in member_pairs(m,'dev_monitor')]
        save(run/'data_audit.json',{'train':len(train),'dev':len(dev),'train_monitor':len(ti),'dev_monitor':len(vi),
                                  'smoke_fixture_reuse':args.smoke,'scientific_training_result':not args.smoke,
                                  'test_payload_reads':0,'tensor_cache_bytes':sum(v.numel()*v.element_size() for rows in (train,dev) for r in rows for v in r.values() if torch.is_tensor(v))})
        for name in args.variants:
            if stop():break
            folder=run/name;folder.mkdir(exist_ok=args.resume);cp=folder/'checkpoints';cp.mkdir(exist_ok=args.resume)
            result_file=folder/'results.json'
            if args.resume and result_file.exists() and json.loads(result_file.read_text())['stop_reason'] in ('validation_early_stop','epoch_cap'):
                previous=json.loads(result_file.read_text())
                terminal_name='fixture5.json' if args.smoke else 'dev2500.json'
                if not (folder/terminal_name).exists():
                    state=torch.load(cp/'best.pt',map_location=args.device,weights_only=False)
                    model=GeometryModel(**state['model_config']).to(args.device);model.load_state_dict(state['model'])
                    full=evaluate(model,dev,list(range(len(dev))),args.batch_size,args.device)
                    save(folder/terminal_name,{'best_step':state['step'],'metrics':full,'scope':'smoke fixtures only' if args.smoke else 'all2500dev, raw geometry only'})
                    previous['terminal_dev']='completed';save(result_file,previous);del model
                continue
            seed_all(args.seed);arch,objective=VARIANTS[name]
            model=GeometryModel(arch,width=96,layers=4).to(args.device)
            opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
            sched=make_scheduler(opt,policy)
            epoch=cursor=step=0;seen=torch.zeros(len(train),dtype=torch.int32)
            early={'best':None,'significant_best':None,'bad':0,'checks':0,'drops':0,'last_drop':-999,'best_step':None}
            spent=0.
            if args.resume and (cp/'latest.pt').exists():
                state=torch.load(cp/'latest.pt',map_location=args.device,weights_only=False)
                model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);sched.load_state_dict(state['scheduler'])
                epoch=state['epoch'];cursor=state['cursor'];step=state['step'];seen=state['seen'].cpu();early=state['early'];spent=state['variant_seconds']
                torch.set_rng_state(state['rng_torch'].cpu());random.setstate(state['rng_python'])
                if torch.cuda.is_available():torch.cuda.set_rng_state_all([v.cpu() for v in state['rng_cuda']])
            variant_begin=time.monotonic()
            def checkpoint(filename,reason):
                state={'model':model.state_dict(),'model_config':model.config,'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),
                       'epoch':epoch,'cursor':cursor,'step':step,'seen':seen,'early':early.copy(),'objective':objective,
                       'variant_seconds':spent+time.monotonic()-variant_begin,'cumulative_seconds':elapsed(),
                       'rng_torch':torch.get_rng_state(),'rng_python':random.getstate(),
                       'rng_cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                       'workset_sha256':m['manifest_sha256'],'stop_reason':reason,'training_policy':policy}
                tmp=cp/(filename+'.tmp');torch.save(state,tmp);replace(tmp,cp/filename)
            def log(row):
                row.update(variant=name,cumulative_seconds=elapsed());append(folder/'run.log',row)
                if row['phase']=='monitor':
                    print(f"[VALIDATION] {name} epoch={row['epoch']:.2f} "
                          f"DEV_REPAIR={row['dev_repair']:.6f} BEST={row['best_dev_repair']:.6f} "
                          f"TRAIN={row['train_repair']:.6f} LR={row['lr']:.3g}",flush=True)
                print(json.dumps(row,allow_nan=False),flush=True)
                status(phase=row['phase'],variant=name,step=step,epoch=epoch+cursor/len(train),test_payload_reads=0)
            def monitor():
                value=evaluate(model,dev,vi,args.batch_size,args.device)
                training=evaluate(model,train,ti,args.batch_size,args.device)
                early['checks']+=1;metric=value['repair'];old_lr=opt.param_groups[0]['lr'];sched.step(metric)
                if opt.param_groups[0]['lr']<old_lr:early['drops']+=1;early['last_drop']=early['checks']
                if early['significant_best'] is None or metric<early['significant_best']*(1-policy['scheduler']['relative_min_delta']):
                    early['significant_best']=metric;early['bad']=0
                else:early['bad']+=1
                if early['best'] is None or metric<early['best']:
                    early['best']=metric;early['best_step']=step;checkpoint('best.pt','dev_monitor_repair_best')
                row=monitor_record(value,training,early,step,epoch+cursor/len(train),opt.param_groups[0]['lr'])
                row.update(train_monitor_seen=int((seen[ti]>0).sum()),train_monitor_unseen=int((seen[ti]==0).sum()),early=early.copy())
                log(row)
                checkpoint('latest.pt','monitor')
            save(folder/'config.json',{'model':model.config,'objective':objective,'parameters':sum(p.numel() for p in model.parameters())})
            if torch.cuda.is_available():torch.cuda.reset_peak_memory_stats()
            if step==0:monitor()
            reason='user_stop';last_save=time.monotonic()
            try:
                while not stop():
                    # A monitored epoch checkpoint stores the next epoch and
                    # cursor0, avoiding an extra patience update on resume.
                    if cursor>=len(train):
                        epoch+=1;cursor=0
                    if args.max_epochs and epoch>=args.max_epochs:reason='epoch_cap';break
                    if early_stop_ready(early,epoch,policy):
                        reason='validation_early_stop';break
                    order=torch.randperm(len(train),generator=torch.Generator().manual_seed(args.seed+epoch*1000003)).tolist()
                    ids=order[cursor:cursor+args.batch_size];g=collate(train,ids,args.device)
                    model.train();opt.zero_grad(set_to_none=True)
                    values=all_components(model(inputs_only(g)),g)
                    loss=objective_loss(values,objective,epoch+cursor/len(train))
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
                    loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                    if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradient')
                    opt.step();cursor+=len(ids);step+=1;seen[ids]+=1
                    if step==1 or step%100==0:log({'phase':'train','step':step,'epoch':epoch+cursor/len(train),
                                                 'loss':float(loss.detach()),'lr':opt.param_groups[0]['lr']})
                    if cursor>=len(train):
                        # Persist completion and monitor together at the next-epoch state.
                        epoch+=1;cursor=0;monitor()
                    if time.monotonic()-last_save>60:checkpoint('latest.pt','periodic');last_save=time.monotonic()
                checkpoint('latest.pt',reason)
                result={'stop_reason':reason,'epoch':epoch+cursor/len(train),'step':step,'best':early,
                        'variant_seconds':spent+time.monotonic()-variant_begin,
                        'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
                        'unique_train_seen':int((seen>0).sum()),'test_payload_reads':0,'terminal_dev':'pending',
                        'scientific_training_result':not args.smoke}
                save(result_file,result)
                if reason!='user_stop' and not stop():
                    state=torch.load(cp/'best.pt',map_location=args.device,weights_only=False);model.load_state_dict(state['model'])
                    full=evaluate(model,dev,list(range(len(dev))),args.batch_size,args.device)
                    save(folder/('fixture5.json' if args.smoke else 'dev2500.json'),{'best_step':state['step'],'metrics':full,
                         'scope':'smoke fixtures only' if args.smoke else 'all2500dev, raw geometry only'})
                    result['terminal_dev']='completed';save(result_file,result)
                log({'phase':'finished',**result})
            except BaseException as exc:
                # Optimizer state may be incomplete; preserve the last completed
                # step rather than overwriting it with an exceptional state.
                save(folder/'error.json',{'type':type(exc).__name__,'message':str(exc),'recovery':'latest completed step only'})
                raise
            del model,opt,sched
            if torch.cuda.is_available():torch.cuda.empty_cache()
            if reason=='user_stop':break
        status(phase='user_stop' if stop() else 'training_complete',test_payload_reads=0,
               completed_variants=[n for n in args.variants if (run/n/'results.json').exists()])
    except InterruptedError:
        status(phase='user_stop_during_preparation',test_payload_reads=0)
    except BaseException as exc:
        status(phase='error',type=type(exc).__name__,message=str(exc),test_payload_reads=0)
        raise


if __name__=='__main__':main()
