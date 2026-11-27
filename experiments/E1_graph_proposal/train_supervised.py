"""Worker-balanced deterministic training with fixed validation and ML controls.

Fresh default training uses workset_v2. No local_test payload is loaded here.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
import uuid
import zipfile

STARTED = time.perf_counter()
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from datasets.paths import ROOT, resolved_data_paths
from datasets.worksets import WORKSET_PATH, workset_manifest, member_pairs, LayoutReader
from .data import graph_features, encode_target
from .model import GraphProposal
from .train import save_json, append_budget, recorded_seconds, replace_with_retry

HERE = Path(__file__).resolve().parent


def source_files():
    return sorted(set([Path(__file__),HERE/'data.py',HERE/'model.py',HERE/'train.py',
                       ROOT/'protocol/split_v1.json',ROOT/'protocol/workset_v2.json',ROOT/'protocol/paths.json',
                       *list((ROOT/'datasets').glob('*.py'))]))


def hashes():
    return {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files()}


def pack_sample(inputs, reference):
    graph=graph_features(inputs,'cpu')
    target=encode_target(reference,graph).float()
    return ({k:graph[k] for k in ('features','adjacency','relations','mask')},target)


def move(sample,device):
    g,t=sample
    out={k:([v.to(device) for v in value] if isinstance(value,list) else value.to(device)) for k,value in g.items()}
    return out,t.to(device)


def components(prediction,target,mask):
    error=(prediction-target).square()*mask
    return (error.sum()/mask.sum().clamp_min(1),
            error[:,:2].sum()/mask[:,:2].sum().clamp_min(1),
            error[:,2].sum()/mask[:,2].sum().clamp_min(1))


@torch.no_grad()
def evaluate(model,samples,indices,device,deadline=None):
    model.eval(); totals=[0.,0.,0.]; count=0
    for i in indices:
        if deadline is not None and time.perf_counter()>=deadline:
            break
        g,t=move(samples[i],device)
        values=components(model(g),t,g['mask'])
        for j,value in enumerate(values): totals[j]+=float(value)
        count+=1
    return {'sample_count':count,'requested_count':len(indices),'complete':count==len(indices),
            'mse':totals[0]/count if count else None,'center_mse':totals[1]/count if count else None,
            'log_ratio_mse':totals[2]/count if count else None,
            'aggregation':'mean of per-layout masked MSE, identical to training objective'}


def prepare(reader,deadline,stop_requested,phase):
    samples=[]; blocks=Counter(); constraints=Counter(); tensor_bytes=0
    started=time.perf_counter()
    for i in range(len(reader)):
        if stop_requested[0] or time.perf_counter()>=deadline:
            raise InterruptedError('Stopped during input preparation; no training or test read was performed after this boundary')
        inputs,reference=reader[i]
        sample=pack_sample(inputs,reference)
        for value in sample[0].values():
            for tensor in (value if isinstance(value,list) else [value]): tensor_bytes+=tensor.numel()*tensor.element_size()
        tensor_bytes+=sample[1].numel()*sample[1].element_size()
        samples.append(sample);blocks[int(inputs[0])]+=1
        for j,key in enumerate(('fixed','preplaced','mib','grouped','boundary')):
            constraints[key]+=int((inputs[5][:,j]!=0).sum())
        if (i+1)%1000==0 or i+1==len(reader):
            print(json.dumps({'phase':phase,'loaded':i+1,'total':len(reader),'seconds':time.perf_counter()-started,
                              'feature_cache_bytes':tensor_bytes}),flush=True)
        if tensor_bytes>6*1024**3:
            raise MemoryError('Feature cache exceeded preregistered 6GiB cap')
    return samples,{'samples':len(samples),'block_count_histogram':dict(sorted(blocks.items())),
                    'block_count_mean':sum(k*v for k,v in blocks.items())/len(samples),
                    'constraint_flag_block_counts':dict(constraints),'cache_tensor_bytes':tensor_bytes,
                    'preparation_seconds':time.perf_counter()-started}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--budget-seconds',type=float,default=3600)
    p.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    p.add_argument('--threads',type=int,default=1)
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--validate-every',type=int,default=250)
    p.add_argument('--early-patience',type=int,default=8)
    p.add_argument('--min-epochs',type=int,default=3)
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    if not 30<=args.budget_seconds<=3600 or min(args.threads,args.batch_size,args.validate_every,args.early_patience,args.min_epochs)<1:
        p.error('Budget must be30..3600s, positive batch/thread/control counts')
    run=args.run_dir.resolve()
    if not run.is_relative_to((HERE/'runs').resolve()):p.error('run-dir must be in this experiment/runs')
    if run.exists() and not args.resume:p.error('Refusing overwrite; new run or explicit --resume')
    if args.resume and not (run/'checkpoints/latest.pt').exists():p.error('Missing latest checkpoint')
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    random.seed(args.seed);torch.manual_seed(args.seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(args.seed)
    device=torch.device(args.device)
    manifest=workset_manifest()
    train_reader,dev_reader=LayoutReader(manifest,'train'),LayoutReader(manifest,'dev')
    train_map={pair:i for i,pair in enumerate(train_reader.members)}
    dev_map={pair:i for i,pair in enumerate(dev_reader.members)}
    ti=[train_map[pair] for pair in member_pairs(manifest,'train_monitor')]
    vi=[dev_map[pair] for pair in member_pairs(manifest,'dev_monitor')]
    paths=resolved_data_paths()
    data_root=Path(paths['training_root'])
    selected_shards=sorted({pair[0] for pair in [*train_reader.members,*dev_reader.members]})
    stats=[(s,(data_root/s).stat().st_size,(data_root/s).stat().st_mtime_ns) for s in selected_shards]
    identity={'source_sha256':hashes(),'workset_sha256':manifest['manifest_sha256'],
              'data_paths':paths,'data_stat_sha256':hashlib.sha256(json.dumps(stats).encode()).hexdigest(),
              'seed':args.seed,'batch_size':args.batch_size,'device':str(device),'threads':args.threads,
              'validate_every':args.validate_every,'early_patience':args.early_patience,'min_epochs':args.min_epochs,
              'torch':torch.__version__,'budget_seconds':args.budget_seconds}
    model=GraphProposal('deterministic').to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=.5,patience=2,
                                                       threshold=.002,threshold_mode='rel',cooldown=1,min_lr=1e-5)
    early={'reference':None,'bad_checks':0,'lr_reductions':0,'min_delta_relative':.002,
           'best_metric':None,'best_step':0,'best_epoch':0.,'validation_checks':0}
    epoch=cursor=steps=samples_seen=0;prior=0.;seen=torch.zeros(len(train_reader),dtype=torch.int32)
    invocations=[]; stop_requested=[False]
    if args.resume:
        state=torch.load(run/'checkpoints/latest.pt',map_location=device,weights_only=False)
        if state['identity']!=identity:raise ValueError('Resume identity mismatch; preserve run and review')
        model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
        early=state['early'];epoch=state['epoch'];cursor=state['cursor'];steps=state['total_steps'];samples_seen=state['samples_seen']
        seen=state['seen_counts'].cpu();prior=max(state['accumulated_seconds'],recorded_seconds(run))
        random.setstate(state['rng_python']);torch.set_rng_state(state['rng_torch'].cpu())
        if device.type=='cuda':torch.cuda.set_rng_state_all([s.cpu() for s in state['rng_cuda']])
        invocations=state['invocations']
    run.mkdir(parents=True,exist_ok=args.resume);(run/'checkpoints').mkdir(exist_ok=True)
    if not args.resume:
        with zipfile.ZipFile(run/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
            for path in source_files():z.write(path,path.relative_to(ROOT).as_posix())
    invocations.append({'command':[sys.executable,*sys.argv],'prior_seconds':prior,'created_utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()})
    config={'identity':identity,'model':model.config,'optimizer':{'name':'AdamW','lr':1e-3,'weight_decay':1e-4,'clip_norm':1.},
            'scheduler':{'name':'ReduceLROnPlateau','factor':.5,'patience':2,'cooldown':1,'threshold_relative':.002,'min_lr':1e-5},
            'early_stop':{'patience':args.early_patience,'min_delta_relative':.002,'minimum_full_epochs':args.min_epochs,'minimum_lr_reductions':2},
            'workset_counts':manifest['counts'],'invocations':invocations,'python':sys.version,
            'device_name':torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU',
            'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'git_status':subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),
            'fresh_initialization':not args.resume,'warm_start':None,'validation_metric':'500dev mean per-layout masked MSE',
            'sampling':'fresh global randperm of all20000 train members each epoch; seed+1000003*epoch',
            'budget_scope':'whole invocation including preparation, validation, checkpoint and terminal evaluation; total across resumes',
            'terminal_reserve':'estimate from initial500dev throughput, at least120s, at most600s; incomplete terminal evaluation explicitly reported',
            'test_payload_reads':0,'source_snapshot':'source_snapshot.zip'}
    save_json(run/'config.json',config)
    def elapsed():return prior+time.perf_counter()-STARTED
    deadline=STARTED+args.budget_seconds-prior
    def request_stop(signum,frame):stop_requested[0]=True
    signal.signal(signal.SIGINT,request_stop)
    if hasattr(signal,'SIGTERM'):signal.signal(signal.SIGTERM,request_stop)
    def log(row):
        row['accumulated_seconds']=elapsed()
        with (run/'run.log').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    def checkpoint(name,reason):
        if device.type=='cuda':torch.cuda.synchronize(device)
        state={'identity':identity,'model_config':model.config,'model':model.state_dict(),'optimizer':opt.state_dict(),
               'scheduler':scheduler.state_dict(),'early':early.copy(),'epoch':epoch,'cursor':cursor,'total_steps':steps,
               'samples_seen':samples_seen,'seen_counts':seen,'accumulated_seconds':elapsed(),'stop_reason':reason,
               'rng_python':random.getstate(),'rng_torch':torch.get_rng_state(),
               'rng_cuda':torch.cuda.get_rng_state_all() if device.type=='cuda' else None,'invocations':invocations}
        temp=run/'checkpoints'/f'{name}.{uuid.uuid4().hex}.tmp'
        torch.save(state,temp);replace_with_retry(temp,run/'checkpoints'/name)
        append_budget(run,{'accumulated_seconds':elapsed(),'completed_steps':steps})
    checkpoint('latest.pt','initialized' if not args.resume else 'resumed')
    try:
        train,audit_train=prepare(train_reader,deadline-10,stop_requested,'prepare_train')
        dev,audit_dev=prepare(dev_reader,deadline-10,stop_requested,'prepare_dev')
        save_json(run/'data_audit.json',{'train':audit_train,'dev':audit_dev,'test_payload_reads':0,
                  'note':'worker balanced does not imply size/constraint balance; membership not changed after seeing model results'})
        train_reader.cached_raw=dev_reader.cached_raw=None
        def monitor(initial=False,observe=True):
            nonlocal early
            t=time.perf_counter()
            validation=evaluate(model,dev,vi,device,deadline-5)
            training=evaluate(model,train,ti,device,deadline-5)
            seen_ti=[i for i in ti if seen[i]>0]
            seen_training=evaluate(model,train,seen_ti,device,deadline-5) if seen_ti else None
            if not validation['complete'] or not training['complete'] or (seen_training and not seen_training['complete']):
                raise TimeoutError('Incomplete fixed monitoring panel at wall boundary')
            value=validation['mse']
            if not math.isfinite(value):raise FloatingPointError('Nonfinite validation metric')
            oldlr=opt.param_groups[0]['lr']
            if observe:
                scheduler.step(value)  # also establish the initial untrained baseline
                if opt.param_groups[0]['lr']<oldlr:early['lr_reductions']+=1
                if early['reference'] is None or value<early['reference']*(1-.002):early['reference']=value;early['bad_checks']=0
                elif not initial:early['bad_checks']+=1
                early['validation_checks']+=1
                improved=early['best_metric'] is None or value<early['best_metric']
                if improved:
                    early.update(best_metric=value,best_step=steps,best_epoch=epoch+cursor/len(train))
                    checkpoint('best.pt','best_fixed_dev_monitor')
            record={'phase':'validation' if observe else 'resume_inspection','step':steps,'epoch_fraction':epoch+cursor/len(train),
                    'train_monitor':training,'train_monitor_seen_only':seen_training,
                    'train_monitor_seen_count':len(seen_ti),'dev_monitor':validation,
                    'learning_rate':opt.param_groups[0]['lr'],'early':early.copy(),'seconds':time.perf_counter()-t}
            log(record);checkpoint('latest.pt','validation_complete')
            return record
        init=monitor(initial=True,observe=not args.resume)
        # Initial monitor includes train/seen extras; overestimate terminal cost conservatively.
        terminal_reserve=min(600.,max(120.,init['seconds']/1000*(len(train)+len(dev))*1.6+15))
        last_save=time.perf_counter();reason='resource_budget';last_batch_seconds=0.
        while True:
            if stop_requested[0]:reason='user_stop_after_batch';break
            if elapsed()+terminal_reserve+max(5.,2*last_batch_seconds)>=args.budget_seconds:reason='resource_budget';break
            if cursor>=len(train):epoch+=1;cursor=0
            order=torch.randperm(len(train),generator=torch.Generator().manual_seed(args.seed+epoch*1000003)).tolist()
            ids=order[cursor:cursor+args.batch_size]
            model.train();t=time.perf_counter();opt.zero_grad(set_to_none=True);loss_total=0.
            for i in ids:
                g,target=move(train[i],device)
                loss=components(model(g),target,g['mask'])[0]/len(ids)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
                loss.backward();loss_total+=float(loss.detach())
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            if device.type=='cuda':torch.cuda.synchronize(device)
            cursor+=len(ids);steps+=1;samples_seen+=len(ids);seen[ids]+=1
            last_batch_seconds=time.perf_counter()-t
            append_budget(run,{'accumulated_seconds':elapsed(),'completed_steps':steps})
            if steps==1 or steps%50==0:log({'phase':'train','step':steps,'epoch_fraction':epoch+cursor/len(train),
                                           'samples_seen':samples_seen,'batch_loss':loss_total,'learning_rate':opt.param_groups[0]['lr'],
                                           'batch_seconds':last_batch_seconds})
            epoch_end=cursor==len(train)
            if steps%args.validate_every==0 or epoch_end:
                monitor()
                epoch_fraction=epoch+cursor/len(train)
                if early['bad_checks']>=args.early_patience and epoch_fraction>=args.min_epochs and early['lr_reductions']>=2:
                    reason='validation_early_stop';break
            if steps==1 or time.perf_counter()-last_save>=60:checkpoint('latest.pt','periodic');last_save=time.perf_counter()
        checkpoint('latest.pt',reason)
        state=torch.load(run/'checkpoints/best.pt',map_location=device,weights_only=False)
        model.load_state_dict(state['model'])
        best_seen=state['seen_counts'].cpu()
        best_seen_ti=[i for i in ti if best_seen[i]>0]
        best_unseen_ti=[i for i in ti if best_seen[i]==0]
        terminal={'evaluated_checkpoint':'checkpoints/best.pt','best_step':state['total_steps'],
                  'dev_monitor':evaluate(model,dev,vi,device,deadline-3),
                  'dev_full':evaluate(model,dev,list(range(len(dev))),device,deadline-3),
                  'train_monitor_seen_at_best':evaluate(model,train,best_seen_ti,device,deadline-3) if best_seen_ti else None,
                  'train_monitor_unseen_at_best':evaluate(model,train,best_unseen_ti,device,deadline-3) if best_unseen_ti else None,
                  'train_full':evaluate(model,train,list(range(len(train))),device,deadline-3),
                  'train_unique_seen_at_best':int((best_seen>0).sum()),
                  'train_unique_seen_at_stop':int((seen>0).sum()),'test_payload_reads':0}
        save_json(run/'terminal_metrics.json',terminal)
        result={'status':'stopped','stop_reason':reason,'converged':False,
                'convergence_note':'Early stopping means no significant fixed-monitor improvement within configured patience; no global-convergence guarantee.',
                'total_steps':steps,'samples_seen':samples_seen,'unique_train_samples_seen':int((seen>0).sum()),
                'epoch_fraction':epoch+cursor/len(train),'best':early,'accumulated_seconds':elapsed(),
                'requested_budget_seconds':args.budget_seconds,'terminal_complete':all(terminal[k]['complete'] for k in ('dev_monitor','dev_full','train_full')),
                'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0,
                'best_checkpoint':'checkpoints/best.pt','latest_checkpoint':'checkpoints/latest.pt','test_payload_reads':0}
        save_json(run/'results.json',result);append_budget(run,{'accumulated_seconds':elapsed(),'completed_steps':steps})
        log({'phase':'finished',**result})
    except Exception as exc:
        # Completed checkpoints only; an exception can interrupt an optimizer update.
        try:
            save_json(run/'error.json',{'type':type(exc).__name__,'message':str(exc),'accumulated_seconds':elapsed(),
                                      'complete_steps_in_memory':steps,'recovery':'latest completed checkpoint only'})
            append_budget(run,{'accumulated_seconds':elapsed(),'completed_steps':steps})
        except Exception as logging_error:print(str(logging_error),file=sys.stderr)
        raise


if __name__=='__main__':main()
