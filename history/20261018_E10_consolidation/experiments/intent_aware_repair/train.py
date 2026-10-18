"""Geometry-only architecture/loss controls and registered local ablation."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
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
import zipfile

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import torch
from datasets.paths import ROOT,resolved_data_paths
from datasets.worksets import workset_manifest,LayoutReader,member_pairs
from .batches import sample,collate,decode,inputs_only
from .inference import prepare_inference
from .model import GeometryModel
from .losses import components,WEIGHTS

HERE=Path(__file__).resolve().parent
VARIANTS=[('baseline_mse','baseline','mse'),('baseline_geometry','baseline','geometry'),
          ('attention_mse','attention','mse'),('attention_geometry','attention','geometry')]
REGISTRY={v[0]:v for v in [*VARIANTS,('baseline_local','baseline','local')]}


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    os.replace(tmp,path)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sources():
    return [*[HERE/name for name in ('train.py','model.py','losses.py','batches.py','data.py','inference.py')],*sorted((ROOT/'datasets').glob('*.py')),
            ROOT/'protocol/split_v1.json',ROOT/'protocol/workset_v2.json',ROOT/'protocol/paths.json']


def initialize(seed):
    random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model,samples,ids,batch,device):
    model.eval();totals={};count=0
    for start in range(0,len(ids),batch):
        chosen=ids[start:start+batch];g=collate(samples,chosen,device)
        values=components(model(inputs_only(g)),g)
        for key,value in values.items():totals[key]=totals.get(key,0.)+float(value.sum())
        count+=len(chosen)
    return {'sample_count':count,**{k:v/max(count,1) for k,v in totals.items()}}


def read_samples(reader):
    rows=[];beg=time.monotonic();tensor_bytes=0
    for i in range(len(reader)):
        inputs,reference=reader[i];row=sample(inputs,reference);rows.append(row)
        tensor_bytes+=sum(v.numel()*v.element_size() for v in row.values() if isinstance(v,torch.Tensor))
        if tensor_bytes>5*1024**3:raise MemoryError('Experiment feature cache exceeds5GiB')
        if (i+1)%5000==0:print(json.dumps({'phase':'prepare','panel':reader.name,'samples':i+1,'seconds':time.monotonic()-beg}),flush=True)
    reader.cached_raw=None
    return rows


def preflight(run,manifest,device,batch,variants=VARIANTS):
    begin=time.monotonic();initialize(0)
    reader=LayoutReader(manifest,'train_monitor')
    ids=[0,120,245,370,495]
    samples=[sample(*reader[i]) for i in ids]
    g=collate(samples,list(range(5)),device)
    checks=[];bench={}
    for name,arch,objective in variants:
        initialize(0);model=GeometryModel(arch).to(device)
        pred=model(inputs_only(g));assert pred.shape==g['target'].shape and pred.shape[-1]==3
        single=torch.cat([model(inputs_only(collate(samples,[i],device)))[0,:samples[i]['n']] for i in range(5)])
        joined=torch.cat([pred[i,:samples[i]['n']] for i in range(5)])
        error=float((single-joined).abs().max())
        assert error<2e-5,(name,error)
        state=g['target'];decoded=decode(state,g)
        locked=~g['mask'][...,2].bool()&g['valid'];pre=~g['mask'][...,0].bool()&g['valid']
        assert torch.equal(decoded[locked][:,2:],g['specs'][locked][:,2:])
        assert torch.equal(decoded[pre],g['specs'][pre])
        # Changing hidden labels must never change the model inputs.
        inp,ref=reader[ids[0]];alter=ref.clone();free=torch.as_tensor(inp[5])[:,1]==0
        alter[free,:2]+=17
        a=sample(inp,ref);b=sample(inp,alter)
        for k in ('features','adjacency','mib','group','known','mask','area','specs'):
            assert torch.equal(a[k],b[k]),k
        public,_=prepare_inference(inp,'cpu');cached=collate([a],[0],'cpu')
        for key,value in public.items():assert torch.equal(value,cached[key]),key
        try:
            model(g)
        except ValueError:
            pass
        else:
            raise AssertionError('Network must reject protected supervision fields')
        opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
        t=time.monotonic()
        for step in range(20):
            model.train();opt.zero_grad(set_to_none=True);loss=components(model(inputs_only(g)),g)[objective].mean()
            assert torch.isfinite(loss);loss.backward()
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        torch.cuda.synchronize()
        bench[name]={'batch5_step_seconds':(time.monotonic()-t)/20,'parameters':sum(p.numel() for p in model.parameters()),
                     'last_loss':float(loss.detach()),'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated()}
        checks.append({'variant':name,'batched_vs_single_max_abs':error,'output_channels':3,
                       'finite_gradients':True,'locked_specs_preserved':True,'masked_label_invariance':True,
                       'strict_inference_matches_training_features':True,'network_rejects_supervision':True})
    result={'status':'passed','real_train_cases':5,'steps_per_variant':20,'checks':checks,'benchmark':bench,
            'seconds':time.monotonic()-begin,'test_payload_reads':0}
    save(run/'preflight.json',result);print(json.dumps(result),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--preflight-only',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--budget-seconds',type=float,default=7200)
    parser.add_argument('--prior-suite-seconds',type=float,default=0.)
    parser.add_argument('--variant-seconds',type=float,default=1500)
    parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--max-epochs',type=int,default=80)
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--variants',nargs='+',choices=sorted(REGISTRY),default=[v[0] for v in VARIANTS])
    args=parser.parse_args();started=time.monotonic()
    if len(args.variants)!=len(set(args.variants)):raise ValueError('Duplicate variants')
    variants=[REGISTRY[name] for name in args.variants]
    torch.set_num_threads(args.threads);torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available():raise RuntimeError('This declared run requires CUDA')
    if not 0<args.budget_seconds<=7200 or args.prior_suite_seconds<0 or args.budget_seconds+args.prior_suite_seconds>7200 or not 0<args.variant_seconds<=1500:
        raise ValueError('Declared cumulative2hour/25minute caps exceeded')
    device='cuda';manifest=workset_manifest();run=args.run_dir.resolve()
    if not run.is_relative_to((HERE/'runs').resolve()):raise ValueError('Run must belong to experiment')
    run.mkdir(parents=True,exist_ok=True)
    if args.preflight_only:
        if (run/'preflight.json').exists():raise ValueError('Refusing preflight overwrite')
        preflight(run,manifest,device,args.batch_size,variants);return
    if not (run/'preflight.json').exists():raise ValueError('Run the five-case preflight first')
    prior=json.loads((run/'budget.json').read_text())['seconds'] if args.resume and (run/'budget.json').exists() else json.loads((run/'preflight.json').read_text())['seconds']
    def elapsed():return prior+time.monotonic()-started
    config={'command':[sys.executable,*sys.argv],'created_utc':datetime.now(timezone.utc).isoformat(),
            'variants':[list(v) for v in variants],'seed':0,'batch_size':args.batch_size,'threads':args.threads,
            'suite_budget_seconds':args.budget_seconds,'variant_budget_seconds':args.variant_seconds,'max_epochs':args.max_epochs,
            'prior_suite_seconds':args.prior_suite_seconds,'combined_budget_seconds':args.prior_suite_seconds+args.budget_seconds,
            'workset_sha256':manifest['manifest_sha256'],'workset_counts':manifest['counts'],
            'data_paths':resolved_data_paths(),'source_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in sources()},
            'device':torch.cuda.get_device_name(0),'torch':torch.__version__,'python':sys.version,
            'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'git_status':subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),
            'scheduler':{'name':'ReduceLROnPlateau','factor':.5,'patience':3,'cooldown':1,'threshold_relative':.002,'min_lr':1e-5},
            'early_stop':{'patience_checks':12,'min_epochs':5,'min_lr_reductions':2,'relative_min_delta':.002},
            'optimizer':{'name':'AdamW','lr':1e-3,'weight_decay':1e-4,'clip_norm':1.},
            'monitor':'same geometry composite for all variants; each layout equal weight;500fixed dev',
            'geometry_weights':WEIGHTS,'training_geometry_warmup':'overlap/contact/boundary/MIB/bbox ramp linearly over first2epochs',
            'validation_every_steps':250,'validation_at_epoch_end':True,'test_payload_reads':0,'model_output':'centerXY and log(w/h) only; no relation head',
            'authorization':'User explicitly authorized long training in current chat, cumulative ceiling updated to2hours.',
            'budget_scope':'preflight + preparation + all variant training + monitor/checkpoint time; separate terminal evaluation after training',
            'fairness':'same member order and wall upper bound; early stopping can differ; steps/epochs and seen members reported; not convergence proof'}
    if args.resume:
        old=json.loads((run/'config.json').read_text(encoding='utf-8'))
        for k in ('source_sha256','workset_sha256','batch_size','threads','variant_budget_seconds','suite_budget_seconds','prior_suite_seconds','variants'):
            if config[k]!=old[k]:raise ValueError('Resume identity mismatch:'+k)
    else:
        if (run/'config.json').exists():raise ValueError('Existing formal run; explicit resume only')
        save(run/'config.json',config)
        with zipfile.ZipFile(run/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as archive:
            for p in sources():archive.write(p,p.relative_to(ROOT).as_posix())
    stopped=[False]
    def request_stop(signum,frame):stopped[0]=True
    signal.signal(signal.SIGINT,request_stop);signal.signal(signal.SIGTERM,request_stop)
    with (run/'invocations.jsonl').open('a',encoding='utf-8') as stream:
        stream.write(json.dumps({'command':[sys.executable,*sys.argv],'resume':args.resume,'prior_seconds':prior,
                                 'pid':os.getpid(),'created_utc':datetime.now(timezone.utc).isoformat()})+'\n')
    save(run/'budget.json',{'seconds':elapsed(),'phase':'preparation'})
    tr=LayoutReader(manifest,'train');dr=LayoutReader(manifest,'dev')
    train=read_samples(tr);dev=read_samples(dr)
    tm={p:i for i,p in enumerate(tr.members)};dm={p:i for i,p in enumerate(dr.members)}
    ti=[tm[p] for p in member_pairs(manifest,'train_monitor')]
    vi=[dm[p] for p in member_pairs(manifest,'dev_monitor')]
    save(run/'data_audit.json',{'train':len(train),'dev':len(dev),'train_monitor_indices':ti,'dev_monitor_indices':vi,
                             'test_payload_reads':0,'preparation_and_prior_seconds':elapsed(),
                             'tensor_cache_bytes':sum(v.numel()*v.element_size() for rows in (train,dev) for r in rows for v in r.values() if isinstance(v,torch.Tensor))})
    all_results={}
    for name,architecture,objective in variants:
        folder=run/name;result_path=folder/'results.json'
        if result_path.exists():
            previous=json.loads(result_path.read_text())
            if not(args.resume and previous['stop_reason']=='user_stop'):
                all_results[name]=previous;continue
        if stopped[0] or (run/'STOP').exists() or elapsed()>=args.budget_seconds-30:break
        initialize(0);torch.cuda.reset_peak_memory_stats()
        model=GeometryModel(architecture).to(device)
        opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,factor=.5,patience=3,cooldown=1,threshold=.002,min_lr=1e-5)
        epoch=cursor=step=0;seen=torch.zeros(len(train),dtype=torch.int32)
        early={'best':None,'reference':None,'bad':0,'reductions':0,'best_step':0,'checks':0}
        prior_variant=0.
        cp=folder/'checkpoints';cp.mkdir(parents=True,exist_ok=True)
        if args.resume and (cp/'latest.pt').exists():
            state=torch.load(cp/'latest.pt',weights_only=False,map_location=device)
            model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
            epoch=state['epoch'];cursor=state['cursor'];step=state['step'];early=state['early'];seen=state['seen'].cpu()
            prior_variant=state['variant_seconds'];torch.set_rng_state(state['rng_torch'].cpu());torch.cuda.set_rng_state_all([s.cpu() for s in state['rng_cuda']])
            random.setstate(state['rng_python'])
        variant_start=time.monotonic()
        def ve():return prior_variant+time.monotonic()-variant_start
        def log(row):
            row.update(variant=name,suite_seconds=elapsed(),variant_seconds=ve())
            with (folder/'run.log').open('a',encoding='utf-8') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')
            save(run/'budget.json',{'seconds':elapsed(),'variant':name,'step':step})
            print(json.dumps(row,allow_nan=False),flush=True)
        def checkpoint(filename,reason):
            state={'model_config':model.config,'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),
                   'epoch':epoch,'cursor':cursor,'step':step,'seen':seen,'early':early.copy(),'variant_seconds':ve(),'suite_seconds':elapsed(),
                   'rng_torch':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state_all(),'rng_python':random.getstate(),
                   'stop_reason':reason,'objective':objective,'workset_sha256':manifest['manifest_sha256']}
            tmp=cp/(filename+'.tmp');torch.save(state,tmp);os.replace(tmp,cp/filename)
        def monitor():
            value=evaluate(model,dev,vi,args.batch_size,device)
            training=evaluate(model,train,ti,args.batch_size,device)
            metric=value['geometry'];oldlr=opt.param_groups[0]['lr'];scheduler.step(metric)
            if opt.param_groups[0]['lr']<oldlr:early['reductions']+=1
            if early['reference'] is None or metric<early['reference']*(1-.002):early['reference']=metric;early['bad']=0
            else:early['bad']+=1
            early['checks']+=1
            if early['best'] is None or metric<early['best']:
                early.update(best=metric,best_step=step,best_epoch=epoch+cursor/len(train))
                checkpoint('best.pt','fixed_dev_geometry_best')
            log({'phase':'monitor','step':step,'epoch':epoch+cursor/len(train),'dev':value,'train':training,
                 'train_monitor_seen':int((seen[ti]>0).sum()),'lr':opt.param_groups[0]['lr'],'early':early.copy()})
            checkpoint('latest.pt','monitor')
        save(folder/'config.json',{'variant':name,'model':model.config,'objective':objective,'parameters':sum(p.numel() for p in model.parameters()),'parent_config':'../config.json'})
        if step==0:monitor()
        reason='resource_budget';last_save=time.monotonic()
        try:
            while ve()<args.variant_seconds-15 and elapsed()<args.budget_seconds-30 and epoch<args.max_epochs:
                if stopped[0] or (run/'STOP').exists():stopped[0]=True;reason='user_stop';break
                if cursor>=len(train):epoch+=1;cursor=0
                if epoch>=args.max_epochs:reason='epoch_cap';break
                order=torch.randperm(len(train),generator=torch.Generator().manual_seed(epoch*1000003)).tolist()
                ids=order[cursor:cursor+args.batch_size];g=collate(train,ids,device)
                model.train();opt.zero_grad(set_to_none=True);values=components(model(inputs_only(g)),g)
                if objective in ('mse','local'):loss=values[objective].mean()
                else:
                    ramp=min(1.,(epoch+cursor/len(train))/2)
                    loss=sum(WEIGHTS[k]*values[k].mean()*(ramp if k in ('overlap','contact','boundary','mib','bbox') else 1.) for k in WEIGHTS)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradients')
                opt.step();cursor+=len(ids);step+=1;seen[ids]+=1
                if step==1 or step%100==0:log({'phase':'train','step':step,'epoch':epoch+cursor/len(train),'loss':float(loss.detach()),'lr':opt.param_groups[0]['lr']})
                if step%250==0 or cursor>=len(train):
                    monitor()
                    if early['bad']>=12 and epoch+cursor/len(train)>=5 and early['reductions']>=2:reason='validation_early_stop';break
                if time.monotonic()-last_save>30:checkpoint('latest.pt','periodic');save(run/'budget.json',{'seconds':elapsed(),'variant':name,'step':step});last_save=time.monotonic()
            checkpoint('latest.pt',reason)
            result={'status':'completed_training','stop_reason':reason,'step':step,'epoch':epoch+cursor/len(train),
                    'unique_train_seen':int((seen>0).sum()),'samples_seen':int(seen.sum()),'best':early.copy(),
                    'variant_seconds':ve(),'suite_seconds':elapsed(),'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),
                    'test_payload_reads':0,'terminal_evaluation':'pending separate evaluator','converged':False}
            save(result_path,result);all_results[name]=result;log({'phase':'finished',**result})
        except BaseException as exc:
            save(folder/'error.json',{'type':type(exc).__name__,'message':str(exc),'suite_seconds':elapsed(),'variant_seconds':ve(),
                                    'recovery':'latest completed optimizer step checkpoint only'})
            save(run/'budget.json',{'seconds':elapsed(),'variant':name,'error':str(exc)})
            raise
        del model,opt,scheduler
        torch.cuda.empty_cache()
        if stopped[0]:break
    save(run/'training_results.json',{'variants':all_results,'complete':len(all_results)==len(variants),'cumulative_seconds':elapsed(),
                                    'combined_cumulative_seconds':args.prior_suite_seconds+elapsed(),
                                    'user_stop':stopped[0],'test_payload_reads':0})
    save(run/'budget.json',{'seconds':elapsed(),'phase':'training_finished'})


if __name__=='__main__':main()
