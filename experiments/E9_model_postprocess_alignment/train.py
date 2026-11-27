"""E9 matched warm-start fine-tuning. Formal long runs are user-started only."""
import argparse
import json
import os
from pathlib import Path
import random
import signal
import time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import torch
from datasets.worksets import LayoutReader,workset_manifest,member_pairs
from datasets.paths import ROOT
from .batches import sample,collate,inputs_only
from .model import GeometryModel
from .losses import all_components,objective_loss,WEIGHTS,ALIGNED_WEIGHTS
from .evaluate import evaluate_quality,model_digest,panel_reader
from .records import save,append,replace,sha,provenance,snapshot,run_path,OLD_CHECKPOINT,OLD_CHECKPOINT_SHA

VARIANTS=('geometry_control','quality_aligned','center_refiner')
POLICY={'lr':1e-4,'weight_decay':1e-4,'factor':.5,'lr_patience':2,'cooldown':1,'min_lr':3e-6,
        'relative_min_delta':.002,'early_patience':10,'min_epochs':10,'checks_after_lr_drop':3,
        'quality_epochs':[5,15,30],'warmup_epochs':2,'monitor':'aligned on fixed dev_monitor500 for BOTH variants'}


def seed_all(seed):
    random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


def load_samples(reader,stop):
    rows=[];size=0
    for index in range(len(reader)):
        if stop():raise InterruptedError('Stopped during data preparation')
        row=sample(*reader[index]);rows.append(row)
        size+=sum(v.numel()*v.element_size() for v in row.values() if torch.is_tensor(v))
        if size>6*1024**3:raise MemoryError('One split feature cache exceeds 6 GiB')
        if (index+1)%5000==0:print(json.dumps({'phase':'prepare','split':reader.name,'samples':index+1,'bytes':size}),flush=True)
    reader.cached_raw=None
    return rows


@torch.no_grad()
def monitor_metrics(model,rows,indices,batch,device,stop=lambda:False):
    model.eval();totals={};count=0
    for start in range(0,len(indices),batch):
        if stop():raise InterruptedError('Stopped during validation')
        ids=indices[start:start+batch];g=collate(rows,ids,device)
        for key,value in all_components(model(inputs_only(g)),g).items():
            totals[key]=totals.get(key,0.)+float(value.sum())
        count+=len(ids)
    return {'sample_count':count,**{key:value/count for key,value in totals.items()}}


def train_arm(name,folder,train,dev,ti,vi,initial,args,manifest,elapsed,global_stop,user_stop):
    folder.mkdir(exist_ok=True);cp=folder/'checkpoints';cp.mkdir(exist_ok=True)
    seed_all(args.seed)
    configuration={**initial['model_config']}
    if name=='center_refiner':configuration['architecture']='center_refiner'
    model=GeometryModel(**configuration).to(args.device)
    missing,unexpected=model.load_state_dict(initial['model'],strict=name!='center_refiner')
    if name=='center_refiner' and (unexpected or any(not key.startswith('center_refiner.') for key in missing)):
        raise ValueError('Frozen backbone initialization mismatch')
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=POLICY['lr'],weight_decay=POLICY['weight_decay'])
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=POLICY['factor'],patience=POLICY['lr_patience'],
        cooldown=POLICY['cooldown'],threshold=POLICY['relative_min_delta'],threshold_mode='rel',min_lr=POLICY['min_lr'])
    epoch=cursor=step=0;seen=torch.zeros(len(train),dtype=torch.int32);spent=0.
    early={'best':None,'significant_best':None,'bad':0,'checks':0,'last_drop':-999,'best_step':None}
    q_done={};selection=None;initial_monitor_done=False;monitor_due=True
    if args.resume and (cp/'latest.pt').exists():
        state=torch.load(cp/'latest.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
        epoch,cursor,step=state['epoch'],state['cursor'],state['step'];seen=state['seen'];spent=state['arm_seconds']
        early=state['early'];q_done=state['quality_done'];selection=state['selection'];initial_monitor_done=state['initial_monitor_done']
        monitor_due=state['monitor_due']
        torch.set_rng_state(state['rng_torch']);random.setstate(state['rng_python'])
        if torch.cuda.is_available():torch.cuda.set_rng_state_all(state['rng_cuda'])
    begin=time.monotonic();arm_elapsed=lambda:spent+time.monotonic()-begin
    arm_stop=lambda:global_stop() or arm_elapsed()>=args.arm_seconds
    def checkpoint(path,reason):
        value={'model':model.state_dict(),'model_config':model.config,'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
               'epoch':epoch,'cursor':cursor,'step':step,'seen':seen,'early':early.copy(),'variant':name,
               'quality_done':q_done.copy(),'selection':selection,'initial_monitor_done':initial_monitor_done,
               'monitor_due':monitor_due,
               'arm_seconds':arm_elapsed(),'cumulative_seconds':elapsed(),'workset_sha256':manifest['manifest_sha256'],
               'fixture_only':bool(initial.get('fixture_only',False)),
               'rng_torch':torch.get_rng_state(),'rng_python':random.getstate(),
               'rng_cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],'stop_reason':reason}
        temporary=path.with_suffix('.tmp');torch.save(value,temporary);replace(temporary,path)
    def log(row):
        record={'variant':name,'step':step,'epoch':epoch+cursor/len(train),'cumulative_seconds':elapsed(),**row}
        append(folder/'run.log',record);print(json.dumps(record,allow_nan=False),flush=True)
        save(folder.parent/'status.json',{'phase':record['phase'],'variant':name,'step':step,'epoch':record['epoch'],
             'cumulative_seconds':elapsed(),'test_payload_reads':0})
    def monitor():
        nonlocal initial_monitor_done,monitor_due
        values=monitor_metrics(model,dev,vi,args.batch_size,args.device,arm_stop)
        training=monitor_metrics(model,train,ti,args.batch_size,args.device,arm_stop)
        metric=values['aligned'];old_lr=optimizer.param_groups[0]['lr'];scheduler.step(metric);early['checks']+=1
        if optimizer.param_groups[0]['lr']<old_lr:early['last_drop']=early['checks']
        if early['significant_best'] is None or metric<early['significant_best']*(1-POLICY['relative_min_delta']):
            early['significant_best']=metric;early['bad']=0
        else:early['bad']+=1
        improved=early['best'] is None or metric<early['best']
        if improved:early['best']=metric;early['best_step']=step
        initial_monitor_done=True;monitor_due=False
        if improved:checkpoint(cp/'best.pt','dev_monitor500_aligned_best')
        log({'phase':'monitor','dev':values,'train':training,'early':early.copy(),'lr':optimizer.param_groups[0]['lr'],
             'train_monitor_seen':int((seen[ti]>0).sum()),'train_monitor_unseen':int((seen[ti]==0).sum())})
        checkpoint(cp/'latest.pt','monitor')
    def quality(label,path,network):
        nonlocal selection
        if label in q_done:return True
        digest=model_digest(network)
        identical=next((v for v in q_done.values() if v['model_digest']==digest),None)
        if identical is not None:
            q_done[label]={**identical,'same_model_as':identical['checkpoint']};return True
        remaining=min(args.budget_seconds-elapsed(),args.arm_seconds-arm_elapsed())
        if remaining<20:return False
        result=evaluate_quality(network,folder/'quality'/label,'dev100',args.device,
            {'checkpoint':str(path.resolve()),'checkpoint_sha256':sha(path),'variant':name},remaining,arm_stop)
        if result['status']!='completed':return False
        q=result['summary']['Q_mean']
        entry={'checkpoint':str(path.resolve()),'model_digest':digest,'Q_mean':q,'evaluation':str((folder/'quality'/label).resolve())}
        q_done[label]=entry
        # Ties keep the earlier model; no candidate or case is selected by its
        # test performance. Official Q is computed only after solving.
        if selection is None or q<selection['Q_mean']-1e-8:selection=entry
        save(folder/'selection.json',selection);log({'phase':'quality_complete','label':label,'Q_mean':q,'selected':selection})
        return True
    if selection is None:
        _,_,_,baseline,records=panel_reader('dev100')
        q=sum(r['variants']['conic_robust']['final']['metrics']['cost'] for r in records.values())/100
        retained=GeometryModel(**initial['model_config']);retained.load_state_dict(initial['model'])
        retained_digest=model_digest(retained);del retained
        selection={'checkpoint':str(OLD_CHECKPOINT),'checkpoint_sha256':OLD_CHECKPOINT_SHA,'model_digest':retained_digest,
                   'Q_mean':q,'evaluation':str(baseline),'source':'retained E7 + unchanged E8, cached control'}
        save(folder/'selection.json',selection)
    if torch.cuda.is_available():torch.cuda.reset_peak_memory_stats()
    reason='user_stop';terminal='pending';last_save=time.monotonic()
    try:
        if monitor_due:monitor()
        while not arm_stop():
            if cursor==len(train):
                # This also completes an interrupted epoch-end monitor exactly
                # once; checkpoint state before validation retains this cursor.
                epoch+=1;cursor=0;monitor_due=True;checkpoint(cp/'latest.pt','before_monitor');monitor()
            if cursor==0 and epoch in POLICY['quality_epochs']:
                label=f'epoch{epoch:03d}';path=cp/(label+'.pt')
                if label not in q_done:
                    if not path.exists():checkpoint(path,'fixed_quality_node')
                    if not quality(label,path,model):reason='quality_incomplete';break
                    checkpoint(cp/'latest.pt','after_quality')
            if epoch>=args.max_epochs:reason='epoch_cap';break
            if (early['bad']>=POLICY['early_patience'] and epoch>=POLICY['min_epochs']
                    and early['checks']-early['last_drop']>=POLICY['checks_after_lr_drop']):
                reason='validation_early_stop';break
            order=torch.randperm(len(train),generator=torch.Generator().manual_seed(args.seed+1000003*epoch)).tolist()
            ids=order[cursor:cursor+args.batch_size];g=collate(train,ids,args.device)
            model.train();optimizer.zero_grad(set_to_none=True)
            values=all_components(model(inputs_only(g)),g);loss=objective_loss(values,name,epoch+cursor/len(train))
            if not torch.isfinite(loss):raise FloatingPointError('Non-finite loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(norm):raise FloatingPointError('Non-finite gradient')
            optimizer.step();cursor+=len(ids);step+=1;seen[ids]+=1
            if step==1 or step%100==0:log({'phase':'train','loss':float(loss.detach()),'lr':optimizer.param_groups[0]['lr']})
            if time.monotonic()-last_save>=60:checkpoint(cp/'latest.pt','periodic');last_save=time.monotonic()
        if arm_stop():reason='user_stop' if user_stop() else 'global_budget' if global_stop() else 'arm_budget'
        # Resource stops are not convergence. Leave interrupted validation and
        # quality evaluations pending; preserve the last trained state first.
        checkpoint(cp/'latest.pt',reason)
        if reason in ('validation_early_stop','epoch_cap') and not arm_stop():
            best=torch.load(cp/'best.pt',map_location='cpu',weights_only=False)
            network=GeometryModel(**best['model_config']).to(args.device);network.load_state_dict(best['model'])
            quality('proxy_best',cp/'best.pt',network)
            del network
        if not arm_stop() and selection is not None:
            chosen=torch.load(selection['checkpoint'],map_location='cpu',weights_only=False)
            network=GeometryModel(**chosen['model_config']).to(args.device);network.load_state_dict(chosen['model'])
            full=monitor_metrics(network,dev,list(range(len(dev))),args.batch_size,args.device,arm_stop)
            save(folder/'dev2500.json',{'checkpoint':selection['checkpoint'],'metrics':full,'scope':'all frozen dev2500, raw geometry only'})
            terminal='completed';del network
        checkpoint(cp/'latest.pt',reason)
    except InterruptedError:
        reason='user_stop' if user_stop() else 'global_budget' if global_stop() else 'arm_budget'
        checkpoint(cp/'latest.pt',reason)
    result={'stop_reason':reason,'step':step,'epoch':epoch+cursor/len(train),'arm_seconds':arm_elapsed(),
            'architecture':model.config,'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
            'total_parameters':sum(p.numel() for p in model.parameters()),
            'best_proxy':early,'selection':selection,'quality_nodes':q_done,'terminal_dev2500':terminal,
            'unique_train_seen':int((seen>0).sum()),'test_payload_reads':0,
            'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0}
    save(folder/'results.json',result);log({'phase':'finished',**result})
    del model,optimizer,scheduler
    if torch.cuda.is_available():torch.cuda.empty_cache()
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--resume',action='store_true')
    p.add_argument('--variants',nargs='+',choices=VARIANTS,default=list(VARIANTS[:2]))
    p.add_argument('--seed',type=int,default=0);p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--device',default='cuda');p.add_argument('--threads',type=int,default=2)
    p.add_argument('--max-epochs',type=int,default=30);p.add_argument('--budget-seconds',type=float,default=14400.)
    p.add_argument('--arm-seconds',type=float,default=5400.)
    args=p.parse_args();begin=time.monotonic();run=run_path(args.run_dir)
    if args.batch_size<1 or args.threads<1 or args.max_epochs<1 or args.arm_seconds<=0 or args.budget_seconds<=0:
        p.error('Positive sizes and budgets required')
    if len(set(args.variants))!=len(args.variants):p.error('Duplicate variants')
    if sha(OLD_CHECKPOINT)!=OLD_CHECKPOINT_SHA:raise ValueError('Frozen initialization changed')
    torch.set_num_threads(args.threads);torch.use_deterministic_algorithms(True)
    if args.device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    manifest=workset_manifest()
    settings={'variants':args.variants,'seed':args.seed,'batch_size':args.batch_size,'threads':args.threads,
              'device_requested':args.device,'max_epochs':args.max_epochs,'budget_seconds':args.budget_seconds,
              'arm_seconds':args.arm_seconds,'policy':POLICY,'geometry_weights':WEIGHTS,'aligned_weights':ALIGNED_WEIGHTS,
              'initial_checkpoint':str(OLD_CHECKPOINT),'initial_checkpoint_sha256':OLD_CHECKPOINT_SHA,
              'initialization':'same retained E7 model weights; fresh optimizer/scheduler and new E9 budget for each arm',
              'trainable_scope':{name:'center module only; original backbone and shape frozen' if name=='center_refiner' else 'all retained attention parameters' for name in args.variants},
              'workset_sha256':manifest['manifest_sha256'],'workset_counts':manifest['counts'],'target_Q_mean':1.20,
              'test_payload_reads':0,**provenance()}
    if args.resume:
        old=json.loads((run/'config.json').read_text(encoding='utf-8'))
        ignored={'command','created_utc','git_status','git_head'}
        for key in settings:
            if key not in ignored and settings[key]!=old.get(key):raise ValueError('Resume configuration/source mismatch: '+key)
    else:
        run.mkdir(parents=True,exist_ok=False);save(run/'config.json',settings);snapshot(run,settings['source_sha256'])
    prior=json.loads((run/'status.json').read_text()).get('cumulative_seconds',0.) if (run/'status.json').exists() else 0.
    if args.resume:
        for path in run.glob('*/checkpoints/latest.pt'):
            prior=max(prior,float(torch.load(path,map_location='cpu',weights_only=False)['cumulative_seconds']))
    elapsed=lambda:prior+time.monotonic()-begin
    requested=[False]
    def handler(*unused):requested[0]=True
    signal.signal(signal.SIGINT,handler);signal.signal(signal.SIGTERM,handler)
    user_stop=lambda:requested[0] or (run/'STOP').exists()
    stop=lambda:user_stop() or elapsed()>=args.budget_seconds
    append(run/'invocations.jsonl',{'phase':'start','resume':args.resume,'pid':os.getpid(),'command':settings['command'],'prior_seconds':prior})
    failed=False
    try:
        tr=LayoutReader(manifest,'train');dr=LayoutReader(manifest,'dev')
        train=load_samples(tr,stop);dev=load_samples(dr,stop)
        tmap={v:i for i,v in enumerate(tr.members)};vmap={v:i for i,v in enumerate(dr.members)}
        ti=[tmap[v] for v in member_pairs(manifest,'train_monitor')];vi=[vmap[v] for v in member_pairs(manifest,'dev_monitor')]
        size=sum(v.numel()*v.element_size() for rows in (train,dev) for r in rows for v in r.values() if torch.is_tensor(v))
        if size>8*1024**3:raise MemoryError('Total feature cache exceeds 8 GiB')
        save(run/'data_audit.json',{'train':len(train),'dev':len(dev),'train_monitor':len(ti),'dev_monitor':len(vi),
             'feature_bytes':size,'test_payload_reads':0,'split_file_sha256':sha(ROOT/'protocol/split_v1.json')})
        initial=torch.load(OLD_CHECKPOINT,map_location='cpu',weights_only=False)
        results={}
        for name in args.variants:
            if stop():break
            previous=run/name/'results.json'
            if args.resume and previous.exists():
                value=json.loads(previous.read_text())
                if value['stop_reason'] in ('epoch_cap','validation_early_stop','arm_budget'):
                    results[name]=value;continue
            results[name]=train_arm(name,run/name,train,dev,ti,vi,initial,args,manifest,elapsed,stop,user_stop)
            if results[name]['stop_reason'] in ('user_stop','global_budget'):break
        save(run/'results.json',{'status':'user_stop' if user_stop() else 'resource_budget' if elapsed()>=args.budget_seconds else 'completed',
             'variants':results,'cumulative_seconds':elapsed(),'test_payload_reads':0,
             'claim':'No target claim unless a complete dev100 evaluation attains Q<1.20; seed-only selection is a retained control.'})
        from .summarize import write_report
        write_report(run)
    except InterruptedError:
        save(run/'results.json',{'status':'stopped_during_preparation','cumulative_seconds':elapsed(),'test_payload_reads':0})
    except BaseException as exc:
        failed=True
        save(run/'error.json',{'type':type(exc).__name__,'message':str(exc),'cumulative_seconds':elapsed(),
             'recovery':'Resume from latest completed checkpoint; in-flight step may be lost.'})
        save(run/'results.json',{'status':'error','cumulative_seconds':elapsed(),'test_payload_reads':0})
        raise
    finally:
        save(run/'status.json',{'phase':'error' if failed else 'stopped' if stop() else 'finished','cumulative_seconds':elapsed(),'test_payload_reads':0})
        append(run/'invocations.jsonl',{'phase':'end','cumulative_seconds':elapsed(),'user_stop':user_stop()})


if __name__=='__main__':main()
