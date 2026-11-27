"""Resumable fixed-workset distillation; long runs are user-launched only."""
import argparse,json,os,random,signal,sys,time,uuid,zipfile,subprocess
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from .io import ROOT,HERE,sha,save
from .student import Student,CHECKPOINT_SHA256
from .data import load_bank,collate,monitor_indices
from .loss import losses

def sources():return {p.relative_to(HERE).as_posix():sha(p) for p in sorted(HERE.rglob('*.py')) if 'runs' not in p.relative_to(HERE).parts}
def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
def rng_state():return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
    'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
def restore_rng(r):
    random.setstate(r['python']);np.random.set_state(r['numpy']);torch.set_rng_state(r['torch'])
    if r['cuda'] is not None and torch.cuda.is_available():torch.cuda.set_rng_state_all(r['cuda'])
def atomic_checkpoint(path,state):
    temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:torch.save(state,temporary);os.replace(temporary,path)
    finally:
        if temporary.exists():temporary.unlink()

@torch.no_grad()
def evaluate(model,samples,indices,device,structure,batch_size=16):
    model.eval();totals={};loss_sums={};count=0
    for start in range(0,len(indices),batch_size):
        selected=indices[start:start+batch_size];g,t=collate(samples,selected,device)
        value,metrics=losses(model(g,structure),g,t,structure)
        for k,v in metrics.items():totals[k]=totals.get(k,0)+v
        for k,v in value.items():loss_sums[k]=loss_sums.get(k,0)+float(v)*len(selected)
        count+=len(selected)
    result={k:v/count for k,v in loss_sums.items()}
    result['center_mse']=totals['center_sq_sum']/max(1,totals['center_count'])
    result['ratio_mse']=totals['ratio_sq_sum']/max(1,totals['ratio_count'])
    result['geometry']=result['center_mse']+result['ratio_mse']
    result.update(center_rmse=result['center_mse']**.5,logratio_rmse=result['ratio_mse']**.5,count=count)
    for name in ('contact','boundary','mib'):
        if name+'_tp' in totals:
            tp,fp,fn=(totals[name+'_'+k] for k in ('tp','fp','fn'))
            result[name+'_precision']=tp/max(1,tp+fp);result[name+'_recall']=tp/max(1,tp+fn)
            result[name+'_tp']=tp;result[name+'_fp']=fp;result[name+'_fn']=fn
    return result

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--train-bank',type=Path,required=True);ap.add_argument('--dev-bank',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--arm',choices=['geometry','structure'],required=True)
    ap.add_argument('--device',default='cuda');ap.add_argument('--max-seconds',type=float,default=14400.)
    ap.add_argument('--max-epochs',type=int,default=200);ap.add_argument('--batch-size',type=int,default=16)
    ap.add_argument('--seed',type=int,default=0);ap.add_argument('--resume',action='store_true')
    ap.add_argument('--implementation-check',action='store_true',help='Five pilot cases only; never a formal experiment')
    args=ap.parse_args();begin=time.perf_counter();seed_all(args.seed)
    if args.max_seconds<=0 or args.max_epochs<=0 or args.batch_size<=0:raise ValueError('Positive budgets required')
    if args.device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; choose --device cpu explicitly')
    if args.resume:
        latest=args.output/'checkpoints/latest.pt'
        checkpoint=torch.load(latest,map_location='cpu',weights_only=False) if latest.exists() else None
        if checkpoint and checkpoint['finished']:print('Training already finished:',checkpoint['status']);return
        (args.output/'checkpoints').mkdir(parents=True,exist_ok=True)
    else:
        args.output.mkdir(parents=True,exist_ok=False);(args.output/'checkpoints').mkdir();checkpoint=None
    save(args.output/'progress.json',{'status':'preparing','training_started':False})
    if args.implementation_check:
        if args.max_epochs>2 or args.max_seconds>120:raise ValueError('Implementation checks are capped at 2 epochs / 120s')
        train,train_id,quality=load_bank(args.train_bank,'train',smoke=True)
        dev,dev_id,dev_quality=train,train_id,quality;mon=list(range(5))
    else:
        train,train_id,quality=load_bank(args.train_bank,'train')
        dev,dev_id,dev_quality=load_bank(args.dev_bank,'dev_monitor');mon=monitor_indices(train)
    structure=args.arm=='structure'
    identity={'sources':sources(),'checkpoint_sha256':CHECKPOINT_SHA256,'train_bank':train_id,'dev_bank':dev_id,
      'split_sha256':sha(ROOT/'protocol/split_v1.json'),'workset_sha256':sha(ROOT/'protocol/workset_v2.json'),
      'arm':args.arm,'device':args.device,'seed':args.seed,'max_seconds':args.max_seconds,'max_epochs':args.max_epochs,
      'batch_size':args.batch_size,'lr':1e-4,'weight_decay':1e-4,'clip_norm':1.,'min_delta':1e-5,
      'lr_bad_epochs':4,'early_bad_epochs':12,'min_lr':1e-6,'teacher_weight':1.,'reference_weight':0.,'score_proxy_weight':0.,
      'structure_weight':.1,'monitor':'dev_geometry','train_count':len(train),'dev_monitor_count':len(dev),'local_test_read':False,
      'implementation_check':args.implementation_check,'shared_train_dev_for_implementation_check':args.implementation_check}
    identity['torch_version']=str(torch.__version__)
    if checkpoint and checkpoint['identity']!=identity:raise ValueError('Resume source/labels/budget identity mismatch')
    if args.resume and (args.output/'config.json').exists() and json.loads((args.output/'config.json').read_text(encoding='utf-8'))['identity']!=identity:
        raise ValueError('Resume preparation identity mismatch')
    if checkpoint is None:
        save(args.output/'config.json',{'identity':identity,'command':[sys.executable,*sys.argv],
          'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
          'git_status':subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True).strip()})
        with zipfile.ZipFile(args.output/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
            for name in identity['sources']:z.write(HERE/name,name)
    def quality_summary(rows):return {'count':len(rows),'improved':sum(r['teacher_legal'] and (not r['reference_legal'] or r['delta'] < -1e-9) for r in rows),
      'worse':sum(r['reference_legal'] and r['delta']>1e-9 for r in rows),'rows':rows}
    save(args.output/'label_quality.json',{'train':quality_summary(quality),'dev_monitor':quality_summary(dev_quality),
      'scope':'Same neutral-runtime official Q; all teacher targets retained, original references are audit-only.'})
    model=Student().to(args.device);model.initialize()
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=.5,patience=3,threshold=1e-5,threshold_mode='abs',min_lr=1e-6)
    progress={'epoch':0,'cursor':0,'order':None,'step':0,'seen':np.zeros(len(train),dtype=bool),
      'best':float('inf'),'bad_epochs':0,'history':[]}
    prior=0.
    if checkpoint:
        model.load_state_dict(checkpoint['model']);optimizer.load_state_dict(checkpoint['optimizer']);scheduler.load_state_dict(checkpoint['scheduler'])
        progress=checkpoint['progress'];prior=checkpoint['elapsed_seconds'];restore_rng(checkpoint['rng'])
    elapsed=lambda:prior+time.perf_counter()-begin
    requested=[False]
    signal.signal(signal.SIGINT,lambda *_:requested.__setitem__(0,True))
    def stopped():return requested[0] or (args.output/'STOP').exists()
    def store(status,finished=False,best=False):
        state={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
          'rng':rng_state(),'progress':progress,'elapsed_seconds':elapsed(),'identity':identity,'arm':args.arm,
          'finished':finished,'status':status}
        atomic_checkpoint(args.output/'checkpoints/latest.pt',state)
        if best:atomic_checkpoint(args.output/'checkpoints/best.pt',state)
        save(args.output/'progress.json',{'status':status,'finished':finished,'elapsed_seconds':elapsed(),
          'epoch':progress['epoch'],'batch_cursor':progress['cursor'],'step':progress['step'],
          'best_dev_geometry':progress['best'],'bad_epochs':progress['bad_epochs'],'seen_count':int(progress['seen'].sum()),
          'history':progress['history'],'training_started':progress['step']>0,'local_test_read':False})
    def monitor(initial=False):
        val=evaluate(model,dev,list(range(len(dev))),args.device,structure,args.batch_size)
        tr=evaluate(model,train,mon,args.device,structure,args.batch_size)
        seen=progress['seen'][mon];improved=val['geometry']<progress['best']-1e-5
        if improved:progress['best']=val['geometry'];progress['bad_epochs']=0
        elif not initial:progress['bad_epochs']+=1
        scheduler.step(val['geometry'])
        row={'epoch':progress['epoch'],'step':progress['step'],'dev':val,'train_monitor':tr,
          'train_monitor_seen_count':int(seen.sum()),'lr':optimizer.param_groups[0]['lr'],'elapsed_seconds':elapsed()}
        if seen.any():row['train_seen']=tr if seen.all() else evaluate(model,train,[i for i,s in zip(mon,seen) if s],args.device,structure,args.batch_size)
        if (~seen).any():row['train_unseen']=tr if not seen.any() else evaluate(model,train,[i for i,s in zip(mon,seen) if not s],args.device,structure,args.batch_size)
        progress['history'].append(row);store('running',best=improved)
        print(f"epoch={progress['epoch']} step={progress['step']} dev_geometry={val['geometry']:.6f} center_rmse={val['center_rmse']:.4f} ratio_rmse={val['logratio_rmse']:.4f} lr={optimizer.param_groups[0]['lr']:.2g}",flush=True)
    if not checkpoint:monitor(initial=True)
    status='running'
    try:
        while progress['epoch']<args.max_epochs:
            if stopped():status='user_stop';break
            if elapsed()>=args.max_seconds:status='budget';break
            if progress['order'] is None:progress['order']=np.random.permutation(len(train));progress['cursor']=0
            model.train()
            while progress['cursor']<len(train):
                if stopped() or elapsed()>=args.max_seconds:break
                ids=progress['order'][progress['cursor']:progress['cursor']+args.batch_size]
                g,t=collate(train,ids,args.device);optimizer.zero_grad(set_to_none=True)
                values,_=losses(model(g,structure),g,t,structure)
                if not torch.isfinite(values['total']):raise RuntimeError('Nonfinite loss')
                values['total'].backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                optimizer.step();progress['seen'][ids]=True;progress['cursor']+=len(ids);progress['step']+=1
                if progress['step']%100==0:store('running')
            if progress['cursor']<len(train):status='user_stop' if stopped() else 'budget';break
            progress['epoch']+=1;progress['order']=None;progress['cursor']=0;monitor()
            if progress['bad_epochs']>=12:status='early_stop';break
        if status=='running':status='max_epochs'
    except Exception:
        store('error');raise
    store(status,finished=status in ('budget','early_stop','max_epochs'))
    print('Stopped:',status,'elapsed:',elapsed(),flush=True)

if __name__=='__main__':main()
