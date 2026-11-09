"""E94 post-training LR reset1e-3 from P36; inherited Adam, fixed F Loss."""
import argparse,copy,gc,json,random,signal,subprocess,sys,threading,time,traceback
from pathlib import Path
import numpy as np
import torch
from .io import ROOT,HERE,sha,save,checkpoint,sources,snapshot,exclusive
from .arms import model_config
from .model import GeometryModel
from .batches import inputs_only
from .data import load_rows,collate,register_resident
from .dataset_view import ReferenceLayouts,REGISTRY
from .resident import ResidentRows
from .cuda_step import CapturedStep
from .loss import per_case
from .loss_config import configuration,COEFFICIENTS
from .schedule import PlateauController,LR
from .budget import branch_charged_seconds, historical_charged_seconds
from .initialization import load_parent,verify as verify_parent

RUN=None

def run_directory(arm):
    if arm!='F':raise ValueError('Only F supported')
    return HERE/'runs/20261101_lr1e3_seed0'
REVISION='E94.P36.inherited_Adam.lr1e3.v1'

def seed_all(seed=0):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)

def rng_state():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state_all()}

def restore_rng(rng):
    random.setstate(rng['python']);np.random.set_state(rng['numpy']);torch.set_rng_state(rng['torch']);torch.cuda.set_rng_state_all(rng['cuda'])

def optimizer_for(model,phase):
    return torch.optim.AdamW(model.parameters(),lr=LR[phase][0],weight_decay=.0003)

def fresh_progress(size,phase,parent):
    return {'phase':phase,'phase_epoch':0,'epoch':parent['data_epoch'],'cursor':0,'step':parent['model_step'],'phase_step':0,
            'order':None,'seen':parent['seen'].copy(),'phase_seen':np.zeros(size,dtype=bool),'pending_monitor':True,
            'best':{phase:None},'best_epoch':{phase:None},
            'history':[],'batch_history':[],'handoff':{k:v for k,v in parent.items() if k not in ('rng','seen','projection')}}

def numerical_check(value,output=None):
    finite=torch.isfinite(torch.stack(list(value.values()))).all()
    if output is not None:finite=finite & torch.isfinite(output).all()
    if not bool(finite & (value['total'].amax()<1000.)):
        raise RuntimeError('Per-case monitoring guard: nonfinite or total >=1000')

@torch.no_grad()
def monitor(model,rows,phase,device,halt=lambda:False,beat=lambda **kw:None):
    model.eval();sums={};totals=[];peaks=[];inference=0.
    for start in range(0,len(rows),32):
        if halt():return None
        selected=rows[start:start+32]
        batch,_=collate(selected,device)
        torch.cuda.synchronize();begin=time.perf_counter()
        output=model(inputs_only(batch));values=per_case(output,batch,phase)
        numerical_check(values,output)
        torch.cuda.synchronize();inference+=time.perf_counter()-begin
        keys=list(values)
        block=torch.stack([values[k] for k in keys],-1).cpu().double().numpy()
        for j,key in enumerate(keys):sums[key]=sums.get(key,0.)+block[:,j].sum()
        totals.extend(values['total'].cpu().tolist())
        peaks.extend(output.abs().amax((1,2)).cpu().tolist())
        beat(monitor_completed=min(start+32,len(rows)),monitor_count=len(rows))
    result={k:float(v/len(rows)) for k,v in sums.items()}
    worst=int(np.argmax(totals))
    result.update(count=len(rows),p95=float(np.quantile(totals,.95)),p99=float(np.quantile(totals,.99)),
                  max=float(max(totals)),worst_member=list(rows[worst]['member']),output_max=float(max(peaks)),
                  per_case_total=totals,per_case_output_max=peaks,inference_seconds=inference,
                  F_diagnostic=result['reference']+COEFFICIENTS['F']['overlap']*result['overlap_total']+COEFFICIENTS['F']['task']*result['task'],
                  weighted_contributions={'reference':result['reference'],
                    'overlap':COEFFICIENTS[phase]['overlap']*result['overlap_total'],
                    'task':COEFFICIENTS[phase]['task']*result['task']})
    return result

def build_capture(model,optimizer,pack,projection_meta=None):
    model.train()
    sample=pack.batch([i%len(pack.rows) for i in range(32)])
    captured=CapturedStep(model,optimizer,sample,optimizer._e92_phase)
    if projection_meta:
        captured.projection.steps=projection_meta['steps']
        captured.projection.clipped.copy_(torch.tensor(projection_meta['clipped_steps_per_group'],device=pack.device))
        captured.projection.last_scales.copy_(torch.tensor(projection_meta['last_scales'],device=pack.device))
    return captured



def parser():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--arm',choices=('F',),default='F')
    ap.add_argument('--resume',action='store_true')
    return ap

def run(args):
    global RUN
    RUN=run_directory(args.arm)
    from .preflight import verify
    verify()
    if (HERE/'runs/STOP').exists() or (RUN/'STOP').exists():raise ValueError('STOP present')
    with exclusive(HERE/'runs/training.lock'):
        started=time.perf_counter()
        prior=json.loads((RUN/'budget.json').read_text())['elapsed_seconds'] if args.resume and (RUN/'budget.json').exists() else 0.
        try:return _run(args)
        except BaseException as error:
            # Include startup/load/capture errors, not only errors in the main loop.
            if RUN.exists():
                path=RUN/'results.json'
                result=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
                result.update(status='error',error=repr(error),test_read=False)
                save(path,result)
            raise
        finally:
            if RUN.exists():
                path=RUN/'budget.json'
                old=json.loads(path.read_text())['elapsed_seconds'] if path.exists() else 0.
                save(path,{'elapsed_seconds':max(old,prior+time.perf_counter()-started)})

def _run(args):
    begin=time.perf_counter();plan=json.loads((HERE/'plan.json').read_text(encoding='utf-8'))
    seed_all(plan['seed'])
    verify_parent()
    identity={'revision':REVISION,'arm':args.arm,'plan':plan,'sources':sources(),'registry_sha256':sha(REGISTRY),
              'paths_sha256':sha(ROOT/'protocol/paths.json'),'torch':str(torch.__version__),
              'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(0),'numpy':np.__version__}
    prior=0.;previous=None
    if args.resume:
        previous=torch.load(RUN/'checkpoints/latest.pt',map_location='cpu',weights_only=False)
        if previous['identity']!=identity:raise ValueError('Resume identity mismatch; explicit migration required')
        if previous['finished']:raise ValueError('Completed run cannot be automatically resumed')
        if previous['status']=='error':raise ValueError('Error run requires diagnosed, explicitly registered recovery')
        prior=max(previous['elapsed_seconds'],json.loads((RUN/'budget.json').read_text())['elapsed_seconds'])
    else:
        RUN.mkdir(parents=True,exist_ok=False)
        save(RUN/'config.json',{'identity':identity,'command':[sys.executable,*sys.argv],
             'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
             'precision':'FP32; no AMP; unchanged TF32 defaults','threads':4,'train_size':32000,'test_read':False})
        snapshot(RUN/'source_snapshot.zip',identity)
    elapsed=lambda:prior+time.perf_counter()-begin
    others=branch_charged_seconds(exclude=RUN)
    model=GeometryModel(**model_config('simple')).cuda()
    if sum(p.numel() for p in model.parameters())!=5829939 or not all(p.requires_grad for p in model.parameters()):
        raise ValueError('Model/trainability differs from registration')
    phase=args.arm;optimizer=optimizer_for(model,phase);optimizer._e92_phase=phase
    parent=load_parent(model,optimizer) if previous is None else None
    progress=previous['progress'] if previous else fresh_progress(32000,phase,parent)
    scheduler=PlateauController(optimizer,phase)
    if previous:
        model.load_state_dict(previous['model']);optimizer.load_state_dict(previous['optimizer'])
        scheduler.load_state_dict(previous['scheduler']);restore_rng(previous['rng'])
    else:restore_rng(parent['rng'])
    captured=None;current_ids=None;signal_stop=[False]
    hb_state={'status':'loading','activity':'initializing','phase':phase,'step':progress['step']}
    lock=threading.RLock();closed=threading.Event();last_hb=[0.]
    def beat(force=False,**fields):
        with lock:
            hb_state.update(fields)
            if force or elapsed()-last_hb[0]>=5.:
                now=elapsed();last_hb[0]=now
                save(RUN/'budget.json',{'elapsed_seconds':now})
                save(RUN/'heartbeat.json',{**hb_state,'elapsed_seconds':now,'global_remaining_seconds':max(0.,plan['maximum_formal_training_seconds']-others-now),
                                         'run_remaining_seconds':max(0.,plan['maximum_hours_per_arm']*3600-now),
                                         'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'test_read':False})
    def daemon():
        while not closed.wait(5.):beat(force=True)
    worker=threading.Thread(target=daemon,daemon=True);worker.start()
    old_signal=signal.signal(signal.SIGINT,lambda *_:signal_stop.__setitem__(0,True))
    def stopped():return signal_stop[0] or (RUN/'STOP').exists() or (HERE/'runs/STOP').exists()
    def exhausted():return elapsed()>=plan['maximum_hours_per_arm']*3600 or others+elapsed()>=plan['maximum_formal_training_seconds']
    def halt():return stopped() or exhausted()
    def state(status,finished=False):
        return {'model':model.state_dict(),'model_config':model_config('simple'),'optimizer':optimizer.state_dict(),
                'scheduler':scheduler.state_dict(),'rng':rng_state(),'progress':progress,'identity':identity,
                'elapsed_seconds':elapsed(),'status':status,'finished':finished,
                'graph_step_control':captured.projection.summary() if captured else (previous.get('graph_step_control') if previous else parent['projection'])}
    def store(status,finished=False,best=False):
        beat(force=True,status=status,phase=progress['phase'],phase_epoch=progress['phase_epoch'],
             epoch=progress['epoch'],step=progress['step'],cursor=progress['cursor'],lr=optimizer.param_groups[0]['lr'])
        checkpoint(RUN/'checkpoints/latest.pt',state(status,finished))
        if best:
            checkpoint(RUN/f'checkpoints/{progress["phase"]}_best.pt',state(status,finished))
        save(RUN/'progress.json',{'status':status,'phase':progress['phase'],'epoch':progress['epoch'],
             'phase_epoch':progress['phase_epoch'],'cursor':progress['cursor'],'step':progress['step'],'new_updates':progress['phase_step'],
             'seen_count':int(progress['seen'].sum()),'best':progress['best'],'best_epoch':progress['best_epoch'],
             'lr':optimizer.param_groups[0]['lr'],'scheduler':scheduler.state_dict(),'handoff':progress['handoff'],
             'history':progress['history'],'elapsed_seconds':elapsed(),'test_read':False})
    def results(status,error=None):
        save(RUN/'results.json',{'status':status,'phase':progress['phase'],'epochs':progress['epoch'],
             'phase_epoch':progress['phase_epoch'],'steps':progress['step'],'new_updates':progress['phase_step'],'best':progress['best'],
             'best_epoch':progress['best_epoch'],'elapsed_seconds':elapsed(),'error':error,
             'postprocessing_calls':0,'full_dev_complete':False,'test_read':False})
    status='running'
    try:
        store('data_preparation_pending')
        beat(force=True,activity='loading frozen32k CPU rows')
        train=load_rows('train');dev=load_rows('dev_monitor500')
        train_records=ReferenceLayouts('train').records
        lookup={tuple(r['member']):i for i,r in enumerate(train_records)}
        monitor_ids=[lookup[tuple(r['member'])] for r in ReferenceLayouts('train_monitor500').records]
        train_monitor=[train[i] for i in monitor_ids]
        sentinel_meta=json.loads((HERE/'runs/sentinels.json').read_text(encoding='utf-8'))
        sentinel_rows=[train[r['index']] for r in sentinel_meta['records']]
        if len(train)!=32000 or len(dev)!=500 or len(train_monitor)!=500:raise ValueError('Frozen cardinality mismatch')
        if {r['member'][0] for r in train} & {r['member'][0] for r in dev}:raise ValueError('Train/dev shard overlap')
        pack=ResidentRows(train,'cuda',max_nodes=120);devpack=ResidentRows(dev,'cuda',max_nodes=120)
        register_resident(pack);register_resident(devpack)
        save(RUN/'panels.json',{'train_count':32000,'updates_per_epoch':1000,
             'train_monitor_members':[list(r['member']) for r in train_monitor],
             'dev_monitor_members':[list(r['member']) for r in dev],'sentinels':sentinel_meta,'test_read':False})
        beat(force=True,activity='capturing train and projection graphs')
        captured=build_capture(model,optimizer,pack,previous.get('graph_step_control') if previous else parent['projection'])
        if captured.projection.steps!=progress['step']:raise ValueError('Projection step mismatch')
        # Check real training footprint including lazily allocated Adam moments.
        free,total=torch.cuda.mem_get_info()
        moment_reserve=sum(p.numel()*p.element_size()*2 for p in model.parameters()) if not optimizer.state else 0
        if free < moment_reserve+512*2**20:raise RuntimeError('Insufficient dedicated CUDA headroom')
        save(RUN/'resources.json',{'gpu_allocated':torch.cuda.memory_allocated(),'gpu_reserved':torch.cuda.memory_reserved(),
             'gpu_free':free,'gpu_total':total,'adam_remaining_reserve':moment_reserve,
             'full_input_resident':False,'batch_input_bytes':32*pack.row_bytes,'threads':4})
        last_save=elapsed()
        while True:
            if halt():status='user_stop' if stopped() else 'budget';break
            if progress['pending_monitor']:
                record={'phase':progress['phase'],'phase_epoch':progress['phase_epoch'],'epoch':progress['epoch'],
                        'step':progress['step'],'weights':configuration(progress['phase']),
                        'lr':optimizer.param_groups[0]['lr'],'train_seen_count':int(progress['seen'][monitor_ids].sum()),'train_seen_this_branch':int(progress['phase_seen'][monitor_ids].sum())}
                preserved=rng_state()
                for label,rows in [('dev_monitor',dev),('train_monitor',train_monitor),('sentinels',sentinel_rows)]:
                    beat(force=True,activity=label,monitor_completed=0,monitor_count=len(rows))
                    record[label]=monitor(model,rows,progress['phase'],'cuda',halt,beat)
                    if record[label] is None:break
                restore_rng(preserved)
                if any(record.get(k) is None for k in ('dev_monitor','train_monitor','sentinels')):
                    status='user_stop' if stopped() else 'budget';break
                value=record['dev_monitor']['total'];phase=progress['phase']
                improved=progress['best'][phase] is None or value<progress['best'][phase]
                if improved:
                    progress['best'][phase]=value;progress['best_epoch'][phase]=progress['phase_epoch']
                scheduler.step(value,progress['phase_epoch'])
                record.update(lr_next=optimizer.param_groups[0]['lr'],scheduler=scheduler.state_dict(),elapsed_seconds=elapsed())
                progress['history'].append(record);progress['pending_monitor']=False
                store('running',best=improved);last_save=elapsed()
                print(f'phase={phase} epoch={progress["phase_epoch"]} global_epoch={progress["epoch"]} step={progress["step"]} train={record["train_monitor"]["total"]:.6f} dev={value:.6f} maxdev={record["dev_monitor"]["max"]:.4f} sentinelmax={record["sentinels"]["max"]:.4f} lr={optimizer.param_groups[0]["lr"]:.2g} drops={scheduler.drops}',flush=True)
                continue
            if scheduler.should_stop():status='early_stop';break
            if progress['phase_epoch']>=plan['maximum_phase_epochs']:status='max_epochs';break
            if progress['order'] is None:
                progress['order']=np.random.default_rng(plan['seed']+progress['epoch']).permutation(len(train))
                progress['cursor']=0
            model.train();beat(force=True,activity='training')
            while progress['cursor']<len(train):
                if halt():break
                current_ids=progress['order'][progress['cursor']:progress['cursor']+32]
                pack.batch(current_ids,out=captured.inputs)
                values,norm=captured.replay()
                progress['seen'][current_ids]=True;progress['phase_seen'][current_ids]=True;progress['cursor']+=len(current_ids)
                progress['step']+=1;progress['phase_step']+=1
                beat(step=progress['step'],cursor=progress['cursor'])
                if elapsed()-last_save>=30.:
                    store('running');last_save=elapsed()
                if progress['step']%100==0:
                    vals=dict(zip(values,torch.stack(list(values.values())).detach().cpu().tolist()))
                    progress['batch_history'].append({'step':progress['step'],'phase':progress['phase'],
                        'phase_epoch':progress['phase_epoch']+1,'losses':vals,'max_case':float(captured.max_case),
                        'preclip_norm':float(norm),'elapsed_seconds':elapsed()})
                    print(f'step={progress["step"]} phase={progress["phase"]} epoch={progress["phase_epoch"]+1} cursor={progress["cursor"]}/32000 loss={vals["total"]:.6f}',flush=True)
            if progress['cursor']<len(train):status='user_stop' if stopped() else 'budget';break
            progress['epoch']+=1;progress['phase_epoch']+=1
            progress.update(order=None,cursor=0,pending_monitor=True)
            store('monitor_pending')
        store(status,finished=status=='early_stop');results(status)
        print('Stopped',status,'phase',progress['phase'],'steps',progress['step'],'elapsed',elapsed(),flush=True)
    except BaseException as error:
        failure={'error':repr(error),'traceback':traceback.format_exc(),'phase':progress['phase'],
                 'activity':hb_state.get('activity'),'accepted_updates':progress['step'],
                 'candidate_train_indices':current_ids.tolist() if current_ids is not None else None,
                 'optimizer_stage':'before Adam if numerical replay guard; otherwise see traceback',
                 'rng_after_failure':rng_state(),'test_read':False}
        if captured is not None:
            failure.update(per_case={k:v.detach().cpu() for k,v in captured.per_layout.items()},
                           preclip_norm=float(captured.norm),batch={k:v.detach().cpu() for k,v in captured.inputs.items()})
        checkpoint(RUN/'failure.pt',failure)
        store('error');results('error',repr(error));raise
    finally:
        closed.set();worker.join(timeout=10.)
        beat(force=True,status=status if status!='running' else 'error',activity='stopped')
        signal.signal(signal.SIGINT,old_signal)

if __name__=='__main__':run(parser().parse_args())
