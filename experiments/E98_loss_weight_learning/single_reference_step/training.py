"""One registered candidate phase, copied from the validated E98 v2 trainer."""
import copy,gc,time
import numpy as np
import torch
from ..io import ROOT,HERE,save,sha,checkpoint
from ..train import rng_state,restore_rng
from ..generalization_baseline.model import GeometryModel
from ..generalization_baseline.augmentation import flags_for
from ..data import collate
from ..batches import inputs_only
from ..dynamic_weight_learning_v2.engine import CapturedStep,per_case
from .common import MODEL,read,PhaseSchedule
class Halt(Exception):
    pass

@torch.no_grad()
def monitor(model, rows, co, check):
    model.eval()
    blocks = []
    for start in range(0,len(rows),32):
        check()
        batch,_ = collate(rows[start:start+32], 'cuda')
        out = model(inputs_only(batch))
        values = per_case(out,batch,co)
        if not bool(torch.isfinite(out).all() & torch.isfinite(torch.stack(list(values.values()))).all() & (values['total'].amax()<1000)):
            raise RuntimeError('Monitoring numerical guard rejected')
        keys = list(values)
        blocks.append(torch.stack([values[k] for k in keys],-1).cpu().double().numpy())
    a = np.concatenate(blocks)
    return {**{k:float(a[:,i].mean()) for i,k in enumerate(keys)}, 'count':len(rows)}

def make_model(state, pack, co, resume=False):
    model = GeometryModel(**MODEL).cuda()
    model.load_state_dict(state['model'],strict=True)
    opt = torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.0003)
    schedule = PhaseSchedule()
    if resume:
        opt.load_state_dict(copy.deepcopy(state['optimizer']))
        schedule.load_state_dict(state['scheduler'])
    restore_rng(state['rng'])
    model.train()
    cap = CapturedStep(model,opt,pack.batch([i%len(pack.rows) for i in range(32)]),co,state['projection'])
    return model,opt,cap,schedule

def train_branch(folder, parent_path, co, round_spec, arm, pack, tr_monitor, dev, identity, check, beat):
    latest = folder/'checkpoints/latest.pt'
    best_path = folder/'checkpoints/best.pt'
    parent_sha = sha(parent_path)
    prior_elapsed = 0.
    if latest.exists():
        state = torch.load(latest,map_location='cpu',weights_only=False)
        assert state['identity']==identity and state['coefficients']==co and state['parent_sha256']==parent_sha
        if state['status']=='error':
            raise RuntimeError('Error phase needs diagnosis; no automatic retry')
        if state['finished']:
            return best_path, read(folder/'results.json')
        p = state['progress']
        prior_elapsed = state['phase_elapsed_seconds']
        resumed = True
    else:
        state = torch.load(parent_path,map_location='cpu',weights_only=False)
        p = dict(epoch=0,cursor=0,step=0,order=None,pending_monitor=True,best=None,best_epoch=None,
                 seen=np.zeros(len(pack.rows),dtype=bool),history=[])
        resumed = False
        save(folder/'registration.json',dict(parent=parent_path.relative_to(ROOT).as_posix(),
             parent_sha256=parent_sha,coefficients=co,optimizer='fresh AdamW for this candidate only; baseline frozen',
             scheduler_source='fixed own-Loss dev500',sampler_seed=round_spec['sampler_seed'],test_read=False))
    torch.set_num_threads(4)
    model,opt,cap,schedule = make_model(state,pack,co,resumed)
    del state
    started = time.perf_counter()
    reason = 'running'
    def store(status, finished=False, best=False):
        value = dict(model=model.state_dict(),model_config=MODEL,optimizer=opt.state_dict(),
             rng=rng_state(),projection=cap.projection.summary(),scheduler=schedule.state_dict(),
             progress=p,coefficients=co,identity=identity,parent_sha256=parent_sha,
             phase_elapsed_seconds=prior_elapsed+time.perf_counter()-started,
             status=status,finished=finished,optimizer_reset_at_phase_start=True,test_read=False)
        checkpoint(latest,value)
        if best:
            checkpoint(best_path,value)
        save(folder/'progress.json',dict(status=status,finished=finished,epoch=p['epoch'],step=p['step'],
             cursor=p['cursor'],seen_count=int(p['seen'].sum()),best=p['best'],best_epoch=p['best_epoch'],
             lr=schedule.lr,scheduler=schedule.state_dict(),history=p['history'],
             phase_elapsed_seconds=value['phase_elapsed_seconds'],test_read=False))
        beat(stage='training',round=round_spec['round'],arm=arm,epoch=p['epoch'],step=p['step'],
             lr=schedule.lr,seen_count=int(p['seen'].sum()))
    try:
        store('running')
        while True:
            check()
            if p['pending_monitor']:
                preserved = rng_state()
                dv = monitor(model,dev,co,check)
                tr = monitor(model,tr_monitor,co,check)
                restore_rng(preserved)
                improved = p['best'] is None or dv['total'] < p['best']-1e-5
                if improved:
                    p['best'],p['best_epoch'] = dv['total'],p['epoch']
                early = schedule.step(dv['total'],p['epoch'])
                for group in opt.param_groups:
                    group['lr'] = schedule.lr
                p['history'].append(dict(epoch=p['epoch'],step=p['step'],train_monitor=tr,
                     dev_monitor=dv,lr_next=schedule.lr,scheduler=schedule.state_dict()))
                p['pending_monitor'] = False
                store('running',best=improved)
                print('TRAIN',round_spec['round'],arm,'epoch',p['epoch'],'dev',dv['total'],
                      'train',tr['total'],'lr',schedule.lr,flush=True)
                if early or p['epoch']>=15:
                    reason = 'early_stop' if early else 'maximum_epochs'
                    break
            if p['order'] is None:
                p['order'] = np.random.default_rng(round_spec['sampler_seed']+p['epoch']).permutation(len(pack.rows))
                p['cursor'] = 0
            model.train()
            interval_start,interval_step = time.perf_counter(),p['step']
            while p['cursor'] < len(pack.rows):
                check()
                ids = p['order'][p['cursor']:p['cursor']+32]
                pack.batch(ids,out=cap.inputs)
                cap.inputs['axis_swap'].copy_(torch.as_tensor(flags_for(round_spec['sampler_seed'],p['epoch'],ids),device='cuda'))
                cap.replay()
                p['step'] += 1
                p['cursor'] += len(ids)
                p['seen'][ids] = True
                if p['step']%100==0:
                    span = time.perf_counter()-interval_start
                    throughput = 32*(p['step']-interval_step)/span
                    store('running')
                    print('UPDATE',round_spec['round'],arm,p['step'],'samples/s',throughput,flush=True)
                    interval_start,interval_step = time.perf_counter(),p['step']
            p['epoch'] += 1
            p['cursor'],p['order'],p['pending_monitor'] = 0,None,True
            store('monitor_pending')
        summary = dict(status=reason,epochs=p['epoch'],updates=p['step'],best_epoch=p['best_epoch'],
             best_dev_loss=p['best'],best_checkpoint=best_path.relative_to(ROOT).as_posix(),
             best_sha256=sha(best_path),phase_elapsed_seconds=prior_elapsed+time.perf_counter()-started,
             coefficient_sum=sum(co.values()),test_read=False)
        save(folder/'results.json',summary)
        store(reason,finished=True)
        return best_path,summary
    except Halt:
        store('user_stop')
        raise
    except BaseException:
        store('error')
        raise
    finally:
        del model,opt,cap
        gc.collect()
        torch.cuda.empty_cache()
