"""Cached scratch controls; validation never calls a layout postprocessor."""
import argparse
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
import numpy as np
import torch
from datasets.reference_layouts import ReferenceLayouts, REGISTRY, registry
from datasets.paths import resolved_data_paths
from .data import load_rows, collate
from .cache import identity as cache_identity
from .io import ROOT, HERE, sha, save, checkpoint as write_checkpoint, sources, snapshot, exclusive
from .loss import losses, weights
from .model import GeometryModel
from .model import decode
from .batches import inputs_only

from .arms import ARMS, model_config
from .diagnostics import geometry_checks, distribution
REVISION = 'E38.capacity.v1'
LR_LEVELS = (1e-3, 2.5e-4, 1e-4, 5e-5)


def seed_all(seed, threads=4):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['torch'])
    if state['cuda'] is not None: torch.cuda.set_rng_state_all(state['cuda'])


def selection(values):
    # The E10 geometry+occupancy proxy is fixed across all training epochs.
    return values['repair']


class PlateauController:
    """Four exact LR levels; lowest tier must complete two epochs before stopping."""
    def __init__(self, optimizer, start_epoch=15):
        self.optimizer = optimizer
        self.start_epoch = start_epoch
        self.best = None
        self.lr_bad = 0
        self.early_bad = 0
        self.drops = 0
        self.last_drop_epoch = None
        self.lowest_validated = False
        self.lowest_completed_epochs = 0

    def step(self, value, epoch):
        if epoch < self.start_epoch:
            return
        if self.drops == 3 and epoch > self.last_drop_epoch:
            self.lowest_completed_epochs = epoch-self.last_drop_epoch
            self.lowest_validated = self.lowest_completed_epochs >= 2
        delta = max(1e-5, abs(self.best)*1e-3) if self.best is not None else 0.
        meaningful = self.best is None or value < self.best-delta
        if meaningful:
            self.best, self.lr_bad, self.early_bad = value, 0, 0
        else:
            self.lr_bad += 1
            self.early_bad += 1
        if self.lr_bad >= 3 and self.drops < 3:
            self.drops += 1
            for group in self.optimizer.param_groups:
                group['lr'] = LR_LEVELS[self.drops]
            self.lr_bad = 0
            self.last_drop_epoch = epoch
            self.lowest_validated = False

    def should_stop(self):
        return self.early_bad >= 3 and self.drops == 3 and self.lowest_validated

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state):
        expected = set(self.state_dict())
        if set(state) != expected:
            raise ValueError('Incompatible LR-controller state')
        for key, value in state.items():
            setattr(self, key, value)


@torch.no_grad()
def monitor_losses(model, rows, device, arm, epoch, batch_size, *, should_stop=lambda: False, start_epoch=6, return_records=False):
    model.eval(); sums = {}; hard_legal = 0; inference_seconds = 0.; diagnostics = []
    for start in range(0, len(rows), batch_size):
        if should_stop():
            return None
        selected = [rows[i] for i in range(start, min(start+batch_size, len(rows)))]
        if str(device).startswith('cuda'): torch.cuda.synchronize(device)
        started = time.perf_counter()
        graph, targets = collate(selected, device)
        state = model(inputs_only(graph))
        positions = (decode(state, graph)*graph['scale'][:, None, None]).cpu().double().numpy()
        if str(device).startswith('cuda'): torch.cuda.synchronize(device)
        inference_seconds += time.perf_counter()-started
        values = losses(state, graph, targets, arm, 2., include_quality=True, start_epoch=start_epoch)
        for row, predicted in zip(selected, positions):
            _, checks, diagnostic = geometry_checks(predicted, row['case'].solver_inputs)
            hard_legal += int(checks['is_feasible'])
            diagnostics.append({'member': list(row['member']), **diagnostic})
        for key, value in values.items(): sums[key] = sums.get(key, 0.) + float(value)*len(selected)
    result = {key: value/len(rows) for key, value in sums.items()}
    result.update(count=len(rows), selection_proxy=selection(result), raw_hard_legal=hard_legal,
                  raw_hard_legal_rate=hard_legal/len(rows),
                  batched_inference_seconds_per_case=inference_seconds/len(rows))
    result['draft_diagnostics'] = distribution(diagnostics, ('overlap_pairs', 'overlap_area_fraction',
        'penetration_squared_per_block', 'max_local_penetration', 'largest_conflict_cluster',
        'largest_conflict_cluster_fraction'))
    if return_records:
        result['diagnostic_records'] = diagnostics
    return result


def gradient_norm(value, parameters):
    grads = torch.autograd.grad(value, parameters, retain_graph=True, allow_unused=True)
    return float(sum(g.square().sum() for g in grads if g is not None).sqrt())


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--arm', choices=tuple(ARMS), default='base96x4')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--max-hours', type=float, default=4.)
    ap.add_argument('--max-epochs', type=int, default=200)
    ap.add_argument('--batch-size', type=int, choices=(32,), default=32)
    ap.add_argument('--scheduler-start-epoch', type=int, choices=(15,), default=15)
    ap.add_argument('--resume', action='store_true')
    return ap


def make_identity(args, implementation_check=False):
    if registry().get('pilot') and not implementation_check:
        raise ValueError('Pilot data cannot be used for formal training')
    config = model_config(args.arm)
    parameter_count = sum(p.numel() for p in GeometryModel(**config).parameters())
    return {'revision': REVISION, 'arm': args.arm, 'initialization': 'random; preserved on explicit continuation',
        **config, 'parameters': parameter_count,
        'sources': sources(), 'registry_sha256': sha(REGISTRY), 'paths_sha256': sha(ROOT/'protocol/paths.json'),
        'cache_identity': cache_identity(), 'seed': args.seed, 'device': args.device,
        'torch': str(torch.__version__), 'numpy': np.__version__, 'cuda': torch.version.cuda,
        'device_name': torch.cuda.get_device_name(args.device) if args.device.startswith('cuda') else 'cpu',
        'max_seconds': args.max_hours*3600, 'max_epochs': args.max_epochs, 'batch_size': args.batch_size,
        'lr': 1e-3, 'weight_decay': 1e-4, 'clip_norm': 1., 'monitor_interval_epochs': 1,
        'schedule': 'E10 geometry: persistent center1 shape.25 relative.25; geometry terms ramp over first2 epochs; no anchor decay',
        'selection': 'dev500 arithmetic mean of E17 geometry + occupancy; fixed full weights for every epoch; no solver calls',
        'lr_patience_checks': 3, 'early_patience_checks': 3,
        'scheduler_start_epoch': getattr(args, 'scheduler_start_epoch', 15),
        'lr_levels': list(LR_LEVELS), 'required_lr_drops': 3,
        'meaningful_relative_delta': 1e-3, 'meaningful_absolute_delta': 1e-5,
        'cpu_threads': 4, 'lowest_lr_requires_completed_epochs': 2,
        'training_postprocessing': False, 'validation_postprocessing': False,
        'implementation_check': implementation_check, 'test_read': False, 'tail_test_read': False}


def fresh_progress(count):
    return {'epoch': 0, 'cursor': 0, 'order': None, 'step': 0, 'seen': np.zeros(count, dtype=bool),
        'best_proxy': None, 'best_epoch': None, 'bad_checks': 0, 'pending_monitor': True,
        'history': [], 'gradient_history': [], 'batch_history': [], 'full_dev_complete': False}


def run(args, **kwargs):
    if not Path(args.output).resolve().is_relative_to(HERE/'runs'):
        raise ValueError('Output must stay in E38/runs')
    with exclusive(Path(args.output).with_suffix('.lock')):
        return _run(args, **kwargs)


def _run(args, *, implementation_check=False, stop_after_steps=None, check_rows=None):
    begin = time.perf_counter(); args.output = args.output.resolve()
    if not args.output.is_relative_to(HERE/'runs'): raise ValueError('Output must stay in E38/runs')
    if min(args.max_hours, args.max_epochs, args.batch_size) <= 0: raise ValueError('Positive budgets required')
    if args.max_hours > 4 or args.max_epochs > 200 or args.batch_size != 32:
        raise ValueError('E38 is limited to four hours/200 epochs per arm, batch32')
    start_epoch = getattr(args, 'curriculum_start_epoch', 6)
    scheduler_start = getattr(args, 'scheduler_start_epoch', 15)
    if min(start_epoch, scheduler_start) < 1: raise ValueError('Positive schedule epochs required')
    if args.device.startswith('cuda') and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    seed_all(args.seed)
    identity = make_identity(args, implementation_check)
    seed_all(args.seed)
    model = GeometryModel(identity['width'], identity['layers']).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = PlateauController(optimizer, start_epoch=scheduler_start)
    previous = None; prior = 0.
    if args.resume:
        previous = torch.load(args.output/'checkpoints/latest.pt', map_location='cpu', weights_only=False)
        if previous['identity'] != identity: raise ValueError('Resume source/data/budget/device identity mismatch')
        if previous['finished']:
            print('Already finished:', previous['status'], flush=True); return
        if (args.output/'STOP').exists(): raise ValueError('Remove STOP before resuming')
        model.load_state_dict(previous['model']); optimizer.load_state_dict(previous['optimizer'])
        scheduler.load_state_dict(previous['scheduler']); prior = previous['elapsed_seconds']
        if (args.output/'budget.json').exists():
            prior = max(prior, json.loads((args.output/'budget.json').read_text(encoding='utf-8'))['elapsed_seconds'])
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        save(args.output/'config.json', {'identity': identity, 'command': [sys.executable, *sys.argv],
            'commit': subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'git_status': subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True).strip(),
            'data_paths': resolved_data_paths(), 'shared_cache_preparation': 'Read-only frozen E25/E17 train/dev tensors',
            'timing': 'Cumulative training-stage wall includes mmap load/validation; inherited old wall is not reset'})
        snapshot(args.output/'source_snapshot.zip', identity)
    elapsed = lambda: prior+time.perf_counter()-begin
    calls_path = args.output/'invocations.json'
    calls = json.loads(calls_path.read_text(encoding='utf-8')) if calls_path.exists() else []
    calls.append({'argv': [sys.executable,*sys.argv], 'resume': args.resume, 'prior_seconds': prior,
                  'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
    save(calls_path,calls)
    train_meta, dev_meta = ReferenceLayouts('train'), ReferenceLayouts('dev')
    mon_meta = ReferenceLayouts('train_monitor500')
    member_map = {tuple(r['member']):i for i,r in enumerate(train_meta.records)}
    if {m[0] for m in member_map} & {r['member'][0] for r in dev_meta.records}:
        raise ValueError('Student source shards overlap')
    if implementation_check:
        if check_rows is None: raise ValueError('Supply the five preloaded check rows')
        train, dev = check_rows; monitor_ids = list(range(len(train)))
    else:
        train = load_rows('train'); dev = load_rows('dev_monitor500')
        monitor_ids = [member_map[tuple(r['member'])] for r in mon_meta.records]
    train_monitor = [train[i] for i in monitor_ids]
    progress = previous['progress'] if previous and previous['progress'] is not None else fresh_progress(len(train))
    if previous: restore_rng(previous['rng'])
    requested = [False]
    old_handler = signal.signal(signal.SIGINT, lambda *_: requested.__setitem__(0,True))
    def stopped():
        return requested[0] or (args.output/'STOP').exists() or (HERE/'runs/STOP').exists() or (stop_after_steps is not None and progress['step']>=stop_after_steps)
    def exhausted(): return elapsed()>=identity['max_seconds']
    def heartbeat(): save(args.output/'budget.json', {'elapsed_seconds':elapsed()})
    def store(status, finished=False, best=False):
        heartbeat()
        state = {'model':model.state_dict(), 'model_config':model_config(args.arm),
            'optimizer':optimizer.state_dict(), 'scheduler':scheduler.state_dict(), 'rng':rng_state(),
            'progress':progress, 'identity':identity, 'elapsed_seconds':elapsed(), 'status':status, 'finished':finished}
        write_checkpoint(args.output/'checkpoints/latest.pt',state)
        if best: write_checkpoint(args.output/'checkpoints/best.pt',state)
        save(args.output/'progress.json', {'status':status,'finished':finished,'epoch':progress['epoch'],
            'cursor':progress['cursor'],'step':progress['step'],'seen_count':int(progress['seen'].sum()),
            'elapsed_seconds':elapsed(),'best_proxy':progress['best_proxy'],'best_epoch':progress['best_epoch'],
            'bad_checks':progress['bad_checks'],'history':progress['history'],'gradients':progress['gradient_history'],
            'lr_drops':scheduler.drops,'lr_tier':scheduler.drops+1,'lr_bad_checks':scheduler.lr_bad,
            'lowest_lr_validated':scheduler.lowest_validated,
            'full_dev_complete':progress['full_dev_complete'],'postprocessing_calls':0,
            'implementation_check':implementation_check,'test_read':False,'tail_test_read':False})
    def monitor():
        if stopped() or exhausted(): return False
        epoch = progress['epoch']
        val = monitor_losses(model,dev,args.device,args.arm,2.,args.batch_size,
                             should_stop=lambda: stopped() or exhausted(), start_epoch=start_epoch)
        if val is None: return False
        tr = monitor_losses(model,train_monitor,args.device,args.arm,2.,args.batch_size,
                            should_stop=lambda: stopped() or exhausted(), start_epoch=start_epoch)
        if tr is None: return False
        seen = progress['seen'][monitor_ids]
        record = {'epoch':epoch,'step':progress['step'],'dev_monitor':val,'train_monitor':tr,
            'train_seen_count':int(seen.sum()),'weights':weights(args.arm,max(1,epoch),start_epoch),
            'lr':optimizer.param_groups[0]['lr'],'elapsed_seconds':elapsed(),'postprocessing_calls':0}
        for label,mask in (('train_seen',seen),('train_unseen',~seen)):
            if mask.any():
                record[label] = tr if mask.all() else monitor_losses(model,
                    [row for row,flag in zip(train_monitor,mask) if flag],args.device,args.arm,2.,args.batch_size,
                    should_stop=lambda: stopped() or exhausted(), start_epoch=start_epoch)
                if record[label] is None: return False
        improved = progress['best_proxy'] is None or val['selection_proxy']<progress['best_proxy']-1e-5
        if improved: progress['best_proxy'],progress['best_epoch'] = val['selection_proxy'],epoch
        scheduler.step(val['selection_proxy'],epoch)
        progress['bad_checks'] = scheduler.early_bad
        record.update(lr_next=optimizer.param_groups[0]['lr'],lr_drops=scheduler.drops,
                      lr_bad_checks=scheduler.lr_bad,early_bad_checks=scheduler.early_bad,
                      meaningful_best=scheduler.best)
        progress['history'].append(record);progress['pending_monitor']=False
        store('running',best=improved)
        print(f'epoch={epoch} step={progress["step"]} dev_proxy={val["selection_proxy"]:.6f} supervision={val["supervision"]:.6f} lr={optimizer.param_groups[0]["lr"]:.2g} (no postprocessing)',flush=True)
        return True
    status = 'running'
    try:
        save(args.output/'panels.json', {'train_count':len(train),'dev_monitor_count':len(dev),
            'train_monitor_members':[list(r['member']) for r in train_monitor],
            'dev_monitor_members':[list(dev[i]['member']) for i in range(len(dev))],
            'test_read':False,'tail_test_read':False,'postprocessing_calls':0})
        if not previous: store('prepared'); monitor()
        while progress['epoch']<args.max_epochs or progress['pending_monitor']:
            if scheduler.should_stop() and not progress['pending_monitor']:
                status='early_stop';break
            if stopped() or exhausted(): status='user_stop' if stopped() else 'budget';break
            if progress['pending_monitor']:
                if not monitor(): status='user_stop' if stopped() else 'budget';break
                if scheduler.should_stop(): status='early_stop';break
                continue
            if progress['order'] is None:
                progress['order']=np.random.default_rng(args.seed+progress['epoch']).permutation(len(train));progress['cursor']=0
            model.train()
            while progress['cursor']<len(train):
                if stopped() or exhausted():break
                ids=progress['order'][progress['cursor']:progress['cursor']+args.batch_size]
                graph,targets=collate([train[i] for i in ids],args.device)
                optimizer.zero_grad(set_to_none=True)
                values=losses(model(inputs_only(graph)),graph,targets,args.arm,progress['epoch']+progress['cursor']/len(train),
                              include_quality=False, start_epoch=start_epoch)
                if not torch.isfinite(torch.stack(list(values.values()))).all():raise RuntimeError('Nonfinite loss')
                if progress['cursor']==0:
                    params=list(model.parameters());diag={'epoch':progress['epoch']+1,'weights':weights(args.arm,progress['epoch']+1,start_epoch)}
                    for key in ('supervision','quality_proxy','overlap'):
                        if key in values:diag[key+'_grad_norm']=gradient_norm(values[key],params)
                    progress['gradient_history'].append(diag)
                values['total'].backward()
                norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True))
                optimizer.step();progress['seen'][ids]=True;progress['cursor']+=len(ids);progress['step']+=1
                heartbeat()
                if progress['step']%100==0:
                    progress['batch_history'].append({'step':progress['step'],'epoch':progress['epoch']+1,
                        'losses':{k:float(v.detach()) for k,v in values.items()},'grad_norm':norm,'elapsed_seconds':elapsed()})
                    store('running');print(f'step={progress["step"]} epoch={progress["epoch"]+1} samples={progress["cursor"]}/{len(train)} total={float(values["total"].detach()):.5f}',flush=True)
            if progress['cursor']<len(train):status='user_stop' if stopped() else 'budget';break
            progress['epoch']+=1;progress['order']=None;progress['cursor']=0;progress['pending_monitor']=True
            store('monitor_pending')
        if status=='running':status='max_epochs'
        store(status,finished=status in ('budget','max_epochs','early_stop'))
        save(args.output/'results.json',{'status':status,'elapsed_seconds':elapsed(),'steps':progress['step'],
            'epochs':progress['epoch'],'best_epoch':progress['best_epoch'],'best_proxy':progress['best_proxy'],
            'full_dev_complete':progress['full_dev_complete'],'postprocessing_calls':0,
            'test_read':False,'tail_test_read':False,'implementation_check':implementation_check,
            'final_postprocessed_Q':'not run; one frozen best assessment is a separate stage'})
        print('Stopped:',status,'wall hours:',elapsed()/3600,'best epoch:',progress['best_epoch'],flush=True)
    except BaseException:
        store('error');raise
    finally:signal.signal(signal.SIGINT,old_handler)


if __name__=='__main__':run(parser().parse_args())
