"""Train one frozen-base refiner; no optimizer, repair, or official Q in monitoring."""
import argparse
import json
import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_key, '4')
import numpy as np
import torch
from datasets.reference_layouts import ReferenceLayouts, public_hard_reference
from datasets.paths import resolved_data_paths
from scoring.checks import independent_hard_checks
from .io import ROOT, HERE, sha, save, checkpoint, sources, snapshot, exclusive
from .data import BASE_CHECKPOINT, CACHE, identity as cache_identity, load_rows, collate
from .model import ARMS, DraftModel, inputs_only, trainable_state, load_trainable
from .base_model import decode
from .loss import components, loss, WEIGHTS
from .policy import PlateauController, LR_LEVELS

REVISION = 'E28.frozen_base_terminal_recurrence.v1'


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)


def random_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_random(state):
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:
        torch.cuda.set_rng_state_all(state['cuda'])


def make_model(arm, device='cpu'):
    model = DraftModel(arm=arm)
    base = torch.load(BASE_CHECKPOINT, map_location='cpu', weights_only=False)
    model.base.load_state_dict(base['model'])
    return model.to(device)


def physical_positions(state, graph, rows):
    boxes = (decode(state, graph) * graph['scale'][:, None, None]).detach().cpu().double().numpy()
    output = []
    for row, predicted in zip(rows, boxes):
        inputs = row['case'].solver_inputs; n = int(inputs[0])
        c, specs = np.asarray(inputs[5]), np.asarray(inputs[6])
        predicted = predicted[:n].copy(); locked = c[:, :2].any(1)
        predicted[locked, 2:] = specs[locked, 2:]
        predicted[c[:, 1] != 0] = specs[c[:, 1] != 0]
        output.append(predicted)
    return output


@torch.no_grad()
def monitor(model, rows, device, *, should_stop=lambda: False, seen=None):
    model.eval(); collected = {}; hard = []; block_counts = []
    for start in range(0, len(rows), 32):
        if should_stop():
            return None
        batch = rows[start:start + 32]; graph = collate(batch, device)
        state = model(inputs_only(graph)); values = components(state, graph)
        if not all(torch.isfinite(v).all() for v in values.values()):
            raise RuntimeError('Nonfinite monitor loss')
        for key, value in values.items():
            collected.setdefault(key, []).extend(value.cpu().tolist())
        for row, predicted in zip(batch, physical_positions(state, graph, batch)):
            inputs = row['case'].solver_inputs
            result = independent_hard_checks(predicted.tolist(), inputs[1], np.asarray(inputs[5]),
                public_hard_reference(inputs[1], inputs[5], inputs[6]))
            hard.append(bool(result['is_feasible']) and bool((predicted[:, :2] >= 0).all()))
            block_counts.append(int(inputs[0]))
    def summarize(mask):
        count = int(mask.sum())
        if not count:
            return None
        result = {k: float(np.asarray(v)[mask].mean()) for k, v in collected.items()}
        n = np.asarray(block_counts)[mask]; w = np.exp((n - n.max()) / 12)
        legal = np.asarray(hard)[mask]
        result.update(count=count, raw_hard_legal=int(legal.sum()), raw_hard_legal_rate=float(legal.mean()),
                      weighted_hard_legal_rate=float(np.average(legal, weights=w)),
                      weighted_physics=float(np.average(np.asarray(collected['physics'])[mask], weights=w)))
        return result
    all_rows = np.ones(len(rows), dtype=bool)
    result = summarize(all_rows)
    if seen is not None:
        result['seen'] = summarize(np.asarray(seen, dtype=bool))
        result['unseen'] = summarize(~np.asarray(seen, dtype=bool))
    return result


def identity(args, model, implementation_check=False):
    return {'revision': REVISION, 'arm': args.arm, 'model_config': {'arm': args.arm, 'width': 96},
            'base_checkpoint_sha256': sha(BASE_CHECKPOINT), 'cache_identity': cache_identity(),
            'sources': sources(), 'seed': args.seed, 'device': args.device,
            'device_name': torch.cuda.get_device_name(args.device) if args.device.startswith('cuda') else 'cpu',
            'torch': str(torch.__version__), 'numpy': np.__version__, 'cuda': torch.version.cuda,
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'frozen_parameters': sum(p.numel() for p in model.parameters() if not p.requires_grad),
            'max_seconds': args.max_hours * 3600, 'max_epochs': args.max_epochs, 'batch_size': 32,
            'lr_levels': list(LR_LEVELS), 'lr_protection_epochs': 14, 'lr_patience': 3,
            'required_drops': 3, 'minimum_completed_lowest_tier_epochs': 2,
            'meaningful_relative_delta': .001, 'meaningful_absolute_delta': 1e-5,
            'weight_decay': 1e-4, 'clip_norm': 1., 'cpu_threads': 4,
            'public_weights': WEIGHTS, 'reference_center_weight': .1, 'compatible_ratio_weight': .025,
            'selection': 'fixed equal-case mean public physics on dev_monitor500; official Q never selects best',
            'initialization': 'frozen E25 base + zero-output residual head; same random refiner initialization across arms',
            'training_postprocessing_calls': 0, 'test_read': False, 'implementation_check': implementation_check}


def run(args, *, check_rows=None, stop_after_steps=None):
    output = args.output.resolve()
    if not output.is_relative_to(HERE / 'runs') or args.batch_size != 32:
        raise ValueError('Use E28/runs and batch size 32')
    if min(args.max_hours, args.max_epochs) <= 0:
        raise ValueError('Positive budgets required')
    if os.environ.get('E16_REGISTRY'):
        raise ValueError('No pilot registry override')
    with exclusive(output.with_suffix('.lock')):
        seed_all(args.seed); begin = time.perf_counter()
        model = make_model(args.arm, args.device)
        ident = identity(args, model, check_rows is not None)
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=1e-3, weight_decay=1e-4)
        scheduler = PlateauController(optimizer); prior = 0.; previous = None
        if args.resume:
            previous = torch.load(output / 'checkpoints/latest.pt', map_location='cpu', weights_only=False)
            if previous['identity'] != ident:
                raise ValueError('Resume source/data/device/budget identity differs')
            if previous['finished']:
                print('Training already finished:', previous['status'], flush=True)
                return previous['status']
            if (output / 'STOP').exists():
                raise ValueError('Remove STOP to resume')
            load_trainable(model, previous['model']); optimizer.load_state_dict(previous['optimizer'])
            scheduler.load_state_dict(previous['scheduler']); prior = previous['elapsed_seconds']
            if (output / 'budget.json').exists():
                prior = max(prior, json.loads((output / 'budget.json').read_text())['elapsed_seconds'])
        else:
            output.mkdir(parents=True, exist_ok=False)
            save(output / 'config.json', {'identity': ident, 'command': [sys.executable, *sys.argv],
                 'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                 'git_status': subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True).strip(),
                 'data_paths': resolved_data_paths(), 'cache': str(CACHE),
                 'timing': 'Cumulative training wall includes cache loading and raw monitoring; separate preparation/final stage'})
            snapshot(output / 'source_snapshot.zip', ident)
        elapsed = lambda: prior + time.perf_counter() - begin
        invocation_path = output / 'invocations.json'
        invocations = json.loads(invocation_path.read_text()) if invocation_path.exists() else []
        invocations.append({'argv': [sys.executable, *sys.argv], 'resume': args.resume,
                            'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'prior_seconds': prior})
        save(invocation_path, invocations)
        train_meta, dev_meta = ReferenceLayouts('train'), ReferenceLayouts('dev')
        if {r['member'][0] for r in train_meta.records} & {r['member'][0] for r in dev_meta.records}:
            raise ValueError('Train/dev source shards overlap')
        if check_rows is None:
            train, dev, train_monitor = load_rows('train'), load_rows('dev_monitor500'), load_rows('train_monitor500')
            if len(train) != 16000 or len(dev) != 500 or len(train_monitor) != 500:
                raise ValueError('Wrong formal panel sizes')
        else:
            train, dev = check_rows; train_monitor = train
        member_to_index = {tuple(r['member']): i for i, r in enumerate(train)}
        monitor_indices = [member_to_index[tuple(r['member'])] for r in train_monitor]
        progress = previous['progress'] if previous else {
            'epoch': 0, 'step': 0, 'cursor': 0, 'order': None, 'seen': np.zeros(len(train), dtype=bool),
            'pending_monitor': True, 'best_epoch': None, 'best_proxy': None, 'history': []}
        if previous:
            restore_random(previous['random'])
        requested = [False]
        handler = signal.signal(signal.SIGINT, lambda *_: requested.__setitem__(0, True))
        stopped = lambda: requested[0] or (output / 'STOP').exists() or (HERE / 'runs/STOP').exists() or (stop_after_steps is not None and progress['step'] >= stop_after_steps)
        exhausted = lambda: elapsed() >= ident['max_seconds']
        def heartbeat():
            save(output / 'budget.json', {'elapsed_seconds': elapsed()})
        def persist(status, *, finished=False, best=False):
            heartbeat()
            value = {'model': trainable_state(model), 'identity': ident, 'model_config': ident['model_config'],
                     'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'random': random_state(),
                     'progress': progress, 'elapsed_seconds': elapsed(), 'status': status, 'finished': finished}
            checkpoint(output / 'checkpoints/latest.pt', value)
            if best:
                checkpoint(output / 'checkpoints/best.pt', value)
            save(output / 'progress.json', {k: v for k, v in progress.items() if k not in ('seen', 'order')}
                 | {'status': status, 'finished': finished, 'seen_count': int(progress['seen'].sum()),
                    'elapsed_seconds': elapsed(), 'lr': optimizer.param_groups[0]['lr'],
                    'scheduler': scheduler.state_dict(), 'test_read': False, 'postprocessing_calls': 0})
        def observe():
            val = monitor(model, dev, args.device, should_stop=lambda: stopped() or exhausted())
            if val is None:
                return False
            seen = progress['seen'][monitor_indices]
            tr = monitor(model, train_monitor, args.device, seen=seen,
                         should_stop=lambda: stopped() or exhausted())
            if tr is None:
                return False
            metric = val['physics']
            improved = progress['best_proxy'] is None or metric < progress['best_proxy'] - 1e-5
            if improved:
                progress['best_proxy'], progress['best_epoch'] = metric, progress['epoch']
            lr_used = optimizer.param_groups[0]['lr']
            scheduler.step(metric, progress['epoch'])
            progress['history'].append({'epoch': progress['epoch'], 'step': progress['step'],
                'train_monitor': tr, 'dev_monitor': val, 'train_seen_count': int(seen.sum()),
                'lr_used': lr_used, 'lr_next': optimizer.param_groups[0]['lr'],
                'scheduler': scheduler.state_dict(), 'elapsed_seconds': elapsed()})
            progress['pending_monitor'] = False
            persist('running', best=improved)
            print(f'epoch={progress["epoch"]} step={progress["step"]} public_proxy={metric:.6f} '
                  f'raw_legal={val["raw_hard_legal"]}/{val["count"]} lr={optimizer.param_groups[0]["lr"]:.2g}', flush=True)
            return True
        status = 'running'
        try:
            save(output / 'panels.json', {'train_count': len(train),
                 'train_monitor': [list(r['member']) for r in train_monitor],
                 'dev_monitor': [list(r['member']) for r in dev], 'test_read': False})
            if not previous:
                persist('prepared')
            while True:
                if stopped() or exhausted():
                    status = 'user_stop' if stopped() else 'budget'; break
                if progress['pending_monitor']:
                    if not observe():
                        status = 'user_stop' if stopped() else 'budget'; break
                    if scheduler.should_stop():
                        status = 'early_stop'; break
                if progress['epoch'] >= args.max_epochs:
                    status = 'max_epochs'; break
                if progress['order'] is None:
                    progress['order'] = np.random.default_rng(args.seed + progress['epoch']).permutation(len(train))
                    progress['cursor'] = 0
                model.train()
                while progress['cursor'] < len(train):
                    if stopped() or exhausted():
                        break
                    indices = progress['order'][progress['cursor']:progress['cursor'] + 32]
                    graph = collate([train[i] for i in indices], args.device)
                    optimizer.zero_grad(set_to_none=True)
                    values = loss(model(inputs_only(graph)), graph)
                    if not torch.isfinite(torch.stack(list(values.values()))).all():
                        raise RuntimeError('Nonfinite training loss')
                    values['total'].backward()
                    norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
                    optimizer.step(); progress['seen'][indices] = True
                    progress['cursor'] += len(indices); progress['step'] += 1
                    heartbeat()
                    if progress['step'] % 100 == 0:
                        persist('running')
                        print(f'step={progress["step"]} total={float(values["total"].detach()):.5f} grad={float(norm):.4f}', flush=True)
                if progress['cursor'] < len(train):
                    status = 'user_stop' if stopped() else 'budget'; break
                progress['epoch'] += 1; progress['cursor'] = 0
                progress['order'] = None; progress['pending_monitor'] = True
                persist('monitor_pending')
            finished = status in ('budget', 'max_epochs', 'early_stop')
            persist(status, finished=finished)
            save(output / 'results.json', {'status': status, 'finished': finished, 'elapsed_seconds': elapsed(),
                 'epochs': progress['epoch'], 'steps': progress['step'], 'best_epoch': progress['best_epoch'],
                 'best_proxy': progress['best_proxy'], 'full_dev_complete': False,
                 'formal_training': check_rows is None, 'postprocessing_calls': 0, 'test_read': False,
                 'quality_claim': 'raw development proxy only; final frozen-best evaluation is a separate stage'})
            return status
        except BaseException:
            persist('error'); raise
        finally:
            signal.signal(signal.SIGINT, handler)


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--arm', choices=ARMS, default='recurrent_full')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--batch-size', type=int, choices=(32,), default=32)
    ap.add_argument('--max-hours', type=float, default=4.)
    ap.add_argument('--max-epochs', type=int, default=200)
    ap.add_argument('--resume', action='store_true')
    return ap


if __name__ == '__main__':
    run(parser().parse_args())
