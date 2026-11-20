"""One authorized random-initialized rich-loss 96x12 placer run."""
import argparse
import copy
import json
import os
import random
import signal
import subprocess
import sys
import threading
import time
import traceback
import numpy as np
import torch
from .io import ROOT, HERE, OWNER, sha, save, checkpoint, sources, snapshot, exclusive
from .model import GeometryModel
from ..batches import inputs_only
from .data import load_rows, collate, register_resident
from ..dataset_view import ReferenceLayouts, REGISTRY
from ..resident import ResidentRows
from .cuda_step import CapturedStep
from .augmentation import flags_for
from ..loss import per_case
from ..loss_config import configuration
from ..schedule import PlateauController, LR

RUN = HERE/'runs/20261104_generalization96_seed0'
MODEL = dict(width=96, layers=12, rectangle=True, ffn_dropout=.10, residual_dropout=.05)

def seed_all(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())

def restore_rng(rng):
    random.setstate(rng['python'])
    np.random.set_state(rng['numpy'])
    torch.set_rng_state(rng['torch'])
    torch.cuda.set_rng_state_all(rng['cuda'])

def optimizer_for(model):
    return torch.optim.AdamW(model.parameters(), lr=LR[0], weight_decay=.0003)

def build_capture(model, optimizer, pack, projection=None):
    model.train()
    sample = pack.batch([i % len(pack.rows) for i in range(32)])
    cap = CapturedStep(model, optimizer, sample, 'baseline')
    if projection:
        if cap.projection.names != projection['group_names']:
            raise ValueError('Projection trainability changed')
        cap.projection.steps = projection['steps']
        cap.projection.clipped.copy_(torch.tensor(projection['clipped_steps_per_group'], device='cuda'))
        cap.projection.last_scales.copy_(torch.tensor(projection['last_scales'], device='cuda'))
    return cap

def numerical_check(values, output):
    ok = torch.isfinite(torch.stack(list(values.values()))).all() & torch.isfinite(output).all()
    if not bool(ok & (values['total'].amax() < 1000.)):
        raise RuntimeError('Monitoring nonfinite/per-case explosion')

@torch.no_grad()
def monitor(model, rows, halted=lambda: False, beat=lambda **kw: None):
    model.eval()
    all_values = []
    members = []
    for start in range(0, len(rows), 32):
        if halted():
            return None
        batch, _ = collate(rows[start:start+32], 'cuda')
        out = model(inputs_only(batch))
        values = per_case(out, batch)
        numerical_check(values, out)
        keys = list(values)
        all_values.append(torch.stack([values[k] for k in keys], -1).cpu().double().numpy())
        members.extend([list(r['member']) for r in rows[start:start+32]])
        beat(activity='monitor', monitor_completed=min(start+32, len(rows)), monitor_count=len(rows))
    array = np.concatenate(all_values)
    result = {key: float(array[:, j].mean()) for j, key in enumerate(keys)}
    totals = array[:, keys.index('total')]
    result.update(count=len(rows), p95=float(np.quantile(totals, .95)),
                  max=float(totals.max()), worst_member=members[int(totals.argmax())],
                  per_case_total=totals.tolist())
    return result

def fresh_progress(size):
    return dict(epoch=0, cursor=0, step=0, order=None, seen=np.zeros(size, dtype=bool),
                pending_monitor=True, best=None, best_epoch=None, history=[], batch_history=[])

def main():
    gate=json.loads((OWNER/"preflight.json").read_text(encoding="utf-8"))
    if gate["status"]!="passed" or gate["sources"]!=sources():
        raise RuntimeError("Implementation gate/source identity mismatch")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if (OWNER/'STOP').exists() or (RUN/'STOP').exists():
        raise RuntimeError('STOP present')
    with exclusive(OWNER/'training.lock'):
        run(args)

def run(args):
    started = time.perf_counter()
    plan = json.loads((OWNER/'plan.json').read_text(encoding='utf-8'))
    seed_all(plan['seed'])
    if REGISTRY.resolve() != (ROOT/'protocol/reference_workset_v5.json').resolve():
        raise ValueError('Unexpected registry environment override')
    identity = dict(revision='E98.generalization96.axis_swap.scratch.v1', plan=plan, sources=sources(),
                    loss=configuration(), registry_sha256=sha(REGISTRY),
                    paths_sha256=sha(ROOT/'protocol/paths.json'), torch=str(torch.__version__),
                    cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0), numpy=np.__version__)
    previous = None
    prior = 0.
    if args.resume:
        previous = torch.load(RUN/'checkpoints/latest.pt', map_location='cpu', weights_only=False)
        if previous['identity'] != identity:
            raise ValueError('Resume identity changed; migration required')
        if previous['finished'] or previous['status'] == 'error':
            raise ValueError('Terminal/error run cannot automatically restart')
        prior = max(previous['elapsed_seconds'], json.loads((RUN/'budget.json').read_text())['elapsed_seconds'])
    else:
        RUN.mkdir(parents=True, exist_ok=False)
        save(RUN/'config.json', dict(identity=identity, argv=[sys.executable, *sys.argv],
             git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
             uncommitted_source_snapshot=True, test_read=False, parent_checkpoint=None))
        snapshot(RUN/'source_snapshot.zip', identity)
    elapsed = lambda: prior+time.perf_counter()-started
    model = GeometryModel(**MODEL).cuda()
    if sum(p.numel() for p in model.parameters()) != plan['parameters'] or not all(p.requires_grad for p in model.parameters()):
        raise ValueError('Architecture/parameter count mismatch')
    optimizer = optimizer_for(model)
    scheduler = PlateauController(optimizer)
    progress = previous['progress'] if previous else fresh_progress(plan['train_count'])
    if previous:
        model.load_state_dict(previous['model'])
        optimizer.load_state_dict(previous['optimizer'])
        scheduler.load_state_dict(previous['scheduler'])
        restore_rng(previous['rng'])
    requested = [False]
    old_signal = signal.signal(signal.SIGINT, lambda *_: requested.__setitem__(0, True))
    cap = None
    current_ids = None
    closed = threading.Event()
    hb_lock = threading.RLock()
    hb = dict(status='starting', activity='initializing', pid=os.getpid())
    def beat(**fields):
        with hb_lock:
            hb.update(fields)
            save(RUN/'budget.json', dict(elapsed_seconds=elapsed(), maximum_seconds=plan['maximum_seconds']))
            save(RUN/'heartbeat.json', dict(**hb, epoch=progress['epoch'], step=progress['step'],
                 cursor=progress['cursor'], elapsed_seconds=elapsed(),
                 utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                 test_read=False, postprocessing_calls=0))
    def heartbeats():
        while not closed.wait(5.):
            beat()
    worker = threading.Thread(target=heartbeats, daemon=True)
    worker.start()
    def stopped():
        return requested[0] or (OWNER/'STOP').exists() or (RUN/'STOP').exists()
    def exhausted():
        return elapsed() >= plan['maximum_seconds']
    def halted():
        return stopped() or exhausted()
    def store(status, finished=False, best=False):
        beat(status=status)
        state = dict(model=model.state_dict(), model_config=MODEL, optimizer=optimizer.state_dict(),
                     scheduler=scheduler.state_dict(), progress=progress, rng=rng_state(), identity=identity,
                     elapsed_seconds=elapsed(), status=status, finished=finished,
                     projection=cap.projection.summary() if cap else (previous['projection'] if previous else None))
        checkpoint(RUN/'checkpoints/latest.pt', state)
        if best:
            checkpoint(RUN/'checkpoints/best.pt', state)
        save(RUN/'progress.json', dict(status=status, finished=finished, epoch=progress['epoch'],
             step=progress['step'], cursor=progress['cursor'], seen_count=int(progress['seen'].sum()),
             best=progress['best'], best_epoch=progress['best_epoch'], lr=optimizer.param_groups[0]['lr'],
             scheduler=scheduler.state_dict(), history=progress['history'], elapsed_seconds=elapsed(),
             test_read=False, postprocessing_calls=0))
        save(OWNER/'state.json', dict(status=status, run=RUN.relative_to(ROOT).as_posix(), pid=os.getpid(),
             epoch=progress['epoch'], step=progress['step'], best=progress['best'],
             test_read=False, postprocessing_calls=0))
    status = 'running'
    error = None
    try:
        store('data_preparation')
        beat(activity='loading frozen train16k CPU cache and public loss ports')
        train = load_rows('train')
        dev = load_rows('dev_monitor500')
        train_records = ReferenceLayouts('train').records
        dev_records = ReferenceLayouts('dev_monitor500').records
        if len(train) != 16000 or len(dev) != 500:
            raise ValueError('Frozen role counts changed')
        if [list(r['member']) for r in train] != [r['member'] for r in train_records]:
            raise ValueError('Cached v6 prefix differs from frozen v5 members')
        if [list(r['member']) for r in dev] != [r['member'] for r in dev_records]:
            raise ValueError('Dev members changed')
        if {r['member'][0] for r in train} & {r['member'][0] for r in dev}:
            raise ValueError('Train/dev shards overlap')
        lookup = {tuple(r['member']): i for i, r in enumerate(train)}
        monitor_ids = [lookup[tuple(r['member'])] for r in ReferenceLayouts('train_monitor500').records]
        train_monitor = [train[i] for i in monitor_ids]
        save(RUN/'panels.json', dict(train_members=[list(r['member']) for r in train],
             train_monitor_members=[list(r['member']) for r in train_monitor],
             dev_monitor_members=[list(r['member']) for r in dev],
             registry_sha256=sha(REGISTRY), test_read=False))
        max_pins = max(1, max(r['graph']['pin_coordinates'].shape[1] for r in [*train, *dev]))
        pack = ResidentRows(train, 'cuda', max_nodes=120, max_pins=max_pins)
        dev_pack = ResidentRows(dev, 'cuda', max_nodes=120, max_pins=max_pins)
        register_resident(pack)
        register_resident(dev_pack)
        beat(activity='capturing FP32 train graph; no optimizer updates during capture')
        cap = build_capture(model, optimizer, pack, previous['projection'] if previous else None)
        if cap.projection.steps != progress['step']:
            raise ValueError('Optimizer/projection cursor mismatch')
        free, total = torch.cuda.mem_get_info()
        reserve = sum(p.numel()*p.element_size()*2 for p in model.parameters()) if not optimizer.state else 0
        if free < reserve+512*2**20:
            raise RuntimeError('Insufficient dedicated VRAM headroom')
        save(RUN/'resources.json', dict(max_nodes=120, max_loss_ports_per_block=max_pins,
             gpu_allocated=torch.cuda.memory_allocated(), gpu_reserved=torch.cuda.memory_reserved(),
             free=free, total=total, adam_remaining_reserve=reserve,
             row_bytes=pack.row_bytes, full_dataset_on_gpu=False, threads=4))
        store('running')
        last_save = elapsed()
        interval_start = time.perf_counter()
        interval_step = progress['step']
        while True:
            if halted():
                status = 'user_stop' if stopped() else 'budget'
                break
            if progress['pending_monitor']:
                preserved = rng_state()
                beat(activity='fixed dev500 monitoring')
                dv = monitor(model, dev, halted, beat)
                beat(activity='fixed train500 monitoring')
                tr = monitor(model, train_monitor, halted, beat) if dv else None
                restore_rng(preserved)
                if dv is None or tr is None:
                    status = 'user_stop' if stopped() else 'budget'
                    break
                old_best = progress['best']
                improved = old_best is None or dv['total'] < old_best-1e-5
                if improved:
                    progress['best'], progress['best_epoch'] = dv['total'], progress['epoch']
                scheduler.step(dv['total'], progress['epoch'])
                progress['history'].append(dict(epoch=progress['epoch'], step=progress['step'],
                     train_monitor=tr, dev_monitor=dv, train_monitor_seen=int(progress['seen'][monitor_ids].sum()),
                     loss=configuration(), lr_next=optimizer.param_groups[0]['lr'],
                     scheduler=scheduler.state_dict(), elapsed_seconds=elapsed()))
                progress['pending_monitor'] = False
                store('running', best=improved)
                last_save = elapsed()
                print(f'epoch={progress["epoch"]} step={progress["step"]} train={tr["total"]:.6f} dev={dv["total"]:.6f} lr={optimizer.param_groups[0]["lr"]:.2g}', flush=True)
                if scheduler.should_stop():
                    status = 'early_stop'
                    break
                if progress['epoch'] >= plan['maximum_epochs']:
                    status = 'max_epochs'
                    break
                interval_start, interval_step = time.perf_counter(), progress['step']
            if progress['order'] is None:
                progress['order'] = np.random.default_rng(plan['seed']+progress['epoch']).permutation(len(train))
                progress['cursor'] = 0
            model.train()
            beat(activity='training')
            while progress['cursor'] < len(train):
                if halted():
                    break
                current_ids = progress['order'][progress['cursor']:progress['cursor']+32]
                pack.batch(current_ids, out=cap.inputs)
                cap.inputs["axis_swap"].copy_(torch.as_tensor(flags_for(plan["seed"],progress["epoch"],current_ids),device="cuda"))
                values, norm = cap.replay()
                progress['seen'][current_ids] = True
                progress['cursor'] += len(current_ids)
                progress['step'] += 1
                if progress['step'] % 100 == 0:
                    torch.cuda.synchronize()
                    now = time.perf_counter()
                    performance = dict(step=progress['step'], epoch=progress['epoch']+1,
                        seconds_per_step=(now-interval_start)/(progress['step']-interval_step),
                        samples_per_second=32*(progress['step']-interval_step)/(now-interval_start),
                        total=float(values['total'].detach()), gradient_norm=float(norm), elapsed_seconds=elapsed())
                    progress['batch_history'].append(performance)
                    with (RUN/'steps.jsonl').open('a', encoding='utf-8') as f:
                        f.write(json.dumps(performance)+'\n')
                    print(f'step={progress["step"]} epoch={progress["epoch"]+1} cursor={progress["cursor"]} loss={performance["total"]:.5f} ms/step={performance["seconds_per_step"]*1000:.1f}', flush=True)
                    if elapsed()-last_save >= 60.:
                        store('running')
                        last_save = elapsed()
                    interval_start, interval_step = time.perf_counter(), progress['step']
            if progress['cursor'] < len(train):
                status = 'user_stop' if stopped() else 'budget'
                break
            progress['epoch'] += 1
            progress['order'] = None
            progress['cursor'] = 0
            progress['pending_monitor'] = True
            store('monitor_pending')
        store(status, finished=status in ('early_stop', 'budget', 'max_epochs'))
    except BaseException as exc:
        status, error = 'error', repr(exc)
        if cap is not None and current_ids is not None:
            checkpoint(RUN/'failure.pt', dict(members=[list(train[i]['member']) for i in current_ids],
                 batch={k: v.cpu() for k, v in cap.inputs.items()}, rng=rng_state(), error=error))
        try:
            store('error')
        except BaseException:
            traceback.print_exc()
        traceback.print_exc()
    finally:
        closed.set()
        worker.join(timeout=6.)
        signal.signal(signal.SIGINT, old_signal)
        history = progress['history']
        last_train = [h['train_monitor']['total'] for h in history[-4:]]
        train_plateau = len(last_train) >= 4 and max(last_train)-min(last_train) <= max(1e-5, abs(last_train[0])*.001)
        result = dict(status=status, error=error, epochs=progress['epoch'], steps=progress['step'],
             best_epoch=progress['best_epoch'], best_fixed_loss=progress['best'],
             elapsed_seconds=elapsed(), train_monitor_last4=last_train,
             train_plateau_observed=train_plateau,
             convergence='operational dev plateau; train plateau separate' if status == 'early_stop' else 'not established',
             test_read=False, postprocessing_calls=0, next_stage='none; report only')
        save(RUN/'results.json', result)
        beat(status=status, activity='exited')
        save(OWNER/'state.json', dict(**result, run=RUN.relative_to(ROOT).as_posix(), pid=os.getpid()))
        print(json.dumps(result, ensure_ascii=False), flush=True)
    if error:
        raise RuntimeError(error)

if __name__ == '__main__':
    main()
