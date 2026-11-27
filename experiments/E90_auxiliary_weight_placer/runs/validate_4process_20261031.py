"""Frozen own-Loss best: four-process flash500, then batched GPU raw dev2000.

Keep training sources/checkpoints intact. Resume only missing cases by rerunning
this script; run STOP/global STOP safely pauses the evaluation. No test access.
"""
import os
os.environ['E74_THREADS'] = '2'
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '2'
os.environ['OMP_MAX_ACTIVE_LEVELS'] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import json
import multiprocessing as mp
from pathlib import Path
import signal
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import torch
from experiments.E90_auxiliary_weight_placer import final
from experiments.E90_auxiliary_weight_placer.io import HERE, sha, save, sources, snapshot, exclusive
from experiments.E90_auxiliary_weight_placer.dataset_view import REGISTRY, ReferenceLayouts

OUTPUT = HERE/'runs/20261031_aux3_seed0'
TERMINAL = OUTPUT/'terminal'


def selection():
    path = OUTPUT/'checkpoints/best.pt'
    best = torch.load(path, map_location='cpu', weights_only=False)
    latest_path = OUTPUT/'checkpoints/latest.pt'
    latest = torch.load(latest_path, map_location='cpu', weights_only=False)
    progress = json.loads((OUTPUT/'progress.json').read_text(encoding='utf-8'))
    assert latest['finished'] and latest['status'] in ('max_epochs', 'budget', 'early_stop')
    assert best['identity'] == latest['identity']
    assert not best['identity']['implementation_check']
    assert best['identity']['sources'] == sources(), 'Training sources changed'
    assert best['identity']['registry_sha256'] == sha(REGISTRY)
    assert best['progress']['epoch'] == progress['best_epoch']
    calibration = best['identity']['calibration']
    assert sha(calibration['path']) == calibration['sha256']
    from experiments.E90_auxiliary_weight_placer.loss_config import load_coefficients
    load_coefficients(calibration['path'])
    identity = dict(checkpoint=path.relative_to(ROOT).as_posix(), checkpoint_sha256=sha(path),
        latest_checkpoint_sha256=sha(latest_path), best_epoch=best['progress']['epoch'],
        model_config=best['model_config'], registry_sha256=sha(REGISTRY), sources=sources(),
        selection='existing unique fixed-own-Loss best; no Q checkpoint selection',
        panel='unchanged v5 dev_monitor500; raw diagnostics on unchanged dev2000',
        count=500, workers=4, threads_per_worker=2, total_worker_compute_threads=8,
        raw_execution='batched single GPU with CPU4 after terminal workers exit',
        baseline=False, implementation_check=False, test_read=False)
    return best, identity


def initialize_worker(checkpoint):
    final.initialize(checkpoint)
    assert torch.get_num_threads() == 2 and torch.get_num_interop_threads() == 1
    save(TERMINAL/f'worker_{os.getpid()}.json', dict(pid=os.getpid(),
        torch_intraop=torch.get_num_threads(), torch_interop=torch.get_num_interop_threads(),
        parallel_info=torch.__config__.parallel_info(), solver_threads=2, test_read=False))


def terminal(identity):
    TERMINAL.mkdir(exist_ok=True)
    config = dict(**identity, steady=final.STEADY,
        timing='public features + CPU inference + decode + all repair/rescue; excludes load, JIT and scoring',
        timing_caveat='four concurrent workers differ from historical runs; do not equate wall time')
    config_path = TERMINAL/'config.json'
    if config_path.exists():
        assert json.loads(config_path.read_text(encoding='utf-8')) == config
    else:
        save(config_path, config)
        snapshot(TERMINAL/'source_snapshot.zip', identity)
        with zipfile.ZipFile(TERMINAL/'source_snapshot.zip', 'a', zipfile.ZIP_DEFLATED) as archive:
            archive.write(__file__, 'evaluation_runner.py')
    if (TERMINAL/'results.json').exists():
        print('Existing complete terminal retained; zero duplicate calls', flush=True)
        return True
    if final.reuse_parent_terminal(OUTPUT, identity):
        return True
    partial = TERMINAL/'partial.json'
    records = json.loads(partial.read_text(encoding='utf-8')) if partial.exists() else []
    metadata = ReferenceLayouts('dev_monitor500').records
    done = {r['index'] for r in records}
    assert len(done) == len(records) and done.issubset(range(500))
    assert all(r['member'] == list(metadata[r['index']]['member']) for r in records)
    todo = iter(i for i in range(500) if i not in done)
    pending = {}
    print('Starting four workers, each with two solver/numeric threads', flush=True)
    with ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context('spawn'),
            initializer=initialize_worker, initargs=(ROOT/identity['checkpoint'],)) as pool:
        def submit():
            if final.stopped(OUTPUT):
                return
            index = next(todo, None)
            if index is not None:
                pending[pool.submit(final.evaluate, index)] = index
        for _ in range(4):
            submit()
        while pending:
            completed, _ = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
            for future in completed:
                pending.pop(future)
                records.append(future.result())
                save(partial, records)
                submit()
                if len(records) % 25 == 0:
                    print(f'Terminal Q: {len(records)}/500', flush=True)
    if len(records) != 500:
        print('Stopped safely; completed cases saved', flush=True)
        return False
    records.sort(key=lambda r: r['index'])
    summary = final.summarize(records)
    save(TERMINAL/'results.json', dict(summary=summary, records=records, test_read=False))
    save(TERMINAL/'summary.json', dict(checkpoint_sha256=identity['checkpoint_sha256'], **summary,
        workers=4, threads_per_worker=2, test_read=False, neutral_runtime_development_only=True))
    partial.unlink()
    print(json.dumps(summary), flush=True)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    with exclusive(OUTPUT.with_suffix('.lock')):
        state, identity = selection()
        if args.preflight:
            print(json.dumps(dict(identity, sources='verified unchanged')), flush=True)
            return
        frozen = OUTPUT/'frozen.json'
        if frozen.exists():
            assert json.loads(frozen.read_text(encoding='utf-8')) == identity, 'Frozen protocol changed'
        else:
            save(frozen, identity)
        if final.stopped(OUTPUT):
            print('STOP present; validation not started', flush=True)
            return
        signal.signal(signal.SIGINT, lambda *_: (OUTPUT/'STOP').touch())
        launch = OUTPUT/'validation_launch.json'
        if not launch.exists():
            save(launch, dict(pid=os.getpid(), command=[sys.executable, *sys.argv],
                utc=datetime.now(timezone.utc).isoformat(), runner_sha256=sha(__file__),
                authorization='训好了，后台启动4进程验证', workers=4, threads_per_worker=2, test_read=False))
        save(HERE/'state.json', dict(status='validation_running', stage='terminal_Q500',
            pid=os.getpid(), best_epoch=identity['best_epoch'], workers=4, threads_per_worker=2,
            formal_training_started=True, training_status=state['status'], test_read=False))
        if not terminal(identity):
            return
        save(HERE/'state.json', dict(status='validation_running', stage='raw_dev2000',
            pid=os.getpid(), best_epoch=identity['best_epoch'], test_read=False))
        print('Q complete; raw dev2000 diagnostics starting', flush=True)
        if not final.full_dev(OUTPUT, state, identity):
            return
        assert sha(ROOT/identity['checkpoint']) == identity['checkpoint_sha256']
        assert sha(OUTPUT/'checkpoints/latest.pt') == identity['latest_checkpoint_sha256']
        save(HERE/'state.json', dict(status='validation_complete', best_epoch=identity['best_epoch'],
            terminal_count=500, raw_count=2000, training_checkpoints_unchanged=True,
            test_read=False, next_stage='await review; no automatic training or Q rerun'))
        print('Validation complete: flash500 and raw dev2000; checkpoints unchanged; tests sealed', flush=True)


if __name__ == '__main__':
    main()
