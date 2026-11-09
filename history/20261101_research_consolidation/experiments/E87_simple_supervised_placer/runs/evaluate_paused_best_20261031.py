"""User-authorized Q only: paused own-Loss best, one process / two threads.

Does not modify checkpoints, train status, or either training STOP marker.
Stop/resume this assessment with terminal/EVAL_STOP and the same command.
"""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'E74_THREADS'):
    os.environ[name] = '2'
os.environ['OMP_MAX_ACTIVE_LEVELS'] = '1'
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import torch
from contextlib import nullcontext
from experiments.E87_simple_supervised_placer import final
from experiments.E87_simple_supervised_placer.io import HERE, sha, save, snapshot, sources, exclusive
from experiments.E87_simple_supervised_placer.dataset_view import REGISTRY, ReferenceLayouts

OUTPUT = HERE/'runs/20261030_simple_seed0'
FOLDER = OUTPUT/'terminal'


def selection():
    best_path = OUTPUT/'checkpoints/best.pt'
    best = torch.load(best_path, map_location='cpu', weights_only=False)
    latest = torch.load(OUTPUT/'checkpoints/latest.pt', map_location='cpu', weights_only=False)
    progress = json.loads((OUTPUT/'progress.json').read_text(encoding='utf-8'))
    assert progress['status'] == 'user_stop', 'Only the explicitly paused run is authorized'
    assert best['identity'] == latest['identity']
    assert best['identity']['sources'] == sources(), 'Training source identity changed'
    assert best['identity']['registry_sha256'] == sha(REGISTRY)
    calibration = best['identity']['calibration']
    assert sha(calibration['path']) == calibration['sha256']
    assert not best['identity']['implementation_check']
    assert best['progress']['epoch'] == progress['best_epoch'] == 11
    return dict(checkpoint=best_path.relative_to(ROOT).as_posix(),
                checkpoint_sha256=sha(best_path), latest_sha256=sha(OUTPUT/'checkpoints/latest.pt'),
                best_epoch=best['progress']['epoch'], sources=sources(), registry_sha256=sha(REGISTRY),
                model_config=best['model_config'], count=500,
                panel='unchanged v5 dev_monitor500', workers=1, threads_per_worker=2,
                total_compute_threads=2, test_read=False,
                selection='existing unique own-Loss best; user ended training and authorized Q assessment',
                training_status='user_stop; resumable checkpoints unchanged',
                steady=final.STEADY,
                timing='public features + CPU inference + decode + all repair/rescue; excludes load, JIT and scoring',
                comparison_caveat='Q uses neutral runtime; historical 2x2 concurrency differs, do not compare wall time directly')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    with exclusive(OUTPUT.with_suffix('.lock')):
        identity = selection()
        if args.preflight:
            print(json.dumps(identity | {'sources': 'verified unchanged'}, ensure_ascii=False), flush=True)
            return
        FOLDER.mkdir(exist_ok=True)
        config = FOLDER/'config.json'
        if config.exists():
            assert json.loads(config.read_text(encoding='utf-8')) == identity, 'Frozen assessment identity differs'
        else:
            save(config, identity)
            snapshot(FOLDER/'source_snapshot.zip', identity)
        launch_path = FOLDER/'evaluation_launch.json'
        if not launch_path.exists():
            save(launch_path, dict(command=[sys.executable, *sys.argv], pid=os.getpid(),
                 utc=datetime.now(timezone.utc).isoformat(), runner_sha256=sha(__file__),
                 authorization='训的可以了，测试一下Q。跑2线程', training_STOP_preserved=True))
        if (FOLDER/'results.json').exists():
            print('Existing terminal result retained; no duplicate assessment', flush=True)
            return
        stop = FOLDER/'EVAL_STOP'
        signal.signal(signal.SIGINT, lambda *_: stop.touch())
        partial = FOLDER/'partial.json'
        records = json.loads(partial.read_text(encoding='utf-8')) if partial.exists() else []
        metadata = ReferenceLayouts('dev_monitor500').records
        done = {r['index'] for r in records}
        assert len(done) == len(records) and done.issubset(range(500))
        assert all(r['member'] == list(metadata[r['index']]['member']) for r in records)
        if stop.exists():
            print('EVAL_STOP present; nothing started', flush=True)
            return
        started = time.perf_counter()
        with nullcontext():
            final.initialize(ROOT/identity['checkpoint'])
            save(FOLDER/'runtime_threads.json', dict(workers=1, torch_intraop=torch.get_num_threads(),
                 torch_interop=torch.get_num_interop_threads(), parallel_info=torch.__config__.parallel_info(),
                 environment={k:os.environ[k] for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
                 solver_threads=2, pid=os.getpid(), test_read=False))
            assert torch.get_num_threads() == 2 and torch.get_num_interop_threads() == 1
            for index in range(500):
                if index in done:
                    continue
                if stop.exists():
                    print('Assessment paused; completed cases saved', flush=True)
                    return
                record = final.evaluate(index)
                records.append(record)
                save(partial, records)
                print(f"Q assessment {len(records)}/500 index={index} Q={record['metrics']['cost']:.6f} "
                      f"seconds={record['seconds']:.3f}", flush=True)
        records.sort(key=lambda r: r['index'])
        summary = final.summarize(records)
        assert sha(OUTPUT/'checkpoints/best.pt') == identity['checkpoint_sha256']
        assert sha(OUTPUT/'checkpoints/latest.pt') == identity['latest_sha256']
        save(FOLDER/'results.json', dict(summary=summary, records=records, test_read=False))
        save(FOLDER/'summary.json', dict(checkpoint_sha256=identity['checkpoint_sha256'], **summary,
             test_read=False, neutral_runtime_development_only=True, workers=1, threads_per_worker=2,
             invocation_wall_seconds=time.perf_counter()-started))
        partial.unlink()
        print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()

