"""Run exactly original4 then new_first; no retries, Q, test or extra inference."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from ..io import OWNER, ROOT, save, exclusive, sources, sha

JOBS = ('original4', 'new_first')
REGISTRATION = Path(__file__).resolve().parent


def now():
    return datetime.now(timezone.utc).isoformat()


def verify_authorization():
    auth = json.loads((OWNER/'authorization.json').read_text(encoding='utf-8'))
    if (auth.get('scope') != 'E116.first_draft_comparison.seed0'
            or not auth.get('training_enabled') or not auth.get('actual_updates_authorized')
            or not auth.get('automatic_queue') or auth.get('allowed_arms') != list(JOBS)):
        raise PermissionError('The exact two-arm training queue is not authorized')
    gate = json.loads((OWNER/'preflight.json').read_text(encoding='utf-8'))
    registration = json.loads((REGISTRATION/'registration.json').read_text(encoding='utf-8'))
    if (gate['status'] != 'passed' or gate['sources'] != sources()
            or gate['sources'] != registration['prepared_core_sources']
            or sha(__file__) != registration['queue_source_sha256']):
        raise RuntimeError('Prepared model or registered queue source changed')


def summarize_completed_training():
    """Consume only existing training logs, never rerun a model or evaluation."""
    summary = dict(scope='E116 two standalone first-draft models',
                   selection='own fixed common final-Loss dev500 best',
                   dataset='frozen random16k; original areas/noMIB/all-module prediction',
                   single_seed=True, extra_inference=False, test_read=False,
                   postprocessing_calls=0, Q_evaluation=False, arms={})
    for arm in JOBS:
        folder = OWNER/'runs'/f'{arm}_seed0'
        result = json.loads((folder/'results.json').read_text(encoding='utf-8'))
        progress = json.loads((folder/'progress.json').read_text(encoding='utf-8'))
        best = next(item for item in progress['history'] if item['epoch'] == result['best_epoch'])
        summary['arms'][arm] = dict(status=result['status'], epochs=result['epochs'],
            steps=result['steps'], best_epoch=result['best_epoch'],
            best_final_loss=result['best_fixed_loss'],
            best_train500=best['train_monitor'], best_dev500=best['dev_monitor'],
            elapsed_seconds=result['elapsed_seconds'],
            train_plateau_observed=result['train_plateau_observed'])
    save(OWNER/'loss_comparison.json', summary)
    rows = [
        '# E116：两组独立首稿 Loss 对照', '',
        '只读取已完成训练日志；不新增推理、后处理、Q或测试评价。', '',
        '|组别|完成轮数|最佳轮|best train500 Loss|best dev500 Loss|总秒数|',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for arm, value in summary['arms'].items():
        rows.append(f'|{arm}|{value["epochs"]}|{value["best_epoch"]}|'
                    f'{value["best_train500"]["total"]:.6f}|{value["best_final_loss"]:.6f}|'
                    f'{value["elapsed_seconds"]:.2f}|')
    rows += ['', '同一固定16k/共同最终Loss，单seed开发结果；结构同时改变反馈、FFN和读出，'
             '不作单因素因果或独立泛化结论。完整分项及逐例值见loss_comparison.json。', '']
    (OWNER/'comparison_report.md').write_text('\n'.join(rows), encoding='utf-8')


def main():
    with exclusive(OWNER/'queue.lock'):
        verify_authorization()
        if (OWNER/'queue_state.json').exists():
            raise FileExistsError('E116 queue already dispatched; no implicit retry')
        if any((OWNER/'runs'/f'{arm}_seed0').exists() for arm in JOBS):
            raise FileExistsError('Scratch queue cannot overwrite or resume an existing arm')
        requested = [False]
        signal.signal(signal.SIGINT, lambda *_: requested.__setitem__(0, True))
        def stopped():
            return requested[0] or (OWNER/'STOP').exists() or (OWNER/'QUEUE_STOP').exists()
        state = dict(status='running', pid=os.getpid(), started_at=now(),
                     jobs=list(JOBS), active=None, completed=[],
                     test_read=False, Q_evaluation=False, postprocessing_calls=0,
                     scope='Exactly original4 scratch -> new_first scratch; seed0; fixed16k',
                     per_arm_maximum_seconds=14400, per_arm_maximum_epochs=200)
        def beat(**fields):
            state.update(fields)
            state['updated_at'] = now()
            save(OWNER/'queue_state.json', state)
        def forward_stop(message):
            if not (OWNER/'STOP').exists():
                (OWNER/'STOP').write_text(message+'\n', encoding='utf-8')
        child = None
        try:
            logs = OWNER/'launch_logs'
            logs.mkdir(exist_ok=True)
            beat()
            for arm in JOBS:
                if stopped():
                    raise InterruptedError('E116 queue stopped; no next arm')
                verify_authorization()
                args = [sys.executable, '-u', '-B', '-m',
                        'experiments.E116_first_draft_comparison.train', '--arm', arm]
                with (logs/f'{arm}.stdout.log').open('x', encoding='utf-8') as out, \
                     (logs/f'{arm}.stderr.log').open('x', encoding='utf-8') as err:
                    child = subprocess.Popen(args, cwd=ROOT, stdout=out, stderr=err,
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
                    beat(active=dict(arm=arm, pid=child.pid, started_at=now()))
                    print(f'{now()} started {arm}, pid={child.pid}', flush=True)
                    while child.poll() is None:
                        if stopped():
                            forward_stop('User stopped E116 queue; save complete state and do not advance.')
                        beat()
                        time.sleep(5)
                if child.returncode != 0:
                    raise RuntimeError(f'{arm} exited {child.returncode}; no retry or next arm')
                result = json.loads((OWNER/'runs'/f'{arm}_seed0'/'results.json').read_text(encoding='utf-8'))
                if stopped() or result['status'] == 'user_stop':
                    raise InterruptedError('User stop; no next arm')
                if result.get('error') or result['status'] not in ('early_stop', 'budget', 'max_epochs'):
                    raise RuntimeError(f'{arm} did not complete normally; no next arm')
                if not result['steps'] or result['best_epoch'] is None:
                    raise RuntimeError(f'{arm} has no trained/selected result; no next arm')
                state['completed'].append(dict(arm=arm, status=result['status'],
                    epochs=result['epochs'], steps=result['steps'], completed_at=now()))
                child = None
                beat(active=None)
            summarize_completed_training()
            beat(status='complete', active=None, completed_at=now(),
                 comparison='loss_comparison.json and comparison_report.md; saved training logs only')
        except BaseException as error:
            if child is not None and child.poll() is None:
                forward_stop('E116 controller stopped; preserve state and do not advance.')
                until = time.monotonic()+120
                while child.poll() is None and time.monotonic() < until:
                    beat(status='stopping', error=repr(error))
                    time.sleep(2)
            beat(status='user_stop' if isinstance(error, InterruptedError) else 'needs_attention',
                 error=repr(error), traceback=traceback.format_exc(),
                 child_exited=child is None or child.poll() is not None)
            raise


if __name__ == '__main__':
    main()
