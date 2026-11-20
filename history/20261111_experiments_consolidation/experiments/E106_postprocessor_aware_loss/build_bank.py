"""Data preparation only: no optimizer, training invocation or follow-on job."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import time
import traceback
import torch
from .common import OWNER, RUN, read, save, sha, sources, checkpoint, setup, check_stop
from .cache import load_rows
from .bank import initialize, evaluate
from .data import load_case
from .objective import calibrate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    setup()
    gate = read(OWNER/'preflight.json')
    current = sources()
    assert gate['status'] == 'passed' and gate['sources'] == current
    assert read(OWNER/'authorization.json')['critic_training'] is False
    identity = dict(sources=current, panel_sha256=sha(OWNER/'panel.json'))
    config = RUN/'bank_config.json'
    if config.exists():
        if not args.resume or read(config)['identity'] != identity:
            raise RuntimeError('Existing data run: explicit reviewed --resume and identical identity required')
    else:
        save(config, dict(identity=identity, data_only=True, training=False))
    started = time.perf_counter()
    cases = read(OWNER/'panel.json')['cases']
    try:
        check_stop()
        save(RUN/'state.json', dict(status='running', stage='preparing_train_inputs', pid=os.getpid(),
                                    total_drafts=25000, training=False, test_read=False))
        missing = [r for r in cases if not (RUN/f'inputs/{r["index"]:05d}.pt').exists()
                   and not (RUN/f'data/{r["index"]:05d}.pt').exists()]
        if missing:
            rows = load_rows('train', [r['train_index'] for r in missing])
            for spec, row in zip(missing, rows):
                check_stop()
                checkpoint(RUN/f'inputs/{spec["index"]:05d}.pt', row)
            del rows
        completed = 0
        with ProcessPoolExecutor(max_workers=4, initializer=initialize, initargs=(identity,)) as pool:
            pending = {pool.submit(evaluate, r['index']): r['index'] for r in cases}
            try:
                for future in as_completed(pending):
                    future.result()
                    completed += 1
                    if completed % 10 == 0 or completed == len(cases):
                        save(RUN/'state.json', dict(status='running', stage='labeling', pid=os.getpid(),
                                                    completed_cases=completed, completed_drafts=completed*5,
                                                    total_drafts=25000, elapsed_seconds=time.perf_counter()-started,
                                                    training=False, optimizer_updates=0, test_read=False))
                        print('BANK', completed, '/5000 cases;', completed*5, '/25000 drafts', flush=True)
            except BaseException as exc:
                if not isinstance(exc, InterruptedError):
                    (RUN/'BANK_ABORT').write_text(repr(exc), encoding='utf-8')
                for future in pending:
                    future.cancel()
                pool.shutdown(wait=True, cancel_futures=True)
                raise
        all_rows = [load_case(r['index']) for r in cases]
        assert len(all_rows) == 5000 and sum(len(r['Q']) for r in all_rows) == 25000
        calibrate([r for r in all_rows if r['split'] == 'fit'])
        qs = torch.cat([r['Q'] for r in all_rows])
        save(RUN/'bank_results.json', dict(status='complete', cases=5000, drafts=25000,
                                           hard_legal=25000, Q_min=float(qs.min()), Q_mean=float(qs.mean()),
                                           Q_max=float(qs.max()), training_started=False, optimizer_updates=0,
                                           elapsed_seconds=time.perf_counter()-started, automatic_next_stage=False,
                                           identity=identity, test_read=False))
        save(RUN/'state.json', dict(status='complete', stage='data_ready_training_disabled', completed_drafts=25000,
                                    total_drafts=25000, training=False, optimizer_updates=0, test_read=False))
    except BaseException as exc:
        save(RUN/f'errors/bank_{int(time.time())}.json', dict(error=repr(exc), traceback=traceback.format_exc()))
        save(RUN/'state.json', dict(status='user_stop' if isinstance(exc, InterruptedError) else 'error',
                                    training=False, test_read=False))
        raise


if __name__ == '__main__':
    main()
