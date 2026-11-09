"""One authorized scratch run followed by its predeclared assessment; no sweep."""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from .io import ROOT, HERE, save, exclusive, sources
from .initialization import verify_protected

OUTPUT = HERE/'runs/20261027_e42_fixedG_scratch_seed0'
STATE = HERE/'state.json'


def commands(resume=False):
    train = [sys.executable, '-u', '-B', '-m', 'experiments.E64_e42_unified_retrain.train',
             '--output', str(OUTPUT), '--max-hours', '4', '--max-epochs', '200',
             '--batch-size', '32', '--seed', '0', '--device', 'cuda', '--engine', 'cuda_graph']
    if resume and (OUTPUT/'checkpoints/latest.pt').exists():
        train.append('--resume')
    return [('train', train),
            ('final', [sys.executable, '-u', '-B', '-m', 'experiments.E64_e42_unified_retrain.final', '--output', str(OUTPUT), '--stage', 'all']),
            ('draft100', [sys.executable, '-u', '-B', '-m', 'experiments.E64_e42_unified_retrain.render_drafts', '--output', str(OUTPUT)]),
            ('report', [sys.executable, '-u', '-B', '-m', 'experiments.E64_e42_unified_retrain.report'])]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    steps = commands(args.resume)
    if args.dry_run:
        print(json.dumps(steps, indent=2))
        return
    with exclusive(HERE/'runs/pipeline.lock'):
        verify_protected()
        preflight = json.loads((HERE/'runs/20261027_preflight/results.json').read_text(encoding='utf-8'))
        if preflight['status'] != 'passed' or preflight['sources'] != sources():
            raise ValueError('Current sources require a passing five-case preflight')
        previous = json.loads(STATE.read_text(encoding='utf-8')) if STATE.exists() else {}
        events = previous.get('events', [])

        def state(status, stage, child=None, **extra):
            utc = datetime.now(timezone.utc).isoformat()
            events.append({'status': status, 'stage': stage, 'utc': utc})
            save(STATE, {'status': status, 'stage': stage, 'updated_utc': utc,
                        'controller_pid': os.getpid(), 'child_pid': child,
                        'output': str(OUTPUT.relative_to(HERE)), 'max_training_hours': 4,
                        'events': events, 'test_read': False, **extra})

        stage = 'prepare'
        try:
            for stage, command in steps:
                if (HERE/'runs/STOP').exists() or (OUTPUT/'STOP').exists():
                    state('stopped', stage)
                    return
                if stage == 'train' and (OUTPUT/'results.json').exists():
                    old = json.loads((OUTPUT/'results.json').read_text(encoding='utf-8'))
                    if old['status'] in ('early_stop', 'budget', 'max_epochs'):
                        continue
                log = HERE/'runs'/f'{stage}.log'
                with log.open('a', encoding='utf-8') as stream:
                    child = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
                    state('running', stage, child.pid)
                    print(f'{stage} pid={child.pid} log={log}', flush=True)
                    code = child.wait()
                if code:
                    raise RuntimeError(f'{stage} failed with exit {code}; see {log}')
                if stage == 'train':
                    value = json.loads((OUTPUT/'results.json').read_text(encoding='utf-8'))
                    if value['status'] not in ('early_stop', 'budget', 'max_epochs'):
                        state('stopped', stage)
                        return
                if stage == 'final' and not (OUTPUT/'terminal/results.json').exists():
                    state('stopped', 'evaluation_incomplete')
                    return
                if stage == 'draft100' and not (OUTPUT/'draft_comparison100/manifest.json').exists():
                    state('stopped', 'gallery_incomplete')
                    return
                state('stage_complete', stage)
            state('complete', 'complete', report='runs/review.md', baseline_promoted=False)
        except BaseException as error:
            state('failed', stage, error=repr(error))
            raise
        finally:
            verify_protected()


if __name__ == '__main__':
    main()
