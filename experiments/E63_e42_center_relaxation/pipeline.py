"""Bounded, resumable three-arm E42 post-training; each stage runs once."""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from .io import ROOT, HERE, save, exclusive, sources
from .arms import ARMS
from .initialization import verify_protected

OUTPUTS = {arm: HERE/f'runs/20261027_{arm}_seed0' for arm in ARMS}
STATE = HERE/'state.json'


def commands(resume=False):
    result = []
    for arm, output in OUTPUTS.items():
        cmd = [sys.executable, '-u', '-B', '-m', 'experiments.E63_e42_center_relaxation.train',
               '--arm', arm, '--output', str(output), '--max-hours', '2', '--max-epochs', '200',
               '--batch-size', '32', '--seed', '0', '--device', 'cuda', '--engine', 'cuda_graph']
        if resume and (output/'checkpoints/latest.pt').exists():
            cmd.append('--resume')
        result.append((arm, 'train', cmd))
    for arm, output in OUTPUTS.items():
        result.extend([(arm, 'final', [sys.executable, '-u', '-B', '-m', 'experiments.E63_e42_center_relaxation.final', '--output', str(output), '--stage', 'all']),
                       (arm, 'draft100', [sys.executable, '-u', '-B', '-m', 'experiments.E63_e42_center_relaxation.render_drafts', '--output', str(output)])])
    result.append(('all', 'report', [sys.executable, '-u', '-B', '-m', 'experiments.E63_e42_center_relaxation.report']))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    steps = commands(args.resume)
    if args.dry_run:
        print(json.dumps(steps, indent=2)); return
    with exclusive(HERE/'runs/pipeline.lock'):
        verify_protected()
        preflight = json.loads((HERE/'runs/20261027_preflight/results.json').read_text(encoding='utf-8'))
        if preflight['status'] != 'passed' or preflight['sources'] != sources():
            raise ValueError('Current frozen sources require a passing five-case preflight')
        previous = json.loads(STATE.read_text(encoding='utf-8')) if STATE.exists() else {}
        events = previous.get('events', [])
        def state(status, arm, stage, child=None, **extra):
            utc = datetime.now(timezone.utc).isoformat()
            events.append({'status': status, 'arm': arm, 'stage': stage, 'utc': utc})
            save(STATE, {'status': status, 'arm': arm, 'stage': stage, 'updated_utc': utc,
                 'controller_pid': os.getpid(), 'child_pid': child, 'outputs': {k: str(v.relative_to(HERE)) for k,v in OUTPUTS.items()},
                 'max_training_hours': 6, 'events': events, 'test_read': False, **extra})
        try:
            for arm, stage, cmd in steps:
                if (HERE/'runs/STOP').exists() or (arm in OUTPUTS and (OUTPUTS[arm]/'STOP').exists()):
                    state('stopped', arm, stage); return
                if stage == 'train' and (OUTPUTS[arm]/'results.json').exists():
                    old = json.loads((OUTPUTS[arm]/'results.json').read_text(encoding='utf-8'))
                    if old['status'] in ('early_stop','budget','max_epochs'):
                        continue
                log = HERE/'runs'/f'{arm}_{stage}.log'
                with log.open('a', encoding='utf-8') as stream:
                    child = subprocess.Popen(cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
                    state('running', arm, stage, child.pid)
                    print(f'{arm}/{stage} pid={child.pid} log={log}', flush=True)
                    code = child.wait()
                if code:
                    raise RuntimeError(f'{arm}/{stage} failed with exit {code}; see {log}')
                if stage == 'train':
                    value = json.loads((OUTPUTS[arm]/'results.json').read_text(encoding='utf-8'))
                    if value['status'] not in ('early_stop','budget','max_epochs'):
                        state('stopped', arm, stage); return
                if stage == 'final' and not (OUTPUTS[arm]/'terminal/results.json').exists():
                    state('stopped', arm, 'evaluation_incomplete'); return
                state('stage_complete', arm, stage)
            state('complete', 'all', 'complete', report='runs/review.md', baseline_promoted=False)
        except BaseException as error:
            state('failed', arm, stage, error=repr(error)); raise
        finally:
            verify_protected()


if __name__ == '__main__':
    main()
