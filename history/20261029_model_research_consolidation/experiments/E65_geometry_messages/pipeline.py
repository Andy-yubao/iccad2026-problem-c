"""Fixed seed0 pair, independently evaluated; hourly agent handles the next mechanism."""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from .io import ROOT, HERE, save, exclusive, sources
from .initialization import verify_protected

ARMS = ('geometry_message', 'feature_message')
OUTPUTS = {arm: HERE/f'runs/20261027_{arm}_seed0' for arm in ARMS}
STATE = HERE/'state.json'


def commands(arm, resume=False):
    output = OUTPUTS[arm]
    train = [sys.executable, '-u', '-B', '-m', 'experiments.E65_geometry_messages.train', '--arm', arm,
             '--output', str(output), '--max-hours', '3', '--max-epochs', '200',
             '--batch-size', '32', '--seed', '0', '--device', 'cuda', '--engine', 'cuda_graph']
    if resume and (output/'checkpoints/latest.pt').exists():
        train.append('--resume')
    return [('train', train),
            ('final', [sys.executable, '-u', '-B', '-m', 'experiments.E65_geometry_messages.final', '--output', str(output), '--stage', 'all']),
            ('draft100', [sys.executable, '-u', '-B', '-m', 'experiments.E65_geometry_messages.render_drafts', '--output', str(output)])]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.dry_run:
        print(json.dumps({arm: commands(arm, args.resume) for arm in ARMS}, indent=2))
        return
    with exclusive(HERE/'runs/pipeline.lock'):
        verify_protected()
        preflight = json.loads((HERE/'runs/20261027_preflight/results.json').read_text(encoding='utf-8'))
        if preflight['status'] != 'passed' or preflight['sources'] != sources():
            raise ValueError('Current sources require a passing five-case preflight')
        prior = json.loads(STATE.read_text(encoding='utf-8')) if STATE.exists() else {}
        events = prior.get('events', [])
        failures = {}

        def state(status, arm, stage, child=None, **extra):
            utc = datetime.now(timezone.utc).isoformat()
            events.append({'status': status, 'arm': arm, 'stage': stage, 'utc': utc})
            save(STATE, {'status': status, 'arm': arm, 'stage': stage, 'updated_utc': utc,
                        'controller_pid': os.getpid(), 'child_pid': child,
                        'outputs': {key: str(value.relative_to(HERE)) for key, value in OUTPUTS.items()},
                        'events': events, 'failures': failures, 'test_read': False, **extra})

        try:
            for arm in ARMS:
                if (HERE/'runs/STOP').exists():
                    state('stopped', arm, 'before_arm')
                    return
                try:
                    for stage, command in commands(arm, args.resume):
                        if (HERE/'runs/STOP').exists() or (OUTPUTS[arm]/'STOP').exists():
                            state('stopped', arm, stage)
                            return
                        if stage == 'train' and (OUTPUTS[arm]/'results.json').exists():
                            old = json.loads((OUTPUTS[arm]/'results.json').read_text(encoding='utf-8'))
                            if old['status'] in ('early_stop', 'budget', 'max_epochs'):
                                continue
                        log = HERE/'runs'/f'{arm}_{stage}.log'
                        with log.open('a', encoding='utf-8') as stream:
                            child = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
                            state('running', arm, stage, child.pid)
                            print(f'{arm}/{stage} pid={child.pid} log={log}', flush=True)
                            code = child.wait()
                        if code:
                            raise RuntimeError(f'{arm}/{stage} exited {code}; see {log}')
                        if stage == 'train':
                            value = json.loads((OUTPUTS[arm]/'results.json').read_text(encoding='utf-8'))
                            if value['status'] not in ('early_stop', 'budget', 'max_epochs'):
                                state('stopped', arm, stage)
                                return
                        if stage == 'final' and not (OUTPUTS[arm]/'terminal/results.json').exists():
                            state('stopped', arm, 'evaluation_incomplete')
                            return
                        state('stage_complete', arm, stage)
                except Exception as error:
                    failures[arm] = repr(error)
                    state('arm_failed', arm, stage, error=repr(error))
                    # Independent predeclared arm may still run; no repair or extra variant is invented.
                    continue
            log = HERE/'runs/report.log'
            with log.open('a', encoding='utf-8') as stream:
                command = [sys.executable, '-u', '-B', '-m', 'experiments.E65_geometry_messages.report']
                child = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
                state('running', 'all', 'report', child.pid)
                if child.wait():
                    raise RuntimeError('E65 report failed; see '+str(log))
            state('complete_with_failures' if failures else 'complete', 'all', 'complete',
                  report='runs/review.md', baseline_promoted=False,
                  next_action='Hourly continuation: review E65, then implement E66 under long_task.md; all seed0')
            with exclusive((HERE/'long_task.json').with_suffix('.lock')):
                overall = json.loads((HERE/'long_task.json').read_text(encoding='utf-8'))
                overall['stages'][0]['status'] = 'complete_with_failures' if failures else 'complete'
                overall['next_action'] = 'Review E65 and implement E66; no multi-seed confirmation'
                save(HERE/'long_task.json', overall)
        finally:
            verify_protected()


if __name__ == '__main__':
    main()
