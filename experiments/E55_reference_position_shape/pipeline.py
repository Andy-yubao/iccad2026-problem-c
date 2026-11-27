"""One authorized scratch run, immediately followed by its sole flash500 check."""
import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from .io import ROOT, HERE, save, exclusive

OUTPUT = HERE/'runs/20261026_reference_position_shape_seed0'
CONTROL = HERE/'runs/20261026_reference_position_shape_seed0_control'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    train = [sys.executable, '-u', '-B', '-m', 'experiments.E55_reference_position_shape.train',
             '--arm', 'reference_position_shape', '--output', str(OUTPUT),
             '--max-hours', '2', '--max-epochs', '200', '--batch-size', '32', '--seed', '0',
             '--device', 'cuda', '--engine', 'cuda_graph']
    if args.resume and (OUTPUT/'checkpoints/latest.pt').exists():
        train.append('--resume')
    commands = [('train', train),
                ('flash500', [sys.executable, '-u', '-B', '-m', 'experiments.E55_reference_position_shape.final',
                              '--output', str(OUTPUT), '--stage', 'terminal']),
                ('draft100', [sys.executable, '-u', '-B', '-m', 'experiments.E55_reference_position_shape.render_drafts', '--output', str(OUTPUT)]),
                ('report', [sys.executable, '-u', '-B', '-m', 'experiments.E55_reference_position_shape.report'])]
    if args.dry_run:
        print(json.dumps({'commands': commands, 'test_read': False, 'starts_processes': False}, indent=2))
        return
    CONTROL.mkdir(parents=True, exist_ok=True)
    with exclusive(CONTROL/'pipeline.lock'):
        events_path = CONTROL/'pipeline.json'
        events = json.loads(events_path.read_text(encoding='utf-8')).get('events', []) if events_path.exists() else []
        def state(status, stage):
            events.append({'status': status, 'stage': stage, 'utc': datetime.now(timezone.utc).isoformat()})
            save(events_path, {'status': status, 'stage': stage, 'events': events, 'output': str(OUTPUT), 'test_read': False})
        try:
            for stage, command in commands:
                if (HERE/'runs/STOP').exists() or (OUTPUT/'STOP').exists():
                    state('stopped', stage); return
                if stage == 'train' and (OUTPUT/'results.json').exists():
                    result = json.loads((OUTPUT/'results.json').read_text(encoding='utf-8'))
                    if result['status'] in ('early_stop', 'budget', 'max_epochs'):
                        continue
                if stage == 'flash500':
                    result = json.loads((OUTPUT/'results.json').read_text(encoding='utf-8'))
                    if result['status'] not in ('early_stop', 'budget', 'max_epochs'):
                        state('stopped', 'training_not_finished'); return
                state('running', stage)
                print('Stage:', stage, 'Command:', subprocess.list2cmdline(command), flush=True)
                subprocess.run(command, cwd=ROOT, check=True)
                if stage == 'flash500' and not (OUTPUT/'terminal/results.json').exists():
                    state('stopped', 'flash500_incomplete'); return
                state('stage_complete', stage)
            state('complete', 'all')
        except BaseException as error:
            state('failed', str(error)); raise


if __name__ == '__main__':
    main()
