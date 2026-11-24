"""Run the registered E164 candidate and freeze its Loss-best on normal completion."""
import argparse
import json
import os
import subprocess
import sys

from .io import HERE, ROOT
from .report import finalize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    environment = dict(os.environ, E164_EXECUTION_PLATFORM='local',
                       CUBLAS_WORKSPACE_CONFIG=':4096:8')
    command = [sys.executable, '-u', '-B', '-m', f'{__package__}.train', '--arm', 'e164_l0']
    if args.resume:
        command.append('--resume')
    result = subprocess.run(command, cwd=ROOT, env=environment)
    if result.returncode:
        return result.returncode
    run = HERE / 'runs/e164_l0_seed0'
    training = json.loads((run / 'results.json').read_bytes())
    if training['status'] != 'user_stop':
        finalize(run)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
