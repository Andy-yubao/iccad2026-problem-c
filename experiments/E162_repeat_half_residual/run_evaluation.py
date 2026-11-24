"""Run the frozen half/full repeated-stage profile on the original dev2000."""
import argparse
from datetime import datetime, timezone
import json
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OWNER = HERE / 'evaluation'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', action='store_true', required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    from .io import save, sha
    auth = json.loads((HERE / 'authorization.json').read_text(encoding='utf-8'))
    if not auth['postprocessing_enabled'] or auth['training_enabled']:
        raise PermissionError('Only the registered inference evaluation is authorized')
    plan = json.loads((OWNER / 'plan.json').read_text(encoding='utf-8'))
    frozen = json.loads((OWNER / 'frozen.json').read_text(encoding='utf-8'))
    for name, expected in plan['sources'].items():
        if sha(ROOT / name) != expected:
            raise ValueError('Frozen evaluation source changed: ' + name)
    if sha(ROOT / frozen['checkpoint']) != frozen['checkpoint_sha256']:
        raise ValueError('Frozen checkpoint changed')
    from .evaluation import run as evaluator
    launch = dict(pid=os.getpid(), started_utc=datetime.now(timezone.utc).isoformat(),
                  checkpoint=frozen['checkpoint'], checkpoint_sha256=frozen['checkpoint_sha256'],
                  profile='L0 -> S0(.5,1) -> S1(.5,1) -> S2(.5,1)',
                  training_updates=0, total=2000, test_read=False, resume=args.resume)
    save(OWNER / 'background/launch.json', launch)
    print(json.dumps(dict(stage='formal_local_dev2000', **launch)), flush=True)
    sys.argv = [str(__file__), '--arm', 'e162_repeat_half'] + (['--resume'] if args.resume else [])
    code = 0
    try:
        evaluator.main()
    except BaseException as error:
        code = 1
        save(OWNER / 'background/entry_error.json', dict(error=repr(error), traceback=traceback.format_exc()))
        raise
    finally:
        path = OWNER / 'output/state.json'
        state = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
        save(OWNER / 'background/exit.json', dict(pid=os.getpid(), exit_code=code,
             ended_utc=datetime.now(timezone.utc).isoformat(), status=state.get('status'),
             completed=state.get('completed', 0)))


if __name__ == '__main__':
    main()
