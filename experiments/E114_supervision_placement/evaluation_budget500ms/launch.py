"""One background evaluation, separate logs, no retries or additional jobs."""
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime,timezone
from ..io import save,exclusive,ROOT,sha

HERE=Path(__file__).resolve().parent

def main():
    with exclusive(HERE/'launch.lock'):
        if (HERE/'launch.json').exists() or (HERE/'output').exists():
            raise FileExistsError('Budget500ms evaluation already dispatched; no automatic retry')
        for path in HERE.rglob('*.py'):
            ast.parse(path.read_text(encoding='utf-8'))
        plan=json.loads((HERE/'plan.json').read_text(encoding='utf-8'))
        model=plan['models']['staged']
        assert sha(ROOT/model['checkpoint'])==model['checkpoint_sha256']
        assert not (HERE/'STOP').exists()
        source_identity={p.relative_to(ROOT).as_posix():sha(p) for p in HERE.rglob('*.py')}
        with (HERE/'stdout.log').open('x',encoding='utf-8') as out, (HERE/'stderr.log').open('x',encoding='utf-8') as err:
            child=subprocess.Popen([sys.executable,'-u','-B','-m',
                'experiments.E114_supervision_placement.evaluation_budget500ms.run','--arm','staged'],
                cwd=ROOT,stdout=out,stderr=err,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            state=dict(status='running',controller_pid=os.getpid(),pid=child.pid,
                started_at=datetime.now(timezone.utc).isoformat(),source_identity=source_identity,
                count=500,role='original fixed dev_monitor500',test_read=False,
                training=False,work_budget_multiplier=3,target_mean_complete_seconds=.5,
                automatic_retry=False)
            save(HERE/'launch.json',state)
            print(f'Background budget500ms evaluation started, pid={child.pid}',flush=True)
            while child.poll() is None:
                time.sleep(5)
            state.update(status='complete' if child.returncode==0 else 'needs_attention',
                returncode=child.returncode,finished_at=datetime.now(timezone.utc).isoformat())
            save(HERE/'launch.json',state)
            if child.returncode:raise RuntimeError('Budget evaluation failed; evidence retained, no retry')

if __name__=='__main__':main()
