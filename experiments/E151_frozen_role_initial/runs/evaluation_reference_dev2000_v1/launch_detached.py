"""Resume the authorized evaluation outside the tool-session Windows job."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]


def save(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def main():
    if os.name!='nt':raise RuntimeError('This launcher is Windows-specific')
    state_path=HERE/'output/state.json'
    state=json.loads(state_path.read_text(encoding='utf-8'))
    background=HERE/'background/detached_resume_504'
    background.mkdir(exist_ok=False)
    (background/'interrupted_state.json').write_bytes(state_path.read_bytes())
    state.update(status='paused',stage='host_process_exited_at_committed_case_boundary',
        interruption='PID30380 no longer exists; stderr empty; 504 committed cases retained; evaluator/model/data/budget unchanged')
    save(state_path,state)
    command=[sys.executable,'-u','-B','-m',
        'experiments.E151_frozen_role_initial.runs.evaluation_reference_dev2000_v1.run',
        '--arm','e151_best45','--resume']
    environment=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONIOENCODING='utf-8')
    startup=subprocess.STARTUPINFO()
    startup.dwFlags|=subprocess.STARTF_USESHOWWINDOW
    startup.wShowWindow=0
    flags=(subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP|
        subprocess.CREATE_BREAKAWAY_FROM_JOB)
    record=dict(command=command,cwd=str(ROOT),completed_before_resume=state['completed'],
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        creationflags=flags,tests_run=False,checkpoint_and_sources_unchanged=True)
    try:
        with (background/'stdout.log').open('ab',buffering=0) as stdout,(background/'stderr.log').open('ab',buffering=0) as stderr:
            process=subprocess.Popen(command,cwd=ROOT,env=environment,stdin=subprocess.DEVNULL,
                stdout=stdout,stderr=stderr,startupinfo=startup,creationflags=flags,close_fds=True)
        record.update(status='process_created',pid=process.pid)
        save(background/'launch.json',record)
        print(json.dumps(record,ensure_ascii=False),flush=True)
    except BaseException as error:
        record.update(status='launch_failed',error=repr(error))
        save(background/'launch.json',record)
        raise


if __name__=='__main__':main()
