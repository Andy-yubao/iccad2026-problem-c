"""Launch the one shared read-only viewer for three frozen saved galleries."""
from pathlib import Path
import datetime
import json
import os
import subprocess
import sys

RUNS=Path(__file__).resolve().parent
ROOT=RUNS.parents[2]
LOGS=ROOT/'.cache/visualization'

def main():
    LOGS.mkdir(parents=True,exist_ok=True)
    workers=[]
    for arm,port in (('e119_A',8784),('e119_B',8785),('e119_C',8786)):
        folder=RUNS/f'{arm}_seed0/visualization_dev15_loss'
        state=json.loads((folder/'state.json').read_text(encoding='utf-8'))
        if state['status']!='complete':raise ValueError('Gallery is incomplete: '+arm)
        command=[sys.executable,'-B','-m','visualization','--data',str(folder/'data.json'),
                 '--port',str(port),'--log',str(LOGS/f'e135_{arm}.service.log')]
        flags=subprocess.CREATE_NO_WINDOW|subprocess.CREATE_NEW_PROCESS_GROUP if os.name=='nt' else 0
        child=subprocess.Popen(command,cwd=ROOT,creationflags=flags)
        workers.append(dict(arm=arm,port=port,pid=child.pid,url=f'http://127.0.0.1:{port}/',
            data=(folder/'data.json').relative_to(ROOT).as_posix()))
    (RUNS/'viewer_services.json').write_text(json.dumps(dict(parent_pid=os.getpid(),
        started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),services=workers,
        read_only=True,automatic_restart=False),indent=2)+'\n',encoding='utf-8')
    # Keep the independent task alive. No restart/retry is performed.
    import time
    while all(_alive(w['port']) for w in workers):time.sleep(10)

def _alive(port):
    # Allow initialization before checking HTTP; use process-independent listeners.
    import socket,time
    for _ in range(10):
        try:
            with socket.create_connection(('127.0.0.1',port),timeout=1):return True
        except OSError:time.sleep(.5)
    return False

if __name__=='__main__':main()
