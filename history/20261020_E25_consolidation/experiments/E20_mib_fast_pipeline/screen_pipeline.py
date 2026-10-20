"""Wait for the active model's primary control, then run three direct screens."""
import json
import os
import subprocess
import sys
import time
from .io import ROOT,HERE,save,exclusive


def main():
    logs=HERE/'runs/20261020_background';logs.mkdir(parents=True,exist_ok=True)
    parent=ROOT/'experiments/E19_mib_consistent_geometry/runs/20261020_background/pipeline.json'
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='2'
    with exclusive(logs/'pipeline.lock'):
        print('Waiting for E19 training and primary dev500 control; no timed competition',flush=True)
        save(logs/'pipeline.json',{'stage':'wait_for_E19','status':'waiting','test_read':False})
        started=time.monotonic()
        while True:
            status=json.loads(parent.read_text())
            if status['stage']=='final_dev500' and status['status']=='complete':break
            if status['status'] in ('failed','paused'):raise RuntimeError(f'E19 needs attention: {status}')
            if time.monotonic()-started>5*3600:raise TimeoutError('E19 did not finish within the waiting protection budget')
            time.sleep(30)
        command=[sys.executable,'-u','-B','-m','experiments.E20_mib_fast_pipeline.run',
            '--tag','20261020_direct_screen','--count','100','--configs','fast','lp_gate','qp_gate','--direct']
        save(logs/'pipeline.json',{'stage':'direct_screen100','status':'running','command':command,'test_read':False})
        print('E19 completed. Running fixed-model direct feature/inference/postprocessing screens',flush=True)
        subprocess.run(command,cwd=ROOT,check=True)
        save(logs/'pipeline.json',{'stage':'direct_screen100','status':'complete','command':command,'test_read':False})


if __name__=='__main__':main()
