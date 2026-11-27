"""Serial retained-model controls, after all E19/E20 CPU measurements finish."""
import json
import os
import subprocess
import sys
import time
from .io import ROOT,HERE,save,exclusive


def main():
    logs=HERE/'runs/20261020_background';logs.mkdir(parents=True,exist_ok=True)
    parent=ROOT/'experiments/E20_mib_fast_pipeline/runs/20261020_background/pipeline.json'
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='2'
    with exclusive(logs/'pipeline.lock'):
        save(logs/'pipeline.json',{'stage':'wait_for_E20','status':'waiting','test_read':False})
        print('Waiting for E20 direct screens; CPU measurements remain serial',flush=True)
        started=time.monotonic()
        while True:
            status=json.loads(parent.read_text())
            if status['stage']=='direct_screen100' and status['status']=='complete':break
            if status['status'] in ('failed','paused'):raise RuntimeError(f'E20 needs attention: {status}')
            if time.monotonic()-started>5*3600:raise TimeoutError('Upstream stage exceeded waiting protection budget')
            time.sleep(30)
        command=[sys.executable,'-u','-B','-m','experiments.E21_original_model_fast.run',
            '--tag','20261020_model_control','--count','100','--configs','lp_gate','qp_gate','weighted','--direct']
        save(logs/'pipeline.json',{'stage':'direct_control100','status':'running','command':command,'test_read':False})
        print('E20 completed. Comparing original retained model on identical public panel',flush=True)
        subprocess.run(command,cwd=ROOT,check=True)
        save(logs/'pipeline.json',{'stage':'direct_control100','status':'complete','command':command,'test_read':False})


if __name__=='__main__':main()
