"""Preparation only; wait for E57 to finish so its runtime evidence is protected."""
import argparse
from datetime import datetime,timezone
import subprocess
import sys
import time
from .io import ROOT,RUN,CACHE,read,save,verify_frozen,exclusive


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--resume',action='store_true');args=ap.parse_args()
    RUN.mkdir(parents=True,exist_ok=True)
    with exclusive(RUN/'pipeline.lock'):
        def state(status,**extra):
            save(RUN/'preparation_status.json',{'status':status,'updated_utc':datetime.now(timezone.utc).isoformat(),
                'model_training_started':False,'test_payloads_read':False,**extra})
        verify_frozen()
        observer=ROOT/'experiments/E57_capacity_data_diagnosis/runs/20261026_study_control/progress.json'
        while observer.exists():
            other=read(observer)
            if other['status'] in ('complete','complete_with_failures','failed','stopped'):break
            state('waiting_for_E57',reason='Avoid CPU/I/O contention during existing frozen training and flash timing',
                E57_task=other.get('current_task'),E57_stage=other.get('stage'))
            if (RUN/'STOP').exists():state('stopped');return
            time.sleep(30)
        if (RUN/'STOP').exists():state('stopped');return
        py=[sys.executable,'-u','-B','-m']
        try:
            state('building_cache',new_train_cases=48000,reused_train_cases=16000)
            subprocess.run(py+['experiments.E58_training_data_v6.cache'],cwd=ROOT,check=True)
            if not (CACHE/'results.json').exists():state('stopped');return
            state('verifying_cache')
            subprocess.run(py+['experiments.E58_training_data_v6.check','--cache-ready'],cwd=ROOT,check=True)
            value=read(CACHE/'results.json');verify_frozen()
            save(RUN/'results.json',{'status':'complete','train_count':64000,'dev_count':2000,
                'local_test_count':2000,'nested_train_views':[16000,32000,64000],
                'cache':CACHE.relative_to(ROOT).as_posix(),'MIB_audit':value['MIB_audit'],
                'old_protocol_unchanged':True,'model_training_started':False,'test_payloads_read':False})
            state('complete',train_count=64000,dev_count=2000,nested_train_views=[16000,32000,64000])
            print('Expanded data and cache ready; no model training launched',flush=True)
        except BaseException as error:
            state('failed',error=str(error));raise


if __name__=='__main__':main()
