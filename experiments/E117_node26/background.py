"""Hidden Windows scheduled task entry, independent of the command window job."""
import os
from pathlib import Path
import sys

def main():
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    os.environ['PYTHONUNBUFFERED']='1'
    folder=Path(__file__).parent/'launch_logs/recovery1';folder.mkdir(parents=True,exist_ok=True)
    sys.stdout=(folder/'controller.stdout.log').open('x',encoding='utf-8',buffering=1)
    sys.stderr=(folder/'controller.stderr.log').open('x',encoding='utf-8',buffering=1)
    import faulthandler
    faulthandler.enable(file=sys.stderr)
    from .run_queue import main as run
    sys.argv=[sys.argv[0],'--authorized-recovery']
    try:run()
    except BaseException:
        import traceback
        traceback.print_exc();raise

if __name__=='__main__':main()
