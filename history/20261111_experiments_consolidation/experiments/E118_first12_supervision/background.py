"""Independent hidden Windows task entry; no dependency on command-window lifetime."""
import os
import sys
from pathlib import Path

def main():
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';os.environ['PYTHONUNBUFFERED']='1'
    logs=Path(__file__).parent/'launch_logs';logs.mkdir(exist_ok=True)
    sys.stdout=(logs/'controller.stdout.log').open('x',encoding='utf-8',buffering=1)
    sys.stderr=(logs/'controller.stderr.log').open('x',encoding='utf-8',buffering=1)
    import faulthandler
    faulthandler.enable(file=sys.stderr)
    from .run_queue import main as run
    try:run()
    except BaseException:
        import traceback
        traceback.print_exc();raise

if __name__=='__main__':main()
