"""Hidden one-shot queue entry with durable output, no retries or automatic resume."""
import argparse
import datetime
import os
import sys
import traceback
from .io import HERE,save,exclusive

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--start',action='store_true')
    mode.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    folder=HERE/'background';folder.mkdir(exist_ok=True)
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    path=folder/f'{stamp}_{os.getpid()}.log'
    with exclusive(folder/'launcher.lock'),path.open('x',encoding='utf-8',buffering=1) as stream:
        sys.stdout=sys.stderr=stream
        save(folder/'active.json',dict(pid=os.getpid(),log=path.relative_to(HERE).as_posix(),
            mode='start' if args.start else 'resume',started_at=stamp))
        try:
            from . import run_queue
            sys.argv=['run_queue','--start' if args.start else '--resume']
            run_queue.main()
        except BaseException:
            traceback.print_exc();raise

if __name__=='__main__':main()
