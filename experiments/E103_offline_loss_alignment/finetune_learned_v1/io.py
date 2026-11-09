"""Atomic checkpoints with the verified Windows replacement retry policy."""
import os
import time
import uuid
from pathlib import Path
import torch
from ..common import ROOT,OWNER as HERE,sha,read,save

def checkpoint(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    torch.save(value,temporary)
    deadline=time.monotonic()+5.;retries=0
    while True:
        try:os.replace(temporary,path);return
        except PermissionError as exc:
            if os.name!='nt' or exc.winerror not in (5,32,33) or time.monotonic()>=deadline:raise
            retries+=1;time.sleep(min(.01*retries,.2,max(0.,deadline-time.monotonic())))
