from pathlib import Path
import hashlib
import json
import os
import time
import uuid
from contextlib import contextmanager
from datasets.paths import ROOT

HERE=Path(__file__).resolve().parent
RUN=HERE/'runs/20261026_reference_v6_64k'
CACHE=RUN/'cache'
REGISTRY=ROOT/'protocol/reference_workset_v6.json'


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def replace(temp,path):
    for i in range(20):
        try:os.replace(temp,path);return
        except PermissionError:
            if i==19:raise
            time.sleep(.05)


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    replace(temp,path)


def verify_frozen():
    expected=read(HERE/'provenance.json')['frozen_existing_protocol']
    for name,digest in expected.items():
        if sha(ROOT/name)!=digest:raise ValueError('Frozen existing file changed: '+name)
    return expected


@contextmanager
def exclusive(path):
    import msvcrt
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+b') as f:
        if f.tell()==0:f.write(b'0');f.flush()
        f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        try:yield
        finally:f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
