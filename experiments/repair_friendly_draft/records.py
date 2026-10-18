"""Local run provenance and atomic writes; no shared algorithm abstraction."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
import platform
import zipfile
from datetime import datetime,timezone
from datasets.paths import ROOT,resolved_data_paths

HERE=Path(__file__).resolve().parent


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    replace(tmp,path)


def replace(source,target):
    # Windows readers/antivirus can briefly open the destination without delete
    # sharing. Retry atomic replacement; never truncate the previous checkpoint.
    for attempt in range(8):
        try:os.replace(source,target);return
        except PermissionError:
            if attempt==7:raise
            time.sleep(min(.03*2**attempt,.5))


def append(path,value):
    with Path(path).open('a',encoding='utf-8') as f:
        f.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n');f.flush()


def source_hashes():
    paths=[*HERE.glob('*.py'),*sorted((ROOT/'datasets').glob('*.py')),
           *sorted((ROOT/'scoring').glob('*.py')),*sorted((ROOT/'protocol').glob('*.json')),
           *sorted(HERE.glob('*.json')),ROOT/'environment.yml',
           ROOT/'third_party/FloorSet/iccad2026contest/iccad2026_evaluate.py']
    return {p.relative_to(ROOT).as_posix():sha(p) for p in paths if p.exists()}


def provenance():
    import torch
    import numpy,scipy,highspy,shapely
    return {'command':[sys.executable,*sys.argv],'created_utc':datetime.now(timezone.utc).isoformat(),
            'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'git_status':subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),
            'data_paths':resolved_data_paths(),'source_sha256':source_hashes(),
            'torch':torch.__version__,'cuda':torch.version.cuda,
            'python':platform.python_version(),'numpy':numpy.__version__,'scipy':scipy.__version__,
            'highs':highspy.Highs().version(),'shapely':shapely.__version__,
            'device':torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu'}


def snapshot(folder,hashes):
    with zipfile.ZipFile(Path(folder)/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in hashes:z.write(ROOT/name,name)


def run_path(path):
    path=Path(path).resolve()
    if not path.is_relative_to(HERE/'runs') or path==HERE/'runs':raise ValueError('Output must be inside this experiment runs/')
    return path
