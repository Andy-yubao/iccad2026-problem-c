"""Experiment-local identities, atomic evidence and process locks."""
import hashlib,json,os,time,uuid,zipfile
from pathlib import Path
from contextlib import contextmanager
import torch
from datasets.paths import ROOT
HERE=Path(__file__).resolve().parent
OWNER=HERE
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.'+uuid.uuid4().hex+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    # Windows readers can briefly deny atomic replacement; bounded metadata-only retry.
    for attempt in range(20):
        try:
            os.replace(tmp,p);break
        except PermissionError:
            if attempt==19:raise
            time.sleep(.05)
def checkpoint(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.'+uuid.uuid4().hex+'.tmp');torch.save(value,tmp);os.replace(tmp,p)
def sources():
    excluded={'runs','data_cache_v1','background','preparation_diagnostics'}
    paths=[p for p in HERE.rglob('*.py') if not set(p.relative_to(HERE).parts)&excluded]
    paths += [HERE/n for n in ('plan.json','README.md','DESIGN.md','evaluation/budget.json') if (HERE/n).exists()]
    paths += [ROOT/n for n in ('protocol/random_mib40k_v1.json','protocol/research_mib_restored_v1.json','protocol/model_prediction_boundary_v1.json','protocol/paths.json','datasets/cases.py')]
    data=ROOT/'datasets/random_mib40k_v1'
    paths += list(data.glob('*.py'))
    paths += [data/n for n in ('manifest.json','report.json','cache/train/manifest.json','cache/dev/manifest.json','cache/profile.json')]
    return {p.relative_to(ROOT).as_posix():sha(p) for p in sorted(paths)}
def snapshot(p,identity):
    with zipfile.ZipFile(p,'w',zipfile.ZIP_DEFLATED) as z:
        for name in identity['sources']:z.write(ROOT/name,name)
@contextmanager
def exclusive(p):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a+b') as f:
        if not f.tell():f.write(b'0');f.flush()
        f.seek(0)
        import msvcrt
        msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        try:yield
        finally:f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
