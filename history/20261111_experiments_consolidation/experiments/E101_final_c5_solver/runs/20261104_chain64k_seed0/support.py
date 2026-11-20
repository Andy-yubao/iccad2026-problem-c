"""Restore the original E42/E74 runtime snapshots and immutable shared inputs."""
from pathlib import Path
import hashlib,json,os,shutil,zipfile,datetime
ROOT=Path(__file__).resolve().parents[4]
BUNDLE=Path(__file__).resolve().parent
ARCHIVE=ROOT/'history/20261103_final_delivery_consolidation/experiments'
E58=ROOT/'history/20261029_model_research_consolidation/experiments/E58_training_data_v6'
STAGES={'E42':('E42_joint_regularization','20261023_rectangle_w96d12_ab_seed0','s42'),'E74':('E74_collision_32k_posttrain','20261029_e42_topology_seed0','s74')}
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def link(a,b):
    a=a.resolve();b=b.absolute();assert a.is_relative_to(ROOT) and b.is_relative_to(BUNDLE)
    b.parent.mkdir(parents=True,exist_ok=True)
    if b.exists():assert os.path.samefile(a,b)
    else:os.link(a,b)
def mirror(a,b):
    for p in a.rglob('*'):
        if p.is_file() and p.suffix in ('.pt','.pth','.json'):link(p,b/p.relative_to(a))
def original(name):
    for base in [ROOT,ROOT/'history/20261103_final_delivery_consolidation',ROOT/'history/20261101_research_consolidation',ROOT/'history/20261029_model_research_consolidation',ROOT/'history/20261026_flash_pro_consolidation']:
        p=base/name
        if p.exists():return p
    raise FileNotFoundError(name)
def junction(a,b):
    # Historical dataset_paths requires resolved files to remain under its
    # isolated ROOT. Directory junctions escape that guard; file hardlinks do not.
    import stat
    a=a.resolve();b=b.absolute();assert a.is_relative_to(ROOT) and b.is_relative_to(BUNDLE)
    if b.exists() and b.lstat().st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT:
        assert b.resolve()==a,'Refuse to replace an unrelated directory mapping'
        os.rmdir(b)  # Remove only the verified junction itself, never its target.
    b.mkdir(parents=True,exist_ok=True);assert b.resolve()==b
    for source in a.rglob('*'):
        if source.is_file():link(source,b/source.relative_to(a))
