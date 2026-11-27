import hashlib,json
from pathlib import Path
from datasets.paths import ROOT
HERE=Path(__file__).resolve().parent
OUT=HERE/'runs/20261027_best51_five_train'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
