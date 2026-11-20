"""Identity and fixed paths for one candidate, with old algorithms left intact."""
import json
from pathlib import Path
from ..io import ROOT,HERE,sha,save
from ..dynamic_weight_learning_v2.common import sources as shared_sources,PhaseSchedule,q_summary

OWNER=Path(__file__).resolve().parent
RUN=HERE/'runs/20261104_weaken_reference120_seed0'
PLAN_PATH=OWNER/'plan.json'
MODEL=dict(width=96,layers=12,rectangle=True,ffn_dropout=.10,residual_dropout=.05)

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def sources():
    files=list(OWNER.glob('*.py'))+[OWNER/'README.md',OWNER/'launch.ps1',PLAN_PATH]
    return {**shared_sources(),**{p.relative_to(ROOT).as_posix():sha(p) for p in files}}
