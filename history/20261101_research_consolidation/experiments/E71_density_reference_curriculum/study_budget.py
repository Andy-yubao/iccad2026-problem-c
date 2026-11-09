"""One bounded combined candidate; never resumes the stopped long task."""
from datetime import datetime,timezone
import json
from .io import HERE,save,exclusive

PATH=HERE/'runs/budget_ledger.json'
ALLOWED={'20261028_full_ports_density_curriculum_v1_seed0'}


def register(output,max_hours,implementation_check=False):
    if implementation_check:return None
    if output.name not in ALLOWED or not 0<max_hours<=3:raise ValueError('Unregistered E71 run or budget')
    with exclusive(PATH.with_suffix('.lock')):
        data=json.loads(PATH.read_text()) if PATH.exists() else {'runs':{},'max_cumulative_seconds':10800}
        data['runs'].setdefault(output.name,{'elapsed_seconds':0.,'status':'registered'})
        other=sum(v['elapsed_seconds'] for k,v in data['runs'].items() if k!=output.name)
        remaining=data['max_cumulative_seconds']-other
        if data['runs'][output.name]['elapsed_seconds']>=remaining:raise RuntimeError('E71 cumulative budget exhausted')
        save(PATH,data)
    return {'remaining_including_this':remaining}


def exhausted(guard,elapsed):return guard is not None and elapsed>=guard['remaining_including_this']


def charge(output,elapsed,status='running',*,implementation_check=False):
    if implementation_check:return
    with exclusive(PATH.with_suffix('.lock')):
        data=json.loads(PATH.read_text());r=data['runs'][output.name]
        r.update(elapsed_seconds=max(elapsed,r['elapsed_seconds']),status=status,updated_utc=datetime.now(timezone.utc).isoformat())
        data['cumulative_seconds']=sum(v['elapsed_seconds'] for v in data['runs'].values());save(PATH,data)
