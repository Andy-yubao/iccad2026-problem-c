"""Explicit scheduler-only source review; never silently accept identity drift."""
import json
from .io import HERE

REVIEWABLE={
    'docs/protocol.md',
    *('experiments/E85_cooperative_space_refinement/'+name for name in
      ('train.py','pipeline.py','freeze.py','initial_audit.py','report.py','source_compatibility.py')),
}

def reviewed_sources(old,current):
    if old==current:return True
    path=HERE/'runs/lr_policy_change/source_compatibility.json'
    if not path.exists():return False
    record=json.loads(path.read_text(encoding='utf-8'))
    changes={k for k in old.keys()|current.keys() if old.get(k)!=current.get(k)}
    return (record['status']=='passed_scheduler_only_review'
            and record['old_sources']==old and record['new_sources']==current
            and changes==set(record['changes']) and changes<=REVIEWABLE)
