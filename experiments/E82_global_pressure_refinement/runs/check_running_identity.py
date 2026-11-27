"""Verify actual updates only touch the three new pressure embeddings."""
import sys,json
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import torch
from experiments.E82_global_pressure_refinement.io import HERE,sources,save
out=HERE/'runs/20261029_e81_pressure_seed0';state=torch.load(out/'checkpoints/latest.pt',map_location='cpu',weights_only=False,mmap=True)
meta=json.loads((HERE/'provenance.json').read_text(encoding='utf-8'));source=torch.load(ROOT/meta['initialization_checkpoint'],map_location='cpu',weights_only=False,mmap=True)
assert state['identity']['sources']==sources() and not state['identity']['implementation_check'] and state['progress']['step']>0
for k,v in source['model'].items():assert torch.equal(v,state['model'][k]),k
new={k:v for k,v in state['model'].items() if '.pressure.' in k}
assert len(new)==3 and sum(v.numel() for v in new.values())==2304 and all(torch.count_nonzero(v)>0 for v in new.values())
assert state['identity']['trainable_parameters']==2304
result={'checked_utc':datetime.now(timezone.utc).isoformat(),'durable_step':state['progress']['step'],
 'frozen_E81_tensors_bit_exact':len(source['model']),'new_weight_tensors_updated':3,'trainable_parameters':2304,
 'sources_match_preflight':True,'source_commit_after_launch':'2f392f7','test_read':False}
save(HERE/'runs/runtime_identity_check.json',result)
study_path=ROOT/'experiments/E81_multisource_refinement/study.json';study=json.loads(study_path.read_text(encoding='utf-8'))
entry=next(r for r in study['run_ledger'] if r['experiment']==HERE.name);entry.update(source_commit='2f392f7',runtime_identity_check=result)
study_path.write_text(json.dumps(study,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result))
