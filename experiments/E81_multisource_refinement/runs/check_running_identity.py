"""Read a durable running checkpoint; verify frozen placers and source snapshot."""
import sys,json
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import torch
from experiments.E81_multisource_refinement.io import HERE,sources,sha,save
from experiments.E81_multisource_refinement.initialization import widen
out=HERE/'runs/20261029_e42_multisource_seed0'
state=torch.load(out/'checkpoints/latest.pt',map_location='cpu',weights_only=False,mmap=True)
p=json.loads((HERE/'provenance.json').read_text(encoding='utf-8'))['parents']
old=torch.load(ROOT/p['correction']['checkpoint'],map_location='cpu',weights_only=False,mmap=True)
placer=torch.load(ROOT/p['placer']['checkpoint'],map_location='cpu',weights_only=False,mmap=True)
assert state['identity']['sources']==sources()
assert not state['identity']['implementation_check'] and state['progress']['step']>0
frozen_count=0
for k,v in placer['model'].items():
    assert torch.equal(state['model']['placer.'+k],v);frozen_count+=1
prefix='source.source.backbone.'
for k,v in old['model'].items():
    if k.startswith(prefix):
        assert torch.equal(state['model']['alternative.'+k[len(prefix):]],v);frozen_count+=1
changes={}
for new,prefix in (('first','source.source.corrector.'),('second','source.corrector.'),('third','corrector.')):
    original=widen({k[len(prefix):]:v for k,v in old['model'].items() if k.startswith(prefix)})
    changes[new]=sum(not torch.equal(state['model']['refiner.'+new+'.'+k],v) for k,v in original.items())
    assert changes[new]>0
value={'checked_utc':datetime.now(timezone.utc).isoformat(),'durable_step':state['progress']['step'],
 'status':state['status'],'source_identity_matches_final_preflight':True,
 'frozen_placer_tensors_bit_exact':frozen_count,'changed_refiner_tensors_by_stage':changes,
 'actual_training_not_smoke':True,'test_read':False,'parent_sha256':{k:v['sha256'] for k,v in p.items()}}
save(HERE/'runs/runtime_identity_check.json',value);print(json.dumps(value))
