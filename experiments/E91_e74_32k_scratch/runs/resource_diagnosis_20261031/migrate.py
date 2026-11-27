"""One-time audited data-residency migration; no neural updates or budget reset."""
import json
from pathlib import Path
import numpy as np
import torch
from experiments.E91_e74_32k_scratch.io import HERE,save,sha,checkpoint,snapshot
from experiments.E91_e74_32k_scratch.preflight import verify
from experiments.E91_e74_32k_scratch.train import parser,make_identity
from experiments.E91_e74_32k_scratch.loss_config import load_coefficients

torch.set_num_threads(4)

def same(a,b):
    if isinstance(a,torch.Tensor):assert torch.equal(a,b)
    elif isinstance(a,np.ndarray):assert np.array_equal(a,b) and a.dtype==b.dtype
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:same(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):same(x,y)
    else:assert a==b

audit=Path(__file__).resolve().parent
assert not (audit/'migration.json').exists(),'Already migrated; never rerun'
verify()
out=HERE/'runs/20261031_scratch32k_seed0'
stop=json.loads((audit/'stop.json').read_text(encoding='utf-8'))
path=out/'checkpoints/latest.pt'
intermediate_sha=sha(path)
assert intermediate_sha=='164803eaf80641744b91c0897fdcb9a8f9481b64dbc6f5ff553c29c242b59844'
c=torch.load(path,map_location='cpu',weights_only=False)
old=torch.load(audit/'before/latest.pt',map_location='cpu',weights_only=False)
assert sha(audit/'before/latest.pt')==stop['checkpoint_sha256']
for key in ('model','model_config','optimizer','scheduler','rng','progress','finished','graph_step_control'):
    same(c[key],old[key])
# First attempt wrote only the host identity, then its generic verifier rejected
# NumPy arrays. Validate that intermediate binding explicitly before the new budget.
host_check=json.loads((audit/'host_preflight_before_budget.json').read_text(encoding='utf-8'))
assert c['identity']['sources']==host_check['sources']
assert c['identity']['max_seconds']==old['identity']['max_seconds']==7200.
assert c['progress']['step']==0 and c['progress']['epoch']==0 and c['progress']['pending_monitor']
assert not c['finished'] and not (out/'checkpoints/best.pt').exists()
args=parser().parse_args(['--output',str(out),'--max-hours','2','--max-epochs','200','--resume'])
identity=make_identity(args)
identity['calibration']=load_coefficients(HERE/'runs/registered_loss.json')
changed=[k for k in set(old['identity'])|set(identity) if old['identity'].get(k)!=identity.get(k)]
assert set(changed)=={'sources','calibration','gpu_resident_inputs','data_transfer','max_seconds',
                       'additional_training_budget_seconds','preserved_prior_training_seconds'},changed
changed_sources=[k for k in identity['sources'] if old['identity']['sources'].get(k)!=identity['sources'][k]]
assert set(changed_sources)=={f'experiments/E91_e74_32k_scratch/{name}' for name in ('resident.py','train.py','preflight.py','plan.json')}
charged=max(stop['charged_seconds'],json.loads((out/'budget.json').read_text())['elapsed_seconds'],c['elapsed_seconds'])
assert charged<identity['max_seconds']
assert identity['max_seconds']-charged==7200.
save(audit/'migration_pending.json',dict(intermediate_checkpoint_sha256=intermediate_sha,
    target_identity=identity,model_optimizer_rng_progress_verified_before_write=True))
c['identity']=identity
c['elapsed_seconds']=charged
c['status']='resource_pause'
checkpoint(path,c)
restored=torch.load(path,map_location='cpu',weights_only=False)
for key in ('model','model_config','optimizer','scheduler','rng','progress','finished','graph_step_control'):
    same(restored[key],old[key])
snapshot(audit/'source_after.zip',identity)
save(audit/'migration.json',dict(status='verified_resumable',changed_identity_fields=sorted(changed),
    changed_sources=changed_sources,old_checkpoint_sha256=stop['checkpoint_sha256'],new_checkpoint_sha256=sha(path),
    preserved=['model','optimizer','scheduler','RNG','progress/order/cursor/seen','graph step control','all32000 members','batch32','fixed Loss and coefficients'],
    epoch=0,step=0,pending_monitor=True,charged_seconds=charged,cap_seconds=identity['max_seconds'],
    remaining_seconds=identity['max_seconds']-charged,
    budget_authorization='用户要求修复后自行后台重启，并恢复完整2h；保留先前费用，累计18h不变',
    intermediate_checkpoint_sha256=intermediate_sha,first_attempt_note='Host identity written before NumPy audit exception; all tensors/Adam/RNG/progress now verified before and after final binding',
    original_run_config_and_snapshot_preserved=True,
    host_batch_bit_exact=True,formal_update_performed_during_migration=False,test_read=False))
save(HERE/'state.json',dict(status='resource_repaired',stage='ready to resume original authorized run',
    epoch=0,step=0,charged_seconds=charged,remaining_seconds=identity['max_seconds']-charged,test_read=False))
print('Migrated exact checkpoint state; remaining training seconds:',identity['max_seconds']-charged)
