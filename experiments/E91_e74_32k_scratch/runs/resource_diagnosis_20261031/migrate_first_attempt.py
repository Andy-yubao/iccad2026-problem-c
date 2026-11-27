"""One-time audited data-residency migration; no neural updates or budget reset."""
import json
from pathlib import Path
import torch
from experiments.E91_e74_32k_scratch.io import HERE,save,sha,checkpoint,snapshot
from experiments.E91_e74_32k_scratch.preflight import verify,same
from experiments.E91_e74_32k_scratch.train import parser,make_identity
from experiments.E91_e74_32k_scratch.loss_config import load_coefficients

torch.set_num_threads(4)
audit=Path(__file__).resolve().parent
assert not (audit/'migration.json').exists(),'Already migrated; never rerun'
verify()
out=HERE/'runs/20261031_scratch32k_seed0'
stop=json.loads((audit/'stop.json').read_text(encoding='utf-8'))
path=out/'checkpoints/latest.pt'
assert sha(path)==stop['checkpoint_sha256']
c=torch.load(path,map_location='cpu',weights_only=False)
assert c['progress']['step']==0 and c['progress']['epoch']==0 and c['progress']['pending_monitor']
assert not c['finished'] and not (out/'checkpoints/best.pt').exists()
args=parser().parse_args(['--output',str(out),'--max-hours','2','--max-epochs','200','--resume'])
identity=make_identity(args)
identity['calibration']=load_coefficients(HERE/'runs/registered_loss.json')
changed=[k for k in set(c['identity'])|set(identity) if c['identity'].get(k)!=identity.get(k)]
assert set(changed)=={'sources','calibration','gpu_resident_inputs','data_transfer'},changed
changed_sources=[k for k in identity['sources'] if c['identity']['sources'].get(k)!=identity['sources'][k]]
assert set(changed_sources)=={f'experiments/E91_e74_32k_scratch/{name}' for name in ('resident.py','train.py','preflight.py','plan.json')}
charged=max(stop['charged_seconds'],json.loads((out/'budget.json').read_text())['elapsed_seconds'],c['elapsed_seconds'])
assert charged<identity['max_seconds']
c['identity']=identity
c['elapsed_seconds']=charged
c['status']='resource_pause'
checkpoint(path,c)
restored=torch.load(path,map_location='cpu',weights_only=False)
old=torch.load(audit/'before/latest.pt',map_location='cpu',weights_only=False)
for key in ('model','model_config','optimizer','scheduler','rng','progress','finished','graph_step_control'):
    same(restored[key],old[key])
snapshot(audit/'source_after.zip',identity)
save(audit/'migration.json',dict(status='verified_resumable',changed_identity_fields=sorted(changed),
    changed_sources=changed_sources,old_checkpoint_sha256=stop['checkpoint_sha256'],new_checkpoint_sha256=sha(path),
    preserved=['model','optimizer','scheduler','RNG','progress/order/cursor/seen','graph step control','all32000 members','batch32','fixed Loss and coefficients'],
    epoch=0,step=0,pending_monitor=True,charged_seconds=charged,cap_seconds=identity['max_seconds'],
    remaining_seconds=identity['max_seconds']-charged,original_run_config_and_snapshot_preserved=True,
    host_batch_bit_exact=True,formal_update_performed_during_migration=False,test_read=False))
save(HERE/'state.json',dict(status='resource_repaired',stage='ready to resume original authorized run',
    epoch=0,step=0,charged_seconds=charged,remaining_seconds=identity['max_seconds']-charged,test_read=False))
print('Migrated exact checkpoint state; remaining training seconds:',identity['max_seconds']-charged)
