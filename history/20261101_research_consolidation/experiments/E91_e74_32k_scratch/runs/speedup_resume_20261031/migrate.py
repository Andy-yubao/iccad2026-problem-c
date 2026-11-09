"""One-time speed implementation and +1h migration; numeric training state retained."""
import json
from pathlib import Path
import torch
from experiments.E91_e74_32k_scratch.io import HERE,save,sha,checkpoint,snapshot
from experiments.E91_e74_32k_scratch.preflight import verify
from experiments.E91_e74_32k_scratch.train import parser,make_identity
from experiments.E91_e74_32k_scratch.loss_config import load_coefficients
from experiments.E91_e74_32k_scratch.budget import charge_preflight,global_charged_seconds
from .check_trained_resume import same

def main():
    torch.set_num_threads(4)
    a=Path(__file__).resolve().parent;out=HERE/'runs/20261031_scratch32k_seed0'
    assert not (a/'migration.json').exists(),'Already migrated; do not rerun'
    verify()
    test=json.loads((a/'trained_resume_checks.json').read_text(encoding='utf-8'))
    assert test['status']=='passed'
    if not (a/'trained_check_charge.json').exists():
        charge_preflight(test['seconds']);save(a/'trained_check_charge.json',dict(charged_seconds=test['seconds']))
    authorization=json.loads((a/'authorization.json').read_text(encoding='utf-8'))
    prior=authorization['prior_charged_seconds']
    assert global_charged_seconds()+3600<64800
    args=parser().parse_args(['--output',str(out),'--max-hours','1','--resume'])
    identity=make_identity(args);identity['calibration']=load_coefficients(HERE/'runs/registered_loss.json')
    assert abs(identity['max_seconds']-prior-3600)<1e-8
    old_states={}
    for name,expected in [('latest.pt',authorization['old_latest_sha256']),('best.pt',authorization['old_best_sha256'])]:
        assert sha(out/'checkpoints'/name)==expected
        old_states[name]=torch.load(a/'before'/name,map_location='cpu',weights_only=False)
        assert sha(a/'before'/name)==expected
    old=old_states['latest.pt']
    assert old['finished'] and old['status']=='budget'
    assert old['progress']['step']==32443 and old['progress']['cursor']==14176
    allowed={'sources','calibration','max_seconds','additional_training_budget_seconds',
             'preserved_prior_training_seconds','graph_step_control'}
    changed={k for k in set(old['identity'])|set(identity) if old['identity'].get(k)!=identity.get(k)}
    assert changed==allowed,changed
    save(a/'migration_pending.json',dict(target_identity=identity,authorization=authorization))
    for name,state in old_states.items():
        state['identity']=identity
        if name=='latest.pt':
            state['finished']=False;state['status']='continuation_prepared';state['elapsed_seconds']=prior
        checkpoint(out/'checkpoints'/name,state)
    for name in old_states:
        original=torch.load(a/'before'/name,map_location='cpu',weights_only=False)
        new=torch.load(out/'checkpoints'/name,map_location='cpu',weights_only=False)
        for key in ('model','model_config','optimizer','scheduler','rng','progress','graph_step_control'):
            same(original[key],new[key])
    save(out/'budget.json',dict(elapsed_seconds=prior))
    snapshot(a/'source_after.zip',identity)
    save(a/'migration.json',dict(status='verified_ready_to_resume',source_step=32443,epoch=32,cursor=14176,
        best_epoch=old['progress']['best_epoch'],LR=old['optimizer']['param_groups'][0]['lr'],
        preserved=['model','Adam','RNG','scheduler','history','order/cursor/seen','projection counters/scales','prior elapsed'],
        changed_identity_fields=sorted(changed),old_latest_sha256=authorization['old_latest_sha256'],
        old_best_sha256=authorization['old_best_sha256'],new_latest_sha256=sha(out/'checkpoints/latest.pt'),
        new_best_sha256=sha(out/'checkpoints/best.pt'),prior_charged_seconds=prior,
        additional_seconds=3600,cumulative_cap_seconds=identity['max_seconds'],
        latest_finished_cleared='authorized continuation after budget stop; no update or progress reset',
        source_before='before/prepared_source_snapshot.zip',source_after='source_after.zip',test_read=False))
    save(HERE/'state.json',dict(status='speedup_prepared',stage='ready for authorized one-hour resume',
        epoch=32,step=32443,cursor=14176,LR=old['optimizer']['param_groups'][0]['lr'],test_read=False))
    print('Migration verified; latest32443 preserved; additional1h; LR unchanged',flush=True)

if __name__=='__main__':main()
