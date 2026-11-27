"""Register exact3x of saved E87 coefficients, without data or recalibration."""
import json
from .io import HERE,ROOT,sha,sources,save,snapshot
from .loss_config import FIXED,load_coefficients
from .initialization import verify_protected

def main():
    output=HERE/'runs/registered_loss.json'
    if output.exists():
        load_coefficients(output);print('Existing x3 registration verified; no recalibration');return
    verify_protected()
    base=ROOT/'experiments/E87_simple_supervised_placer/runs/registered_loss.json'
    old=json.loads(base.read_text(encoding='utf-8'));assert old['status']=='registered'
    assert {k:3*v for k,v in old['coefficients'].items()}==FIXED
    protected=[ROOT/'experiments/E87_simple_supervised_placer/runs/20261030_simple_seed0/checkpoints'/f'{name}.pt' for name in ('best','latest')]
    value=dict(status='registered',sources=sources(),base_artifact=base.relative_to(ROOT).as_posix(),
        base_artifact_sha256=sha(base),base_coefficients=old['coefficients'],multiplier=3.,coefficients=FIXED,
        gradient_medians_inherited=old['gradient_medians'],gradient_target_fractions=dict(overlap=.9,task=.6),
        method='exact3x of original train-only calibration; no gradient rerun or search',parameter_updates=0,
        dev_read=False,test_read=False,protected_E87_checkpoints={q.relative_to(ROOT).as_posix():sha(q) for q in protected})
    save(output,value);snapshot(HERE/'runs/prepared_source_snapshot.zip',value);load_coefficients(output)
    print('Registered:',FIXED,'; no training or inference',flush=True)

if __name__=='__main__':main()
