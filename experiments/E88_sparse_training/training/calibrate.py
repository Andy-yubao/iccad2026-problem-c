"""Register unchanged E90 fixed Loss baseline; no new calibration or data reads."""
import json
from .io import HERE,ROOT,sha,sources,save,snapshot
from .loss_config import FIXED,load_coefficients
from .initialization import verify_protected
from .sampler import manifest

def main():
    output=HERE/'runs/registered_training_loss.json'
    if output.exists():load_coefficients(output);print('Fixed E90 baseline verified');return
    protected=verify_protected();manifest()
    base=ROOT/'experiments/E90_auxiliary_weight_placer/runs/registered_loss.json'
    original=json.loads(base.read_text(encoding='utf-8'));assert original['coefficients']==FIXED
    value=dict(status='registered',sources=sources(),base_artifact=base.relative_to(ROOT).as_posix(),base_artifact_sha256=sha(base),
        coefficients=FIXED,method='Unchanged user-adopted E90 fixed Loss baseline; no recalibration',
        protected_E90_checkpoints=protected,parameter_updates=0,dev_read=False,test_read=False)
    save(output,value);snapshot(HERE/'runs/training_prepared_source_snapshot.zip',value)
    load_coefficients(output);print('Registered E90 fixed baseline unchanged',FIXED,flush=True)

if __name__=='__main__':main()
