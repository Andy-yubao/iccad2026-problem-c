"""Write source provenance and architecture metadata, without training or tests."""
import json
from .io import ROOT,HERE,sha,save
from .model import GeometryModel
from .train import MODEL


def main():
    parent=ROOT/'experiments/E98_loss_weight_learning'
    names=['__init__.py','features.py','batches.py','data.py','dataset_view.py','cache.py',
           'ports.py','resident.py','simple_loss.py','group_tree.py','loss_config.py','io.py',
           'schedule.py','captured_projection.py','step_control.py','cuda_step.py','train.py','launch.ps1']
    copied={name:{'source':(parent/name).relative_to(ROOT).as_posix(),'source_sha256':sha(parent/name)} for name in names}
    copied['base_model.py']={'source':(parent/'model.py').relative_to(ROOT).as_posix(),'source_sha256':sha(parent/'model.py')}
    model=GeometryModel(**MODEL)
    count=sum(p.numel() for p in model.parameters())
    plan=json.loads((HERE/'plan.json').read_text(encoding='utf-8'))
    if count!=plan['parameters']:
        raise ValueError(f'Registered parameter count differs: {count}')
    save(HERE/'provenance.json',dict(experiment='E100',copied_sources=copied,
         policy='copied locally and maintained independently; no cross-experiment private imports',
         new_files=['model.py','loss.py','preflight.py','prepare.py'],
         source_checkpoint=None,initialization='all random seed0 in actual training',
         preserved_input_semantics=['features.py','batches.py'],parameters=count,
         minimal_checks='five train members, one actual update; no restore checks',test_read=False))
    print(json.dumps({'parameters':count,'provenance':str(HERE/'provenance.json')}))


if __name__=='__main__':
    main()
