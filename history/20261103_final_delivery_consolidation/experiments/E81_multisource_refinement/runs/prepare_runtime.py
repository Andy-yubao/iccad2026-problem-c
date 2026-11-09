"""Copy the proven runtime independently, preserving public tensor encoding."""
import shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];OLD=ROOT/'experiments/E79_center_refinement';NEW=ROOT/'experiments/E81_multisource_refinement'
skip={'model.py','initialization.py','preflight.py','report.py','arms.py','loss.py','loss_config.py'}
for p in OLD.glob('*.py'):
    if p.name not in skip:
        if p.name in ('features.py','batches.py'):shutil.copy2(p,NEW/p.name)
        else:(NEW/p.name).write_text(p.read_text(encoding='utf-8').replace('E79','E81'),encoding='utf-8')
for p in (OLD/'backend').rglob('*.py'):
    q=NEW/p.relative_to(OLD);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
shutil.copy2(OLD/'requirements-native.txt',NEW/'requirements-native.txt')
shutil.copy2(ROOT/'experiments/E80_modular_transfer/model.py',NEW/'old_model.py')
print('Independent E81 runtime copied, no training started.')
