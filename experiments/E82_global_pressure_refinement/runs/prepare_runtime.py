"""Independent E81 runtime copy; no edits to the completed producer."""
import shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];OLD=ROOT/'experiments/E81_multisource_refinement';NEW=ROOT/'experiments/E82_global_pressure_refinement'
skip={'initialization.py','preflight.py'}
for p in OLD.glob('*.py'):
    if p.name not in skip:
        if p.name in ('features.py','batches.py'):shutil.copy2(p,NEW/p.name)
        else:
            t=p.read_text(encoding='utf-8').replace('E81_multisource_refinement','E82_global_pressure_refinement').replace('E81','E82')
            t=t.replace('20261029_e42_multisource_seed0','20261029_e81_pressure_seed0')
            for a,b in (('1713086','1715390'),('7543025','7545329'),('241547','243851')):t=t.replace(a,b)
            (NEW/p.name).write_text(t,encoding='utf-8')
for p in (OLD/'backend').rglob('*.py'):
    q=NEW/p.relative_to(OLD);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
shutil.copy2(OLD/'requirements-native.txt',NEW/'requirements-native.txt')
for name in ('corrector.py','center_corrector.py'):shutil.copy2(OLD/name,NEW/('old_'+name))
t=(OLD/'model.py').read_text(encoding='utf-8').replace('from .corrector import','from .old_corrector import').replace('from .center_corrector import','from .old_center_corrector import')
(NEW/'old_model.py').write_text(t,encoding='utf-8')
print('Independent global-pressure experiment prepared; no training started.')
