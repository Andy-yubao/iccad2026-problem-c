from pathlib import Path
import json,shutil,hashlib
root=Path.cwd();dst=root/'experiments/E103_offline_loss_alignment';src=root/'experiments/E98_loss_weight_learning'
manifest=[]
def cp(a,b):
 b.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(a,b)
 digest=hashlib.sha256(a.read_bytes()).hexdigest();assert hashlib.sha256(b.read_bytes()).hexdigest()==digest
 manifest.append(dict(source=a.relative_to(root).as_posix(),copy=b.relative_to(root).as_posix(),sha256=digest))
for name in ['features.py','batches.py','ports.py','simple_loss.py','group_tree.py','loss_config.py']:
 cp(src/name,dst/name)
for a in (src/'depth_evaluation').rglob('*.py'):
 if '__pycache__' not in a.parts and a.name!='preflight.py':cp(a,dst/'postprocessor'/a.relative_to(src/'depth_evaluation'))
cp(src/'depth_evaluation/budget.json',dst/'postprocessor/budget.json')
cache=(src/'cache.py').read_text(encoding='utf-8').replace('from .io import ROOT,HERE,sha','from .common import ROOT,OWNER as HERE,sha')
cache=cache.replace("target=(ROOT/name).resolve()","target=(ROOT/'history/20261103_final_delivery_consolidation'/name).resolve()")
(dst/'cache.py').write_text(cache,encoding='utf-8')
manifest.append(dict(source=(src/'cache.py').relative_to(root).as_posix(),copy=(dst/'cache.py').relative_to(root).as_posix(),adaptation='local identity import and explicit archived E42 cache path mapping; original semantics and train16k prefix retained'))
(dst/'implementation_provenance.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
# Must precede numerical libraries in coordinator and spawned workers.
(dst/'__init__.py').write_text('''"""E103 frozen-source offline alignment research."""
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS','RAYON_NUM_THREADS'):
    os.environ[_key]='1'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
''',encoding='utf-8')
print('Copied independent depth evaluator and Loss components; E98 sources unchanged.')
