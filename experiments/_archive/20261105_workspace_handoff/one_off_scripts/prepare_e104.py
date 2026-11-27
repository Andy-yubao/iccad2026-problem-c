from pathlib import Path
import hashlib,json,shutil
root=Path.cwd(); src=root/'experiments/E98_loss_weight_learning'; dst=root/'experiments/E104_scaling32k'; dst.mkdir(exist_ok=False)
manifest=[]
def copied(source,target):
    p=src/source;q=dst/target;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
    manifest.append(dict(source=p.relative_to(root).as_posix(),copy=q.relative_to(root).as_posix(),source_sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
for name in ['__init__.py','features.py','batches.py','data.py','cache.py','ports.py','resident.py','loss.py','simple_loss.py','loss_config.py','group_tree.py','captured_projection.py','step_control.py','io.py','schedule.py']:
    copied(name,name)
for name in ['model.py','train.py','cuda_step.py','augmentation.py','preflight.py']:
    copied('generalization_baseline/'+name,name)
for p in (src/'depth_evaluation').rglob('*'):
    if p.is_file() and (p.suffix=='.py' or p.name=='budget.json'):
        copied(p.relative_to(src).as_posix(),p.relative_to(src).as_posix())
copied('qualification/metrics.py','metrics.py')
# Extract public-only inference implementation, without importing another experiment.
s=(src/'qualification/evaluate.py').read_text(encoding='utf-8'); graph=s[s.index('def public_graph('):s.index('class Predictor:')]
s=(src/'dynamic_weight_learning_v2/evaluation.py').read_text(encoding='utf-8'); pred=s[s.index('class Predictor:'):s.index('@torch.inference_mode()')]
(dst/'predictor.py').write_text('import numpy as np\nimport torch\nfrom .model import GeometryModel\nfrom .features import graph_features\nfrom .batches import decode\n\n'+graph+pred,encoding='utf-8')
for name in ['train.py','cuda_step.py','preflight.py']:
    p=dst/name;s=p.read_text(encoding='utf-8').replace('from ..','from .');p.write_text(s,encoding='utf-8')
p=dst/'data.py';s=p.read_text().replace('from .dataset_view import ReferenceLayouts','from .dataset_view import ReferenceLayouts');p.write_text(s)
p=dst/'cache.py';s=p.read_text().replace('v5prefix16k','v6prefix32k').replace("'train_size':16000","'train_size':32000").replace('role_records[:16000]','role_records[:32000]').replace("target=(ROOT/name).resolve()","target=(ROOT/'history/20261103_final_delivery_consolidation'/name).resolve()")
p.write_text(s,encoding='utf-8')
(dst/'dataset_view.py').write_text('from datasets.expanded_reference import ExpandedReferenceLayouts, REGISTRY, registry\n\nclass ReferenceLayouts(ExpandedReferenceLayouts):\n    def __init__(self, role):\n        super().__init__(role, train_size=32000)\n',encoding='utf-8')
p=dst/'augmentation.py';p.write_text(p.read_text().replace('rng.random(16000)','rng.random(32000)'),encoding='utf-8')
p=dst/'train.py';s=p.read_text().replace('ROOT, HERE, OWNER, sha, save, checkpoint, sources, snapshot, exclusive','ROOT, HERE, OWNER, sha, save, checkpoint, sources, snapshot, exclusive')
s=s.replace('20261104_generalization96_seed0','20261104_scaling128x16_seed0').replace('width=96, layers=12','width=128, layers=16').replace('E98.generalization96.axis_swap.scratch.v1','E104.scaling32k128x16.scratch.v1').replace('protocol/reference_workset_v5.json','protocol/reference_workset_v6.json').replace('train16k','train32k').replace('len(train) != 16000','len(train) != 32000').replace('Cached v6 prefix differs from frozen v5 members','Cached v6 prefix differs from frozen v6 32k members')
s=s.replace("return elapsed() >= plan['maximum_seconds']","return plan['maximum_seconds'] is not None and elapsed() >= plan['maximum_seconds']")
s=s.replace("if progress['epoch'] >= plan['maximum_epochs']:","if plan['maximum_epochs'] is not None and progress['epoch'] >= plan['maximum_epochs']:")
s=s.replace("next_stage='none; report only'","next_stage='frozen best dev500 only if normal early_stop; managed by pipeline'")
p.write_text(s,encoding='utf-8')
p=dst/'preflight.py';s=p.read_text().replace('1471539','PARAMETERS').replace('from .train import seed_all','from .train import seed_all')
s=s.replace('def main():',"PARAMETERS=json.loads((OWNER/'plan.json').read_text(encoding='utf-8'))['parameters']\n\ndef main():").replace('import ast','import ast\nimport json')
# Use exactly five cases, including a new32k member. Fixed/preplaced coverage retained from the original fixtures.
s=s.replace('1683,10027,5474,10140,15001','1683,10027,5474,10140,31001')
s=s.replace("cap=build_capture(model,opt,pack)","cap=build_capture(model,opt,pack)")
s=s.replace("gradient_norm=float(norm),elapsed_seconds=", "gradient_norm=float(norm),peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),elapsed_seconds=")
p.write_text(s,encoding='utf-8')
# Keep atomic persistence implementation; restrict source identity to this experiment and shared public infrastructure.
p=dst/'io.py';s=p.read_text();a=s.index('def sources():');b=s.index('\ndef snapshot(',a)
s=s[:a]+'''OWNER = HERE

def sources():
    files=[p for p in HERE.rglob('*.py') if 'runs' not in p.relative_to(HERE).parts]
    files += [HERE/'plan.json', HERE/'README.md', HERE/'launch.ps1', HERE/'provenance.json', HERE/'depth_evaluation/budget.json']
    for folder in ('datasets','scoring'):
        files += list((ROOT/folder).glob('*.py'))
    files += [ROOT/'datasets/expanded_reference/__init__.py',ROOT/'visualization/layouts.py']
    files += [ROOT/p for p in ('protocol/reference_workset_v5.json','protocol/reference_workset_v6.json','protocol/split_v1.json','protocol/paths.json','third_party/FloorSet/iccad2026contest/iccad2026_evaluate.py','third_party/FloorSet/cost.py','third_party/FloorSet/utils.py')]
    return {p.relative_to(ROOT).as_posix():sha(p) for p in sorted(set(files))}

''' +s[b:];p.write_text(s,encoding='utf-8')
(dst/'provenance.json').write_text(json.dumps(dict(copies=manifest,derived_predictor=['qualification/evaluate.py:public_graph','dynamic_weight_learning_v2/evaluation.py:Predictor'],private_cross_experiment_imports=False),indent=2),encoding='utf-8')
plan=json.loads((src/'generalization_baseline/plan.json').read_text())
plan.update(experiment='E104_scaling32k',run='20261104_scaling128x16_seed0',authorization='User authorized new32k wider/deeper scratch generalization baseline, background training and automatic fixed dev500 evaluation; dev500 explicitly confirmed.',scope='one candidate; normal early stop then freeze unique dev-Loss best, evaluate fixed dev500 once using copied E98.depth_budget.v1; report then stop',architecture=dict(width=128,layers=16,rectangle=True,ffn_dropout=.10,residual_dropout=.05),train_count=32000,maximum_seconds=None,maximum_epochs=None,revision='E104.scaling32k128x16.scratch.v1',data='reference_workset_v6 frozen train first32000; original v5 first16000 nested unchanged; identical module-count histogram; prior E58/E74/E101 expansion exposure, not new unseen data',test_read=False,postprocessing_calls='500 terminal dev500 only; no training feedback',future_weight_learning='none',source_copies='provenance.json',preflight_train_indices=[1683,10027,5474,10140,31001])
plan['comparison']={'baseline':'experiments/E98_loss_weight_learning/runs/20261104_generalization96_seed0','interpretation':'joint data+width+depth scaling pilot, one seed; cannot fit scaling exponent or isolate causal factors','loss':'same ten original fixed E98 coefficients, not E103 learned coefficients','metrics':'same fixed train500/dev500 Loss, absolute and relative gaps; unique best terminal threeQ, raw/final legality, overlap/bbox, full mean/P95/max latency and >250ms; cold/disk/bulk separate'}
# Parameter count is independent of CUDA and training.
import sys;sys.path.insert(0,str(root))
from experiments.E104_scaling32k.model import GeometryModel
params=sum(p.numel() for p in GeometryModel(**plan['architecture']).parameters());plan['parameters']=plan['trainable_parameters']=params
(dst/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(dict(directory=str(dst),parameters=params,parameter_ratio=params/1471539))
