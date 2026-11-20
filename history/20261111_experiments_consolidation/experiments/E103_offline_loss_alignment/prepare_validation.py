from pathlib import Path
import ast,json,hashlib,shutil
root=Path.cwd();owner=root/'experiments/E103_offline_loss_alignment';dst=owner/'validate_learned_v1';dst.mkdir(exist_ok=False)
old=root/'experiments/E98_loss_weight_learning'
def select(path,names):
 text=path.read_text(encoding='utf-8');tree=ast.parse(text)
 return '\n\n'.join(ast.get_source_segment(text,node) for node in tree.body if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in names)+'\n'
header='''"""Unchanged frozen-baseline public graph / CUDA inference path, locally copied."""
import hashlib
import numpy as np
import torch
from ..sources.e98_generalization.model import GeometryModel
from ..features import graph_features
from ..batches import decode
'''
code=header+select(old/'qualification/evaluate.py',{'public_graph'})+'\n'+select(old/'dynamic_weight_learning_v2/evaluation.py',{'model_hash','input_hash','Predictor'})
(dst/'predictor.py').write_text(code,encoding='utf-8')
shutil.copy2(old/'qualification/metrics.py',dst/'metrics.py')
(dst/'__init__.py').write_text('"""Frozen learned-v1 best29 terminal dev500 verification."""\n',encoding='utf-8')
provenance=[dict(source=p.relative_to(root).as_posix(),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in [old/'qualification/evaluate.py',old/'dynamic_weight_learning_v2/evaluation.py',old/'qualification/metrics.py']]
(dst/'copy_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n',encoding='utf-8')
plan=dict(authorization='User: training completed, please validate; run same original fixeddev500 once with E98 fixed-depth postprocessor and compare frozen parent.',
 train_run='experiments/E103_offline_loss_alignment/runs/20261104_learned_v1_lr1e3_seed0',
 candidate_checkpoint='experiments/E103_offline_loss_alignment/runs/20261104_learned_v1_lr1e3_seed0/checkpoints/best.pt',candidate_sha256='85d21d2dee2a6266894a9bf0036210c3a0ff7c2290ca0a6ec0b4e32784c84325',
 parent_checkpoint='experiments/E103_offline_loss_alignment/assets/e98_generalization/best.pt',parent_sha256='3f7ca0086af8a13bd52eedf0635117312e5c0a72e80e423e9cee3db5ca9b87ea',
 parent_evaluation='experiments/E98_loss_weight_learning/runs/20261104_generalization96_seed0/evaluation_depth_dev500',
 panel='original frozen dev_monitor500',cases=500,selection='frozen new-Loss best29 before Q, never reselect epochs',
 numerical_path='Same GPU FP32 deterministic batch1 pad120 CUDA graph and public graph as completed parent evaluation; exact-source E98 fixed-depth CPU1 repair.',
 parent_reuse='Verify all source/budget/input hashes; five-size-spanning dev cases must exactly reproduce parent drafts, repaired coordinates and Q; reuse saved parent500 only after pass.',
 timing='Candidate complete per-case input construction/transfer/model/repair/independent hard check measured sequentially; init/scoring/disk/plots excluded. Parent timing historical, not matched concurrent benchmark.',
 parallelism='Timed inference/repair sequential to preserve complete latency scope; plot same100 with4 independent CPU processes after all timed solves.',
 output='training run/evaluation_depth_dev500',terminal='report three Qs, hard legality, paired deltas/tails, raw overlap/bbox/empty and complete timing; same100 images; no training or tests',test_read=False)
(dst/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('Prepared exact local parent evaluation path and frozen candidate29.')
