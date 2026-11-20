"""Freeze test membership from metadata only, before any test-payload read."""
import ast
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT))
from datasets.random_reference import registry
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
if (HERE/'plan.json').exists():raise FileExistsError('Frozen selection already exists; no resampling')
reg=registry();owner=HERE.parent
frozen=read(owner/'runs/global2_seed0/evaluation_no_mib_dev500/frozen.json')
assert frozen['best_epoch']==55 and sha(ROOT/frozen['checkpoint'])==frozen['checkpoint_sha256']
sources={}
for name,digest in frozen['source_identity'].items():
    assert sha(ROOT/name)==digest,('Historical evaluation source drift',name)
    sources[name]=digest
for p in list(owner.glob('*.py'))+[HERE/'run.py',HERE/'prepare.py']+[ROOT/n for n in (
    'datasets/random_reference.py','datasets/cases.py','datasets/reference_layouts.py','datasets/paths.py',
    'scoring/research_no_mib.py','scoring/evaluation.py','scoring/checks.py','scoring/official.py',
    'third_party/FloorSet/iccad2026contest/iccad2026_evaluate.py','protocol/paths.json')]:
    if p.suffix=='.py':ast.parse(p.read_text(encoding='utf-8'))
    sources[p.relative_to(ROOT).as_posix()]=sha(p)
seed=20261109
indices=sorted(np.random.default_rng(seed).choice(2000,size=500,replace=False).tolist())
records=[dict(**reg['sets']['test']['records'][i],frozen_test_index=i) for i in indices]
members={tuple(r['member']) for r in records}
assert len(members)==500 and not members&{tuple(r['member']) for role in ('train','dev') for r in reg['sets'][role]['records']}
folder=(ROOT/read(ROOT/'protocol/paths.json')['random_reference40k_root']).resolve()
assert read(folder/'report.json')['manifest_sha256']==reg['manifest_sha256']
plan=dict(authorized_test_evaluation=True,authorization='User explicitly requested E119 on another500 test cases with original postprocessor budget, background, 2026-11-09',
    model='E119 best55',checkpoint=frozen['checkpoint'],checkpoint_sha256=frozen['checkpoint_sha256'],count=500,
    registry='protocol/random_reference40k_v1.json',registry_sha256=sha(ROOT/'protocol/random_reference40k_v1.json'),
    payload_checksums_sha256=sha(folder/'checksums.jsonl'),records=records,sources=sources,
    sampling=dict(method='Uniform random without replacement from original random_reference40k test2000; indices sorted for execution',seed=seed,
        selection_uses='Frozen metadata only, no payload/labels/Loss/Q/N filtering'),
    test_payload_scope='Only the frozen selected500; other1500 remain sealed',
    preparation_test_payload_reads=0,training=False,checkpoint_reselection=False,automatic_retry=False,
    postprocessor='E98.depth_budget.v1 original work caps, unchanged',budget_sha256=sha(owner/'evaluation/budget.json'),
    dev_result='experiments/E119_global_initial_generator/runs/global2_seed0/evaluation_no_mib_dev500/results.json')
(HERE/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(HERE/'README.md').write_text('# E119 best55：一次授权的独立test500评价\n\n'
    '用户2026-11-09明确要求后台运行另外500例测试，使用原预算后处理。仅开启random_reference40k test2000中seed20261109随机无放回冻结的500例；其余1500例保持封存。名单在任何测试载荷读取前冻结，不按N/标签/Loss/Q选题。\n\n'
    '复用原NPZ，不生成数据或构建训练缓存。E119 best55、原E98.depth_budget.v1、原面积/无MIB本地Q/原硬检查；不训练、不重选模型、不新增图。错误/停止不自动重试。\n\n'
    '实时进度output/state.json；日志stdout.log、stderr.log；完成结果output/results.json及output/report.md。本次测试500例运行后记为已曝光，不用于选择权重或调预算。\n',encoding='utf-8')
print('Ready: selected500 frozen from metadata; zero test payload reads; original checkpoint/budget/source identity checked.')
