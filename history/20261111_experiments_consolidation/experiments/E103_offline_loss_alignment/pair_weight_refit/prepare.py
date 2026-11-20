from pathlib import Path
import json,hashlib
root=Path.cwd();exp=root/'experiments/E103_offline_loss_alignment';owner=exp/'pair_weight_refit'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
parent=exp/'assets/e98_generalization/best.pt';current=exp/'runs/20261104_learned_v1_lr1e3_seed0/checkpoints/best.pt'
assert sha(parent)=='3f7ca0086af8a13bd52eedf0635117312e5c0a72e80e423e9cee3db5ca9b87ea'
assert sha(current)=='85d21d2dee2a6266894a9bf0036210c3a0ff7c2290ca0a6ec0b4e32784c84325'
assert read(exp/'runs/20261104_learned_v1_lr1e3_seed0/results.json')['best_epoch']==29
plan=dict(revision='E103.enhanced_current_pair.fixed_lambda.v1',authorization='User requested one weight calculation using only E98 enhanced and current version, existing postprocessor results reusable; launch in background.',
 enhanced_checkpoint=parent.relative_to(root).as_posix(),enhanced_sha256=sha(parent),
 current_checkpoint=current.relative_to(root).as_posix(),current_sha256=sha(current),
 source_models=['E98 enhanced best71','current learned-v1 best29'],models=2,
 panel='Existing original frozen train1000, same members, split600/200/200',panel_sha256=sha(exp/'panel.json'),
 reused_enhanced_records=1000,current_records_to_generate=1000,
 dev500_reuse='Previously completed current dev500 does not overlap train1000 and must not enter weight fitting; validation results are not reused as train labels.',
 inference='Match existing bank CPU FP32 deterministic single-thread exactN inference with original source features/decoder; four processes across cases.',
 postprocessor='Unmodified E98 fixed-depth v1 already copied into E103; same repair budget and all hard guards; official neutral-time Q.',
 loss_basis='Same ten raw E98 terms at original network states for both models; independent of their training coefficients.',
 weight_fits=1,regularization=1.,regularization_source='Fixed at previous chosen value1, no new sweep; preserve prior fit method to isolate source-population change.',
 prior='Original E98 coefficients, identical prior to first five-source fit; current-v1 is a reported comparison, not an additional optimized candidate.',
 fit_cases=600,tune_cases=200,check_cases=200,selection='One fit on600 only. Freeze proposal before any200-panel reports, no tune/check coefficient selection.',
 constraints='Nonnegative scaled simplex, fit-only75th percentile within-pair difference scales, total physical coefficient sum matches original E98.',
 aggregation='Case weights=.5 equal+.5 official exp(n/12); mixedQ report all three.',
 mathematical_boundary='Two related source versions, at most one informative pair per problem. No claim that obsolete models caused prior failure. Existing check200 already exposed; descriptive only.',
 terminal='Save proposal, normalized coefficient comparison and reports. No automatic neural training, extra source models, local perturbations, dev Q fitting or tests.',
 stop='Newrun STOP or package STOP prevents next case/fit; retain completed rows. No removal of any old E98 STOP.',test_read=False)
(owner/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
p=root/'AGENTS.md';s=p.read_text(encoding='utf-8')
needle='除此不得擅自新增候选、恢复旧实验或自动唤醒。'
s=s.replace(needle,'用户最新授权pair_weight_refit：仅E98增强best71与当前best29，在原训练1000题上计算一次权重；复用增强版1000结果，补当前版1000，四路后台，固定lambda1不扫描，仅fit600拟合，原200面板只报告，dev/test不参与，完成保存提案即停、不自动训练。'+needle,1)
p.write_text(s,encoding='utf-8')
entry='\nE103最新授权：仅使用E98增强best71和当前新Loss best29进行一次权重拟合。原1000训练题保持不变，增强版结果复用；当前版只有已完成dev500，故补齐同1000训练题后处理，不把验证Q拿来拟合。四路后台补齐、固定lambda1仅fit600一次拟合，200/200只报告，结束保存提案即停。入口experiments/E103_offline_loss_alignment/pair_weight_refit，状态runs/20261104_pair_enhanced_current_v1/state.json。\n'
for name in ('README.md','docs/protocol.md','docs/workspace_structure.md','experiments/README.md'):
 p=root/name;s=p.read_text(encoding='utf-8');p.write_text(s+entry,encoding='utf-8')
p=exp/'README.md';s=p.read_text(encoding='utf-8');s+='\n## 仅增强版与当前版的单次权重重算\n\n用户最新授权[pair_weight_refit](pair_weight_refit/README.md)：排除其他四个旧模型，只用增强best71与当前best29，原同1000训练题。增强结果复用，只补当前模型；固定lambda1拟合一次，报告后停止，不自动训练。新日志pair_weight_refit/stdout.log，新状态runs/20261104_pair_enhanced_current_v1/state.json。\n';p.write_text(s,encoding='utf-8')
print('Registered two-source single-fit research; train data only, no new training.')
