from pathlib import Path
import ast
root=Path.cwd();exp=root/'experiments/E103_offline_loss_alignment';owner=exp/'reference075_posttrain'
p=owner/'train.py';s=p.read_text(encoding='utf-8').replace('"""One warmed-start learned-Loss training run, no automatic postprocessing."""','"""One reference075 training phase; enclosing pipeline performs terminal dev500."""')
s=s.replace('automatic_next_stage=False','automatic_next_stage=True').replace('training_complete_no_next_stage','training_complete_ready_for_dev500')
p.write_text(s,encoding='utf-8')
p=owner/'common.py';s=p.read_text(encoding='utf-8');needle="    return {p.relative_to(ROOT).as_posix():sha(p) for p in sorted(set(files))}"
insert="""    files += list((EXPERIMENT/'postprocessor').rglob('*.py'))+[EXPERIMENT/'postprocessor/budget.json']
    files += list((ROOT/'scoring').glob('*.py'))
    files += [ROOT/n for n in ('visualization/layouts.py','third_party/FloorSet/iccad2026contest/iccad2026_evaluate.py',
        'third_party/FloorSet/cost.py','third_party/FloorSet/utils.py')]
"""
assert needle in s;s=s.replace(needle,insert+needle);p.write_text(s,encoding='utf-8')
for p in owner.rglob('*.py'):ast.parse(p.read_text(encoding='utf-8-sig'))
p=root/'AGENTS.md';s=p.read_text(encoding='utf-8');needle='除此不得擅自新增候选、恢复旧实验或自动唤醒。'
s=s.replace(needle,'用户最新另授权一个reference075_posttrain小实验：从E98增强best71原权重开始，仅原position/shape各乘.75（.75/.1875），其余8项不变且不归一化；LR1e-4→5e-5、无轮数/时间上限、验证Loss正常早停，后台单流水线训练后冻结唯一best并自动固定dev500评价及报告，完成停止。旧E98 STOP不解除，不开放测试。'+needle,1);p.write_text(s,encoding='utf-8')
entry='\n用户新增并明确授权E103/reference075_posttrain小实验：E98原增强best71，position=0.75、shape=0.1875、其他8项原值，不归一化；LR从倒数第二档1e-4至5e-5，无15轮或时间上限，按验证Loss正常早停；后台训练→冻结唯一best→自动E98固定计算深度dev500→报告即停。已有父模型500结果及已通过精确复现桥复用，测试封存。run为experiments/E103_offline_loss_alignment/runs/20261104_reference075_seed0，总体状态pipeline_state.json，日志reference075_posttrain/stdout.log。\n'
for name in ('README.md','docs/protocol.md','docs/workspace_structure.md','experiments/README.md'):
 p=root/name;s=p.read_text(encoding='utf-8');p.write_text(s+entry,encoding='utf-8')
p=exp/'README.md';s=p.read_text(encoding='utf-8');p.write_text(s+'\n## 参考监督0.75倍单候选实验\n\n用户最新授权[reference075_posttrain](reference075_posttrain/README.md)：回到E98原增强best71，仅位置/形状原系数乘.75；1e-4起，无轮数上限，验证Loss正常早停，训练结束自动dev500和报告。不使用统计拟合系数，不恢复旧搜索。\n',encoding='utf-8')
print('Registered one reference075 pipeline, fixed parent and automatic dev500; no scientific source changes to old runs.')
