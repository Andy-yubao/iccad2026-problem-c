from pathlib import Path
root=Path.cwd(); dst=root/'experiments/E98_loss_weight_learning/dynamic_weight_learning_v3'
p=dst/'evaluation.py';s=p.read_text(encoding='utf-8')
start=s.index('def fingerprint():');end=s.index('class Predictor:',start)
s=s[:start]+s[end:];p.write_text(s,encoding='utf-8')
# Adapt the already validated one-update check to the real, frozen baseline and new panel.
old=root/'experiments/E98_loss_weight_learning/dynamic_weight_learning_v2/preflight.py'
s=old.read_text(encoding='utf-8').replace('from ..io import save','from ..io import ROOT,HERE,save,sha')
a=s.index("    assert len(set(plan['feedback_pool_train_indices']))==1000")
b=s.index("    assert abs(plan['log_perturbation']",a)
s=s[:a]+'''    assert len(set(plan['feedback_train_indices']))==500
    assert all(0<=i<16000 for i in plan['feedback_train_indices'])
    assert len(plan['feedback_members'])==500 and len(plan['rounds'])==10
    assert [r['kind'] for r in plan['rounds']]==['designed']*5+['seeded_random']*5
    assert (HERE/'runs/20261104_dynamic_mixedq_seed0/STOP').exists()
'''+s[b:]
a=s.index('    initial=GeometryModel(**MODEL).cuda()');b=s.index('    co=coefficients',a)
s=s[:a]+'''    parent=torch.load(ROOT/plan['initial_checkpoint'],map_location='cpu',weights_only=False)
    assert sha(ROOT/plan['initial_checkpoint'])==plan['initial_sha256']
    assert parent['progress']['epoch']==71
    before_hash=model_hash(parent['model'])
'''+s[b:]
s=s.replace('dict(hard_legal=100,mixed_Q=1.1)','dict(hard_legal=500,mixed_Q=1.1)')
s=s.replace("    assert choose(same,plan)[0]=='control'", """    assert choose(same,plan)[0]=='control'
    lower={a:[dict(hard_legal=500,mixed_Q=q)] for a,q in [('control',1.1),('plus',1.05),('minus',1.08)]}
    assert choose(lower,plan)[0]=='plus'
    lower['plus'][0]['hard_legal']=499
    assert choose(lower,plan)[0]=='minus'
    from .common import PhaseSchedule
    schedule=PhaseSchedule()
    assert not any(schedule.step(1.,epoch) for epoch in range(6))
    assert schedule.drop_epoch==5 and schedule.lr==5e-5
    for spec in plan['rounds']:
        for sign in (-1,1):
            moved=coefficients(plan['initial_coefficients'],spec['direction'],sign,plan['log_perturbation'])
            assert abs(sum(moved.values())-sum(plan['initial_coefficients'].values()))<1e-12
    assert all(int(v['step'])==35500 for v in parent['optimizer']['state'].values())
    assert sha(ROOT/plan['initial_checkpoint'])==plan['initial_sha256']""")
s=s.replace('feedback_pool=1000,each_feedback_panel=100','feedback_source_population=16000,each_feedback_panel=500')
s=s.replace('baseline_freeze_pending=True','baseline_frozen_best_epoch=71,control_zero_updates=True,control_summary_reused=True')
s=s.replace('Implementation check while independent baseline trains; not a throughput benchmark','One update from actual frozen baseline; five train cases only; no dev/test or search feedback')
s=s.replace('Prepared v2 passed: one update, frozen parent unchanged, strict100/1000, single feedback, 15epoch cap, factor1.2; no search launched.','Prepared v3 passed: actual best71 parent unchanged, one fresh Adam update, fixed500/16000, selection and schedule checks; no search launched.')
(dst/'preflight.py').write_text(s,encoding='utf-8')
# Current boundary supersedes the former single-candidate authorization, without altering old evidence.
p=root/'AGENTS.md';s=p.read_text(encoding='utf-8')
s=s.replace('用户另授权E98从原固定基线重启多轮Loss调权并后台运行，旧pilot不作初始化或选型依据；训练反馈严格100/1000、三支共用，混合Q=(加权Q+平均Q)/2接纳；','用户最新授权E98从泛化增强best71重启5组设计方向后5组随机方向的多轮Loss调权并后台运行，旧pilot、旧搜索与single_reference_step不作初始化或选型依据；训练反馈固定从train16k随机冻结500例，三支及各轮共用，control零训练且直接复用当前父模型的同500结果（初始基线补齐一次），混合Q=(加权Q+平均Q)/2接纳；')
a=s.index('当前用户仅授权single_reference_step');b=s.index('\n- GPU工程',a)
s=s[:a]+'当前用户授权dynamic_weight_learning_v3从泛化增强best71重启5组设计方向＋5组随机方向：plus/minus各自最多15轮、验证Loss选best/LR/早停、固定train500/16000反馈、control不训练且复用同面板结果。十轮后冻结停止，不自动追加方向或dev/test后处理Q；新run的STOP可阻止后续阶段。'+s[b:]
p.write_text(s,encoding='utf-8')
entry='''
用户最新授权[E98 train500调权重启](experiments/E98_loss_weight_learning/dynamic_weight_learning_v3/README.md)：从泛化增强best71重新开始，舍弃刚完成单步削弱候选及旧搜索的初始化/选型作用，原证据保留。前5组设计方向后5组固定种子随机方向；plus/minus每支最多15轮，验证Loss选best/LR/早停。反馈从完整train16k随机冻结同500，三支及各轮共用；初始control补齐同500一次，之后零训练且复用当前父模型已有结果。按混合Q接纳，所有best/latest保留，十轮冻结后停止，不自动dev/test Q。 [运行状态](experiments/E98_loss_weight_learning/runs/20261104_dynamic_train500_v3_seed0/state.json)。
'''
p=root/'README.md';s=p.read_text(encoding='utf-8');at=s.index('\n用户另行授权[E102');s=s[:at]+'\n'+entry+s[at:];p.write_text(s,encoding='utf-8')
entry=entry.replace('(experiments/','(../experiments/')
p=root/'docs/protocol.md';s=p.read_text(encoding='utf-8');at=s.index('\n## ');s=s[:at]+'\n'+entry+s[at:];p.write_text(s,encoding='utf-8')
p=root/'docs/workspace_structure.md';s=p.read_text(encoding='utf-8');s+='\nE98最新授权入口为dynamic_weight_learning_v3，run为20261104_dynamic_train500_v3_seed0：五组设计后五组随机、固定train500/16000反馈、冻结control结果复用；旧single_reference_step已完成且排除出新搜索。旧dynamic运行STOP保留。\n';p.write_text(s,encoding='utf-8')
p=root/'experiments/README.md';s=p.read_text(encoding='utf-8');s+='\nE98最新授权：[从泛化增强best71重启，五组设计＋五组随机，固定train500反馈](E98_loss_weight_learning/dynamic_weight_learning_v3/README.md)，control零训练且结果复用。旧搜索与单步候选不采用。\n';p.write_text(s,encoding='utf-8')
p=root/'experiments/E98_loss_weight_learning/.gitignore';s=p.read_text(encoding='utf-8');s+='\n# New authorized fixed500 search local generated payloads.\nruns/20261104_dynamic_train500_v3_seed0/evaluation_cache/\nruns/20261104_dynamic_train500_v3_seed0/**/feedback/\nruns/20261104_dynamic_train500_v3_seed0/baseline_feedback/\nruns/20261104_dynamic_train500_v3_seed0/*.lock\ndynamic_weight_learning_v3/*.log\n';p.write_text(s,encoding='utf-8')
print('Registered latest authorization; implemented real-parent five-case gate.')
