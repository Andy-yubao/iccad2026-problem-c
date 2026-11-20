"""Independent E142 control: replace hidden context only after supervised readouts."""
import ast
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
ENGINE=ROOT/'experiments/E146_layout_only'
BASE=ROOT/'experiments/E142_dual_geometry'
ARM='e147_supervision_hidden_reset'
SCOPE='E147.supervision_hidden_reset.v1.seed0'

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def mapped(s):
    for old,new in [('experiments.E146_layout_only','experiments.E147_supervision_hidden_reset'),
        ('E146.layout_only.v1.seed0',SCOPE),('e146_layout_only',ARM),('E146','E147'),("'e146'","'e147'")]:s=s.replace(old,new)
    return s
def replace_function(s,name,replacement):
    node=next(n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name==name)
    lines=s.splitlines(keepends=True)
    return ''.join(lines[:node.lineno-1])+replacement.rstrip()+'\n'+''.join(lines[node.end_lineno:])

def main():
    if (HERE/'provenance.json').exists():raise FileExistsError('Already prepared')
    gate=read(ENGINE/'preflight.json');assert gate['status']=='passed'
    for name,digest in gate['sources'].items():assert sha(ROOT/name)==digest,name
    copied={}
    for name,digest in gate['sources'].items():
        src=ROOT/name
        if not src.is_relative_to(ENGINE):continue
        rel=src.relative_to(ENGINE)
        if src.suffix!='.py' and rel.as_posix()!='evaluation/budget.json':continue
        if rel.as_posix() in ('prepare.py','preflight.py','model.py'):continue
        dst=HERE/rel;dst.parent.mkdir(parents=True,exist_ok=True)
        dst.write_text(mapped(src.read_text(encoding='utf-8')),encoding='utf-8')
        copied[rel.as_posix()]=dict(parent=name,sha256=digest)
    baseline_gate=read(BASE/'preflight.json');assert baseline_gate['status']=='passed'
    for name,digest in baseline_gate['sources'].items():assert sha(ROOT/name)==digest,name
    s=(BASE/'model.py').read_text(encoding='utf-8')
    s=s.replace("architecture='e142'","architecture='e147'").replace("architecture!='e142'","architecture!='e147'")
    s=s.replace('from .geometry_feedback import module_features,pair_features','from .geometry_feedback import module_features,pair_features\nfrom .loss_config import SUPERVISED_LAYERS')
    s=s.replace("        h=self.input(g['features'])*g['valid'][...,None]", "        static=self.input(g['features'])*g['valid'][...,None]\n        h=static")
    s=s.replace('        for layer in self.layers:', '        for depth,layer in enumerate(self.layers,1):')
    s=s.replace("            h=layer(h+self.state_embed(state*g['mask']),g,state,demand,pair)",
        "            context=static if depth-1 in SUPERVISED_LAYERS else h\n            h=layer(context+self.state_embed(state*g['mask']),g,state,demand,pair)")
    (HERE/'model.py').write_text(s,encoding='utf-8')
    copied['model.py']=dict(parent=(BASE/'model.py').relative_to(ROOT).as_posix(),sha256=sha(BASE/'model.py'))
    plan=read(ENGINE/'plan.json')
    plan.update(experiment='E147 hidden reset after supervised layers only',revision=SCOPE,status='authorized_preparation',
        authorization='2026-11-17 user confirmed only S2->S3, S6->S7, S8->S9, S10->S11 reset hidden transmission, other layers retain it; prepare and launch in background. Same E142 settings and automatic original dev500.',
        arms={ARM:plan['arms']['e146_layout_only']},architecture={'e147':'E142 with hidden reset immediately after supervised readouts only'},
        parameters={'e147':895123},trainable_parameters={'e147':895123},
        hidden_reset_after_layers=[2,6,8,10],hidden_retained_after_layers=[0,1,3,4,5,7,9,11],
        hidden_propagation='After S2/S6/S8/S10, next layer starts from public static embedding, prior-layout embedding and current geometric demand. All other edges carry original hidden context, including S0->S1. Layout and its original gradient channels always retained; no detach.',
        preflight='Exact E142 scratch parameters, unchanged feedback/pair18/E139 objective and gradients, interventions on all 12 hidden edges holding emitted layouts fixed, original five train cases, short guarded CUDA64/tail32 run. Resume engine inherited unchanged from verified E146/E142.',
        preflight_parent_sha256=sha(ENGINE/'preflight.json'),
        queue='Single train -> frozen own-Loss-best -> original fixed dev500/postprocessing/hard checks -> immediate Q logs -> E142 comparison',
        comparison='E142 control with same parameters, seed0,36k,batch64, objective and schedule; only supervised-layer hidden resets differ; adaptive single-seed result.')
    save(HERE/'plan.json',plan)
    ev=read(ENGINE/'evaluation/plan.json');ev.update(revision=SCOPE+'.dev500',authorization=plan['authorization'],models={});save(HERE/'evaluation/plan.json',ev)
    save(HERE/'provenance.json',dict(engine_parent='experiments/E146_layout_only',model_parent='experiments/E142_dual_geometry',copied_originals=copied,
        changes=['Only hidden context crossing supervised readouts is replaced by public static embedding; no parameter/geometry/Loss/optimizer change']))
    save(HERE/'authorization.json',dict(scope=SCOPE,training_enabled=False,allowed_arms=[],prepared_arms=[ARM],automatic_evaluation=True,automatic_retry=False,test_read=False,note=plan['authorization']))
    save(HERE/'queue_state.json',dict(status='prepared',completed=[],active=None))
    p=HERE/'run_queue.py';s=p.read_text(encoding='utf-8').replace('layout_only versus E142','supervised hidden reset versus E142')
    s=s.replace('Same E142 batch64, panels, seed0, supervision, schedule and per-arm4h/200epoch budget; independent single-factor arms, one seed/adaptive stopping.',plan['comparison'])
    p.write_text(s,encoding='utf-8')
    text='''# E147：只在监督层之后清除隐藏状态传递

2026-11-17 用户确认定义并授权准备完成后后台启动。E142为独立对照；从头训练，不加载已训练权重。

只在S2→S3、S6→S7、S8→S9、S10→S11清除上下文隐藏状态。下一层由公开静态属性编码＋上一层布局的state_embed＋当前需求/pair几何重建输入；不使用S0隐藏状态替代。其他层间照常传布局及隐藏状态，S0→S1也保留。S12无下一层。

每段内部照常积累隐藏状态：S0–S2；S3–S6；S7–S8；S9–S10；S11–S12。布局始终传递，不detach，保留通过布局、pair和state_embed的梯度；它不是截断反传实验。动态几何摘要保持E142原49维及原detach策略。

原两初始化块、12层width64、模块需求编码、pair18 bias、ψ关系value、局部消息、残差、FFN和共享读出头全部保留；895123参数，逐值对应E142同seed scratch初值。

配置复用E142：冻结36k及现有CPU缓存、seed0、batch64/尾批32、确定性FP32/TF32关闭、原增强/AdamW/更新门禁/LR/早停、4h/200epoch上限。E139监督不变：S2/S6去直接overlap；S8/S10完整，前四项各.05；S12完整.8。其他层不加直接监督；不读test。

训练、监测、best、LR、早停使用同一Loss。训练正常结束后冻结唯一own-Loss-best，自动原固定dev500、原后处理预算及独立硬检查；立即在评价日志、后台日志和background/quality.log打印[Q]，不按Q选权重。

入口：python -u -B -m experiments.E147_supervision_hidden_reset.run_queue --start。实时状态在queue_state.json和runs/e147_supervision_hidden_reset_seed0/heartbeat.json；后台日志background/queue.stdout.log及queue.stderr.log。STOP/QUEUE_STOP/异常阻断后续，不自动重试或追加实验。
'''
    (HERE/'README.md').write_text(text,encoding='utf-8')
    (HERE/'DESIGN.md').write_text(text+'''
## 实现

static=input(public_features)。初始化h经原两图块得到S0。循环生成S_depth时，仅若depth-1属于监督层集合，选择context=static，否则context=h；再调用原layer(context+state_embed(previous_layout), current_demand, pair)。监督集合含12但没有S13，故实际只有四处reset。不增加随机参数，不复制/缓存S0上下文，不改变原布局读出与几何计算。

## 准备检查

固定各读出的真实布局，对S0及S1–S11隐藏输出逐个施加扰动，检查下一层输入：只有监督层后的四处完全不受影响，其余八处必须响应。四处reset后下游输出与未扰动运行逐位一致。核对S1/S2与同参数E142逐位一致；验证最终目标仍可通过布局通路向四个reset层读出反传。短检使用原登记五训练例，丢弃全部短检更新，正式重新scratch；不提前读取dev/test载荷或做后处理。
''',encoding='utf-8')
    s=mapped((ENGINE/'preflight.py').read_text(encoding='utf-8')).replace("VARIANT='layout_only'","VARIANT='supervision_hidden_reset'")
    s=replace_function(s,'ablation_check',CHECK)
    s=s.replace('ablation_check(model,public,states)','ablation_check(model,public,states,small)')
    (HERE/'preflight.py').write_text(s,encoding='utf-8')
    print('Prepared E147; training remains disabled until preflight passes.',flush=True)


CHECK='''def ablation_check(model,public,states,batch):
    from experiments.E142_dual_geometry.model import GeometryModel as ParentModel
    from .loss_config import SUPERVISED_LAYERS
    expected_cuts=[2,6,8,10]
    assert [d for d in range(12) if d in SUPERVISED_LAYERS]==expected_cuts
    baseline_inputs=[];readouts=[]
    hooks=[layer.register_forward_pre_hook(lambda module,args:baseline_inputs.append(args[0].detach().clone())) for layer in model.layers]
    hooks.append(model.head.register_forward_hook(lambda module,args,value:readouts.append(value.detach().clone())))
    try:
        with torch.no_grad():model(public)
    finally:
        for hook in hooks:hook.remove()
    cuts=[];retained=[]
    for depth in range(12):
        source=model.initial_blocks[-1] if depth==0 else model.layers[depth-1]
        calls=[0];incoming=[]
        def fixed_layout(module,args,value):
            d=calls[0];calls[0]+=1
            return readouts[depth].clone() if d==depth else value
        perturb=source.register_forward_hook(lambda module,args,value:value+.37*public['valid'][...,None])
        fixed=model.head.register_forward_hook(fixed_layout)
        capture=model.layers[depth].register_forward_pre_hook(lambda module,args:incoming.append(args[0].detach().clone()))
        try:
            with torch.no_grad():altered=model(public)
        finally:perturb.remove();fixed.remove();capture.remove()
        if depth in expected_cuts:
            torch.testing.assert_close(incoming[0],baseline_inputs[depth],rtol=0,atol=0)
            torch.testing.assert_close(altered[depth:],states[depth:],rtol=0,atol=0)
            cuts.append(depth)
        else:
            assert not torch.equal(incoming[0],baseline_inputs[depth]),depth
            retained.append(depth)
    assert cuts==expected_cuts and retained==[0,1,3,4,5,7,9,11]
    parent=ParentModel(**train.MODEL).cuda().eval();parent.load_state_dict(model.state_dict(),strict=True)
    with torch.no_grad():original=parent(public)
    torch.testing.assert_close(states[:2],original[:2],rtol=0,atol=0)
    del parent,original
    differentiable=[]
    hook=model.head.register_forward_hook(lambda module,args,value:differentiable.append(value))
    try:
        output=model(public)
        loss=per_case(output,batch,ARMS[0])['final_total'].mean()
        gradients=torch.autograd.grad(loss,[differentiable[d] for d in expected_cuts],allow_unused=False)
        assert all(bool(torch.isfinite(v).all()) and bool(v.abs().sum()>0) for v in gradients)
    finally:hook.remove()
    return dict(reset_after_layers=cuts,retained_after_layers=retained,all_12_edges_intervened=True,
        reset_receivers_and_downstream_bitwise_unchanged=True,S0_hidden_retained_to_S1=True,
        first_two_layouts_bitwise_E142=True,layout_gradient_across_all_resets=True)
'''

if __name__=='__main__':main()
