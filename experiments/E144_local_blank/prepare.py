"""Prepare three independent E142-controlled experiments, without starting training."""
import ast
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PARENT=ROOT/'experiments/E142_dual_geometry'
SPECS=(
    ('E144_local_blank','e144_local_blank','local_blank','有上限的脱离距离与局部空白反馈'),
    ('E145_no_relation_value','e145_no_relation_value','no_relation_value','移除关系 value 消息'),
    ('E146_layout_only','e146_layout_only','layout_only','层间仅传布局，不传隐藏状态'),
)

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def replace_function(s,name,replacement):
    node=next(n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name==name)
    lines=s.splitlines(keepends=True)
    return ''.join(lines[:node.lineno-1])+replacement.rstrip()+'\n'+''.join(lines[node.end_lineno:])

def main():
    gate=read(PARENT/'preflight.json');assert gate['status']=='passed'
    for name,digest in gate['sources'].items():assert sha(ROOT/name)==digest,name
    for directory,arm,variant,title in SPECS:
        dst=ROOT/'experiments'/directory
        if (dst/'provenance.json').exists():raise FileExistsError(dst)
        code=arm.split('_')[0];number=code.upper();scope=f'{number}.{variant}.v1.seed0'
        def mapped(s):
            for old,new in [('experiments.E142_dual_geometry','experiments.'+directory),
                ('E142.dual_geometry.v1.seed0',scope),('e142_dual_geometry',arm),('E142',number),("'e142'",repr(code))]:s=s.replace(old,new)
            return s
        copied={}
        for name,digest in gate['sources'].items():
            src=ROOT/name
            if not src.is_relative_to(PARENT):continue
            rel=src.relative_to(PARENT)
            if src.suffix!='.py' and rel.as_posix()!='evaluation/budget.json':continue
            if rel.as_posix() in ('prepare.py','preflight.py'):continue
            path=dst/rel;path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(mapped(src.read_text(encoding='utf-8')),encoding='utf-8')
            copied[rel.as_posix()]=dict(parent=name,sha256=digest)
        plan=read(PARENT/'plan.json')
        result=PARENT/'runs/e142_dual_geometry_seed0/evaluation_no_mib_dev500/results.json'
        weight=PARENT/'runs/e142_dual_geometry_seed0/checkpoints/best.pt'
        plan.update(experiment=f'{number} {title}',revision=scope,status='authorized_preparation',
            authorization='2026-11-16 user requested three independent E142 controls, preparation then background scratch runs with original dev500. Latest clarification: E146 carries layouts only; neither previous hidden states nor S0 hidden states enter refinement layers.',
            arms={arm:plan['arms']['e142_dual_geometry']},architecture={code:variant},
            parameters={code:None},trainable_parameters={code:None},
            dynamic_module_features=45 if variant=='local_blank' else 49,
            module_encoder_input=61 if variant=='local_blank' else 65,
            initialization='Scratch seed0. Retained parameters equal E142 scratch initialization; no trained checkpoint or optimizer loaded.',
            node_feedback='45 dynamic features: capped axis escape / corresponding side, newly entered .25-side strip blank fraction / predicted area; remove old probe025/probe100; remaining E142 fields unchanged' if variant=='local_blank' else plan['node_feedback'],
            pair_feedback='Original pair18 attention bias retained. '+('Ordinary attention values only; remove psi network and its parameters.' if variant=='no_relation_value' else 'Original receiver-aware relation values retained.'),
            hidden_propagation='Only prior layout crosses layers. Each layer starts with public static embedding + prior-layout state embedding, current demand and pair geometry. No previous contextual hidden or S0 hidden bypass; no detach.' if variant=='layout_only' else 'Original E142 hidden and layout propagation retained.',
            preflight='Independent geometry union/strip checks for E144; exact scratch retained parameters, E139 Loss and gradients, pair18, ablation semantics, original five train cases, short guarded CUDA graph64/tail32 check. Unchanged engine/resume gates inherited from E142.',
            comparison_control=dict(result=result.relative_to(ROOT).as_posix(),result_sha256=sha(result),checkpoint=weight.relative_to(ROOT).as_posix(),checkpoint_sha256=sha(weight)),
            comparison='Same E142 batch64, panels, seed0, supervision, schedule and per-arm4h/200epoch budget; independent single-factor arms, one seed/adaptive stopping.',
            queue='Sequential E144 -> E145 -> E146. Each: train -> unique own-Loss-best freeze -> original fixed dev500 -> original postprocessing and hard checks -> immediate Q logs. Error/STOP blocks later arms.',
            relation_width=0 if variant=='no_relation_value' else 8,
            preflight_parent_sha256=sha(PARENT/'preflight.json'))
        save(dst/'plan.json',plan)
        ev=read(PARENT/'evaluation/plan.json');ev.update(revision=scope+'.dev500',authorization=plan['authorization'],models={});save(dst/'evaluation/plan.json',ev)
        save(dst/'provenance.json',dict(parent='experiments/E142_dual_geometry',parent_preflight_sha256=sha(PARENT/'preflight.json'),copied_originals=copied,variant=variant))
        save(dst/'authorization.json',dict(scope=scope,training_enabled=False,allowed_arms=[],prepared_arms=[arm],automatic_evaluation=True,automatic_retry=False,test_read=False,note=plan['authorization']))
        save(dst/'queue_state.json',dict(status='prepared',completed=[],active=None))
        path=dst/'run_queue.py';s=path.read_text(encoding='utf-8').replace("'E139_existing_batch32'","'E142_existing_batch64'")
        s=s.replace('Same supervision/Loss/panel; changed architecture and batch32->64, historical control, single seed; not architecture-only causal comparison.',plan['comparison'])
        s=s.replace('dual geometry versus historical E139',variant+' versus E142');path.write_text(s,encoding='utf-8')
        path=dst/'model.py';s=path.read_text(encoding='utf-8')
        if variant=='local_blank':
            needle='        self.layers=nn.ModuleList([CoordinatedBlock(base,width) for base in self.layers])'
            s=s.replace(needle,needle+'''
        # Initialize the complete E142 model first, preserving every retained weight.
        original=self.demand_encoder[0]
        with torch.random.fork_rng(devices=[]):reduced=nn.Linear(45+16,width)
        with torch.no_grad():
            reduced.weight.copy_(original.weight[:,list(range(16))+list(range(20,65))])
            reduced.bias.copy_(original.bias)
        self.demand_encoder[0]=reduced
''')
        elif variant=='no_relation_value':
            start=s.index('        content,gate=self.edge(pair).chunk(2,-1)')
            end=s.index('        local=self.local',start)
            s=s[:start]+'        attended=(attention@v).transpose(1,2).reshape(b,n,w)\n'+s[end:]
            needle='        self.layers=nn.ModuleList([CoordinatedBlock(base,width) for base in self.layers])'
            s=s.replace(needle,needle+'''
        # Delete psi only after full scratch initialization; retained RNG is identical.
        for layer in self.layers:
            for name in ('edge','receiver','sender','relation_value'):delattr(layer,name)
''')
        else:
            s=s.replace("        h=self.input(g['features'])*g['valid'][...,None]", "        static=self.input(g['features'])*g['valid'][...,None]\n        h=static")
            s=s.replace("h=layer(h+self.state_embed(state*g['mask'])", "h=layer(static+self.state_embed(state*g['mask'])")
        path.write_text(s,encoding='utf-8')
        text=f'''# {number}：{title}

用户授权准备通过后后台从头训练并自动验证原固定500例。本组以E142为独立对照，不叠加其他两组修改。

{plan['node_feedback']}

{plan['pair_feedback']}

{plan['hidden_propagation']}

其余完整沿用E142：原36k/cache、seed0、batch64/末批32、FP32 deterministic、原优化器/数值门禁/早停、每组4h/200epoch上限。E139监督：S2/S6不含overlap，S8/S10完整，各.05；S12完整.8。原reference及公开硬约束Loss不变，不读test。网络预测全部模块，由原后处理满足硬规格。

训练、监测、best、LR、早停使用同一Loss，Q不参与选模。完成后冻结own-Loss-best，自动执行原固定dev500及原后处理预算、独立硬检查，立即在评价日志、队列日志和background/quality.log输出[Q]摘要。STOP/错误阻断后续，不自动重试或恢复。

实时记录：preflight.json、queue_state.json、runs/{arm}_seed0/heartbeat.json、launch_logs/。本轮整体入口为E144_local_blank.background.run_suite，依次E144→E145→E146。
'''
        (dst/'README.md').write_text(text,encoding='utf-8')
        (dst/'DESIGN.md').write_text(text+'\n所有动态摘要只取当前预测及公开输入，detach；布局的原state_embed/pair梯度保持。原两层初始化仅用于产生S0。\n',encoding='utf-8')
        print('Prepared sources: '+directory,flush=True)

if __name__=='__main__':main()
