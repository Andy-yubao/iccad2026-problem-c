from pathlib import Path
import json,hashlib
root=Path.cwd();here=root/'experiments/E98_loss_weight_learning';old=here/'dynamic_weight_learning_v2';dst=here/'dynamic_weight_learning_v3'
def write(name,text):(dst/name).write_text(text,encoding='utf-8')
def replace(text,a,b):
    assert a in text,a[:100]
    return text.replace(a,b)
plan=json.loads((dst/'plan.json').read_text(encoding='utf-8'))
run=(old/'run.py').read_text(encoding='utf-8')
run=replace(run,"optimizer='fresh AdamW including control'","optimizer='fresh AdamW for plus/minus only; frozen control has no optimizer'")
run=replace(run,'from ..qualification import evaluate as evaluator','from ..qualification import evaluate as evaluator\nfrom ..depth_evaluation.repair.canonical_native import warmup')
run=replace(run,"parent_path,center = ROOT/plan['initial_checkpoint'],plan['initial_coefficients']\n            summaries = []", """parent_path,center = ROOT/plan['initial_checkpoint'],plan['initial_coefficients']
            feedback_rows = [rows[j] for j in plan['feedback_train_indices']]
            assert len(feedback_rows)==500 and len({tuple(r['member']) for r in feedback_rows})==500
            assert [list(r['member']) for r in feedback_rows]==plan['feedback_members']
            for row in feedback_rows:
                case=row['case']
                case.reference_positions=public_hard_reference(case.solver_inputs[1],case.solver_inputs[5],case.solver_inputs[6])
            torch.set_num_threads(1)
            with torch.inference_mode(),threadpool_limits(limits=1):
                warmup()
                predictor=Predictor(parent_path,feedback_rows[0]['case'].solver_inputs)
                if (RUN/'baseline_feedback/results.json').exists():
                    parent_score=read(RUN/'baseline_feedback/results.json')
                else:
                    parent_score=evaluate_panel(predictor,parent_path,feedback_rows,RUN/'baseline_feedback',
                        check,beat,'initial_baseline_train500',round=0,arm='control')
            del predictor
            gc.collect();torch.cuda.empty_cache()
            summaries = []""")
run=replace(run,"summaries.append(read(folder/'results.json'))\n                    continue", """completed=read(folder/'results.json')
                    parent_score=completed['train_feedback'][selected['selected_arm']][0]
                    summaries.append(completed)
                    continue""")
start=run.index('                feedback_rows = [rows[j] for j in spec[')
end=run.index('                winner,accepted = choose(scores,plan)',start)
run=run[:start]+"""                torch.set_num_threads(1)
                with torch.inference_mode(),threadpool_limits(limits=1):
                    predictor = Predictor(parent_path,feedback_rows[0]['case'].solver_inputs)
                scores={'control':[dict(parent_score)]}
                assert scores['control'][0]['count']==500
                from .evaluation import model_hash
                parent_state=torch.load(parent_path,map_location='cpu',weights_only=False)
                assert model_hash(parent_state['model'])==parent_score['model_hash']
                del parent_state
                save(folder/'feedback/control/reused.json',dict(
                    checkpoint=parent_path.relative_to(ROOT).as_posix(),checkpoint_sha256=sha(parent_path),
                    panel_sha256=plan['feedback_panel_sha256'],summary=parent_score,
                    actual_solver_calls_this_round=0,training_updates=0,test_read=False))
                for arm in plan['training_arms']:
                    check()
                    scores[arm]=[evaluate_panel(predictor,paths[arm],feedback_rows,
                        folder/f'feedback/{arm}',check,beat,'train500_feedback',round=spec['round'],arm=arm)]
"""+run[end:]
run=replace(run,'feedback_unique_cases=100,feedback_pool_cases=1000','feedback_unique_cases=500,feedback_source_population=16000,control_result_reused=True')
run=replace(run,"'train100/1000 mixed_Q only; dev used for own-Loss epoch selection, not Q'","'fixed train500/16000 mixed_Q only; dev used for own-Loss epoch selection, not Q'")
run=replace(run,"parent_path,center = paths[winner],co[winner]","parent_path,center = paths[winner],co[winner]\n                parent_score = scores[winner][0]")
run=run.replace('train100/1000 mixedQ','fixed train500/16000 mixedQ').replace('调权v2','调权v3')
run=run.replace('\\\\n','\\n')
write('run.py',run)
plan['feedback_panel_sha256']=hashlib.sha256(json.dumps(dict(indices=plan['feedback_train_indices'],members=plan['feedback_members']),sort_keys=True).encode()).hexdigest()
write('plan.json',json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
write('freeze_baseline.py',(old/'freeze_baseline.py').read_text(encoding='utf-8'))
launch=(here/'single_reference_step/launch.ps1').read_text(encoding='utf-8').replace('20261104_weaken_reference120_seed0','20261104_dynamic_train500_v3_seed0').replace('single_reference_step.run','dynamic_weight_learning_v3.run').replace('User explicitly requested continuing only one weakening of position/shape; previous search results excluded','User explicitly authorized five designed plus five seeded random rounds from generalization best71; fixed train500 feedback; frozen control reuse')
write('launch.ps1',launch)
write('README.md','''# E98从泛化增强基线重启：固定train500调权

2026-11-04用户明确授权准备后后台运行。唯一初始权重为泛化增强best71；旧多轮搜索、旧pilot及单步削弱reference候选均不作初始化、选型或设计依据，原证据与STOP保护。

预算10轮：沿用已登记5组设计方向（reference/公开几何、重叠/线长紧凑、最差/平均重叠、线长/紧凑、边界/分组），随后5组固定种子随机方向。每轮plus/minus从当前父模型出发，log(1.2)正反扰动、正系数与系数总和保持；无收益保留父模型与原系数。

训练16k、固定train/dev各500监测、96×12、FFN Dropout .10、残差Dropout .05、50%xy交换（固定位置/形状和公开约束同步变换）不变。plus/minus新建AdamW状态，decay .0003、clip1、模块相对更新界 .005；每阶段own-Loss用于反传/监测/best/LR/早停，epoch0可选。LR1e-4→5e-5，保护2轮、最早第5轮末降档，低档停滞3轮早停，最多15轮。所有分支best/latest保留，阶段内恢复完整状态；不专测恢复。

反馈从完整train16k均匀无放回随机冻结500例（seed2026110421），替代100/1000。所有轮、plus/minus/control共用相同500，不能把不同样本的Q直接比较。基线control先补齐该500一次，以后完全复用父模型结果，零训练、零重复求解；被接纳候选的同500结果成为下一control。按单次固定计算深度评价，混合Q=(加权Q+平均Q)/2改善>.0001且500/500硬合法才接纳；三个Q均报告，平局保留control。固定反馈面板反复使用有自适应过拟合风险，不称未见泛化；收益包含额外训练。

缓存按模型张量/公开输入/member/源码环境身份保存；计时包含输入构造/传输/网络/后处理/独立硬检查，评分在计时外。缓存耗时保留原测量，不称新的同场速度比较。验证后处理Q不参与搜索，本次结束只冻结最终模型和完整反馈，不自动dev/test Q；测试始终封存。

运行目录：`runs/20261104_dynamic_train500_v3_seed0`。训练日志在本目录stdout.log/stderr.log，状态见run/state.json。写入run/STOP或本目录STOP，批次边界安全保存latest并禁止下一阶段。旧运行STOP不移除，不自动唤醒，不超出10轮。
''')
base=here/'runs/20261104_generalization96_seed0'; result=json.loads((base/'results.json').read_text(encoding='utf-8'))
assert result['status'] in ('early_stop','max_epochs','budget')
write('baseline_frozen.json',json.dumps(dict(checkpoint=plan['initial_checkpoint'],checkpoint_sha256=plan['initial_sha256'],source_results_sha256=hashlib.sha256((base/'results.json').read_bytes()).hexdigest(),selection='completed enhanced baseline own devLoss best71; user restart authorization; no Q selection',stop_reason=result['status'],best_epoch=result['best_epoch'],test_read=False),indent=2)+'\n')
print('Prepared v3 sources and fixed500 membership; previous searches untouched.')
