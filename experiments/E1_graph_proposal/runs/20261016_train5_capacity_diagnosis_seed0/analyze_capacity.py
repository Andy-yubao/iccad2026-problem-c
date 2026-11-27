"""Report only saved train5 diagnosis; no tensors or model execution."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

run=Path(__file__).resolve().parent
r=json.loads((run/'results.json').read_text(encoding='utf-8'))
assert r['status']=='completed'
rows=r['cases'];s=r['summary']
def q(name):return s[name]['neutral_runtime_weighted_cost_all_cases']
analysis={'case_count':len(rows),'workers':[int(x['source']['shard'].split('/')[0][7:]) for x in rows],
          'current_train5_mse':r['current_train5_mse'],
          'reference_weighted_Q':q('reference'),'current_weighted_Q':q('deterministic_final'),
          'known_optimum_interval':[1,q('reference')],
          'certified_current_additive_regret_lower_bound':q('deterministic_final')-q('reference'),
          'reference_repacked_Q':q('reference_repacked'),
          'probes':{},'cases':[],'test_payload_reads':0}
for name,v in r['probes'].items():
    cpu_mse=float(np.mean([x[name+'_prediction_metrics']['mse'] for x in rows]))
    analysis['probes'][name]={'parameters':v['parameters'],'best_cuda_mse':v['best_mse'],
       'cpu_mse_for_layout_evaluation':cpu_mse,'final_Q':q(name+'_final'),
       'mse_reduction_vs_current':1-cpu_mse/r['current_train5_mse']['mse'],
       'best_step':v['best_step'],'completed_steps':v['completed_steps'],'seconds':v['seconds']}
for row in rows:
    ref=np.asarray(row['variants']['reference']['positions'])
    decoded=np.asarray(row['variants']['target_decode']['positions'])
    depth=[]
    for i in range(len(ref)):
        for j in range(i):
            ov=np.minimum(decoded[i,:2]+decoded[i,2:],decoded[j,:2]+decoded[j,2:])-np.maximum(decoded[i,:2],decoded[j,:2])
            if (ov>1e-6).all():depth.append(float(ov.min()))
    a={'case_id':row['case_id'],'n':len(ref),'reference_Q':row['variants']['reference']['metrics']['cost'],
       'current_Q':row['variants']['deterministic_final']['metrics']['cost'],
       'reference_repacked_Q':row['variants']['reference_repacked']['metrics']['cost'],
       'target_decode_max_coordinate_or_dimension_error':float(np.abs(ref-decoded).max()),
       'target_decode_overlap_count':row['variants']['target_decode']['metrics']['overlap_violations'],
       'target_decode_max_min_axis_overlap':max(depth,default=0),
       'symmetry_lower_bound':row['symmetry']['proven_masked_mse_lower_bound']}
    a['certified_additive_regret_lower_bound']=a['current_Q']-a['reference_Q']
    analysis['cases'].append(a)
(run/'diagnostic_summary.json').write_text(json.dumps(analysis,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(14,5.5))
for name,color in [('small','#2563eb'),('large','#ea580c')]:
    c=r['probes'][name]['curve']
    axes[0].plot([v['step'] for v in c],[v['mse'] for v in c],label=f"{name}: {r['probes'][name]['parameters']:,} parameters",color=color)
axes[0].axhline(r['current_train5_mse']['mse'],color='#64748b',ls='--',label='Frozen model, same train5')
axes[0].set(yscale='log',xlabel='Optimizer updates (each uses all five layouts)',ylabel='Mean masked MSE',title='Can these models memorize five training layouts?')
axes[0].legend(fontsize=8);axes[0].grid(alpha=.2)
specs=[('reference','Reference','#15803d'),('reference_repacked','Perfect draft + packing','#94a3b8'),
       ('deterministic_final','Frozen model + packing','#7c3aed'),('small_final','Small memory fit + packing','#2563eb'),
       ('large_final','Large memory fit + packing','#ea580c')]
x=np.arange(5);width=.15
for j,(name,label,color) in enumerate(specs):
    axes[1].bar(x+(j-2)*width,[v['variants'][name]['metrics']['cost'] for v in rows],width,label=label,color=color)
axes[1].axhline(1,color='#15803d',ls=':',label='Universal neutral-Q lower bound')
axes[1].set_ylim(0,4.7)
axes[1].set(xticks=x,xticklabels=[f"W{w}\nn={row['input_summary']['block_count']}" for w,row in zip(analysis['workers'],rows)],
            ylabel='Neutral-runtime Q (lower is better)',title='Even a perfect draft loses quality during packing')
axes[1].legend(fontsize=7,ncol=2);axes[1].grid(axis='y',alpha=.2)
fig.suptitle('Train-only diagnosis: capacity, supervision, and postprocessing',fontsize=15,fontweight='bold')
fig.text(.06,.035,'Five cases chosen before scores; memory probes train on these same cases. Oracle variants use train labels for diagnosis only.\nReference Q is an attainable upper bound on the unknown optimum; these results do not establish full-train or test performance.',fontsize=9,color='#475569')
fig.tight_layout(rect=(0,.13,1,.94));fig.savefig(run/'capacity_diagnostics.png',dpi=170);plt.close(fig)

table='\n'.join(f"| {w} | {row['n']} | {row['reference_Q']:.4f} | {row['reference_repacked_Q']:.4f} | {row['current_Q']:.4f} | {row['certified_additive_regret_lower_bound']:.4f} |" for w,row in zip(analysis['workers'],analysis['cases']))
probe_table='\n'.join(f"| {name} | {v['parameters']:,} | {v['best_step']} / {v['completed_steps']} | {v['cpu_mse_for_layout_evaluation']:.8f} | {v['final_Q']:.4f} |" for name,v in analysis['probes'].items())
stages=['reference','target_decode','reference_corrected','reference_repacked','heuristic_final','deterministic_raw','deterministic_corrected','deterministic_final','small_raw','small_corrected','small_final','large_raw','large_corrected','large_final']
stage_table='\n'.join(f"| {k} | {s[k]['feasible_count']}/5 | {q(k):.6f} | {s[k]['mean_hpwl_gap_returned_geometry']:+.4f} | {s[k]['mean_area_gap_returned_geometry']:+.4f} | {s[k]['mean_boundary_violations_returned_geometry']:.2f} / {s[k]['mean_grouping_violations_returned_geometry']:.2f} / {s[k]['mean_mib_violations_returned_geometry']:.2f} |" for k in stages)
report=f'''# 训练集上的容量、参考分数与后处理诊断

2026-10-16。用户明确要求检查当前模型表达能力、训练解能否接近最优Q，以及MSE与Q不一致的原因。本次是独立的短诊断，不恢复已停止的20,000例训练，不读取dev/test。5例均在读取标签和分数前选定：冻结train中worker0、24、49、74、99的首个成员。不能将五例结果称为完整训练集表现。

## 1. 当前结构与表达边界

确定性模型 **80,131参数**。每节点17维输入→Linear(17,64)→SiLU→LayerNorm；3个残差消息层，每层拼接自身、连接邻居加权均值、MIB组均值、grouping组均值、全图均值，320→64→64，加SiLU、残差与LayerNorm；最后64→64→3，输出归一化中心XY、log(w/h)。输入包括相对面积、公开尺寸/锚点、fixed/preplaced标志、四种边界位、MIB/group规模、连线度、pin权重和加权中心。

解码由面积与宽高比恢复尺寸，fixed尺寸和preplaced位置/尺寸按公开规格覆盖。随后12轮重叠修正与贪心pack；训练没有穿过这两步反向传播，也未直接优化Q。pin空间信息被压缩为总权重/均值，平均聚合丢失部分细节，对称节点不能自动区分；加宽同一结构不保证消除表示歧义。不能称严格三跳感受野，因为每层还有全图均值。此次5例的精确节点交换对称性检查未得到非零不可消除MSE下界，不代表排除了所有表达限制。

## 2. “最优Q”可以知道多少

中性时间、硬合法时，Q=min((1+0.5(max(0,HPWL_gap)+max(0,bbox_gap)))*exp(2V_rel),9.999999)；硬非法时Q=10。因此普遍下界为1。若现成合法解的两项gap不为正且软违反为零，则可证明该例Q最优为1。这是评分最优，不能解释为HPWL和面积各自全局最优。正式含时间分数的下界与此不同。

实际参考布局的Q并不自动等于1：此次5例全部硬合法，但按未修改的官方评分仍有boundary/grouping/MIB违反，reference Q范围1.0370～1.6099，HPWL和bbox gap约为零。未得到任何一例Q*=1的可达证明，也未求解精确全局布局优化。

仍然可以给出有效界：**1 ≤ Q* ≤ Q_reference**。只要模型Q高于一个已知合法参考解，其次优差距至少为Q_model−Q_reference；无需知道真最优也能证明模型还有差距。此结论对固定五例的同权重聚合也成立。

| worker | 模块数 | 参考Q | 完美参考经同后处理Q | 当前冻结模型最终Q | 距真最优的差距下界 |
|---|---:|---:|---:|---:|---:|
{table}

加权聚合：参考 **{q('reference'):.6f}**，当前模型 **{q('deterministic_final'):.6f}**，真最优聚合分在[1,{q('reference'):.6f}]内；当前模型的次优差距至少 **{analysis['certified_current_additive_regret_lower_bound']:.6f}**。5例全部明显差于各自参考，故这些已训练样例上没有贴近已有优质解，不能靠“test MSE与train接近”证明求解质量好。

## 3. 后处理存在直接反例，不能把高Q全归因模型容量

把参考XYWH原样当作完美提议，经过生产流程相同的correct+pack后，5/5例Q都变差，加权Q从{q('reference'):.6f}变为 **{q('reference_repacked'):.6f}**。参考经过correct保持分数，主要损失发生在重新pack阶段。脚本在冻结模型的5个提议上验证复制的诊断后处理与生产Optimizer最终坐标逐项一致（绝对容差1e-10）；这不等于它是最优的几何修复。

生产pack会重新安排所有可移动块，并非只在必要时微调冲突；完美坐标也会被重排。因此即使网络已经精确输出参考布局，当前整个求解链也不保证保留其Q。这个反例说明后处理确实是瓶颈之一，但不排除网络预测仍有问题。

另一个必须分开的反例是encode_target→decode_layout：监督状态完全正确时，5例均出现微小重叠，被硬约束评分为10。坐标/尺寸最大差异与重叠穿透量保存在diagnostic_summary.json；这是面积重建/浮点与严格接触容差的数值敏感性，不能据此声称参考数据本身硬非法，也不能把Q10当网络未学会。未通过放宽检查规避此问题。

## 4. 小样本记忆与加大容量对照

两个独立诊断模型均seed0从头初始化、每步训练同5例、AdamW(.001,wd=.0001)、clip1，检查频率100步；训练误差平台降学习率和早停仅用于记忆诊断，不进行dev选择。每个最多1500步/150秒，总任务预算600秒，实际{r['wall_seconds']:.2f}秒。原结构64×3，对照128×6；这是同时改变深宽的容量探针，不能归因单个组件。两者都只按5例train MSE选best，未依据Q选模型或超参。小模型达到步数上限，大模型在1000步达到预设MSE<1e-5的记忆目标；两者未触发平台降学习率，不声称是等耗时或等步数的完整容量基准。

当前冻结模型在相同5例MSE为 **{r['current_train5_mse']['mse']:.8f}**。下表CPU MSE与最终布局评分同设备；CUDA训练选中时的微小数值差异另记，不把GPU指标冒充CPU评估。

| 探针 | 参数 | best步 / 完成步 | 同5例CPU MSE | 最终加权Q |
|---|---:|---:|---:|---:|
{probe_table}

小模型能够将五例监督误差压得远低于当前全量训练权重，说明当前架构有记忆这些实例的能力，原训练过程并非已经碰到这些五例的不可突破表达下界。但五例记忆不证明能拟合20,000例，也不能排除全量容量不足、优化不足、表示歧义或多解监督问题；扩大网络的结果也仅限这一预算/优化器和五例。没有训练新的正式候选，没有用这些probe替换原best或重测test。

关键对照是：小模型MSE降低约99.918%，最终Q仍为{q('small_final'):.4f}，没有优于原模型{q('deterministic_final'):.4f}；大模型MSE降低约99.992%，最终Q反而为{q('large_final'):.4f}。这不是“增加容量必然有害”的统计结论，而是一个直接反例：把监督误差压得接近零，仍不足以保证现有完整求解链得到低Q。

## 5. 全阶段结果

Q是官方中性时间口径，块数指数加权；HPWL/bbox gap与软违反数是逐例等权均值。失败仍保留在Q中，不能用非法raw的HPWL较低就称质量好。

| 变体 | 硬合法 | 加权Q | 平均HPWL gap | 平均bbox gap | boundary / grouping / MIB违反 |
|---|---:|---:|---:|---:|---:|
{stage_table}

MSE不直接监督接触/边界/MIB等离散条件；微小形状差异就可能破坏接触，软违反进入指数惩罚；bbox又受外围极值影响。MSE对布局等权，Q还偏重大布局。因而train/test MSE相近只说明监督误差泛化差距不大，不能说明模型在任一集合上的布局已经好。

## 6. 研究判断与下一步

这些检查必要，已经排除了“只要增加参数、继续降低MSE就会自然得到优质最终布局”的充分性假设。应先在train/dev检验保持已有合法优质提议、只修必要冲突的后处理，以及对软关系/最终质量更一致的目标；再通过固定dev对照判断是否需要扩大网络。当前不改生产模型、后处理或评分定义，不重开test。

全部证据见[results.json](results.json)、[紧凑诊断数据](diagnostic_summary.json)、[图](capacity_diagnostics.png)、[config.json](config.json)、[源码快照](source_snapshot.zip)。完整几何只在results.json保存；probe权重仅用于复核本次记忆诊断。原best哈希复核未变。报告生成命令为项目Python执行本目录analyze_capacity.py；只读已保存结果。
'''
(run/'analysis.md').write_text(report,encoding='utf-8')
print(json.dumps(analysis,ensure_ascii=False,indent=2))
