"""Report saved sparse-anchor evidence and concrete, unexecuted research proposals."""
import html
import re
import zipfile
import numpy as np
from .io import HERE,ROOT,read,save,sha
from .diagnose import OUT,OLD


def charts(result,local,sensitivity):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    groups=['low','middle','high'];labels=['Low pin coverage','Middle','High pin coverage'];x=np.arange(3)
    fig,axes=plt.subplots(2,2,figsize=(13,9))
    for j,(key,label) in enumerate((('pin','Pin-connected'),('no_pin','No direct pin'))):
        axes[0,0].bar(x+(j-.5)*.33,[result['summary'][g][key]['case_mean_local_error'] for g in groups],.33,label=label)
    axes[0,0].set(xticks=x,xticklabels=labels,ylabel='Decoded center error / sqrt(area)',title='Error concentrates in blocks without direct pins');axes[0,0].legend()
    for g,label in zip(groups,labels):
        steps=[0,3,6,9,12];axes[0,1].plot(steps,[result['summary'][g]['layer'+str(k)]['spread_ratio'] for k in steps],'o-',label=label)
    axes[0,1].axhline(1,color='gray',ls='--');axes[0,1].set(xlabel='State prediction after layer',ylabel='Predicted/reference RMS spread',title='Final layouts do not globally collapse');axes[0,1].legend()
    for j,(key,label) in enumerate((('pin_pin','Both pin-connected'),('mixed','Mixed'),('no_pin_no_pin','Neither pin-connected'))):
        axes[1,0].bar(x+(j-1)*.24,[local['summary'][g]['pairs'][key]['near_relative_local_error_case_mean'] for g in groups],.24,label=label)
    axes[1,0].set(xticks=x,xticklabels=labels,ylabel='Reference-near pair relative center error',title='Local structure is less accurate without direct pins');axes[1,0].legend(fontsize=8)
    grid=result['pin_density_cross_strata']
    for j,(suffix,label) in enumerate((('_sparse','Sparse relative to exact-N median'),('_dense','Dense relative to exact-N median'))):
        axes[1,1].bar(x+(j-.5)*.33,[grid[g+suffix]['G'] for g in groups],.33,label=label)
    axes[1,1].set(xticks=x,xticklabels=labels,ylabel='Mean fixed G',title='Both pin coverage and graph density matter');axes[1,1].legend(fontsize=8)
    for ax in axes.flat:ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'mechanism.png',dpi=155);plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(14,5.5))
    variants=['full','pins_half','pins_none','edges_half','edges_none','pins_edges_none']
    for model,records in sensitivity.items():
        axes[0].plot(np.arange(6),[100*np.mean([r['variants'][v]['overlap'] for r in records]) for v in variants],'o-',label=model)
    axes[0].set(xticks=np.arange(6),xticklabels=['Full','Half pins','No pins','Half edges','No edges','Neither'],yscale='log',
                ylabel='Pair-overlap area / total area (%)',title='Five-case deletion probes: frozen-model sensitivity');axes[0].tick_params(axis='x',labelrotation=25);axes[0].legend()
    for j,(branch,label) in enumerate((('weight075','Center weight0.75'),('softband005','Center softband0.05'))):
        axes[1].bar(x+(j-.5)*.33,[result['saved_E63_controls']['strata'][g][branch]['delta_percentage_points'] for g in groups],.33,label=label)
    axes[1].axhline(0,color='gray',lw=1);axes[1].set(xticks=x,xticklabels=labels,ylabel='Overlap change versus matched control (percentage points)',title='Existing E63 center relaxation: modest subgroup effects');axes[1].legend(fontsize=8)
    for ax in axes:ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'interventions.png',dpi=155);plt.close(fig)


def render_html(report):
    blocks=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>E69 稀疏位置线索进一步诊断</title><style>body{max-width:1120px;margin:35px auto;padding:0 22px;font:17px/1.8 system-ui,"Microsoft YaHei",sans-serif;background:#f6f8fa;color:#203044}h1,h2,h3{line-height:1.4}img{width:100%;border:1px solid #dde3e8;background:white}table{border-collapse:collapse;background:white;width:100%}th,td{border:1px solid #ccd6df;padding:8px}a{color:#1769aa}p{margin:.8em 0}</style>'];table=False
    for line in report.splitlines():
        if line.startswith('|'):
            cells=[c.strip() for c in line.strip('|').split('|')]
            if all(re.fullmatch(':?-+:?',c) for c in cells):continue
            tag='td' if table else 'th'
            if not table:blocks.append('<table>');table=True
            blocks.append('<tr>'+''.join(f'<{tag}>{html.escape(c)}</{tag}>' for c in cells)+'</tr>');continue
        if table:blocks.append('</table>');table=False
        if not line:continue
        if line.startswith('!['):
            src=line.split('](',1)[1][:-1];blocks.append('<p><img src="'+html.escape(src)+'"></p>');continue
        tag='p';value=line
        for prefix,t in (('### ','h3'),('## ','h2'),('# ','h1')):
            if line.startswith(prefix):tag=t;value=line[len(prefix):];break
        value=html.escape(value);value=re.sub(r'\*\*(.+?)\*\*',r'<strong>\1</strong>',value)
        value=re.sub(r'\[([^]]+)\]\(([^)]+)\)',r'<a href="\2">\1</a>',value)
        blocks.append(f'<{tag}>{value}</{tag}>')
    if table:blocks.append('</table>')
    blocks.append('</html>');return '\n'.join(blocks)


def main():
    r=read(OUT/'results.json');loc=read(OUT/'local_structure.json');sensitivity=read(OUT/'input_sensitivity.json')
    s=r['summary'];low=s['low'];high=s['high'];c=r['saved_E63_controls']['strata'];lp=loc['summary']['low']['pairs']
    report=f'''# E69后续：端口覆盖不足与连接稀疏为何更难

**结论：目前证据更支持“缺少直接位置线索时，内部相对结构恢复和联合避碰不足”，没有支持“整图收缩”“消息无法到达”或“弱端口任务特别被中心梯度压倒”作为主要解释。** 同时，现有输入确有端口信息压缩缺口，固定G与真实连接目标也存在距离；这些是值得针对性修改的机制。

本轮只诊断。固定E65 seed0 best在既有dev_monitor500上补逐层状态与输出梯度；复用dev2000和E63保存结果；对5个公开条件选出的富端口案例做E42/E65删输入敏感性；同E62的5个训练成员只计算参数梯度，不作一步更新。没有优化器、训练、后处理、选模、测试或自动唤醒。新前向逐例G与E69原文件最大差{r['stored_E65_G_max_abs_difference']:.2g}。

## 1. 先把“弱约束”说清楚

端口连接与模块连接给出加权线长目标及位置线索，不直接规定模块的参考中心。端口少、图稀疏并不减少面积、预放置、固定形状和不重叠等硬要求。对优化问题而言，可选布局可能更多；对逐点拟合特定reference而言，可用线索反而可能更少。单一reference不能用来证明多解、不可识别或不可约误差，本轮不宣称已经证明这些。

## 2. 错误发生在哪里：整体尺度正常，内部相对位置较差

端口覆盖≤30%为低组80例，≥80%为高组206例，其余214例。阈值在本次运行前记录；这些仍是已暴露开发集的探索性子群。

| E65 dev500 | 低端口组 | 高端口组 |
|---|---:|---:|
| 平均G | {low['G']:.4f} | {high['G']:.4f} |
| 平均真实交面积比例 | {low['overlap']:.2%} | {high['overlap']:.2%} |
| 最终整体空间展开/参考展开 | {low['layer12']['spread_ratio']:.3f} | {high['layer12']['spread_ratio']:.3f} |
| 有端口模块局部中心误差，逐例等权 | {low['pin']['case_mean_local_error']:.3f} | {high['pin']['case_mean_local_error']:.3f} |
| 无端口模块局部中心误差，逐例等权 | {low['no_pin']['case_mean_local_error']:.3f} | {high['no_pin']['case_mean_local_error']:.3f} |

展开量为去均值后的中心RMS距离之比，排除预放置模块。低组最终约0.990，拟合reference的整体尺度系数约0.962。逐层从初始接近中心的小尺度输出逐步展开，第6层短暂过展、末层回到约1；**不能把低组的主要问题描述成整图平均化坍缩。** 局部多模态或小片区错位仍未排除。

低组无端口模块4009个，其中2861个与有端口模块只有一条模块边之隔，1143个需两跳及以上，仅5个不可达。一跳模块局部误差约0.676，两跳以上约0.731。且现有网络有全局attention，并不只沿稀疏模块边通信。因此，简单增加层数以扩大可达范围，证据不足；更值得研究的是怎样提取和传递有用的位置关系。

进一步检查reference中的近邻对：低组“均有端口/一有一无/均无端口”的局部相对中心误差分别为{lp['pin_pin']['near_relative_local_error_case_mean']:.3f}/{lp['mixed']['near_relative_local_error_case_mean']:.3f}/{lp['no_pin_no_pin']['near_relative_local_error_case_mean']:.3f}。对reference轴间隔足够清晰的邻对，均无端口者发生轴向次序翻转约{lp['no_pin_no_pin']['near_axis_flip_fraction']:.1%}，均有端口者为{lp['pin_pin']['near_axis_flip_fraction']:.1%}。均无端口对贡献低组约{lp['no_pin_no_pin']['overlap_area_share_case_mean']:.1%}交面积；该份额受模块数量影响，不能单独当风险比。

按精确N的开发连接密度中位数分稀/密，低端口组的G分别0.799/0.463，高端口组0.214/0.167。端口覆盖与稀疏性不是同一个维度，但此表仍未完全配平其他输入差异。

![错误位置、逐层展开与分组](mechanism.png)

## 3. 已确认的表示缺口：端口分布被压成了一个均值

源码features.py只向每个模块提供端口总权重及加权平均坐标，模型没有读取完整p2b/pins。不同的端口分布可能压成完全相同的输入。

构造三个端口x=1、2、4：A的权重为2、0、1，B为0、3、0。两者总权重均3、加权均值均2，其余公开输入相同，现有编码逐位相同。但单模块端口线长子问题的最优x分别为1和2；在x=1/2处，A成本3/4，B成本3/0。可见均值无法保留加权L1线长所需的分布信息。这里只证明编码可能丢失任务信息，没有测定真实数据中这类冲突的频率，也不是完整布局最优性证明。

当前G同样没有直接的实际b2b/p2b线长项；relative_local的近邻由reference几何确定，contact还指定了reference中的接触方向。保持输出state不变时，删除端口/模块边并不会改变G，本轮5例核验成立。于是训练主要借参考去学习输入和布局的映射，而不是直接对每个实际连线目标求解。

## 4. 删输入对照：网络确实依赖线索，但不能解读成新任务最优性

5例仅按公开高端口覆盖和N接近30/50/70/90/110选取。保持标签和几何规格不变，嵌套固定删除端口边或模块边。E65平均交面积：完整5.57%，保留一半端口边7.16%，删全部端口边10.35%；保留一半模块边6.32%，删全部模块边8.78%。两者都删时67.82%。E42表现出相似依赖。

这支持输入线索影响局部几何的机制，不只是原始跨案例混杂。但删边同时改变了真实优化任务，旧reference不再代表新目标的最优布局，全部删除也是明显分布外压力；不能称为端口的独立因果效应、自然弱组的泛化误差或新增正式质量成绩。

## 5. 中心Loss是否牵扯过强

**存在与避碰争夺方向的局部证据，但本轮不支持把弱端口劣势主要归因于更强的中心梯度。** 不能从中心项占标量G约53%直接推出梯度主导。

在500例临时输出state空间，低组加权中心梯度范数/其他项合成梯度范数的逐例均值为{low['center_gradient_ratio_to_rest']:.2f}，高组为{high['center_gradient_ratio_to_rest']:.2f}；中心与重叠梯度余弦为{low['center_overlap_cosine']:.3f}/{high['center_overlap_cosine']:.3f}，总梯度与重叠仍同向。输出空间与参数空间不同，这不推翻E62关于共享参数和裁剪冲突的局部发现。

又在原5个训练成员对E65参数作分项梯度检查：中心与重叠在4/5例反向；总梯度在2/5例反向，两例端口覆盖分别100%及96.5%。降低中心系数可改善这5例的一阶避碰对齐，但这组成员没有覆盖≤30%的弱组，没有模拟Adam、裁剪或长期更新，不能据此选新权重。

更直接的既有训练对照来自E63，其完整2000例按端口覆盖重新分组：

| E63相对同起点续训control | 低端口324例 | 中间868例 | 高端口808例 |
|---|---:|---:|---:|
| 中心权重0.75：交面积相对变化 | {c['low']['weight075']['relative_change']:.2%} | {c['middle']['weight075']['relative_change']:.2%} | {c['high']['weight075']['relative_change']:.2%} |
| 中心小误差容忍带0.05：交面积相对变化 | {c['low']['softband005']['relative_change']:.2%} | {c['middle']['softband005']['relative_change']:.2%} | {c['high']['softband005']['relative_change']:.2%} |

0.75对低组仅改善约2.8%，仍从8.32%变为8.09%，远未解决；容忍带在低组略退化。该分析为E42起点的历史单seed分支，不能当成E65已验证的效果。**不建议把下一轮主要预算继续花在全局中心权重微调，也不建议按“端口少”直接取消中心监督。**

![删输入敏感性及已有Loss对照](interventions.png)

## 6. 解决方案与执行优先级

### A. 优先：保留完整端口分布，并显式传播位置线索，固定G不变

把完整公开端口坐标、连接权重作为端口token，通过带真实连接权重的模块—端口消息读入；保留无直连模块与其他模块的交互。最低成本版本可先用多方向分箱/加权分位数补充均值，完整token版本保留更多信息。不要把端口均值当作硬位置，也不要将reference自由坐标引入输入。

给传播过程提供公开可算的一跳/多跳端口信息及预放置位置，并区分“直接观测、间接传播、无直接信息”。边界方向只提供方向约束，不伪装成精确坐标。使用稳定归一化与有界残差；保留实际连线权重，不能把避碰用的全连接关系冒充物理连线。

最小可辨识消融为同参数的“只见总量/均值”对照和“保留端口分布”候选，二者其余模型、G、数据、预算一致；若再加入传播标记，另做独立消融。本轮不启动。它针对已证实的输入压缩缺口；是否足以改善弱覆盖无直连模块仍需实验。

### B. 若A仍无法改善内部几何：引入明确的联合避碰步骤

保持公开规格解码，针对当前矩形的穿透和相对空间关系，设计固定少量步数、可解释的协同位移/空间分配模块，并在训练图内评价最终草稿。预放置模块不可移动，固定形状/MIB等仍受约束；公开有线连接的吸引和所有矩形的排斥分开处理。区别于E66通用残差块循环，它应显式使用几何冲突和可行方向，而非只增加迭代深度。区别于E65已有几何消息，仅把几何作为神经消息特征不等于保证联合避碰。

先用最多5个相关训练案例的输出级诊断检验“消碰、相对关系、边界、固定规格”取舍，再决定是否训练。有限步投影不保证彻底合法或Q更好；若采用真正无重叠的结构化构造，还需专门解决预放置与分组兼容，不能把普通bin packing直接套用。

### C. Loss可以调整，但应修复任务目标与单reference模仿的距离

首选研究方向是固定的混合目标：L新 = L几何可行性 + λ线长·L实际连接线长 + λ紧凑·L紧凑度 + λ参考·L参考正则。线长直接使用完整b2b/p2b权重和端口坐标，按公开总权重和sqrt(总面积)归一化；几何项包括碰撞及公开分组/边界/MIB，硬规格优先由构造保证。参考项用于保护原先学到的结构，但不应把reference中的每个接触边和精确中心都当作唯一任务答案。紧凑目标不能单独使用，否则会鼓励重叠；线长也不能抵消硬不可行。

这属于新的固定Loss版本，而非在原G里临时变权。具体常数只在训练面板上预先定标后冻结；训练、监测、best、LR、早停使用同一个新Loss。旧G和原始几何作为共同报告指标，不能拿不同分支自身总Loss直接比较，也不能用Q偷偷重选checkpoint。

如果希望先做最小中心项消融，可比较“固定中心系数0.25”与原系数1.0，其他项完全相同；0.25是待验证的强对照而非诊断得出的最佳值，不做0.9/0.8/0.7式连续试探。它检验强中心牵制是否仍有实质影响，不能弥补端口信息丢失。容忍带0.05已有低组无益证据，不优先重试；按低端口自适应减权会改变困难任务的有效权重、可能掩盖难例，不建议直接采用。

若进一步证实同一公开输入附近存在多个可行高质量布局，再研究K个联合草稿/集合监督：模型给出多个完整布局而非对各模块独立取平均，参考只匹配其中合适的候选，所有候选仍接受几何/真实连接目标。不能任意交换不同模块身份、不能把旋转/平移当成总是合法的等价类；现有证据尚不证明必须使用多假设。

### 建议的最小顺序

先做A的完整端口表示单因素验证，沿用固定G；重点检查原固定低端口且稀疏子群、无直连模块的近邻相对误差/次序翻转/重叠，以及同100例人工观感。整体均值和富端口组不能用来掩盖难组退化。

随后依据结果，在B的几何结构改进和C的固定任务目标之间选择一个，不同时叠加。若需严格分辨表示与目标的贡献，可预先登记“均值/完整端口 × 原G/新固定目标”的2×2单seed0实验；不做多种子，不按开发结果临时扩大权重搜索。原G继续作共同诊断口径；旧E42/E65保留。

## 证据、范围与复现

- [逐例逐层与输出梯度](case_diagnostics.json)、[局部结构](local_structure.json)、[5训练例参数梯度](parameter_gradients.json)、[5例×2模型删输入](input_sensitivity.json)、[汇总](results.json)、[配置](config.json)、[执行前计划](../../diagnosis_plan.json)、[来源清单](input_manifest.json)、[源码快照](source_snapshot.zip)。此前[同例图文报告](../20261028_dev2000/index.html)保留原样，没有新增人工评分。
- dev500嵌套于dev2000；E63来自已完成seed0实验，都是自适应开发证据。梯度诊断不是实际Adam更新，输入删除不是重训，编码反例不是实际数据中的不可识别证明。AI图表核对不代表用户认可草稿。
- 初次执行在全部模型证据已保存后，仅合成编码反例的FP32精确相等检查因1ULP舍入失败；修正合成面积使归一化尺度精确为2，复用全部保存结果完成，未重复推理。原脚本和哈希保留。
- 源权重哈希核对不变。训练、自动唤醒保持中止，测试封存。以上训练方案为提案，未执行。
'''
    charts(r,loc,sensitivity)
    (OUT/'report.md').write_text(report,encoding='utf-8');(OUT/'index.html').write_text(render_html(report),encoding='utf-8')
    inputs=[OLD/'features.json',OLD/'forwards.json',ROOT/'experiments/E65_geometry_messages/runs/20261027_geometry_message_seed0/terminal/results.json']
    inputs += [ROOT/p for p in r['saved_E63_controls']['input_sha256']]
    inputs += [ROOT/'protocol/reference_workset_v5.json',ROOT/'references/official/problem_statement.md']
    save(OUT/'input_manifest.json',{'inputs':{p.relative_to(ROOT).as_posix():sha(p) for p in inputs},
          'E42_checkpoint_sha256':r['E42_checkpoint_sha256'],'E65_checkpoint_sha256':r['E65_checkpoint_sha256'],
          'test_read':False,'note':'Original execution sources in config; corrected synthetic example and completion source in snapshot.'})
    sources=[*HERE.glob('*.py'),HERE/'diagnosis_plan.json',HERE/'README.md',OUT/'diagnose_initial.py',
             *ROOT.glob('datasets/*.py'),ROOT/'docs/protocol.md',ROOT/'protocol/reference_workset_v5.json']
    with zipfile.ZipFile(OUT/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(sources):archive.write(p,p.relative_to(ROOT).as_posix())
    save(OUT/'report_provenance.json',{'source_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in sources},
            'snapshot_sha256':sha(OUT/'source_snapshot.zip'),'report_sha256':sha(OUT/'report.md'),
            'new_inference':False,'training_started':False,'test_read':False})
    print('Sparse-anchor report and figures complete; no model inference or training.')


if __name__=='__main__':main()
