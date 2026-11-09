"""Write the full500 saved-data report after analyze_full500; no model execution."""
from .analyze_full500 import OUT,read
import csv
import numpy as np

def main():
    a=read(OUT/'analysis.json')
    with (OUT/'paired_metrics.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    assert len(rows)==500 and len(read(OUT/'manifest.json')['cases'])==35
    shape=float(np.median([float(r['fraction_aspect_changed_over20pct']) for r in rows]))
    selected_good=[r for r in rows if float(r['new_Q'])<1.03]
    farther=sum(float(r['new_D'])>1.2*float(r['min_found_good_D']) for r in selected_good)
    lines=['# E50完整500例结果与草稿修复负担','',
        '本轮仅分析已完成的20261025_full500_parallel_v3：冻结dev_monitor500全部500例，冻结E42 best51，单例5秒、2+2数值线程。未重新推理、求解或读取封存测试。原始逐例记录不改写。', '',
        '## 通俗原理','',
        '它像两组人同时整理同一张草稿：一组沿用常规方法，另一组多尝试几套左右、上下关系，再一起调整模块的位置和形状。某套关系无解时，查看冲突证据，改掉卡住的关系；对少量残余边界/接触冲突再试针对性修复。过程中保存多个候选，逐个独立检查硬约束，截止时从已检查的合法候选中返回一个。', '',
        '选解使用可见的面积、连线、软约束及草稿距离代理，无法在线获得真实Q。真实Q只在事后评分。因此“生成了好方案”与“选中了好方案”是两件事。最终选择器的紧密装填分支有很弱的D罚项，其他分支D权重为0；生成阶段仍有位置和形状先验。它目前是质量优先求解器，不能称为最小改动求解器。', '',
        '## 结果','',
        '| 指标 | 常规后处理 | E50 |','|---|---:|---:|',
        f"| 加权Q | {a['baseline_weighted_Q']:.8f} | {a['weighted_Q']:.8f} |",
        f"| 普通平均Q | {a['baseline_mean_Q']:.8f} | {a['mean_Q']:.8f} |",
        f"| P95 Q | {a['baseline_p95_Q']:.6f} | {a['p95_Q']:.6f} |",
        f"| 最差Q | {a['baseline_max_Q']:.6f} | {a['max_Q']:.6f} |",
        f"| Q<1.03 | {a['baseline_Q_lt1_03']}/500 | {a['Q_lt1_03']}/500 |",'',
        f"E50全500例硬合法；逐例胜/平/负为{a['win_tie_loss']}，容差1e-8。加权超额Q下降{a['excess_Q_reduction_fraction']:.2%}。平均完整求解{a['mean_seconds']:.4f}秒，最长{a['max_seconds']:.4f}秒，0例超5秒。Q>1.2为3例（307、450、467），满足扩展到500例后的最多5例规则。但有12例回退，最大+{a['max_regression']:.8f}，所以严格联合验收仍未通过。",'',
        '加权Q沿用既定exp(N/12)模块数权重；不是普通平均，也不是正式比赛总分。本面板是已经参与开发的冻结开发集，不能当作封存测试。常规时间来自不同历史并发口径，未据此宣称速度提升。5秒包含推理、搜索、检查、调度及最终评分；冷启动、写盘和所有非选中候选的事后Q诊断单列。', '',
        f"本轮共保存{a['candidate_count']}个候选，离线全候选Q诊断另耗时{a['total_offline_diagnostics_seconds']:.1f}秒。12例回退中，10例候选库已有不差于历史基线的点，反映选解误差；185和393的候选库最好值也略差，不能一并归因于选择器。",'',
        '| 回退index | 常规Q | E50 Q | 差值 | 候选库最好Q |','|---:|---:|---:|---:|---:|']
    for r in a['regressions']:lines.append(f"| {r['index']} | {r['old_Q']:.7f} | {r['new_Q']:.7f} | +{r['delta_Q']:.7f} | {r['best_found_Q']:.7f} |")
    lines += ['', '## 能否衡量草稿修复复杂度','',
        '可以得到当前算法和预算下的修复负担画像，但不能得到草稿固有难度或全局最小修复代价。当前控制器主动使用大部分预算，多候选数也受控制器与单次求解速度影响；总耗时和候选数量不宜直接变成草稿分数。', '',
        '建议保留三个独立维度：①找到合法/高质量见证的时间；②满足某个质量阈值时，已找到的最小位移、改形和局部关系变动；③不可行求解、换关系与修复次数，并按模块数及公开约束复杂度对照。轨迹次数反映当前算法遇到的阻力，既可能来自草稿，也可能来自关系构造或数值条件。', '',
        f"所有500例首次合法候选核验时间中位数{a['first_legal_seconds']['median']:.3f}秒。候选库中有403例找到Q<1.03，其中338例在0.5秒内、368例在1秒内、392例在2秒内找到；该403例首次达标时间中位数{a['first_good_seconds']['median']:.3f}秒。剩余97例在本预算内未找到，不能推断不存在。Q是求解结束后补标，在线无法用这个真实Q阈值直接停止；截断轨迹也不等于真实短预算性能。",'',
        f"在403个有达标见证的案例中，已找到最近达标解D中位数{a['min_found_good_D']['median']:.3f}；其中86例D≤0.2、236例D≤0.3。D由0.5×平均中心位移+0.2×P95中心位移+0.2×平均绝对对数长宽比变化+0.1×近邻方向关系变化比例构成。中心位移按各自sqrt(规定面积)归一化；这些权重未经人工反馈校准，不能把D=0.3解释为移动了30%图幅。",'',
        '已找到的最小D只提供“至少存在这么近的好解”的见证，是未知真实最小代价的上界。缺少见证不能反推草稿差。后续用于模型调整时，应优先保存质量—改动—发现时间的整条前沿，再结合人工评价校准，避免过早压成一个强化学习奖励。', '',
        '## 后处理是否改得过大','',
        f"相对同一草稿，E50有{a['closer_than_ordinary']}/500例比常规结果更近；D中位数{a['old_D']['median']:.3f}→{a['new_D']['median']:.3f}，P95 {a['old_D']['p95']:.3f}→{a['new_D']['p95']:.3f}。但相对常规更近，不代表绝对改动已经很小。",'',
        f"每例平均中心移动量的跨案例中位数为{a['mean_movement_module_units']['median']:.3f}个模块尺度；每例P95移动量的中位数为{a['p95_movement_module_units']['median']:.3f}个模块尺度。平均移动按整幅草稿对角线计仅{a['mean_movement_chip_diagonal']['median']:.2%}，但这一全图比例会掩盖小模块的大幅移动。每例可改形模块中长宽比改变超过20%的比例，中位数达{shape:.1%}；目前改形相当普遍。",'',
        f"402个最终Q<1.03案例中，{farther}例的最终D，比候选库里最近的达标点还高20%以上。说明最终选解仍会为质量代理继续牺牲距离；用于评估草稿潜力，应该另报最近达标见证，而不是把最终解的全部变化都记为必要修复。",'',
        '## 35例三相图','',
        '[打开只读图集](index.html)。左为原E42草稿，中为保存的常规结果，右为E50。每张共享模块ID和坐标尺度；可逐张切换、单列放大、缩放和打开原高清图。当前不是盲评轮，也不会覆盖既有人工评分。', '',
        '选择规则先冻结最差10例，再取剩余Q<1.03中改善最大的5例，最后从余下Q<1.03中以种子20261025随机抽20例；三组不重复。“正常组”是明确限定的高质量子集，不代表全500例随机样本。']
    for g in a['groups']:lines.extend(['',f"- {g['label']}："+'、'.join(map(str,g['indices']))])
    lines += ['', '| 常规差、新结果好 | 常规Q | E50 Q | D旧→新 |','|---:|---:|---:|---:|']
    for r in a['improved5']:lines.append(f"| {r['index']} | {r['old_Q']:.5f} | {r['new_Q']:.5f} | {r['old_D']:.3f}→{r['new_D']:.3f} |")
    lines += ['', '| 新结果最差10例 | 常规Q | E50 Q | 已找到最好Q | 面积超参考 | 线长超参考 | 软违约数 |','|---:|---:|---:|---:|---:|---:|---:|']
    for r in a['worst10']:lines.append(f"| {r['index']} | {r['old_Q']:.5f} | {r['new_Q']:.5f} | {r['best_found_Q']:.5f} | {r['new_area_gap']:.1%} | {r['new_wire_gap']:.1%} | {r['new_soft']} |")
    lines += ['', '最差10例中，8例仍有软约束违约；368和493无软违约，主要是面积与线长偏高。这10例的候选库最好Q与最终Q几乎一致，主要不是末端漏选好方案；只能说明当前搜索没给出更好的见证，尚不能证明草稿本身没有潜力。', '',
        '视觉抽查：489保留整体模块次序，做压缩和局部改形；443常规结果出现明显拉伸和大空洞，新结果恢复紧凑，但仍有可见改形；450常规与新结果均出现细长模块和大空洞，平均移动1.35个模块尺度、P95达3.61，显然不能称为微调。这里应同时检查草稿、预放置/分组约束和求解关系选择，不能只归咎于重叠面积。', '',
        '## 后续可用证据','',
        '[paired_metrics.csv](paired_metrics.csv)保留500例配对质量、距离分量、首次达标时间、轨迹计数和选解差；[frontiers.json](frontiers.json)保留全部候选的Q/D/核验时间非支配点及原候选ID，坐标仍保留原运行中；[analysis.json](analysis.json)为机器可读汇总。前沿来自有限预算实际候选，不是最优性证明；一次500例结果也不足以证明运行间稳定性。', '',
        '复现：在仓库根目录执行 `python -m experiments.E50_anytime_quality.analyze_full500 --render`，再执行 `python -m experiments.E50_anytime_quality.report_full500`。只读取已保存结果及对应开发公开规格，无推理或求解。']
    (OUT/'analysis.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4.2),layout='constrained')
    old=np.array([float(r['old_Q']) for r in rows]);new=np.array([float(r['new_Q']) for r in rows])
    axes[0].scatter(old,new,s=13,alpha=.65);axes[0].plot([1,2],[1,2],'--',c='gray');axes[0].set(xlabel='Ordinary Q',ylabel='E50 Q',title='Paired quality: all 500')
    axes[1].scatter([float(r['old_D']) for r in rows],[float(r['new_D']) for r in rows],s=13,alpha=.65);axes[1].plot([0,4.2],[0,4.2],'--',c='gray');axes[1].set(xlabel='Ordinary distance D',ylabel='E50 distance D',title='Repair burden: all 500')
    hit=sorted(float(r['first_good_seconds']) for r in rows if r['first_good_seconds']);axes[2].step(hit,np.arange(1,len(hit)+1)/500,where='post');axes[2].set(xlim=(0,5),ylim=(0,1),xlabel='Candidate independently checked (s)',ylabel='Fraction of all 500',title='Posthoc first Q<1.03 witness')
    for ax in axes:ax.grid(alpha=.2)
    fig.savefig(OUT/'summary.png',dpi=160);plt.close(fig)
    print('Saved full500 report and summary figure; no inference/solver calls')

if __name__=='__main__':main()
