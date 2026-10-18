"""Render saved dev100 diagnosis; no dataset reads, inference, or solving."""
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
BASE=HERE/'runs/20261018_stageA_dev100_seed0'
REPLAY=HERE/'runs/20261018_dev100_guard_replay_seed0'
BEAM=HERE/'runs/20261018_beam_diagnostic5_seed0'


def main():
    result=json.loads((BASE/'results.json').read_text(encoding='utf-8'))
    replay=json.loads((REPLAY/'results.json').read_text(encoding='utf-8'))
    changed=[json.loads(s) for s in (REPLAY/'changed_cases.jsonl').read_text(encoding='utf-8').splitlines()]
    rows=[json.loads(s) for s in (BASE/'cases.jsonl').read_text(encoding='utf-8').splitlines()]
    beam=json.loads((BEAM/'results.json').read_text(encoding='utf-8'))
    baseline=result['variants']['control120'];before=replay['before'];after=replay['after_cached_guard_replay']
    overrides={r['case_id']:r['after']['metrics'] for r in changed}
    old=np.array([r['variants']['control120']['final']['metrics']['cost'] for r in rows])
    new=np.array([overrides.get(r['case_id'],r['variants']['control120']['final']['metrics'])['cost'] for r in rows])
    times=np.array([r['variants']['control120']['solve_seconds'] for r in rows])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans']
    figure,axes=plt.subplots(1,3,figsize=(15,4.2))
    axes[0].plot(range(1,101),np.sort(old),label='原输出',marker='.',markersize=3)
    axes[0].plot(range(1,101),np.sort(new),label='末端修复重放',marker='.',markersize=3)
    axes[0].axhline(1.5,color='gray',ls='--',lw=.8);axes[0].set_title('全部100例：Q从低到高排序')
    axes[0].set_xlabel('排序位置');axes[0].set_ylabel('Q（越低越好）');axes[0].legend(fontsize=8)
    axes[1].hist(times,bins=[0,1,2,5,10,20,30,40,50,65],edgecolor='white')
    axes[1].axvline(baseline['postprocess_seconds_mean'],color='tab:red',label='均值10.47秒')
    axes[1].set_title('原后处理：实际单例耗时');axes[1].set_xlabel('秒');axes[1].set_ylabel('例数');axes[1].legend(fontsize=8)
    parts=replay['baseline_analysis']['seconds_by_component']
    total=baseline['postprocess_seconds_total']
    vals=[parts['shape_solver_only'],parts['fixed_solver_only'],
          (parts['shape']-parts['shape_solver_only'])+(parts['fixed_lp']-parts['fixed_solver_only'])]
    vals.append(total-sum(vals))
    labels=['联合形状求解','固定尺寸求解','问题构建等','构造/清理/其余']
    axes[2].pie(vals,labels=labels,autopct='%1.1f%%',textprops={'fontsize':8})
    axes[2].set_title('78%的时间在优化器内部')
    figure.tight_layout();figure.savefig(REPLAY/'diagnosis.png',dpi=160);plt.close(figure)
    text=['# 完整dev100结论：先修复尾部回退，再重设加速研究重点','',
          '2026-10-18。E7 attention_geometry冻结，固定dev_quality100、100个worker；结果100/100完整，无执行错误、无测试数据读取。Q为官方中性时间质量分，越低越好。','',
          '## 质量与合法性','',
          '| 指标 | 原完整运行 | 修复末端后的缓存重放 |','|---|---:|---:|',
          f'| Q算术均值 | {before["Q_mean"]:.6f} | {after["Q_mean"]:.6f} |',
          f'| 官方块数加权Q | {before["Q_weighted"]:.6f} | {after["Q_weighted"]:.6f} |',
          f'| Q的P95 | {before["Q_p95"]:.6f} | {after["Q_p95"]:.6f} |',
          f'| 最大Q | {before["Q_max"]:.6f} | {after["Q_max"]:.6f} |',
          f'| 最差10例平均Q | {before["worst10_mean"]:.6f} | {after["worst10_mean"]:.6f} |',
          f'| Q>1.5 | {before["Q_above1_5"]} | {after["Q_above1_5"]} |',
          '| 硬合法 | 100/100 | 100/100 |','',
          '**右列是缓存末端重放，不是重新运行100次求解。** 只有2个被守卫拒绝的候选发生变化并重新执行官方/独立硬检查；其余98例坐标及评分直接引用原结果。另检查3个未变化的软约束尾部案例，归零函数保持其坐标和Q逐值不变。没有新增模型前向或LP求解，修复后的在线耗时未重新测量。','',
          '## 两个极端差解的原因','',
          'worker45与worker65的候选中分别有1个、2个坐标为−1.45519×10⁻¹¹。内部容差接受它们，最终严格非负守卫拒绝它们，于是替换成质量差的保底布局。候选没有其他面积、固定规格或重叠问题。','',
          '| 案例 | 原回退Q | 微小负坐标归零并复核后的Q |','|---|---:|---:|']
    for row in changed:text.append(f'| {row["case_id"]} | {row["before_metrics"]["cost"]:.6f} | {row["after"]["metrics"]["cost"]:.6f} |')
    text+=['',
        '修复仅限可移动位置中[-1e-9,0)的数值误差。归零后仍须通过原严格守卫，预放置规格不动，面积与重叠容差不改；修复失败仍使用原有回退。实现位于实验内guard.py，在线配置名为safe120；原control120及frozen源码保持不变。','',
        '## 速度瓶颈','',
        f'- 整次墙钟{result["wall_seconds_this_invocation"]:.1f}秒，约20.2分钟；核心后处理均值{baseline["postprocess_seconds_mean"]:.3f}秒，P95为{baseline["postprocess_seconds_p95"]:.3f}秒，最长{baseline["postprocess_seconds_max"]:.3f}秒。',
        '- 固定尺寸求解246.63秒，联合形状求解567.67秒，共占核心后处理77.8%。其余包含建模、构造、清理与观测开销。',
        '- 即使完全消除求解器外开销，理想加速仅约1.29倍。事后已知“最后一次有效代理改善”而立即停下的理想化诊断也仅约1.33倍，不能支撑8–10倍目标。该停止点需要未来信息，不能当作可部署策略。',
        '- 8–10倍对应平均约1.05–1.31秒/例。旧长预算轨迹在1.3秒处的严格合法候选平均Q约2.481；这只是离线轨迹前缀诊断，不是短预算算法实测。',
        '- 93/100例最初草稿固定尺寸的子问题出现不可行。固定方向后反复调整形状、再投影的位置/形状交替过程是应重点替换的计算结构。','',
        '![完整dev100质量与耗时诊断](diagnosis.png)','',
        '## 后续短探针及决策','',
        f'在同一批已访问的5个诊断样本（worker37/45/59/65/73）上，测试保留多个方向修改分支的构造机制。1.3秒配置平均Q为{beam["variants"]["beam_short"]["Q_mean"]:.6f}，最大Q为{beam["variants"]["beam_short"]["Q_max"]:.6f}；3秒配置没有改善质量。该机制仍把困难样本退化成差保底，停止扩测。诊断样本为事后选取，不能称均衡验证或泛化结论。',
        '',
        '下一轮优先检验**位置、形状和分离方向共同调整的表示**，例如以中心坐标与宽高比直接进行联合几何优化、让面积由表示保证，再执行少量严格合法化。要检验能否替代昂贵的多轮形状LP与面积补切，不继续靠Python优化、缩短旧预算或增加固定形状方向枚举来凑8倍。先固定5例做机制反例；有质量信号才进入dev100。模型继续冻结。','',
        '## 质量目标的可达边界','',
        '软违反不全是可修复错误。worker37的同形组面积为48、72、90、160、440、528，各自合法面积范围互不相交；若按真实宽高一致判断同形，就至少有5个MIB违反。官方软约束分母为26，对应工程质量参照exp(2×5/26)=1.469049，正好等于该例现有Q。官方实际按宽高四位小数计同形，本轮没有对任意退化长宽比与舍入组合做全局最优证明，因此不把这一面积兼容性诊断称为官方Q的无条件数学下界。后续训练目标需要同时核验真实尺寸、评分精度与可达性，不默认每例都能接近Q=1。',
        '',
        '这不表示所有MIB违反都无法改善。例如worker73含多个面积35的组成员，仍有合并尺寸类别的潜在空间。后续区分可修复软违反和规格导致的下界，再决定训练目标。上述下界只检查了5个诊断样本，未宣称覆盖全部100例。','',
        '## 证据与边界','',
        '- [原始完整dev100结果](../20261018_stageA_dev100_seed0/results.json)、[逐例记录](../20261018_stageA_dev100_seed0/cases.jsonl)',
        '- [末端重放配置](config.json)、[重放结果](results.json)、[发生变化的两例](changed_cases.jsonl)',
        '- [方向分支探针](../20261018_beam_diagnostic5_seed0/analysis.md)',
        '',
        '原完整运行的config、results、坐标、日志与源码快照保持原样。新修复单独留证；模型、训练数据、冻结划分和评分器均未改。当前质量尾部已有具体修复，8–10倍加速仍未实现。','']
    (REPLAY/'analysis.md').write_text('\n'.join(text),encoding='utf-8')
    print(str(REPLAY/'analysis.md'))


if __name__=='__main__':main()
