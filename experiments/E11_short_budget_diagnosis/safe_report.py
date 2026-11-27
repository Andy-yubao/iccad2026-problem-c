"""Final saved-evidence dev100 report including the numerical acceptance check."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def load_rows(root):
    rows = [json.loads(p.read_text(encoding='utf-8')) for p in sorted((root/'cases').glob('*.json'))]
    for r in rows: r['by_arm'] = {t['arm']: t for t in r['trials']}
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('main-run', 'representation-run', 'audit-run', 'run'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args(); root=args.run
    paths=[args.main_run,args.representation_run,args.audit_run,root]
    parent,representation,audit,current=[json.loads((p/'results.json').read_text(encoding='utf-8')) for p in paths]
    assert all(r['status']=='completed' for r in [parent,representation,audit,current])
    assert [parent['trial_count'],representation['trial_count'],current['trial_count']]==[700,200,300]
    rows=load_rows(root); old_main=load_rows(args.main_run); old_rep=load_rows(args.representation_run)
    assert len(rows)==len(old_main)==len(old_rep)==100
    labels={'raw_short_safe':'原草图＋合法候选保留', 'teacher_relations_safe':'教师关系＋合法候选保留',
            'decoder_relations_safe':'教师中心/形状经当前解码＋教师关系＋合法候选保留'}
    old_arms={'raw_short_safe':'raw_short','teacher_relations_safe':'teacher_relations','decoder_relations_safe':'decoder_oracle_relations'}
    stats=current['summary']; baseline=parent['summary']['raw_full']; pairs={}
    table=['| 方法 | 原路径平均Q | 修正后平均Q | 修正后P95 Q | 修正后总秒/例 | 保底数 原→新 | 质量容差且≤0.5秒 |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for arm,old_arm in old_arms.items():
        old_stats=representation['summary'][old_arm] if arm=='decoder_relations_safe' else parent['summary'][old_arm]
        old_rows=old_rep if arm=='decoder_relations_safe' else old_main
        qdelta=np.array([r['by_arm'][arm]['metrics']['cost']-o['by_arm'][old_arm]['metrics']['cost'] for r,o in zip(rows,old_rows)])
        pairs[arm]={'mean_Q_difference':float(qdelta.mean()),'wins':int((qdelta<-1e-9).sum()),
                    'ties':int((np.abs(qdelta)<=1e-9).sum()),'losses':int((qdelta>1e-9).sum())}
        s=stats[arm]
        table.append(f"| {labels[arm]} | {old_stats['Q_mean']:.6f} | {s['Q_mean']:.6f} | {s['Q_p95']:.6f} | {s['charged_total_seconds_mean']:.3f} | {old_stats['initial_row_retained_count']}→{s['initial_row_retained_count']} | {s['quality_and_05_count']}/100 |")
    audit_table=['| worker | 原候选硬合法 | 规范化后硬合法 | 规范化后重叠对数 | 最大坐标改变量 | 原候选官方Q |',
                 '|---|---:|---:|---:|---:|---:|']
    confirmed=0
    for r in audit['cases']:
        before=r['variants']['before_canonical'];after=r['variants']['after_canonical']
        a=before['after_same_tiny_negative_clip_hard_valid'];b=after['after_same_tiny_negative_clip_hard_valid']
        confirmed+=a and not b
        native_q=before.get('scoring',{}).get('metrics',{}).get('cost')
        audit_table.append(f"| {r['panel_index']} | {a} | {b} | {after['after_clip_reasons']['overlap_pairs']} | {r['canonical_max_abs_change']:.3g} | {f'{native_q:.6f}' if native_q is not None else '未评分：硬不可行'} |")
    details=['| 方法 | 最大Q | 最差10% Q | HPWL gap | bbox gap | B/G/M违反均值 | 使用保留分支例数 | 硬合法 |',
             '|---|---:|---:|---:|---:|---|---:|---:|']
    times=['| 方法 | 核心秒均值 | 总秒P95 | 总秒最大 | >0.5秒 | 锥迭代均值 |', '|---|---:|---:|---:|---:|---:|']
    for arm,s in stats.items():
        details.append(f"| {labels[arm]} | {s['Q_max']:.6f} | {s['worst_10_percent_mean_Q']:.6f} | {s['hpwl_gap_mean']:.6f} | {s['area_gap_mean']:.6f} | {s['boundary_mean']:.2f}/{s['grouping_mean']:.2f}/{s['mib_mean']:.2f} | {s['cases_using_native_retention']} | {s['hard_legal_count']}/100 |")
        times.append(f"| {labels[arm]} | {s['core_seconds_mean']:.3f} | {s['charged_total_seconds_p95']:.3f} | {s['charged_total_seconds_max']:.3f} | {s['over_05_seconds_count']}/100 | {s['conic_calls_mean']:.2f} |")
    worst=sorted(rows,key=lambda r:r['by_arm']['decoder_relations_safe']['Q_delta_vs_live_full_raw'],reverse=True)[:10]
    bad_table=['| worker | 完整基线Q | 几何oracle修正后Q | Q差 | 证据 |', '|---|---:|---:|---:|---|']
    for r in worst:
        t=r['by_arm']['decoder_relations_safe']
        bad_table.append(f"| {r['panel_index']} | {r['baseline_q']:.6f} | {t['metrics']['cost']:.6f} | {t['Q_delta_vs_live_full_raw']:+.6f} | [逐例](cases/{r['panel_index']:03d}.json) |")
    paired={'strict_legal_candidate_invalidated_by_canonicalization_count_in_audit':int(confirmed),
            'before_after_pairs':pairs,'teacher_input_geometry_not_learned':True,
            'all_phases_shared_elapsed_seconds':current['shared_elapsed_seconds'],
            'worst_geometry_oracle_workers':[r['panel_index'] for r in worst]}
    (root/'paired_analysis.json').write_text(json.dumps(paired,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(14,5.5),layout='constrained')
    arm_list=list(old_arms);names=['Original draft','Teacher relations','Teacher geometry + relations']
    oldmeans=[(representation['summary'] if a=='decoder_relations_safe' else parent['summary'])[old_arms[a]]['Q_mean'] for a in arm_list]
    newmeans=[stats[a]['Q_mean'] for a in arm_list];x=np.arange(3)
    axes[0].bar(x-.18,oldmeans,.36,color='#a5a5a5',label='Original acceptance')
    axes[0].bar(x+.18,newmeans,.36,color='#2483b7',label='Retain strictly legal native candidate')
    axes[0].axhline(baseline['Q_mean'],color='black',linestyle=':',label='Full-budget baseline')
    axes[0].set_xticks(x,names,rotation=12,fontsize=9);axes[0].set(ylabel='Mean Q, all 100 cases',title='Preserving legal candidates still needs quality checks')
    axes[0].legend(fontsize=8)
    for a,name,color in zip(arm_list,names,['#888888','#d98930','#2483b7']):
        delta=np.sort([r['by_arm'][a]['Q_delta_vs_live_full_raw'] for r in rows])
        axes[1].step(delta,np.arange(1,101)/100,where='post',label=name,color=color)
    axes[1].axvline(.002,color='black',linestyle=':',label='+0.002 tolerance')
    axes[1].set_xscale('symlog',linthresh=.002)
    axes[1].set(xlabel='Q difference from same-case full baseline',ylabel='Fraction of 100 cases',title='All quality tails retained')
    axes[1].legend(fontsize=8);axes[1].grid(alpha=.2)
    fig.savefig(root/'safe_acceptance_quality.png',dpi=170);plt.close(fig)
    geometry=stats['decoder_relations_safe'];relation=stats['teacher_relations_safe'];raw=stats['raw_short_safe']
    goal=(geometry['Q_mean']<=baseline['Q_mean']+.002 and geometry['charged_total_seconds_mean']<=.5)
    all_legal=sum(s['hard_legal_count'] for r in [parent,representation,current] for s in r['summary'].values())
    all_errors=sum(s['error_count'] for r in [parent,representation,current] for s in r['summary'].values())
    text=f'''# E11研究总报告：扩大样本后，先修正蒸馏接口与几何接纳

2026-10-18。本轮同一固定dev_quality100完成700次结构消融、200次当前解码器oracle、300次最小数值修正复核，共1200次正式对照；另对5个真实失败例各重放首步以核查规范化。全部正式返回布局官方及独立硬检查{all_legal}/1200合法，执行错误{all_errors}。各阶段累计计算{current['shared_elapsed_seconds']:.2f}秒，共用预先固定900秒上限，没有启动训练或读取封存测试。

**扩大数据后，原先5例的乐观结论不成立。** 在线完整基线平均Q={baseline['Q_mean']:.6f}、记账时间{baseline['charged_total_seconds_mean']:.3f}秒。原路径只给教师关系时，短预算平均Q={parent['summary']['teacher_relations']['Q_mean']:.6f}；给准教师中心与形状、再经当前真实解码器和教师关系时，也有5例退回Q封顶保底。

进一步核查确认：{confirmed}/5个所查反例中的原生候选已经严格硬合法，却被坐标规范化引入新重叠。这是一个具体接纳缺陷；但硬合法的原生候选仍可能有较多软违反，因此修复它不等于达成最终质量目标。

## 1. 最小修正的全100例结果

{chr(10).join(table)}

修正后，给准教师中心/形状、经当前解码器并配教师关系的平均Q={geometry['Q_mean']:.6f}，平均记账时间{geometry['charged_total_seconds_mean']:.3f}秒；同例Q＋0.002容差与0.5秒同时满足{geometry['quality_and_05_count']}/100例。{'均值条件通过，但这是理想化输入结果，尚无学生模型。' if goal else '仍未达到0.5秒内维持原Q的目标。'} 原草图与关系oracle的修正后均值分别为{raw['Q_mean']:.6f}/{relation['Q_mean']:.6f}。

这里的“平均时间”含在线实测核心/通信和统一沿用的原模型推理时间，oracle构造不计入；未来新网络时延未知。所有组总目标0.40秒，不把实际超时隐藏为硬实时保证。不按Q挑选样例，合法但很差的布局仍完整计入。

## 2. 数值反例与修正边界

{chr(10).join(audit_table)}

以上是按原组全部Q≥5例选出的事后反例，不是随机代表面板。首步重放给原生求解1秒保护以检查数值，不用其耗时宣称加速；规范化前后比较使用同一份候选。4例新重叠来自规范化，worker52在规范化前已存在重叠，不能把它也归为同一根因。所有候选均经原严格hard_valid；硬合法原候选又通过官方及独立检查，Q仍约2.2–2.5，说明硬合法和软接触质量需要分开。

修正只保存在E11独立conic_geometry_safe/robust_safe/worker_safe：仍先使用原规范化结果；该结果不合法时，才检查原候选经原有微小负坐标裁切后是否严格硬合法。容差、重叠阈值、面积约束、公开规格与官方评分均未改。原E10、原E11路径及历史结果保持不变。该候选不自动提升为基线。

## 3. 质量组成、尾部与时间

{chr(10).join(details)}

{chr(10).join(times)}

保留原候选可能改善某例，也可能因软接触未恢复、代理权衡或截止前迭代差异而劣于原路径；各组配对胜平负及平均差均保存在paired_analysis.json，不声称逐例单调改善。软违反统计B/G/M分别为边界/分组/MIB，不能用“全部硬合法”替代这些质量指标。

修正后几何oracle退化最大的10例：

{chr(10).join(bad_table)}

## 4. 对下一阶段的判断

- 教师蒸馏仍有价值，但单独的关系头不是已获充分支持的方案。分离与接触的完整消融和失败率见主报告，不能沿用5例中约0.4%的平均退化推断全100例。
- 必须把教师目标、学生输出形式和轻量修复共同设计。当前解码自由块面积固定为A，而教师成品通常使用允许的面积容差；补充oracle经过了真实decode，仍暴露几何兼容与数值接触问题。这个教师投影不是当前输出族的最优解，不能据此断言网络表达能力不足。
- 优先验证一个能可靠接纳、修复接近合法草图的廉价的几何入口，例如一次有明确约束的投影，或生成本身就符合学生输出形式的教师标签。然后再训练联合中心/形状/结构预测，避免把数值接纳问题当成模型容量问题。
- 本轮不再扩大参数或启动训练；保留E10完整基线，E11安全接纳作为实验候选及最小反例证据。下一轮方案应先证明在理想化输入下能保持质量，再投入训练。

## 5. 证据强度与复现

100例覆盖全部100个worker，每worker1例，仍为已参与开发的面板。后两阶段由前序结果触发，属于自适应开发，不能作为独立未见泛化证明。单seed、分阶段顺序执行、墙钟截止会影响末次迭代；正负结果均保留，不以时间噪声为由择优重跑。全部失败仍计入均值与尾部。

研究计算累计包含主阶段、解码阶段、5例审计和本修正复核；没有拆分规避900秒上限。原冻结split/workset、E10源码与权重未改，数据处理、官方评分及独立检查复用公共模块。所有图和本报告只读取已保存证据。

![数值接纳复核](safe_acceptance_quality.png)

本阶段：[配置](config.json)、[结果](results.json)、[配对统计](paired_analysis.json)、[逐例坐标与轨迹](cases/)、[源码快照](source_snapshot.zip)。此前阶段：[完整结构消融](../{args.main_run.name}/analysis.md)、[当前解码器诊断](../{args.representation_run.name}/analysis.md)、[5例接纳审计](../{args.audit_run.name}/results.json)。
'''
    (root/'analysis.md').write_text(text,encoding='utf-8')
    with zipfile.ZipFile(root/'report_source.zip','w',zipfile.ZIP_DEFLATED) as archive:
        archive.write(Path(__file__),Path(__file__).name)
    (root/'report_provenance.json').write_text(json.dumps({'report_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'input_results':{str(p):hashlib.sha256((p/'results.json').read_bytes()).hexdigest() for p in paths},
        'dataset_reads':False,'solver_calls':0},indent=2)+'\n',encoding='utf-8')
    print('Saved final all-stage analysis.md and figure; no new solver/data reads')


if __name__=='__main__':main()
