"""Render saved training/evaluation evidence only; never run inference or solvers."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parents[1]
ROOT=HERE.parents[1]
RUN=HERE/'runs/20261103_relation_guided_seed0'
OUT=RUN/'evaluation_dev500'

def read(path):return json.loads(path.read_text(encoding='utf-8'))
def pct(x):return f'{100*x:.2f}%'

def main():
    result=read(OUT/'results.json');s=result['summary']
    history=read(RUN/'progress.json')['history']
    audit=read(RUN/'training_audit.json');training=audit['training'];best=audit['best_dev_monitor']
    previous=ROOT/'experiments/E98_loss_weight_learning/runs/20261102_fixed_rich96_seed0'
    oldhistory=read(previous/'progress.json')['history']
    oldbest=oldhistory[58]['dev_monitor']
    epochs=[h['epoch'] for h in history]
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    ax=axes[0,0]
    ax.plot(epochs,[h['train_monitor']['total'] for h in history],label='E100 train500')
    ax.plot(epochs,[h['dev_monitor']['total'] for h in history],label='E100 dev500')
    ax.set(yscale='log',xlabel='Epoch',ylabel='Fixed total loss',title='Same registered E100 loss')
    ax=axes[0,1]
    ax.plot(epochs,[h['dev_monitor']['legacy_total'] for h in history],label='E100 common 10 terms')
    ax.plot([h['epoch'] for h in oldhistory],[h['dev_monitor']['total'] for h in oldhistory],label='E98 common 10 terms')
    ax.set(yscale='log',xlabel='Epoch',ylabel='Common layout loss',title='Relation supervision excluded for comparison')
    ax=axes[1,0]
    ax.plot(epochs,[100*h['dev_monitor']['relation_reference_accuracy'] for h in history],label='Reference-direction accuracy')
    ax.plot(epochs,[100*h['dev_monitor']['relation_realized_fraction'] for h in history],label='Directions realized in draft')
    ax.set(xlabel='Epoch',ylabel='Percent',title='All-pair relation monitoring')
    ax=axes[1,1]
    ax.plot(epochs,[100*h['dev_monitor']['pair_intersection_fraction'] for h in history],label='E100')
    ax.plot([h['epoch'] for h in oldhistory],[100*h['dev_monitor']['pair_intersection_fraction'] for h in oldhistory],label='E98')
    ax.set(xlabel='Epoch',ylabel='Pair intersection / total module area (%)',title='Draft overlap')
    for ax in axes.flat:
        ax.axvline(training['best_epoch'],color='#555',ls=':',lw=1)
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(OUT/'training_curves.png',dpi=160);plt.close(fig)
    lines=['# E100冻结best61检验报告','',
        '固定v5 dev500；测试封存。模型只读取公开规格，reference只用于监督和诊断。只评价既定best，不按Q重新选择权重，也未新增训练或调参。','',
        f'训练正常早停于第{training["epochs"]}轮、{training["steps"]}步，用时{training["elapsed_seconds"]/60:.2f}分钟；best为第{training["best_epoch"]}轮。错误为空，权重和训练记录一致，保存身份中的源码无漂移。早停表示开发集平台，不声称训练目标已收敛。','',
        '## 直接草稿','',
        '| 指标（同固定dev500） | E98 best58 | E100 best61 |','|---|---:|---:|',
        f'| 共同10项Loss，不含方向监督 | {oldbest["total"]:.6f} | {best["legacy_total"]:.6f} |',
        f'| 成对交面积/总模块面积，逐例均值 | {pct(s["baseline"]["pair_intersection_fraction"]["mean"])} | {pct(s["guided"]["pair_intersection_fraction"]["mean"])} |',
        f'| 精确覆盖并集≤10%的模块，占全部模块 | {pct(s["baseline"]["all_modules_below10_fraction"])} | {pct(s["guided"]["all_modules_below10_fraction"])} |',
        f'| 每个模块都达到覆盖≤10%的实例 | {s["baseline"]["all_modules_pass10"]}/500 | {s["guided"]["all_modules_pass10"]}/500 |',
        f'| 草稿硬合法 | {s["baseline"]["hard_legal"]}/500 | {s["guided"]["hard_legal"]}/500 |',
        f'| 外框面积/总模块面积，均值 | {s["baseline"]["bbox_over_A"]["mean"]:.5f} | {s["guided"]["bbox_over_A"]["mean"]:.5f} |',
        f'| 线长相对reference分母的gap，均值 | {pct(s["baseline"]["hpwl_gap"]["mean"])} | {pct(s["guided"]["hpwl_gap"]["mean"])} |','',
        '覆盖并集不会重复计算同一覆盖区域；训练中的pair-sum负荷会重复累加，可能超过100%，两者不可混用。草稿不合法时，不把其官方惩罚分当作可比的高质量Q。','',
        '## 关系是否转化为几何','',
        f'训练监测的全体关系准确率为{pct(best["relation_reference_accuracy"])}，输出满足预测方向的比例为{pct(best["relation_realized_fraction"])}。下表按模块对计数汇总，因此与训练时逐例平均会有差异。','',
        '| 模块对范围 | 模块对数 | 网络方向正确率 | 仅初始坐标的最小分离方向正确率 |','|---|---:|---:|---:|']
    labels={'all':'全部可监督模块对','near_reference':'reference近邻','initial_overlap':'初始草稿发生重叠','connected':'有直接电连接'}
    for key,v in s['relation'].items():
        lines.append(f'| {labels[key]} | {v["count"]} | {pct(v["accuracy"])} | {pct(v["coordinate_heuristic_accuracy"])} |')
    lines += ['',
        'reference近邻定义：两矩形的边间欧氏距离≤该模块对平均面积平方根的0.25倍。坐标启发式只从初始坐标选择当前所需移动最小的分离方向，不读取reference；reference只用于判断正确与否。','',
        f'同一冻结网络中，4层更新前后的成对交面积均值为{pct(s["initial"]["pair_intersection_fraction"]["mean"])} → {pct(s["guided"]["pair_intersection_fraction"]["mean"])}；{s["comparison"]["initial_to_guided_overlap_improved"]}例降低、{s["comparison"]["initial_to_guided_overlap_worsened"]}例增加。',
        f'精确覆盖≤10%的模块占比为{pct(s["initial"]["all_modules_below10_fraction"])} → {pct(s["guided"]["all_modules_below10_fraction"])}。',
        '这是同一模型内部的前后诊断；几何层同时包含方向分离、连接吸引和边界引导，不能把前后差异全部归因于学到的关系，也不能代替独立训练的消融对照。','',
        '## 相同完整后处理','',
        f'- E100：{s["final"]["hard_legal"]}/500硬合法，{s["final"]["fallbacks"]}次回退，加权中性时间Q={s["final"]["weighted_Q"]:.9f}。',
        f'- E98已保存的同成员、同完整配置结果：Q={s["comparison"]["baseline_weighted_Q"]:.9f}；本次只读复用。逐例Q优/劣为{s["comparison"]["final_Q_wins"]}/{s["comparison"]["final_Q_losses"]}。',
        f'- E100完整逐例耗时：均值{s["final"]["complete_seconds"]["mean"]*1000:.2f}ms，P95 {s["final"]["complete_seconds"]["p95"]*1000:.2f}ms，最大{s["final"]["complete_seconds"]["maximum"]*1000:.2f}ms；{s["final"]["at_least250ms"]}例≥250ms。',
        f'- 网络请求耗时（公开输入构造、传输、网络、输出还原）：均值{s["final"]["draft_seconds"]["mean"]*1000:.2f}ms，P95 {s["final"]["draft_seconds"]["p95"]*1000:.2f}ms。',
        '完整计时包括公开输入构造、传输、网络、全部后处理和必要硬检查；模型加载/预热单列。CPU单线程、GPU逐例；诊断、reference读取和绘图不计入求解。历史基线耗时不是本次配对重测，不据此声称严格加速。','',
        '## 最坏模块','']
    for key,label in (('initial','E100更新前'),('guided','E100草稿'),('baseline','E98草稿')):
        w=s[key]['worst'];lines.append(f'- {label}：dev index {w["index"]}，成员{w["member"]}，模块ID {w["module"]}，精确覆盖{pct(w["coverage"])}。')
    draft_better=s['guided']['pair_intersection_fraction']['mean']<s['baseline']['pair_intersection_fraction']['mean']
    final_better=s['final']['weighted_Q']<s['comparison']['baseline_weighted_Q']
    accepted=s['final']['hard_legal']==500 and s['final']['weighted_Q']<1.01
    lines += ['', '## 判断','',
        f'相对E98，本轮草稿成对重叠{"改善" if draft_better else "未改善"}；相同后处理后的加权Q{"改善" if final_better else "未改善"}。最终质量目标{"达到" if accepted else "未达到"}。',
        '这是一次seed0、反复用于监测和选择的开发集结果。E100与E98训练步数不同，尚无匹配预算的direct/auxiliary独立训练对照；不能推广为方向关系建模整体无效或有效。局部方向正确、关系被满足与高质量整体布局应继续分开判断。','',
        '[训练曲线](training_curves.png) · [固定100例同图比较](gallery/index.html) · [完整统计](results.json)','']
    (OUT/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'report':str(OUT/'report.md'),'draft_improved':draft_better,'final_Q_improved':final_better,'quality_target_met':accepted}))

if __name__=='__main__':main()
