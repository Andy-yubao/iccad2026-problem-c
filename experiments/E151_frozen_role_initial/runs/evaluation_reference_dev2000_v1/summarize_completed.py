"""Read completed saved results only; no inference, repair, or checkpoint selection."""
import json
from pathlib import Path
import numpy as np
from scoring.official import official_evaluator

HERE=Path(__file__).resolve().parent


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def main():
    out=HERE/'output';result=read(out/'results.json')
    if result['status']!='complete' or result['count']!=2000:
        raise RuntimeError('A completed original dev2000 evaluation is required')
    records=[read(out/f'cases/{i:04d}.json') for i in range(2000)]
    monitor=set(read(HERE/'plan.json')['monitor500_indices'])
    subsets={}
    for label,rows in (('monitor500',[r for r in records if r['index'] in monitor]),
                       ('extension1500',[r for r in records if r['index'] not in monitor]),
                       ('full2000',records)):
        costs=[r['metrics']['cost'] for r in rows]
        weighted=official_evaluator().compute_total_score(costs,[r['n'] for r in rows])
        times=[r['complete_seconds'] for r in rows]
        subsets[label]=dict(count=len(rows),weighted_Q=weighted,mean_Q=float(np.mean(costs)),
            mixed_Q=.5*(weighted+float(np.mean(costs))),
            hard_legal=sum(r['independent_hard_checks']['is_feasible'] for r in rows),
            grouping_satisfied=sum(r['metrics']['grouping_violations']==0 for r in rows),
            boundary_satisfied=sum(r['metrics']['boundary_violations']==0 for r in rows),
            all_soft_satisfied=sum(r['metrics']['total_soft_violations']==0 for r in rows),
            grouping_violations=sum(r['metrics']['grouping_violations'] for r in rows),
            boundary_violations=sum(r['metrics']['boundary_violations'] for r in rows),
            mean_complete_seconds=float(np.mean(times)),p95_complete_seconds=float(np.quantile(times,.95)),
            over250ms=sum(t>.25 for t in times))
    (out/'subsets.json').write_text(json.dumps(subsets,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    report=['# E151 best45：原reference、无MIB dev2000','',
        '冻结权重、原始面积/reference与原后处理预算；不读test、不重选模型。',
        '执行被宿主中断于504例后从连续保存前缀恢复，全部2000例结果汇总如下。','',
        '| 面板 | 例数 | 混合Q | 硬合法 | grouping满足 | 贴边满足 | 全软约束满足 | 均时/P95 ms |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for label,v in subsets.items():
        report.append(f"| {label} | {v['count']} | {v['mixed_Q']:.9f} | {v['hard_legal']} | {v['grouping_satisfied']} | {v['boundary_satisfied']} | {v['all_soft_satisfied']} | {1000*v['mean_complete_seconds']:.1f}/{1000*v['p95_complete_seconds']:.1f} |")
    report+=['','Q使用中性时间因子，耗时单独报告；旧监测500及扩展1500不宣称为全新未见测试。',
        '原冻结身份和完整汇总分别保留在frozen.json、results.json；本报告只读取保存结果，没有新推理或后处理。','']
    (out/'terminal_report.md').write_text('\n'.join(report),encoding='utf-8')
    print(json.dumps(subsets,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
