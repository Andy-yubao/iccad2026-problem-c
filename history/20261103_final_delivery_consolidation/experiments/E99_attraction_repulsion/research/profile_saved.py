"""Read existing500 results/trace; no inference, scoring or repeated solve."""
import collections,hashlib,json
from pathlib import Path
import numpy as np
from datasets.paths import ROOT

HERE=Path(__file__).resolve().parent


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def summary(v):
    x=np.asarray(v,float)
    return dict(mean=float(x.mean()),p95=float(np.percentile(x,95)),max=float(x.max()))


def main():
    study=read(HERE/'study.json');folder=ROOT/study['source_control'];rows=[read(folder/f'{i:04d}.json') for i in range(500)]
    phase=collections.defaultdict(list);counts=collections.Counter();diagnostic=collections.defaultdict(list)
    for r in rows:
        for t in r['trace']:
            name=t['phase'];counts[name]+=1
            if 'seconds' in t:phase[name].append(t['seconds'])
            for key in ('assembly_seconds','setup_seconds','native_seconds','iterations','variables','constraints'):
                if key in t:diagnostic[name+'.'+key].append(t[key])
    weights=np.exp((np.array([r['n'] for r in rows])-120)/12);weights/=weights.sum()
    result=dict(status='complete',source=study['source_control'],count=500,
        source_sha256={f'{i:04d}.json':hashlib.sha256((folder/f'{i:04d}.json').read_bytes()).hexdigest() for i in range(500)},
        weighted_Q=float(sum(w*r['metrics']['cost'] for w,r in zip(weights,rows))),
        complete_seconds=summary([r['complete_seconds'] for r in rows]),draft_seconds=summary([r['draft_seconds'] for r in rows]),repair_seconds=summary([r['repair_seconds'] for r in rows]),
        cases_over_250ms=sum(r['complete_seconds']>=.25 for r in rows),phase_calls=dict(counts),
        phase_call_seconds={k:summary(v) for k,v in phase.items()},phase_diagnostics={k:summary(v) for k,v in diagnostic.items()},
        slowest=[dict(index=r['index'],n=r['n'],complete_seconds=r['complete_seconds'],draft_seconds=r['draft_seconds'],repair_seconds=r['repair_seconds'],Q=r['metrics']['cost'],phases=[x['phase'] for x in r['trace']]) for r in sorted(rows,key=lambda r:-r['complete_seconds'])[:20]],
        weighted_soft_counts={k:float(sum(w*r['metrics'][k] for w,r in zip(weights,rows))) for k in ('boundary_violations','grouping_violations','mib_violations')},
        caveat='Clarabel reported native_seconds can overlap setup accounting; do not add diagnostic timers as disjoint phases. These are old measured rows, not a fresh latency claim.',
        no_new_solve=True,test_read=False)
    path=HERE/'saved_control_profile.json';path.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    lines=['# 冻结C3既有500例开销诊断','',f"Q={result['weighted_Q']:.9f}，完整均时/P95/最大={result['complete_seconds']}，超250ms={result['cases_over_250ms']}/500。",'',
        f"NN草稿阶段={result['draft_seconds']}；后处理阶段={result['repair_seconds']}。",'',
        '逐调用阶段：','', '| 阶段 | 调用数 | 每例调用均数 | 单次均时ms | 单次P95 ms |','|---|---:|---:|---:|---:|']
    for k,n in counts.items():
        if k in phase:
            x=result['phase_call_seconds'][k];lines.append(f"| {k} | {n} | {n/500:.3f} | {x['mean']*1000:.3f} | {x['p95']*1000:.3f} |")
    lines+=['',f"按公开N权重汇总的软违例数量={result['weighted_soft_counts']}。",'',result['caveat'],'',
        '后续优先：固定全局组织的快速并行前置、减少可行性失败/复杂修复重复；新的显式装配/初始化/solve墙钟分开计时。不能用旧阶段计时直接宣称新工程提速。']
    (HERE/'saved_control_profile.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('source_sha256','phase_diagnostics','slowest')},indent=2))


if __name__=='__main__':main()
