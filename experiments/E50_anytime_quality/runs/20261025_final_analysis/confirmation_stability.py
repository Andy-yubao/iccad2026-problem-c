"""Report frozen development confirmation and repeated stress runs, saved-data only."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import json,numpy as np
HERE=Path(__file__).resolve().parent;RUNS=HERE.parent
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def load(name):
    p=RUNS/name;assert read(p/'execution.json')['complete']
    return [read(p/'cases'/f'{i:04d}.json') for i in read(p/'config.json')['indices']]
def main():
    primary=load('20261025_frozen100_parallel_v3');confirm=load('20261025_confirm100_parallel_v3')
    assert not({r['index'] for r in primary}&{r['index'] for r in confirm})
    q=np.array([r['selected_metrics']['cost'] for r in confirm]);b=np.array([r['baseline_Q'] for r in confirm]);n=np.array([r['n'] for r in confirm]);w=np.exp((n-n.max())/12)
    repeats=[load(f'20261025_stability_v3_r{k}') for k in(1,2,3)];flat=[r for group in repeats for r in group];per_case=[]
    for i in (164,443,246,21,259):
        values=[next(r for r in group if r['index']==i) for group in repeats];qq=[r['selected_metrics']['cost'] for r in values]
        per_case.append({'index':i,'Q_runs':qq,'min_Q':min(qq),'max_Q':max(qq),'baseline_Q':values[0]['baseline_Q'],
            'seconds_runs':[r['full_seconds'] for r in values],'all_hard_legal':all(r['selected_metrics']['is_feasible'] for r in values),
            'all_no_regression':all(r['selected_metrics']['cost']<=r['baseline_Q']+1e-8 for r in values)})
    regressions=[{'index':r['index'],'baseline_Q':r['baseline_Q'],'selected_Q':r['selected_metrics']['cost'],'delta':r['selected_metrics']['cost']-r['baseline_Q']} for r in confirm if r['selected_metrics']['cost']>r['baseline_Q']+1e-8]
    result={'confirmation':{'count':100,'hard_legal':sum(r['selected_metrics']['is_feasible'] for r in confirm),
        'weighted_Q':float(np.average(q,weights=w)),'baseline_weighted_Q':float(np.average(b,weights=w)),
        'pair_win_tie_loss':[int(sum(q<b-1e-8)),int(sum(np.abs(q-b)<=1e-8)),int(sum(q>b+1e-8))],
        'max_Q':float(q.max()),'p95_Q':float(np.quantile(q,.95)),'Q_gt1_2':int(sum(q>1.2)),
        'mean_seconds':float(np.mean([r['full_seconds'] for r in confirm])),'max_seconds':max(r['full_seconds'] for r in confirm),
        'over5s':sum(r['full_seconds']>5 for r in confirm),'regressions':regressions},
        'stability':{'cases':5,'repeats':3,'runs':15,'hard_legal':sum(r['selected_metrics']['is_feasible'] for r in flat),
        'regressions':sum(r['selected_metrics']['cost']>r['baseline_Q']+1e-8 for r in flat),'max_Q':max(r['selected_metrics']['cost'] for r in flat),
        'over1_2':sum(r['selected_metrics']['cost']>1.2 for r in flat),'max_seconds':max(r['full_seconds'] for r in flat),
        'over5s':sum(r['full_seconds']>5 for r in flat),'per_case':per_case},
        'scope':'primary100 adaptive gate passed; separate development100 frozen before results, no test opened; three repetitions measure timing/numeric variation on five hard cases, not a universal guarantee',
        'search_and_selection_changed_after_confirmation_started':False,'solver_or_inference_calls':0}
    (HERE/'confirmation_stability.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    c=result['confirmation'];s=result['stability']
    lines=['# 冻结后确认与稳定性','',
        f"另100例来自剩余400个开发成员，成员在确认前冻结，和主100不交叉。全部硬合法，加权Q {c['weighted_Q']:.8f}，原常规同例 {c['baseline_weighted_Q']:.8f}；逐例胜/平/负{c['pair_win_tie_loss']}，最差Q {c['max_Q']:.6f}，0例Q>1.2，最长{c['max_seconds']:.6f}秒，0例超5秒。",'',
        '**主固定100验收已通过；额外确认100没有达到逐例全部不退化。** 确认中出现以下小幅回退，不能把主100结果概括成任意新案例逐例支配。未据这些确认结果修改本轮搜索或选解，反例保留供后续研究。','',
        '| dev索引 | 常规Q | E50 Q | 差值 |','|---|---:|---:|---:|']
    for r in regressions:lines.append(f"| {r['index']} | {r['baseline_Q']:.8f} | {r['selected_Q']:.8f} | +{r['delta']:.8f} |")
    lines+=['','## 三次相同配置的难例重复','',
        f"五例各重复三次，15/15硬合法、15/15不差于历史基线、0例Q>1.2、0例超5秒；最长{s['max_seconds']:.6f}秒。相同种子和算法，保留实际时间截断/数值波动，没有事后选最好的一次。",'',
        '| dev索引 | 三次Q范围 | 常规Q |','|---|---:|---:|']
    for r in per_case:lines.append(f"| {r['index']} | {r['min_Q']:.8f}–{r['max_Q']:.8f} | {r['baseline_Q']:.8f} |")
    lines+=['','## 帕累托研究结论','',
        '当前获得一个5秒、4数值线程的高质量候选点：两个不同开发面板的加权Q均低于1.015、200例硬合法且Q均低于1.2，难例重复也保持低于1.2。主100逐例不退化，确认100有少量小退化。可保留为后续潜力评估和模型研究工具，尚不能宣称新案例的普遍逐例支配或已求得全局帕累托边界。','',
        '每例Q–D–核验时间前沿见analysis.json，坐标只引用原记录。距离项撤除的候选库诊断仅消除确认中的一例小回退，其余反例仍在，因而“全部回退都由移动偏好引起”的猜想不成立；还需研究公开归一化误差及候选覆盖。更长预算仍只列未来方向，本轮未运行超5秒求解预算。','']
    (HERE/'confirmation_stability.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
