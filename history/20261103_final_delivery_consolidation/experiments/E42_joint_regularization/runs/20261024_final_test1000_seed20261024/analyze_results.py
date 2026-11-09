"""Audit saved results and compare the two existing runs; no inference or scoring."""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def main():
    c = read(HERE/'config.json')
    base = (ROOT/c['comparison_config']).parent
    b = read(base/'config.json')
    current, prior = read(HERE/'summary.json'), read(base/'summary.json')
    assert c['records'] == b['records'] and c['steady'] == b['steady']
    assert c['time_scope'] == b['time_scope'] and c['registry_sha256'] == b['registry_sha256']
    rows = [read(HERE/f'cases/{i:04d}.json') for i in range(1000)]
    old = [read(base/f'cases/{i:04d}.json') for i in range(1000)]
    assert len(list((HERE/'cases').glob('*.json'))) == 1000
    assert all(r['member'] == o['member'] == m['member'] and r['n'] == o['n'] == m['n']
               and r['index'] == o['index'] == i for i,(r,o,m) in enumerate(zip(rows,old,c['records'])))
    assert len({tuple(r['member']) for r in rows}) == 1000
    assert not {tuple(r['member']) for r in rows} & {tuple(m) for m in c['excluded_members']}
    assert all(r['torch_threads'] == 1 and all(set(v)<={1} for v in r['solver_threads'].values()) for r in rows)
    assert all(len(r['positions']) == r['n'] and np.isfinite(r['positions']).all() for r in rows)
    assert all(r['metrics']['is_feasible'] == r['independent_hard_checks']['is_feasible'] for r in rows)
    assert len({r['worker_pid'] for r in rows}) == 4
    assert all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest() == h for n,h in c['sources'].items())
    assert hashlib.sha256((ROOT/c['checkpoint']).read_bytes()).hexdigest() == c['checkpoint_sha256']
    assert hashlib.sha256((ROOT/'protocol/split_v1.json').read_bytes()).hexdigest() == c['split_sha256']
    frozen = read(HERE.parent/'20261023_rectangle_w96d12_ab_seed0/frozen.json')
    assert all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest() == h for n,h in frozen['sources'].items())
    q = np.array([r['metrics']['cost'] for r in rows]); q0 = np.array([r['metrics']['cost'] for r in old])
    t = np.array([r['seconds'] for r in rows]); t0 = np.array([r['seconds'] for r in old])
    n = np.array([r['n'] for r in rows]); w = np.exp((n-n.max())/12)
    dq,dt = q-q0,t-t0
    assert abs(float(np.average(q,weights=w))-current['weighted_Q']) < 1e-12
    assert abs(float(t.mean())-current['seconds_mean']) < 1e-12
    workers = sorted({r['member'][0].split('/')[0] for r in rows})
    aggregates = []
    for worker in workers:
        mask = np.array([r['member'][0].split('/')[0] == worker for r in rows])
        aggregates.append([sum(w[mask]*dq[mask]),sum(w[mask]),sum(dt[mask]),sum(mask)])
    values = np.array(aggregates)
    counts = np.random.default_rng(20261024).multinomial(len(workers),[1/len(workers)]*len(workers),size=2000)
    boot = counts @ values
    comparison = {
        'count':1000,'exact_same_ordered_panel':True,'workers':4,'threads_per_worker':1,
        'E25':prior,'E42_regular':current,
        'weighted_Q_delta_E42_minus_E25':float(np.average(dq,weights=w)),
        'seconds_mean_delta_E42_minus_E25':float(dt.mean()),
        'Q_excess_over1_reduction_fraction':float((prior['weighted_Q']-current['weighted_Q'])/(prior['weighted_Q']-1)),
        'mean_time_reduction_fraction':float(1-t.mean()/t0.mean()),
        'Q_wins_ties_losses_E42': [int((dq < -1e-9).sum()),int((np.abs(dq)<=1e-9).sum()),int((dq>1e-9).sum())],
        'weighted_Q_delta_worker_cluster_bootstrap95':np.percentile(boot[:,0]/boot[:,1],[2.5,97.5]).tolist(),
        'seconds_delta_worker_cluster_bootstrap95':np.percentile(boot[:,2]/boot[:,3],[2.5,97.5]).tolist(),
        'bootstrap_replicates':2000,'bootstrap_seed':20261024,
        'E42_dominates_E25_aggregate_point_estimates':bool(current['weighted_Q']<prior['weighted_Q'] and t.mean()<t0.mean()),
        'E42_meets_Q_lt1_04_and_mean_time_lt0_25':bool(current['weighted_Q']<1.04 and t.mean()<.25),
        'inference_or_rescoring_repeated':False,
        'exposure':'User requested the frozen E42 comparison after seeing E25 test results; this is an already exposed shared panel, not a new blind test.',
        'time_uncertainty':'Separate single CPU runs; cluster intervals cover case sampling, not repeated-run frequency/load/solver-time noise.'}
    save(HERE/'comparison.json',comparison)
    save(HERE/'audit.json',{'status':'passed','count':1000,'all_frozen_members_and_source_hashes_match':True,
         'original_E42_source_files_unchanged':len(frozen['sources']),
         'all_official_and_independent_hardchecks_agree':True,'actual_workers':4,'all_threads_one':True,
         'prior100_excluded':True,'coordinates_finite_and_count_matched':True,'inference_or_rescoring_repeated':False})
    with (HERE/'paired_metrics.csv').open('w',newline='',encoding='utf-8-sig') as handle:
        writer=csv.writer(handle);writer.writerow(['index','shard','row','n','E25_Q','E42_Q','delta_Q','E25_seconds','E42_seconds','delta_seconds'])
        for i,r in enumerate(rows):writer.writerow([i,*r['member'],r['n'],q0[i],q[i],dq[i],t0[i],t[i],dt[i]])
    section=f'''\n## 与E25同1000例的逐例配对比较

| 指标 | E25 | E42＋常规后处理 |
|---|---:|---:|
| 硬合法 | {prior['hard_legal']}/1000 | {current['hard_legal']}/1000 |
| 加权Q | {prior['weighted_Q']:.9f} | {current['weighted_Q']:.9f} |
| 算术Q | {prior['mean_Q']:.9f} | {current['mean_Q']:.9f} |
| P95 Q | {prior['p95_Q']:.6f} | {current['p95_Q']:.6f} |
| 最差5%平均Q | {prior['CVaR95_Q']:.6f} | {current['CVaR95_Q']:.6f} |
| 最大Q | {prior['max_Q']:.6f} | {current['max_Q']:.6f} |
| 平均完整秒/例 | {prior['seconds_mean']:.6f} | {current['seconds_mean']:.6f} |
| P95 / 最大完整秒 | {prior['seconds_p95']:.6f} / {prior['seconds_max']:.6f} | {current['seconds_p95']:.6f} / {current['seconds_max']:.6f} |

E42加权Q差为{comparison['weighted_Q_delta_E42_minus_E25']:+.9f}；超额Q(Q−1)下降{comparison['Q_excess_over1_reduction_fraction']:.2%}。平均耗时变化{comparison['seconds_mean_delta_E42_minus_E25']:+.6f}秒，下降{comparison['mean_time_reduction_fraction']:.2%}。逐例Q胜/平/负：{comparison['Q_wins_ties_losses_E42']}。

按来源worker聚类的2000次配对bootstrap95%区间：加权Q差{comparison['weighted_Q_delta_worker_cluster_bootstrap95']}；平均时间差{comparison['seconds_delta_worker_cluster_bootstrap95']}秒。区间反映病例采样，不包含重复运行的CPU频率/负载及限时算法波动。本次两个指标的点估计均优于E25：{comparison['E42_dominates_E25_aggregate_point_estimates']}。原Q<1.04且平均时间<0.25秒联合目标通过：{comparison['E42_meets_Q_lt1_04_and_mean_time_lt0_25']}。

E42权重和常规算法均在本轮测试前冻结；这是看到E25测试结果后按用户要求追加的既有候选比较，不能宣称新的独立盲测。未改模型/算法、未重抽成员、未重跑E25、未重复推理或重新评分。完整坐标仍各自在原运行保存一份，比较CSV仅保存逐例指标。

证据：[comparison.json](comparison.json)、[paired_metrics.csv](paired_metrics.csv)、[audit.json](audit.json)。离线重建仅运行本目录analyze_results.py，不调用模型、求解器或评分器。
'''
    report=HERE/'analysis.md'; original=report.read_text(encoding='utf-8').split('\n## 与E25同1000例')[0]
    report.write_text(original+section,encoding='utf-8')
    print(json.dumps({k:v for k,v in comparison.items() if k not in ('E25','E42_regular')},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
