"""Audit saved E27/E17 records only; no inference or dataset access."""
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    root = Path(__file__).resolve().parents[2]
    run = root / 'experiments/E27_overlap_weight/runs/20261020_overlap1_v5_seed0'
    base = root / 'experiments/E25_integrated_baseline/runs/20261020_e10_geometry_v5_seed0'
    read = lambda p: json.loads(p.read_text(encoding='utf-8'))
    new, old = [read(p / 'final_evaluation/results.json') for p in (run, base)]
    fn, fo = [read(p / 'final_evaluation/frozen.json') for p in (run, base)]
    nr, br = new['records'], old['records']
    assert new['status'] == old['status'] == 'complete'
    assert len(nr) == len(br) == 500
    assert [r['member'] for r in nr] == [r['member'] for r in br]
    assert len({tuple(r['member']) for r in nr}) == 500
    assert fn['members'] == fo['members']
    assert fn['postprocess'] == fo['postprocess'] and fn['execution'] == fo['execution']
    assert fn['identity']['cache_identity'] == fo['identity']['cache_identity']
    def normalized(sources):
        return {k.split('/backend/', 1)[1]: v for k, v in sources.items() if '/backend/' in k}
    assert normalized(fn['evaluation_sources']) == normalized(fo['evaluation_sources'])
    for k, v in fn['evaluation_sources'].items():
        if k.startswith(('scoring/', 'datasets/', 'third_party/')):
            assert fo['evaluation_sources'][k] == v
    sha = hashlib.sha256((run / 'checkpoints/best.pt').read_bytes()).hexdigest()
    assert sha == new['checkpoint_sha256'] == fn['checkpoint_sha256']
    n = np.array([r['block_count'] for r in nr], dtype=float)
    assert np.array_equal(n, [r['block_count'] for r in br])
    w = np.exp((n - n.max()) / 12)
    a, b = [np.array([r['metrics']['cost'] for r in rs]) for rs in (nr, br)]
    delta = a - b
    assert np.isclose(np.average(a, weights=w), new['summary']['weighted_Q'])
    rng = np.random.default_rng(20261021)
    idx = rng.integers(0, 500, size=(4000, 500))
    ci = np.quantile((delta[idx] * w[idx]).sum(1) / w[idx].sum(1), [.025, .975]).tolist()
    prog_n, prog_o = [read(p / 'progress.json') for p in (run, base)]
    dn, do = [next(h['dev_monitor'] for h in p['history'] if h['epoch'] == p['best_epoch']) for p in (prog_n, prog_o)]
    monitor_keys = ['repair', 'overlap', 'overlap_area', 'crowding', 'center_mse', 'relative_local', 'bbox', 'mib']
    monitor = {k: {'E17': do[k], 'E27': dn[k], 'relative_change': dn[k] / do[k] - 1} for k in monitor_keys}
    strata = []
    for lo, hi in [(0, 50), (51, 100), (101, 100000)]:
        mask = (n >= lo) & (n <= hi)
        if mask.any():
            strata.append({'blocks': f'{lo}-{hi}', 'count': int(mask.sum()), 'weighted_delta': float(np.average(delta[mask], weights=w[mask]))})
    def detail(i):
        x, y = nr[i], br[i]
        keys = ['area_gap', 'hpwl_gap', 'boundary_violations', 'grouping_violations', 'mib_violations', 'violations_relative']
        return {'member': x['member'], 'blocks': x['block_count'], 'E17_Q': float(b[i]), 'E27_Q': float(a[i]), 'delta_Q': float(delta[i]), 'weighted_contribution': float(delta[i] * w[i] / w.sum()), 'E17_fallback': y['fallback'], 'E27_fallback': x['fallback'], 'metric_delta': {k: x['metrics'][k] - y['metrics'][k] for k in keys}}
    audit = {'status': 'complete', 'checkpoint_sha256': sha, 'count': 500, 'same_order_members': True, 'same_backend_hashes': True, 'same_data_cache': True, 'same_scorer_sources': True, 'same_execution_config': True, 'E17': old['summary'], 'E27': new['summary'], 'paired': {'wins': int((delta < -1e-9).sum()), 'ties': int((abs(delta) <= 1e-9).sum()), 'losses': int((delta > 1e-9).sum()), 'weighted_delta': float(np.average(delta, weights=w)), 'bootstrap_95_interval': ci, 'bootstrap_seed': 20261021, 'bootstrap_resamples': 4000, 'strata': strata}, 'dev_monitor500': monitor, 'top_weighted_regressions': [detail(i) for i in np.argsort(delta * w)[-10:][::-1]], 'top_weighted_improvements': [detail(i) for i in np.argsort(delta * w)[:10]], 'worst_E27': detail(int(a.argmax())), 'training_status': prog_n['status'], 'full_dev2000_complete': False, 'test_read': False, 'tail_test_read': False}
    (run / 'completion_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    lines = ['# E27固定500例结果核查（2026-10-21）', '', '结论：加倍重叠惩罚改善了草稿冲突程度，但没有改善最终加权Q；保留E25当前最佳，不据此推广三次方惩罚。', '', '## 同版后处理对照', '', '| 指标 | E17 | E27 |', '|---|---:|---:|']
    for k in ['hard_legal', 'mean_Q', 'weighted_Q', 'p95_Q', 'max_Q', 'fallback_count', 'solve_seconds_mean']:
        lines.append(f'| {k} | {old["summary"][k]:.9g} | {new["summary"][k]:.9g} |')
    lines += ['', f'逐例胜/平/负={audit["paired"]["wins"]}/{audit["paired"]["ties"]}/{audit["paired"]["losses"]}；按exp(n/12)加权Q差(E27−E17)={np.average(delta, weights=w):+.9f}，配对样例bootstrap95%区间[{ci[0]:+.9f}, {ci[1]:+.9f}]。区间仅表示此开发面板的样例重采样波动，不涵盖训练种子或适应性开发偏差。', '', '## 同一dev_monitor500的best草稿诊断', '', '| 指标 | E17 best51 | E27 best69 | 变化 |', '|---|---:|---:|---:|']
    for k, v in monitor.items():
        lines.append(f'| {k} | {v["E17"]:.9g} | {v["E27"]:.9g} | {v["relative_change"]:+.2%} |')
    lines += ['', '重叠穿透平方、交叠面积和拥挤均降低，同时中心、相对位置、bbox和MIB代理变差。这与多目标权衡一致，但不足以确立具体退化机制；单seed、不同best轮次也限制因果解释。训练total因权重变化不能直接比较。', '', '## 加权退化最大的五例', '', '| 样例 | 模块数 | E17 Q | E27 Q | 加权差贡献 |', '|---|---:|---:|---:|---:|']
    for d in audit['top_weighted_regressions'][:5]:
        lines.append(f'| {d["member"]} | {d["blocks"]} | {d["E17_Q"]:.6f} | {d["E27_Q"]:.6f} | {d["weighted_contribution"]:+.6f} |')
    lines += ['', '分模块数统计：`' + json.dumps(strata, ensure_ascii=False) + '`。逐例面积、线长、软约束差及救援状态见completion_audit.json。', '', '## 边界与身份', '', '核对500例成员及顺序、数据缓存、backend哈希、scorer源哈希、CPU 2进程×2库线程、后处理预算与checkpoint哈希一致。E27验证complete、零错误、500/500硬合法；原训练仍为用户停止，未续训，未改写成自动早停。按明确要求直接验证500例，未运行完整dev2000。测试保持封存。', '', '此处使用E17原后处理隔离模型变化，不能与E25 steady快速路径直接解释为模型优劣。历史运行耗时受负载影响，1.915→1.849秒仅为观察值，不构成独立加速证据；两项目标尚未同时达成。固定500例也是训练监测面板，属于开发证据，不是独立泛化测试或正式比赛分数。', '', '后续若继续，应先在固定快速后处理下验证冲突减轻是否能转化为最终收益，并检查少数加权退化案例的面积/线长与结构；目前不启动新实验。', '']
    (run / 'analysis.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'paired': audit['paired'], 'worst': audit['worst_E27'], 'top_regression': audit['top_weighted_regressions'][0]}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
