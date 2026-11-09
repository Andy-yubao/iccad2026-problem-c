"""Read saved capacity results; paired uncertainty never triggers checkpoint selection."""
import argparse
import json
from pathlib import Path
import numpy as np
from .arms import ARMS
from .io import HERE, save


def analyze(tag, seed):
    output = HERE/'runs'/f'{tag}_pipeline_seed{seed}'
    output.mkdir(parents=True, exist_ok=True)
    results = {}; reference_path = HERE/'runs'/f'{tag}_e25_reference/terminal/results.json'
    reference = json.loads(reference_path.read_text(encoding='utf-8'))['records'] if reference_path.exists() else None
    for arm in ARMS:
        folder = HERE/'runs'/f'{tag}_{arm}_seed{seed}'
        progress_path = folder/'progress.json'
        if not progress_path.exists():
            continue
        progress = json.loads(progress_path.read_text(encoding='utf-8'))
        config = json.loads((folder/'config.json').read_text(encoding='utf-8'))['identity']
        row = {k: progress[k] for k in ('status', 'finished', 'epoch', 'step', 'best_epoch', 'best_proxy', 'elapsed_seconds')}
        row['parameters'] = config['parameters']
        best = next((h for h in progress['history'] if h['epoch'] == progress['best_epoch']), None)
        if best:
            row['best_dev'] = best['dev_monitor']; row['best_train'] = best['train_monitor']
        full = folder/'full_dev.json'
        if full.exists():
            row['full_dev'] = json.loads(full.read_text(encoding='utf-8'))['losses']
        terminal = folder/'terminal/results.json'
        if terminal.exists():
            final = json.loads(terminal.read_text(encoding='utf-8'))
            row['terminal'] = final['summary']
            if reference:
                baseline = {tuple(r['member']): r for r in reference}
                candidate = sorted(final['records'], key=lambda r: r['index'])
                if set(baseline) != {tuple(r['member']) for r in candidate}:
                    raise ValueError('Paired terminal members differ')
                delta = np.array([r['metrics']['cost']-baseline[tuple(r['member'])]['metrics']['cost'] for r in candidate])
                seconds = np.array([r['seconds']-baseline[tuple(r['member'])]['seconds'] for r in candidate])
                n = np.array([r['n'] for r in candidate]); weights = np.exp((n-n.max())/12)
                rng = np.random.default_rng(3800); ids = rng.integers(0, len(delta), size=(2000, len(delta)))
                boot = np.sum(delta[ids]*weights[ids], axis=1)/np.sum(weights[ids], axis=1)
                row['paired_vs_E25'] = {'weighted_Q_delta': float(np.average(delta, weights=weights)),
                    'weighted_Q_bootstrap95': np.percentile(boot, [2.5, 97.5]).tolist(),
                    'mean_seconds_delta': float(seconds.mean()), 'worst_case_Q_delta': float(delta.max())}
        results[arm] = row
    save(output/'comparison.json', {'arms': results, 'test_read': False,
        'scope': 'Adaptive development; rank by fixed raw validation proxy, report diagnostic and deployment tradeoffs separately',
        'fixed_Q_or_runtime_acceptance_threshold': None})
    lines = ['# E40端点与约束交互对照进度', '', '训练目标为尽可能优秀的模型；本轮不设固定Q/耗时通过线。终末评价不用于换checkpoint。', '',
             '| 配置 | 参数 | 状态 | 轮数 | best轮 | 验证代理 |', '|---|---:|---|---:|---:|---:|']
    for arm, row in results.items():
        proxy = row['best_proxy']
        lines.append(f'| {arm} | {row["parameters"]} | {row["status"]} | {row["epoch"]} | {row["best_epoch"]} | {proxy:.6f} |' if proxy is not None
                     else f'| {arm} | {row["parameters"]} | {row["status"]} | {row["epoch"]} | — | — |')
    lines += ['', '精确分项、草稿冲突分布、完整dev核对和配对终末结果见同目录comparison.json。',
              '预算停止不代表收敛；单种子观察用于探索，确认须用匹配种子对照。', '']
    (output/'comparison.md').write_text('\n'.join(lines), encoding='utf-8')
    return results


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tag', default='20261023_tokens'); ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args(); analyze(args.tag, args.seed)
