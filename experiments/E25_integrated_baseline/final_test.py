"""One frozen E25 evaluation on 1,000 previously sealed v5 local-test members."""
import os
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_name] = '1'
os.environ['OMP_MAX_ACTIVE_LEVELS'] = '1'

import argparse
from collections import defaultdict
import ctypes
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import importlib.metadata
import json
import math
import multiprocessing as mp
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import torch
from datasets.paths import ROOT, dataset_paths
from datasets.reference_layouts import REGISTRY, registry
from datasets.public_member_inputs import public_member_inputs
from datasets.worksets import FrozenMemberReader
from datasets.cases import _build_case
from scoring.evaluation import evaluate_case
from scoring.official import official_evaluator
from .cases import WEIGHTS, predict, seed_all
from .io import HERE, save, sha, sources, snapshot, exclusive
from .model import GeometryModel

OUT = HERE / 'runs/20261024_final_test1000_seed20261024'
GALLERY = HERE / 'runs/20261022_test100_draft_gallery/config.json'
EXPECTED_WEIGHT = '3de22809e831edf11b56a85647a0bd9fe625e8b7617abb6eb1f4b27969cfe00a'
SEED = 20261024
STEADY = dict(relaxed=False, mode=0, adaptive=True, fast_budget=False,
              late_extension=True, plateau=True, robust_seeds=True, certified=True,
              quick_project=True, quadratic_project=True, incumbent_project=True, snap=1e-4)


def select_panel():
    """Hamilton quotas by n, then fixed SHA ranks; metadata only, no test tensors."""
    reg = registry()
    exposed = json.loads(GALLERY.read_text(encoding='utf-8'))
    assert exposed['registry_sha256'] == sha(REGISTRY)
    excluded = {tuple(r['member']) for r in exposed['members']}
    assert len(excluded) == 100
    groups = defaultdict(list)
    for record in reg['sets']['local_test']['records']:
        if tuple(record['member']) not in excluded:
            groups[record['n']].append(record)
    assert sum(map(len, groups.values())) == 1900
    quota = {n: len(rows)*1000//1900 for n, rows in groups.items()}
    ranking = sorted(groups, key=lambda n: (-(len(groups[n])*1000 % 1900), n))
    for n in ranking[:1000-sum(quota.values())]:
        quota[n] += 1
    def rank(record):
        return hashlib.sha256((f'E25.final_test1000:{SEED}:' +
                               json.dumps(record['member'])).encode()).hexdigest()
    chosen = [r for n in sorted(groups) for r in sorted(groups[n], key=rank)[:quota[n]]]
    assert len(chosen) == len({tuple(r['member']) for r in chosen}) == 1000
    assert not {tuple(r['member']) for r in chosen} & excluded
    return chosen, exposed, {str(n): quota[n] for n in sorted(quota)}


def prepare():
    records, exposed, quotas = select_panel()
    assert sha(WEIGHTS) == EXPECTED_WEIGHT
    identity = {
        'revision': 'E25.final_test1000.v1',
        'authorization': '2026-10-24 user requested current strongest frozen version, 1000 test cases, CPU 4 processes',
        'selection': 'Hamilton proportional allocation by actual n from remaining1900; SHA256 rank seed20261024; exclude all100 previously inspected draft-gallery members',
        'seed': SEED, 'count': 1000, 'records': records, 'quotas_by_n': quotas,
        'excluded_members': [r['member'] for r in exposed['members']],
        'prior_exposure_config_sha256': sha(GALLERY),
        'checkpoint': WEIGHTS.relative_to(ROOT).as_posix(), 'checkpoint_sha256': sha(WEIGHTS),
        'registry_sha256': sha(REGISTRY), 'split_sha256': sha(ROOT/'protocol/split_v1.json'),
        'sources': sources(), 'solver': 'E25 E17 best51 + original steady; no algorithm/weight change',
        'steady': STEADY, 'workers': 4, 'threads_per_worker': 1,
        'resource_override': 'process-local adapters cap Clarabel/HiGHS/SciPy at1; BLAS/OpenMP/Torch/Numba at1',
        'data_paths': {k: str(v) for k, v in dataset_paths().items()},
        'task': 'v5 MIB mean-area derived task with original-reference denominators',
        'scoring': 'unmodified official evaluator, runtime=median_runtime=1; independent hard checks; errors Q10 included',
        'time_scope': 'public graph features + CPU model + decode + full steady optimization/rescue; exclude file/model loading, fixed task transformation, JIT initialization, scoring and evidence IO',
        'budget_seconds': 1100, 'test_feedback_policy': 'single frozen candidate; no tuning, checkpoint selection or reranking from this test',
    }
    config = OUT/'config.json'
    if config.exists():
        if json.loads(config.read_text(encoding='utf-8')) != identity:
            if (OUT/'progress.json').exists() or list((OUT/'cases').glob('*.json')):
                raise ValueError('Frozen run identity changed; do not rerun or silently migrate')
            save(OUT/'preparation_migration.json', {
                'reason': 'Dependency preflight found no threadpoolctl; use actual bundled OpenBLAS APIs instead. No test tensors or model inference had run.',
                'previous_identity': json.loads(config.read_text(encoding='utf-8')),
                'test_tensors_read': False})
            save(config, identity)
            snapshot(OUT/'source_snapshot.zip', identity)
    else:
        save(config, identity)
        snapshot(OUT/'source_snapshot.zip', identity)
    if not (OUT/'environment.json').exists():
        save(OUT/'environment.json', {
            'python': sys.executable, 'python_version': sys.version, 'platform': platform.platform(),
            'processor': platform.processor(), 'cpu_identifier': os.environ.get('PROCESSOR_IDENTIFIER'),
            'packages': {n: importlib.metadata.version(n) for n in
                         ('torch', 'numpy', 'scipy', 'clarabel', 'highspy', 'numba', 'shapely')},
            'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'git_status': subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True),
            'prepared_at': time.strftime('%Y-%m-%dT%H:%M:%S%z')})
    return identity


def blas_threads():
    """Read and cap the bundled Windows OpenBLAS DLLs without extra packages."""
    result = []
    for library in sorted((Path(sys.prefix)/'Lib/site-packages').glob('*.libs/*openblas*.dll')):
        dll = ctypes.CDLL(str(library))
        for prefix, suffix in (('openblas_', '64_'), ('scipy_openblas_', ''), ('openblas_', '')):
            try:
                setter = getattr(dll, prefix+'set_num_threads'+suffix)
                getter = getattr(dll, prefix+'get_num_threads'+suffix)
            except AttributeError:
                continue
            setter.argtypes = [ctypes.c_int]
            getter.restype = ctypes.c_int
            setter(1)
            result.append({'library': str(library), 'num_threads': getter()})
            break
        else:
            raise RuntimeError('Cannot verify OpenBLAS thread count: '+str(library))
    assert result and all(v['num_threads'] == 1 for v in result)
    return result


def resource_caps():
    """Apply only resource settings; leave frozen numerical source files intact."""
    global SOLVER_THREADS
    import clarabel
    import highspy
    import scipy.optimize
    import numba
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    numba.set_num_threads(1)
    blas_threads()
    SOLVER_THREADS = {'clarabel': set(), 'highspy': set(), 'scipy_highs': set()}
    original_conic = clarabel.DefaultSolver
    def conic(*args, **kwargs):
        settings = kwargs['settings'] if 'settings' in kwargs else args[-1]
        settings.max_threads = 1
        SOLVER_THREADS['clarabel'].add(settings.max_threads)
        return original_conic(*args, **kwargs)
    clarabel.DefaultSolver = conic
    original_option = highspy.Highs.setOptionValue
    def option(self, name, value):
        if name == 'threads':
            value = 1
            SOLVER_THREADS['highspy'].add(value)
        return original_option(self, name, value)
    highspy.Highs.setOptionValue = option
    original_lp = scipy.optimize.linprog
    def lp(*args, **kwargs):
        if kwargs.get('method', 'highs').startswith('highs'):
            kwargs['options'] = dict(kwargs.get('options') or {}, threads=1)
            SOLVER_THREADS['scipy_highs'].add(1)
        return original_lp(*args, **kwargs)
    scipy.optimize.linprog = lp


def initialize(identity):
    global MODEL, READER, RECORDS, OPTIMIZE
    started = time.perf_counter()
    resource_caps()
    from .canonical_native import warmup
    from .solver import optimize
    OPTIMIZE = optimize
    warmup()
    assert sha(WEIGHTS) == identity['checkpoint_sha256']
    state = torch.load(WEIGHTS, map_location='cpu', weights_only=False)
    MODEL = GeometryModel(**state['model_config']).eval()
    MODEL.load_state_dict(state['model'], strict=True)
    RECORDS = identity['records']
    READER = FrozenMemberReader([r['member'] for r in RECORDS], 'local_test',
                               identity=identity['registry_sha256'], allow_local_test=True)
    save(OUT/f'workers/{os.getpid()}.json', {
        'pid': os.getpid(), 'initialization_seconds': time.perf_counter()-started,
        'torch_threads': torch.get_num_threads(), 'interop_threads': torch.get_num_interop_threads(),
        'threadpools': blas_threads(), 'checkpoint_sha256': sha(WEIGHTS)})


def execute(index, deadline):
    path = OUT/f'cases/{index:04d}.json'
    record = RECORDS[index]
    result = json.loads(path.read_text(encoding='utf-8')) if path.exists() else None
    if result is not None and 'metrics' in result:
        return index
    if result is None:
        if time.monotonic() >= deadline:
            return None
        inputs = public_member_inputs(record, 'local_test', authorize_local_test=True)
        seed_all(SEED+index, 1)
        begin = time.perf_counter()
        try:
            draft, inference_seconds = predict(MODEL, inputs)
            positions, stats = OPTIMIZE(draft, inputs, **STEADY)
            elapsed = time.perf_counter()-begin
            result = {'index': index, 'member': record['member'], 'n': record['n'],
                      'positions': positions.tolist(), 'seconds': elapsed,
                      'inference_seconds': inference_seconds, 'postprocess_seconds': stats['seconds'],
                      'row_fallback': stats['row_retained'], 'trace': stats['trace'],
                      'worker_pid': os.getpid(), 'torch_threads': torch.get_num_threads(),
                      'solver_threads': {k: sorted(v) for k, v in SOLVER_THREADS.items()}}
            # Save the immutable solve before reading scoring labels; resume can score it without re-solving.
            save(path, result)
        except Exception as error:
            result = {'index': index, 'member': record['member'], 'n': record['n'],
                      'seconds': time.perf_counter()-begin, 'error': repr(error),
                      'metrics': {'is_feasible': False, 'cost': 10.}, 'worker_pid': os.getpid()}
    if 'metrics' not in result:
        try:
            raw = READER.case(index)  # Labels are first indexed only after the completed solve.
            case = _build_case(raw.case_id, raw.source, raw.solver_inputs[1:6],
                               raw.reference_positions, raw.stored_metrics)
            case.solver_inputs = public_member_inputs(record, 'local_test', authorize_local_test=True)
            scored = evaluate_case(case, result['positions'], result['seconds'])
            result['metrics'] = scored['metrics']
            result['independent_hard_checks'] = scored['independent_hard_checks']
        except Exception as error:
            result['scoring_error'] = repr(error)
            result['metrics'] = {'is_feasible': False, 'cost': 10.}
    save(path, result)
    return index


def summarize(identity, elapsed):
    rows = [json.loads((OUT/f'cases/{i:04d}.json').read_text(encoding='utf-8')) for i in range(1000)]
    assert all('metrics' in r for r in rows)
    q = np.array([r['metrics']['cost'] for r in rows])
    n = np.array([r['n'] for r in rows])
    times = np.array([r['seconds'] for r in rows])
    w = np.exp(n/12 - (n/12).max())
    weighted = float(np.average(q, weights=w))
    assert math.isclose(weighted, official_evaluator().compute_total_score(q.tolist(), n.tolist()), rel_tol=1e-12)
    summary = {'status': 'complete', 'count': 1000, 'hard_legal': sum(r['metrics']['is_feasible'] for r in rows),
               'weighted_Q': weighted, 'mean_Q': float(q.mean()), 'median_Q': float(np.median(q)),
               'p95_Q': float(np.percentile(q, 95)), 'CVaR95_Q': float(np.sort(q)[-50:].mean()),
               'max_Q': float(q.max()), 'seconds_mean': float(times.mean()),
               'seconds_median': float(np.median(times)), 'seconds_p95': float(np.percentile(times, 95)),
               'seconds_max': float(times.max()), 'over_2s': int((times > 2).sum()),
               'sum_solve_seconds': float(times.sum()), 'evaluation_wall_seconds': elapsed,
               'row_fallbacks': sum(r.get('row_fallback', False) for r in rows),
               'execution_errors': sum('error' in r for r in rows),
               'scoring_errors': sum('scoring_error' in r for r in rows),
               'workers': 4, 'threads_per_worker': 1,
               'worker_pids': sorted({r['worker_pid'] for r in rows}),
               'checkpoint_sha256': identity['checkpoint_sha256'], 'test_read': True,
               'scope': identity['task'], 'time_scope': identity['time_scope'],
               'official_contest_score': False, 'previously_exposed_100_excluded': True}
    groups = {}
    for lower, upper in ((21, 40), (41, 60), (61, 80), (81, 100), (101, 120)):
        mask = (n >= lower) & (n <= upper)
        groups[f'{lower}-{upper}'] = {'count': int(mask.sum()), 'hard_legal': sum(r['metrics']['is_feasible'] for r in rows if lower <= r['n'] <= upper),
                                    'weighted_Q': float(np.average(q[mask], weights=w[mask])),
                                    'mean_Q': float(q[mask].mean()), 'seconds_mean': float(times[mask].mean())}
    summary['by_n'] = groups
    save(OUT/'summary.json', summary)
    report = f'''# E25 最终本地留出测试：1000例

冻结版本：E17 best51（497,299参数）＋E25原版steady后处理。单一候选，测试后不选权重、不调参。

| 指标 | 结果 |
|---|---:|
| 硬合法 | {summary['hard_legal']}/1000 |
| 模块数加权Q（越低越好） | {weighted:.9f} |
| 算术平均Q | {summary['mean_Q']:.9f} |
| P95 Q / 最差5%平均Q / 最大Q | {summary['p95_Q']:.6f} / {summary['CVaR95_Q']:.6f} / {summary['max_Q']:.6f} |
| 平均完整求解秒/例 | {summary['seconds_mean']:.6f} |
| P95 / 最大完整求解秒 | {summary['seconds_p95']:.6f} / {summary['seconds_max']:.6f} |
| 超过2秒 | {summary['over_2s']} |
| 执行错误 / 评分错误 | {summary['execution_errors']} / {summary['scoring_errors']} |
| 评价墙钟秒（含初始化、IO及评分） | {elapsed:.3f} |

资源：CPU 4进程，每进程1数值线程；模型和求解器驻留。完整逐例延迟包含公开图特征、CPU推理、解码、全部后处理和救援；文件/模型读取、固定派生任务转换、首次JIT、评分和结果写入另计。批量墙钟除1000是吞吐口径，不能冒充单例求解延迟。历史E25开发结果为2进程×2线程，本次资源口径不同，不直接归因为算法加速。

面板：原冻结v5 local_test2000中排除2026-10-22已可视检查的100例，从余下1900例按实际模块数比例分层，用Hamilton配额和种子20261024的SHA256排名固定1000例。成员及代码/权重哈希在任何测试张量读取前写入config.json。仍遵守原split_v1完整分片隔离，剩余900例继续封存。本轮1000例自此已暴露。

这是MIB平均面积派生任务、沿用原参考分母的本地中性时间Q；采用未修改官方评分器和独立硬约束复核，runtime=median_runtime=1。**不是官方隐藏测试成绩或正式比赛总分，也不代表未经兼容筛选的原始任务分布。** 单次CPU计时会受系统负载及限时优化路径影响。

复现与证据：config.json为冻结成员/输入协议/源码身份；source_snapshot.zip为本机源码快照；environment.json为环境及Git状态；workers/记录初始化和实际线程池；cases/逐例只保存一份完整坐标，原始参考标签不落盘；summary.json保存精确汇总。中断恢复只处理未完成求解，已存坐标的样本仅补评分。

```powershell
& .\\.conda\\envs\\iccad2026\\python.exe -u -B -m experiments.E25_integrated_baseline.final_test --resume
```
'''
    (OUT/'analysis.md').write_text(report, encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prepare-only', action='store_true')
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()
    identity = prepare()
    if args.prepare_only:
        print('Frozen metadata only; no test tensors read:', OUT, flush=True)
        return
    with exclusive(OUT/'evaluation.lock'):
        if (OUT/'summary.json').exists():
            print('Already complete; no repeated evaluation:', OUT, flush=True)
            return
        prior = json.loads((OUT/'progress.json').read_text()) if (OUT/'progress.json').exists() else {'elapsed_seconds': 0}
        if prior['elapsed_seconds'] and not args.resume:
            raise ValueError('Interrupted run requires --resume')
        start = time.monotonic()
        deadline = start + identity['budget_seconds'] - prior['elapsed_seconds']
        if deadline <= start:
            raise TimeoutError('Cumulative1100s budget exhausted; no automatic restart')
        complete = set()
        for i in range(1000):
            p = OUT/f'cases/{i:04d}.json'
            if p.exists() and 'metrics' in json.loads(p.read_text()):
                complete.add(i)
        todo = iter(i for i in range(1000) if i not in complete)
        invocation = OUT/f'invocations/{time.time_ns()}.json'
        save(invocation, {'argv': sys.argv, 'started': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                          'previous_completed': len(complete), 'previous_elapsed_seconds': prior['elapsed_seconds']})
        try:
            with ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context('spawn'),
                                     initializer=initialize, initargs=(identity,)) as pool:
                pending = {}
                def submit():
                    if time.monotonic() >= deadline-15 or (OUT/'STOP').exists():
                        return
                    i = next(todo, None)
                    if i is not None:
                        pending[pool.submit(execute, i, deadline)] = i
                for _ in range(4):
                    submit()
                while pending:
                    finished, _ = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
                    for f in finished:
                        pending.pop(f)
                        index = f.result()
                        if index is not None:
                            complete.add(index)
                        submit()
                    elapsed = prior['elapsed_seconds'] + time.monotonic()-start
                    save(OUT/'progress.json', {'completed': len(complete), 'elapsed_seconds': elapsed,
                                               'status': 'running' if pending else 'stopped'})
                    if finished and len(complete) % 25 == 0:
                        print(f'Final test {len(complete)}/1000; wall {elapsed:.1f}s', flush=True)
        finally:
            elapsed = prior['elapsed_seconds'] + time.monotonic()-start
            save(OUT/'progress.json', {'completed': len(complete), 'elapsed_seconds': elapsed,
                                       'status': 'complete' if len(complete) == 1000 else 'paused'})
        if len(complete) == 1000:
            summarize(identity, elapsed)
        else:
            print('Paused with saved evidence:', len(complete), OUT, flush=True)


if __name__ == '__main__':
    main()
