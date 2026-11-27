"""One registered check of five exposed saved drafts; no training/test access."""
import os
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import time
import zipfile
import numpy as np
from threadpoolctl import threadpool_limits, threadpool_info
from datasets.paths import ROOT
from datasets.reference_layouts import registry, public_hard_reference
from datasets.public_member_inputs import public_member_inputs
from scoring.checks import independent_hard_checks
from visualization.layouts import plot_case, geometry_details
from .relaxation import Config
from .pipeline import preprocess, solve
from .diagnostics import measures
from .planner.solver import optimize
from .planner.canonical_native import warmup

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / 'runs/20261102_translation_v1_check5_attempt02'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def check_final(final, inputs):
    check = independent_hard_checks(final.tolist(), np.asarray(inputs[1]), np.asarray(inputs[5]),
                                    public_hard_reference(inputs[1], inputs[5], inputs[6]))
    if not check['is_feasible'] or (final[:, :2] < 0).any():
        raise AssertionError('downstream returned invalid geometry')
    return check


def main():
    OUTPUT.mkdir(parents=True, exist_ok=False)
    (OUTPUT / 'coordinates').mkdir()
    (OUTPUT / 'gallery').mkdir()
    begin = time.perf_counter()
    plan = read(HERE / 'plan.json')
    config = Config(**plan['config'])
    planner_config = read(HERE / 'planner_config.json')
    reg = registry()
    manifest = read(HERE / 'planner_provenance.json')
    for row in manifest['files']:
        assert sha(ROOT / row['source']) == row['source_sha256'] == sha(ROOT / row['copy'])
    paths = [*HERE.rglob('*.py'), HERE/'plan.json', HERE/'planner_config.json', HERE/'planner_provenance.json', ROOT/'visualization/layouts.py', ROOT/'datasets/public_member_inputs.py', ROOT/'datasets/reference_layouts.py', ROOT/'datasets/paths.py', ROOT/'scoring/checks.py']
    identity = {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(paths)}
    save(OUTPUT/'config.json', dict(registered_plan=plan, source_sha256=identity,
         git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
         working_tree='Existing unrelated user changes preserved; this run bound to hashes and source snapshot',
         registry_sha256=sha(ROOT/'protocol/reference_workset_v5.json'),
         manifest_sha256=reg['manifest_sha256'], frozen_hashes=reg['frozen_original_hashes'],
         data_paths_sha256=sha(ROOT/'protocol/paths.json'),
         created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),test_read=False))
    with zipfile.ZipFile(OUTPUT/'source_snapshot.zip', 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, path.relative_to(ROOT).as_posix())
    records = []
    with threadpool_limits(limits=1):
        import torch, numba
        torch.set_num_threads(1); torch.set_num_interop_threads(1); numba.set_num_threads(1)
        start = time.perf_counter(); warmup(); cold = time.perf_counter() - start
        for index in range(5):
            if (HERE/'runs/STOP').exists() or (OUTPUT/'STOP').exists():
                raise RuntimeError('STOP: no subsequent members will be started')
            member = reg['sets']['dev_monitor500']['records'][index]
            source = ROOT / plan['short_check']['source_draft'].format(index=index)
            saved = read(source)
            assert saved['member'] == member['member'] and saved['n'] == member['n']
            inputs = public_member_inputs(member, 'dev')
            draft = np.asarray(saved[plan['short_check']['source_field']], dtype=float)
            original = draft.copy()
            # One retained trajectory supports invariant checks and all figures.
            revised, relaxed = preprocess(draft, inputs, config=config, return_history=True)
            history = relaxed.pop('history')
            short = draft[:, 2:].min(1)
            steps = np.linalg.norm(np.diff(history[:, :, :2], axis=0), axis=-1)
            assert (steps <= config.step_fraction * short + 1e-10).all()
            assert (steps.sum(0) <= config.total_fraction * short + 1e-10).all()
            np.testing.assert_array_equal(draft, original)
            results = {}
            # Alternating order is fixed before observing outcomes.
            order = ('control', 'relaxed') if index % 2 == 0 else ('relaxed', 'control')
            for arm in order:
                start = time.perf_counter()
                if arm == 'control':
                    final, trace = optimize(draft, inputs, **planner_config)
                else:
                    final, trace = solve(draft, inputs, optimize, config=config, planner_options=planner_config)
                hard = check_final(final, inputs)
                seconds = time.perf_counter() - start
                results[arm] = dict(final=final, hard=hard, trace=trace, seconds=seconds)
            np.savez_compressed(OUTPUT/'coordinates'/f'{index:04d}.npz', positions=draft,
                relaxed=revised, history=history, preplaced=np.asarray(inputs[5])[:,1] != 0,
                constraints=np.asarray(inputs[5]), control_final=results['control']['final'],
                relaxed_final=results['relaxed']['final'])
            record = dict(index=index,member=member['member'],n=int(member['n']),source=source.relative_to(ROOT).as_posix(),
                          source_sha256=sha(source),arm_order=order,
                          raw_before=measures(draft),raw_after=measures(revised),relaxation=relaxed,
                          raw_module_width_height_unchanged=True,preplaced_unchanged=True,
                          control=dict(seconds=results['control']['seconds'],hard=results['control']['hard'],
                                       geometry=measures(results['control']['final']),trace=results['control']['trace']),
                          relaxed=dict(seconds=results['relaxed']['seconds'],hard=results['relaxed']['hard'],
                                       geometry=measures(results['relaxed']['final']),trace=results['relaxed']['trace']))
            records.append(record)
            save(OUTPUT/'progress.json',dict(completed=len(records),records=records,test_read=False))
            print(json.dumps(dict(index=index,n=record['n'],before=record['raw_before']['pair_intersection_over_area_sum'],
                       after=record['raw_after']['pair_intersection_over_area_sum'],
                       movement=relaxed['max_displacement_fraction'],preprocessor_ms=relaxed['seconds']*1000,
                       control_seconds=results['control']['seconds'],relaxed_seconds=results['relaxed']['seconds'])),flush=True)
        pools = threadpool_info()
        assert all(p['num_threads'] <= 1 for p in pools)
    # Rendering happens after all timed requests and reuses saved coordinates.
    for record in records:
        index = record['index']
        with np.load(OUTPUT/'coordinates'/f'{index:04d}.npz') as data:
            constraints = data['constraints'].tolist()
            variants = {name:dict(positions=data[key].tolist(),soft_details=geometry_details(data[key],constraints))
                        for name,key in (('before','positions'),('after','relaxed'),('control','control_final'),('final','relaxed_final'))}
        plot_case(dict(case_id=f'dev500:{index}',panel='E99 / five-case implementation check / not formal evaluation',
                       constraints=constraints,variants=variants),OUTPUT/'gallery'/f'{index:04d}.png',
                  stages=(('before','Saved C3 draft'),('after','Central forces / 4 steps'),
                          ('control','Planner control'),('final','Central forces + planner')),show_metrics=False)
    html = '<!doctype html><meta charset="utf-8"><title>E99 five-case check</title><style>body{font:16px system-ui}img{width:100%}</style><h1>E99 / five-case implementation check</h1>'
    html += ''.join(f'<p>dev500:{i}</p><img loading="lazy" src="{i:04d}.png">' for i in range(5))
    (OUTPUT/'gallery/index.html').write_text(html,encoding='utf-8')
    def timing(values):
        return dict(mean=float(np.mean(values)),p95=float(np.percentile(values,95)),max=float(np.max(values)))
    result = dict(status='passed',finished=True,count=5,test_read=False,no_training=True,
                  quality_scope='Implementation diagnostics on five already-exposed members, no Q evaluation or generalization claim',
                  timing_scope='CPU resident draft through planner and hard check; excludes NN, public construction and transfer; five samples only',
                  cold_native_warmup_seconds=cold,threads=pools,
                  raw_overlap_improved_cases=sum(r['raw_after']['pair_intersection_over_area_sum'] < r['raw_before']['pair_intersection_over_area_sum'] for r in records),
                  max_displacement_fraction=max(r['relaxation']['max_displacement_fraction'] for r in records),
                  max_path_fraction=max(r['relaxation']['max_path_fraction'] for r in records),
                  preprocessor_seconds=timing([r['relaxation']['seconds'] for r in records]),
                  control_seconds=timing([r['control']['seconds'] for r in records]),
                  relaxed_seconds=timing([r['relaxed']['seconds'] for r in records]),
                  records=records,elapsed_seconds=time.perf_counter()-begin)
    save(OUTPUT/'results.json',result)
    lines = ['# E99 五例实施短检', '', '固定 v5 dev_monitor500 的前五例；已暴露开发成员，不是正式 dev500 评价。没有读取测试、训练网络或扫描系数。',
             '', '默认4轮；单轮路程≤各模块初始短边2%；累计路程及位移≤6%。保持尺寸与预放置不变。',
             '', '| ID | N | 交面积/总面积：前→后 | 覆盖<10%模块：前→后 | 最坏覆盖：前→后 | 最大位移/短边 | 前置步骤ms |', '|---|---:|---:|---:|---:|---:|---:|']
    for r in records:
        a,b=r['raw_before'],r['raw_after']
        lines.append(f"| {r['index']} | {r['n']} | {a['pair_intersection_over_area_sum']:.4%} → {b['pair_intersection_over_area_sum']:.4%} | {a['fraction_strictly_below_10pct']:.2%} → {b['fraction_strictly_below_10pct']:.2%} | {a['worst_module_coverage']:.2%} (ID {a['worst_module']}) → {b['worst_module_coverage']:.2%} (ID {b['worst_module']}) | {r['relaxation']['max_displacement_fraction']:.2%} | {r['relaxation']['seconds']*1000:.3f} |")
    lines += ['', '两臂各五例均通过最终独立硬检查。整图草稿指标只解释机制，不保证最终Q改善。未评价Q，也没有正式质量升级结论。', '',
              f"前置步骤均时 {result['preprocessor_seconds']['mean']*1000:.3f} ms，P95 {result['preprocessor_seconds']['p95']*1000:.3f} ms。", '',
              f"CPU草稿到规划器返回及硬检查：控制均时 {result['control_seconds']['mean']:.6f} s / P95 {result['control_seconds']['p95']:.6f} s；前置+规划均时 {result['relaxed_seconds']['mean']:.6f} s / P95 {result['relaxed_seconds']['p95']:.6f} s。", '',
              '以上仅各五次单线程短检，包含运行时预算求解的波动，不能宣称提速；不含网络、公开输入构建和传输，不代表完整请求延迟。冷启动单列于results。', '', '[同例四列图](gallery/index.html)；坐标与每轮轨迹保存在coordinates，未自动打开界面。']
    (OUTPUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('records','threads')},indent=2))


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        if OUTPUT.exists() and not (OUTPUT/'results.json').exists():
            save(OUTPUT/'error.json',dict(error=repr(error),test_read=False,no_automatic_retry=True))
        raise
