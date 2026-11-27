"""Authorized fixed dev100 flash run; immutable training sources are reused."""
import os
os.environ.setdefault('E51_THREADS', '2')
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[key] = os.environ['E51_THREADS']
os.environ['MPLBACKEND'] = 'Agg'
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import json
import multiprocessing as mp
import random
import signal
import time
import zipfile
import numpy as np
import torch
from datasets.reference_layouts import ReferenceLayouts, REGISTRY
from experiments.E51_module_overlap.io import HERE, save, sha, sources, snapshot, exclusive
from experiments.E51_module_overlap.final import STEADY, evaluate_row, summarize
from experiments.E51_module_overlap.data import load_rows, collate
from experiments.E51_module_overlap.model import GeometryModel
from experiments.E51_module_overlap.batches import inputs_only
from experiments.E51_module_overlap.loss import losses
from experiments.E51_module_overlap.train import seed_all
from experiments.E51_module_overlap.paired import paired

OUT = Path(__file__).resolve().parent
TRAIN = HERE/'runs/20261025_rectangle_w96d12_unified_v2_main_seed0'
CONTROL = ROOT/'experiments/E42_joint_regularization/runs/20261023_rectangle_w96d12_ab_seed0'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def stopped():
    return (OUT/'STOP').exists() or (HERE/'runs/STOP').exists()


def initialize(checkpoint, indices):
    global MODEL, ROWS, INITIALIZATION
    started = time.perf_counter()
    from experiments.E51_module_overlap.canonical_native import warmup
    seed_all(0, 2); warmup()
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    MODEL = GeometryModel(**state['model_config']).eval(); MODEL.load_state_dict(state['model'])
    ROWS = dict(zip(indices, load_rows('dev_monitor500', indices)))
    INITIALIZATION = time.perf_counter()-started


@torch.no_grad()
def evaluate(index):
    row = ROWS[index]
    record = evaluate_row(index, row, MODEL)
    # Separate diagnostics use the identical fixed Loss, after timed flash solve.
    started = time.perf_counter()
    graph, targets = collate([row], 'cpu')
    values = losses(MODEL(inputs_only(graph)), graph, targets, 'main', 2., include_quality=True)
    record['raw_loss'] = {k: float(v) for k, v in values.items()}
    record['loss_diagnostic_seconds'] = time.perf_counter()-started
    record['resident_initialization_seconds'] = INITIALIZATION
    record['threads'] = {'torch': torch.get_num_threads(), 'interop': torch.get_num_interop_threads(),
                         'blas_environment': os.environ['OMP_NUM_THREADS']}
    return record


def review_initialize(folder):
    from experiments.E51_module_overlap import prepare_review as review
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    review.ROWS = {r['index']: r for r in [read(p) for p in (Path(folder)/'cases').glob('*.json')]}
    review.DATA = ReferenceLayouts('dev_monitor500'); review.RUN = Path(folder)


def render(payload):
    from experiments.E51_module_overlap.prepare_review import render as render_saved
    return render_saved(payload)


def identity():
    checkpoint = TRAIN/'checkpoints/best.pt'
    best = torch.load(checkpoint, map_location='cpu', weights_only=False)
    latest = torch.load(TRAIN/'checkpoints/latest.pt', map_location='cpu', weights_only=False)
    if not latest['finished'] or latest['identity'] != best['identity']:
        raise ValueError('Training incomplete or identities differ')
    if best['identity']['sources'] != sources() or best['identity']['registry_sha256'] != sha(REGISTRY):
        raise ValueError('Frozen trained source/data changed')
    if best['identity']['implementation_check']:
        raise ValueError('Implementation checkpoint is not a formal candidate')
    indices = random.Random(20261025).sample(range(500), 100)
    meta = ReferenceLayouts('dev_monitor500').records
    return {'checkpoint': checkpoint.relative_to(ROOT).as_posix(), 'checkpoint_sha256': sha(checkpoint),
            'best_epoch': best['progress']['best_epoch'], 'model_config': best['model_config'],
            'sources': sources(), 'runner_sha256': sha(Path(__file__)), 'registry_sha256': sha(REGISTRY),
            'paths_sha256': best['identity']['paths_sha256'], 'seed': 20261025, 'count': 100,
            'panel': 'v5 dev_monitor500 uniform sample without replacement; sampled order frozen before evaluation',
            'selected_indices': indices, 'selected_members': [list(meta[i]['member']) for i in indices],
            'postprocessor': 'flash / original steady, unchanged E51 local kernel', 'steady': STEADY,
            'workers': 2, 'threads_per_worker': 2, 'review_workers': 4, 'review_threads_per_worker': 1,
            'selection': 'training unique best by unified fixed dev500 total; no Q-based checkpoint choice',
            'timing': 'features + CPU inference + decode + flash/repair/rescue; excludes initialization/JIT/scoring and extra Loss diagnostics',
            'control_result_sha256': sha(CONTROL/'terminal/results.json'),
            'authorization': '2026-10-25 user explicitly requested background100-case flash evaluation',
            'test_read': False, 'training_calls': 0, 'neutral_runtime_development_only': True}


def run():
    with exclusive(OUT/'run.lock'):
        config = identity()
        if (OUT/'config.json').exists():
            if read(OUT/'config.json')['identity'] != config:
                raise ValueError('Evaluation source/panel/checkpoint identity changed')
        else:
            save(OUT/'config.json', {'identity': config, 'command': [sys.executable, *sys.argv],
                                    'created_utc': datetime.now(timezone.utc).isoformat()})
            snapshot(OUT/'source_snapshot.zip', config)
            with zipfile.ZipFile(OUT/'source_snapshot.zip', 'a', zipfile.ZIP_DEFLATED) as archive:
                archive.write(Path(__file__), Path(__file__).relative_to(ROOT).as_posix())
        if (OUT/'summary.json').exists():
            print('Existing complete evaluation reused; no rerun.', flush=True); return
        (OUT/'cases').mkdir(exist_ok=True)
        indices = config['selected_indices']; metadata = dict(zip(indices, config['selected_members']))
        records = {r['index']: r for r in [read(p) for p in (OUT/'cases').glob('*.json')]}
        if any(i not in metadata or r['member'] != metadata[i] for i, r in records.items()):
            raise ValueError('Saved case/panel mismatch')
        started = time.perf_counter()
        def progress(status, **extra):
            save(OUT/'progress.json', {'status': status, 'completed': len(records), 'target': 100,
                'elapsed_this_invocation': time.perf_counter()-started, 'test_read': False, **extra})
        progress('evaluating')
        print('Frozen best52 verified; fixed dev100 flash starting, 2 workers x 2 threads.', flush=True)
        with ProcessPoolExecutor(max_workers=2, mp_context=mp.get_context('spawn'), initializer=initialize,
                                 initargs=(ROOT/config['checkpoint'], indices)) as pool:
            todo = iter(i for i in indices if i not in records); pending = {}
            def submit():
                if stopped(): return
                index = next(todo, None)
                if index is not None: pending[pool.submit(evaluate, index)] = index
            for _ in range(2): submit()
            while pending:
                done, _ = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
                for future in done:
                    index = pending.pop(future); record = future.result()
                    if record['member'] != metadata[index]: raise ValueError('Evaluated member changed')
                    save(OUT/f'cases/{index:04d}.json', record); records[index] = record
                    progress('evaluating'); print(f'Completed {len(records)}/100', flush=True); submit()
        if len(records) != 100:
            progress('user_stop'); return
        ordered = [records[i] for i in indices]
        summary = summarize(ordered)
        raw_loss = {k: float(np.mean([r['raw_loss'][k] for r in ordered])) for k in ordered[0]['raw_loss']}
        controls = {r['index']: r for r in read(CONTROL/'terminal/results.json')['records']}
        old = [controls[i] for i in indices]
        if [r['member'] for r in old] != [r['member'] for r in ordered]: raise ValueError('E42 pairing mismatch')
        stats = {'summary': summary, 'raw_unified_loss': raw_loss, 'E42_flash_control': summarize(old),
                 'paired_to_E42': paired(ordered, old), 'checkpoint_sha256': config['checkpoint_sha256'],
                 'selected_indices': indices, 'test_read': False, 'neutral_runtime_development_only': True,
                 'note': 'Same samples; historical control timing is not a concurrent speed benchmark. No new E42 solver runs.'}
        save(OUT/'evaluation_summary.json', stats)
        progress('rendering_review', rendered=0)
        review = OUT/'visual_review100'
        for name in ('images', 'image_meta'): (review/name).mkdir(parents=True, exist_ok=True)
        save(review/'config.json', {'checkpoint_sha256': config['checkpoint_sha256'], 'seed': 20261025,
            'selected_indices': indices, 'selected_members': config['selected_members'], 'no_Q_filtering': True,
            'role': 'v5 dev_monitor500', 'solver_calls': 0, 'model_calls': 0, 'test_read': False})
        os.environ['E51_THREADS'] = '1'
        try:
            with ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context('spawn'), initializer=review_initialize,
                                     initargs=(str(OUT),)) as pool:
                cases = []
                for value in pool.map(render, [(i, str(review)) for i in indices]):
                    cases.append(value); progress('rendering_review', rendered=len(cases))
        finally:
            os.environ['E51_THREADS'] = '2'
        save(review/'manifest.json', {'schema_version': 1, 'session_id': OUT.name,
            'title': 'E51 best52 + flash · 开发100例草稿评分', 'created_utc': datetime.now(timezone.utc).isoformat(),
            'checkpoint_sha256': config['checkpoint_sha256'],
            'display_note': '固定随机开发100例，只评价中间draft。参考列在MIB平均面积变换之前，最终列为flash。',
            'rubric': {'version': 1, 'target': 'raw_draft_only', 'scale': {'1': '差', '2': '一般', '3': '好'},
                       'dimensions': '保留相对方向/邻接意图与局部冲突的自由观察原话。'},
            'selection': {'seed': 20261025, 'selected_indices': indices, 'role': config['panel'], 'no_Q_filtering': True},
            'cases': cases})
        save(review/'preparation_summary.json', {'count': 100, 'human_ratings_complete': False,
            'status': 'ready_for_human_ratings', 'solver_calls': 0, 'model_calls': 0, 'test_read': False})
        text = ['# E51 best52 + flash：固定开发100例', '',
                f'硬合法 {summary["hard_legal"]}/100，加权Q {summary["weighted_Q"]:.9f}，算术Q {summary["mean_Q"]:.9f}。',
                f'完整均时 {summary["seconds_mean"]:.6f} 秒，P95 Q {summary["p95_Q"]:.6f}，最大Q {summary["max_Q"]:.6f}。',
                f'统一原始Loss {raw_loss["total"]:.9f}；错误 {summary["errors"]}。', '',
                f'同例E42 flash加权Q {stats["E42_flash_control"]["weighted_Q"]:.9f}；胜/平/负 {stats["paired_to_E42"]["Q_better_equal_worse"]}。',
                '', '逐例坐标只保存在cases；配对与分项见summary.json。时间对照为历史运行，不能直接归因为模型加速。',
                '这是已参与开发监测面板的结果，不是盲测或正式比赛分数；没有访问封存测试。',
                '100张三相图及草稿评分入口在visual_review100；准备完成不代表用户已评分。']
        (OUT/'analysis.md').write_text('\n'.join(text)+'\n', encoding='utf-8')
        stats['review'] = 'visual_review100/manifest.json'
        stats['status'] = 'complete'; stats['elapsed_this_invocation'] = time.perf_counter()-started
        save(OUT/'summary.json', stats); progress('complete', rendered=100)
        print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('--check', action='store_true')
    args = ap.parse_args()
    if args.check:
        value = identity(); print(json.dumps({k: value[k] for k in ('best_epoch', 'checkpoint_sha256', 'count', 'workers', 'threads_per_worker', 'test_read')}, indent=2))
    else:
        old = signal.signal(signal.SIGINT, lambda *_: (OUT/'STOP').touch())
        try:
            run()
        except BaseException as error:
            save(OUT/'failure.json', {'error': repr(error), 'test_read': False})
            raise
        finally:
            signal.signal(signal.SIGINT, old)
