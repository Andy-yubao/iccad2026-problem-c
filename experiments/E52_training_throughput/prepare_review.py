"""Fixed random dev100 review from saved drafts/finals; no model or solver calls."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
import multiprocessing as mp
import os
from pathlib import Path
import random
import time
from .io import HERE, ROOT, save, sha, exclusive


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def initialize(run_folder):
    global ROWS, DATA, RUN
    import torch
    from datasets.reference_layouts import ReferenceLayouts
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    RUN = Path(run_folder)
    ROWS = {r['index']: r for r in read(RUN/'terminal/results.json')['records']}
    DATA = ReferenceLayouts('dev_monitor500')


def render(payload):
    index, output_string = payload
    from visualization.layouts import plot_case, geometry_details
    import numpy as np
    output = Path(output_string); name = f'case_{index:04d}'
    target = output/'images'/(name+'.png'); meta = output/'image_meta'/(name+'.json')
    if target.exists() and meta.exists():
        value = read(meta)
        if sha(target) != value['image_sha256']: raise ValueError('Review image changed')
        return value
    row = ROWS[index]; item = DATA[index]
    if list(item['member']) != row['member']: raise ValueError('Review member changed')
    constraints = np.asarray(item['case'].solver_inputs[5]).tolist()
    variants = {k: {'positions': p, 'soft_details': geometry_details(p, constraints) if p is not None else {}} for k, p in
                (('reference', item['positions'].tolist()), ('raw', row.get('draft_positions')), ('final', row.get('positions')))}
    adapted = {'panel': 'E52 / dev review', 'case_id': name, 'constraints': constraints, 'variants': variants,
               'caption': 'Original reference before mean-area transform | Saved timed draft | Saved flash/steady final' + (' | FAILED: '+row['error'] if 'error' in row else '')}
    desc = plot_case(adapted, target, show_metrics=False, stages=(('reference', 'Original reference'),
        ('raw', 'E52 draft / RATE THIS'), ('final', 'Flash / steady postprocessing')))
    value = {'id': name, 'source_index': index, 'member': row['member'], 'n': row['n'],
             'image': target.relative_to(output).as_posix(), 'image_sha256': sha(target),
             'axes_bounds': desc['axes_bounds'], 'auxiliary': {'后处理Q': row['metrics']['cost'],
             '最终硬合法': row['metrics']['is_feasible'], '完整求解秒数': row['seconds'],
             '执行异常': row.get('error'),
             '草稿来源': '终末评价保存的实际草稿；没有补生成或重跑后处理',
             '参考列': '原任务参考，尚未应用MIB平均面积变换'}}
    save(meta, value); return value


def prepare(output, seed=20261025):
    output = Path(output).resolve()
    if not output.is_relative_to(HERE/'runs'): raise ValueError('Output must stay in E52/runs')
    target = output/'visual_review100'; started = time.perf_counter()
    frozen = read(output/'frozen.json'); source = output/'terminal/results.json'
    terminal = read(source)
    if len(terminal['records']) != 500: raise ValueError('Require completed once-only dev500 terminal')
    selected = random.Random(seed).sample(range(500), 100)
    records = {r['index']: r for r in terminal['records']}
    identity = {'seed': seed, 'selected_indices': selected, 'selected_members': [records[i]['member'] for i in selected],
        'checkpoint_sha256': frozen['checkpoint_sha256'], 'source_sha256': sha(source),
        'renderer_sha256': sha(ROOT/'visualization/layouts.py'), 'preparer_sha256': sha(Path(__file__)),
        'source_run': output.relative_to(ROOT).as_posix(), 'role': 'v5 dev_monitor500',
        'selection': 'uniform sample without replacement; order frozen before rendering; no Q/appearance filtering',
        'workers': 4, 'threads_per_worker': 1, 'model_calls': 0, 'solver_calls': 0, 'test_read': False}
    with exclusive(target.with_suffix('.lock')):
        target.mkdir(exist_ok=True)
        config = target/'config.json'
        if config.exists():
            if read(config) != identity: raise ValueError('Review source/selection changed')
        else: save(config, identity)
        if (target/'manifest.json').exists():
            for row in read(target/'manifest.json')['cases']:
                if sha(target/row['image']) != row['image_sha256']: raise ValueError('Review image changed')
            print('Existing review reused:', target); return
        for name in ('images', 'image_meta'): (target/name).mkdir(exist_ok=True)
        previous = os.environ.get('E52_THREADS'); os.environ['E52_THREADS'] = '1'
        try:
            with ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context('spawn'), initializer=initialize,
                                     initargs=(str(output),)) as pool:
                cases = []
                for value in pool.map(render, [(i, str(target)) for i in selected]):
                    cases.append(value)
                    if time.perf_counter()-started > 1100: raise TimeoutError('Review preparation budget; completed images retained')
        finally:
            if previous is None: os.environ.pop('E52_THREADS', None)
            else: os.environ['E52_THREADS'] = previous
        save(target/'manifest.json', {'schema_version': 1, 'session_id': output.name+'_dev100',
            'title': 'E52 · 逐模块重叠 / 草稿评分', 'created_utc': datetime.now(timezone.utc).isoformat(),
            'checkpoint_sha256': frozen['checkpoint_sha256'],
            'display_note': '固定随机开发100例；只评价中间draft。原参考在MIB面积变换之前，最终布局采用固定flash/steady。',
            'rubric': {'version': 1, 'target': 'raw_draft_only', 'scale': {'1': '差', '2': '一般', '3': '好'},
                'dimensions': '关注相对方向/邻接意图及局部冲突，保留自由观察原话；不将位移大直接视为拓扑破坏。'},
            'selection': identity, 'cases': cases})
        save(target/'preparation_summary.json', {'count': 100, 'seconds': time.perf_counter()-started,
            'model_calls': 0, 'solver_calls': 0, 'test_read': False, 'human_ratings_complete': False,
            'status': 'ready_for_human_ratings', 'manifest_sha256': sha(target/'manifest.json')})
        print('Ready for human ratings:', target)


def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args(); prepare(args.output)


if __name__ == '__main__': main()
