"""Build the user-authorized 100-case visual feedback round from exposed test1000.

Frozen model forward only; no training, postprocessing, or scoring. The reusable
viewer lives under visualization; dataset selection and inference stay here.
"""
from __future__ import annotations
import os
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[_name] = '1'
os.environ['MPLBACKEND'] = 'Agg'

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'runs/20261024_final_test1000_seed20261024'
TRAIN = HERE / 'runs/20261023_rectangle_w96d12_ab_seed0'
DEFAULT_OUTPUT = HERE / 'runs/20261024_visual_review100_seed20261024'
SHA = '12878108e9a58b8c195b314ba6586686c2a6e2796c1f4b6443e102b4db4224a6'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    os.replace(temporary, path)


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def initialize(checkpoint):
    global MODEL
    import torch
    from .model import GeometryModel
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    MODEL = GeometryModel(**state['model_config']).eval()
    MODEL.load_state_dict(state['model'], strict=True)


def render_task(payload):
    config, row, output_string = payload
    output = Path(output_string)
    index = row['index']
    name = f'case_{index:04d}'
    image = output / 'images' / (name + '.png')
    cache = output / 'drafts_local' / (name + '.json')
    info = output / 'image_meta' / (name + '.json')
    if image.exists() and info.exists() and cache.exists():
        saved = read(info)
        if saved['image_sha256'] != sha(image) or saved['draft_sha256'] != sha(cache):
            raise ValueError('Saved image/draft changed')
        return saved
    import numpy as np
    from datasets.public_member_inputs import public_member_inputs
    from datasets.worksets import FrozenMemberReader
    from visualization.layouts import geometry_details, plot_case
    from .inference import predict
    inputs = public_member_inputs(row['record'], 'local_test', authorize_local_test=True)
    if cache.exists():
        draft = read(cache)
        if draft['member'] != row['record']['member'] or draft['checkpoint_sha256'] != SHA:
            raise ValueError('Draft cache identity changed')
    else:
        positions, seconds = predict(MODEL, inputs)
        draft = {'member': row['record']['member'], 'checkpoint_sha256': SHA,
                 'raw_positions': positions.tolist(), 'reconstruction_seconds': seconds,
                 'purpose': 'Reconstructed for visual review, not the timed benchmark draft',
                 'reference_used_by_model': False}
        save(cache, draft)
    source_case = SOURCE / f'cases/{index:04d}.json'
    final = read(source_case)
    if final['member'] != row['record']['member'] or final['n'] != inputs[0]:
        raise ValueError('Saved final/member mismatch')
    reader = FrozenMemberReader([row['record']['member']], 'local_test',
                               identity=config['registry_sha256'], allow_local_test=True)
    reference = reader.case(0).reference_positions
    constraints = np.asarray(inputs[5]).tolist()
    variants = {key: {'positions': positions, 'soft_details': geometry_details(positions, constraints)}
                for key, positions in [('reference', reference), ('raw', draft['raw_positions']),
                                       ('final', final['positions'])]}
    adapted = {'panel': 'E42 / visual review', 'case_id': name, 'constraints': constraints,
               'variants': variants,
               'caption': 'Original reference (before mean-area transform) | Reconstructed frozen draft | Saved steady final'}
    temporary = image.with_name(image.stem + '.tmp.png')
    description = plot_case(adapted, temporary, show_metrics=False, stages=(
        ('reference', 'Original reference'), ('raw', 'E42 draft / RATE THIS'),
        ('final', 'Regular steady postprocessing')))
    os.replace(temporary, image)
    value = {'id': name, 'source_index': index, 'member': final['member'], 'n': final['n'],
             'image': image.relative_to(output).as_posix(), 'image_sha256': sha(image),
             'draft_source': rel(cache), 'draft_sha256': sha(cache),
             'final_source': rel(source_case), 'final_source_sha256': sha(source_case),
             'axes_bounds': description['axes_bounds'],
             'auxiliary': {'后处理Q（已有记录）': final['metrics']['cost'],
                           '最终硬合法（已有记录）': final['metrics']['is_feasible'],
                           '完整求解秒数（原运行）': final['seconds'],
                           '完整开发集Loss（非本例）': config['full_dev_losses'],
                           '草稿来源': '冻结best51为本次视觉审阅补生成；未重跑后处理',
                           '参考列': '原任务几何，尚未应用MIB平均面积变换，不保证满足派生面积'}}
    save(info, value)
    return value


def prepare(output, seed=20261024, workers=4):
    started = time.perf_counter()
    output = output.resolve()
    if not output.is_relative_to(HERE / 'runs'):
        raise ValueError('Output must remain in E42/runs')
    if not 1 <= workers <= 4:
        raise ValueError('Use 1..4 workers, each one thread')
    if (output / 'manifest.json').exists():
        manifest = read(output / 'manifest.json')
        if manifest['selection']['seed'] != seed or manifest['checkpoint_sha256'] != SHA:
            raise ValueError('Existing round has different identity; use another output directory')
        for row in manifest['cases']:
            if sha(output / row['image']) != row['image_sha256']:
                raise ValueError('Review image changed')
        print(f'Existing complete round reused: {output}', flush=True)
        return
    source = read(SOURCE / 'config.json')
    if source['count'] != 1000 or len(source['records']) != 1000 or source['checkpoint_sha256'] != SHA:
        raise ValueError('Expected the frozen exposed E42 test1000')
    checkpoint = ROOT / source['checkpoint']
    if sha(checkpoint) != SHA:
        raise ValueError('Checkpoint changed')
    frozen = read(TRAIN / 'frozen.json')
    inference_sources = ['model.py', 'features.py', 'inference.py']
    for name in inference_sources:
        path = HERE / name
        if sha(path) != frozen['sources'][rel(path)]:
            raise ValueError(f'Frozen inference source changed: {name}')
    selected = random.Random(seed).sample(range(1000), 100)
    config = {'schema_version': 1, 'seed': seed, 'count': 100, 'role': 'local_test',
              'authorization': '2026-10-24 user explicitly chose 100 uniformly random cases from the already exposed E42 test1000 for draft ratings',
              'selection': 'Python random.Random(seed).sample(range(1000),100); sampled order is presentation order; no Q filtering',
              'source_run': rel(SOURCE), 'source_config_sha256': sha(SOURCE / 'config.json'),
              'checkpoint_sha256': SHA, 'registry_sha256': source['registry_sha256'],
              'selected_indices': selected, 'selected_members': [source['records'][i]['member'] for i in selected],
              'workers': workers, 'threads_per_worker': 1, 'inference_budget_seconds': 1200,
              'full_dev_losses': {k: read(TRAIN / 'full_dev.json')['losses'][k] for k in ('total', 'selection_proxy', 'center_local', 'shape', 'overlap', 'occupancy')},
              'frozen_sources': {rel(HERE / n): sha(HERE / n) for n in inference_sources},
              'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'git_dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True)),
              'command': subprocess.list2cmdline(__import__('sys').argv),
              'data_policy': 'These 100 already exposed test cases are now adaptive research feedback; remaining 900 sealed cases are not accessed',
              'postprocess_calls': 0, 'training_calls': 0, 'official_scoring_calls': 0}
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / 'config.json'
    if config_path.exists():
        previous = read(config_path)
        for key in ('seed', 'selected_members', 'source_config_sha256', 'checkpoint_sha256', 'frozen_sources'):
            if previous[key] != config[key]:
                raise ValueError(f'Review preparation identity changed: {key}')
        config = previous
    else:
        save(config_path, config)
    for directory in ('images', 'drafts_local', 'image_meta'):
        (output / directory).mkdir(exist_ok=True)
    tasks = [(config, {'index': i, 'record': source['records'][i]}, str(output)) for i in selected]
    results = {}
    with ProcessPoolExecutor(max_workers=workers, initializer=initialize, initargs=(checkpoint,)) as pool:
        futures = [pool.submit(render_task, task) for task in tasks]
        for future in as_completed(futures):
            row = future.result()
            results[row['source_index']] = row
            print(f'Rendered {len(results)}/100 ({time.perf_counter()-started:.1f}s)', flush=True)
            if time.perf_counter()-started > 1100:
                for pending in futures:
                    pending.cancel()
                raise TimeoutError('Visual-review preparation budget reached; completed drafts/images retained')
    cases = [results[i] for i in selected]
    manifest = {'schema_version': 1, 'session_id': output.name, 'title': 'E42 · 常规后处理 / 草稿评分',
                'created_utc': datetime.now(timezone.utc).isoformat(), 'checkpoint_sha256': SHA,
                'display_note': '随机100例，来自已暴露测试1000例。草稿为冻结模型补生成；最终布局沿用原记录。原参考处于MIB面积变换之前。',
                'rubric': {'version': 1, 'target': 'raw_draft_only', 'scale': {'1': '差', '2': '一般', '3': '好'},
                           'dimensions': 'Open-ended observation tags and verbatim notes; dimensions may evolve in later rounds'},
                'selection': {k: config[k] for k in ('seed', 'selection', 'source_run', 'source_config_sha256', 'selected_indices', 'data_policy')},
                'cases': cases}
    save(output / 'manifest.json', manifest)
    save(output / 'preparation_summary.json', {'count': 100, 'unique_members': len({tuple(r['member']) for r in cases}),
         'seconds': time.perf_counter()-started, 'postprocess_calls': 0, 'training_calls': 0,
         'official_scoring_calls': 0, 'image_sha_verified': True,
         'checkpoint_sha256': SHA, 'manifest_sha256': sha(output / 'manifest.json'),
         'source_sha256': {rel(p): sha(p) for p in [Path(__file__), ROOT/'visualization/layouts.py', ROOT/'visualization/review.py']},
         'status': 'ready_for_human_ratings', 'human_ratings_complete': False})
    print(f'Ready: {output}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--seed', type=int, default=20261024)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    prepare(args.output, args.seed, args.workers)


if __name__ == '__main__':
    main()
