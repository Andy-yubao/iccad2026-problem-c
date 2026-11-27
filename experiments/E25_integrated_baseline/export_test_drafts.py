"""Authorized v5 local-test first100: raw drafts only, four one-thread workers."""
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[name] = '1'
os.environ['OMP_MAX_ACTIVE_LEVELS'] = '1'
import json
import time
import hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import numpy as np
import torch
from datasets.paths import ROOT
from datasets.reference_layouts import REGISTRY, registry
from datasets.public_member_inputs import public_member_inputs
from visualization.layouts import plot_case, geometry_details
from .cases import predict, WEIGHTS
from .model import GeometryModel

HERE = Path(__file__).resolve().parent
OUT = HERE / 'runs/20261022_test100_draft_gallery'
IMAGES = OUT / 'images'
MODEL = None


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(temp, path)


def initialize():
    global MODEL
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    state = torch.load(WEIGHTS, map_location='cpu', weights_only=False)
    MODEL = GeometryModel(**state['model_config']).eval()
    MODEL.load_state_dict(state['model'])


def render(record):
    constraints = np.asarray(record['constraints'])
    plot_case({'panel': f'v5 local test | draft {record["ordinal"]:03d}/100',
               'case_id': record['case_id'], 'constraints': constraints.tolist(),
               'caption': 'Frozen E25 model output. Module IDs, grouping colors and fixed/preplaced markers are shown.',
               'variants': {'draft': {'positions': record['draft'],
                                      'soft_details': geometry_details(record['draft'], constraints)}}},
              IMAGES / record['image'], stages=(('draft', 'Raw model draft'),), show_metrics=False)


def execute(ordinal, record, deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError('15-minute cumulative export budget reached')
    inputs = public_member_inputs(record, 'local_test', authorize_local_test=True)
    draft, seconds = predict(MODEL, inputs)
    assert np.isfinite(draft).all() and (draft[:, 2:] > 0).all()
    shard, row = record['member']
    image = f'{ordinal:03d}_{Path(shard).parent.name}_{Path(shard).stem}_row{row:03d}_n{record["n"]}.png'
    result = {'ordinal': ordinal, 'case_id': record['case_id'], 'member': record['member'], 'n': record['n'],
              'image': image, 'draft': draft.tolist(), 'constraints': np.asarray(inputs[5]).tolist(),
              'inference_seconds': seconds, 'pid': os.getpid(), 'torch_threads': torch.get_num_threads()}
    render(result)
    return result


def main():
    started = time.monotonic()
    IMAGES.mkdir(parents=True, exist_ok=True)
    os.environ['MPLCONFIGDIR'] = str(ROOT / '.cache/matplotlib')
    records = registry()['sets']['local_test']['records'][:100]
    identity = {'authorization': '2026-10-22 user requested 100 test-set draft images only',
                'selection': 'first100 in frozen v5 local_test record order; no quality filtering',
                'registry_sha256': sha(REGISTRY), 'checkpoint_sha256': sha(WEIGHTS),
                'members': records, 'sources': {p.relative_to(ROOT).as_posix(): sha(p) for p in
                    [Path(__file__), HERE/'cases.py', HERE/'features.py', HERE/'model.py',
                     ROOT/'datasets/public_member_inputs.py', ROOT/'visualization/layouts.py']},
                'workers': 4, 'threads_per_worker': 1, 'postprocessing': False, 'expansion': False,
                'scoring': False, 'free_reference_geometry_indexed': False,
                'test_exposure': 'these 100 members are now exposed through visual model-output inspection; remaining tests sealed'}
    assert identity['checkpoint_sha256'] == '3de22809e831edf11b56a85647a0bd9fe625e8b7617abb6eb1f4b27969cfe00a'
    config = OUT/'config.json'
    if config.exists():
        assert json.loads(config.read_text(encoding='utf-8')) == identity
    else:
        save(config, identity)
    drafts_path = OUT/'drafts_local.json'
    results = json.loads(drafts_path.read_text(encoding='utf-8')) if drafts_path.exists() else []
    prior = json.loads((OUT/'progress.json').read_text(encoding='utf-8')) if (OUT/'progress.json').exists() else {'elapsed_seconds': 0}
    deadline = started + 900 - prior['elapsed_seconds']
    done = {r['ordinal'] for r in results}
    for r in results:
        if not (IMAGES/r['image']).exists():
            render(r)
    try:
        with ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context('spawn'), initializer=initialize) as pool:
            pending = {pool.submit(execute, k+1, r, deadline): k+1 for k, r in enumerate(records) if k+1 not in done}
            for f in as_completed(pending):
                results.append(f.result())
                results.sort(key=lambda r: r['ordinal'])
                save(drafts_path, results)
                save(OUT/'progress.json', {'completed': len(results), 'elapsed_seconds': prior['elapsed_seconds']+time.monotonic()-started})
                if len(results) % 10 == 0:
                    print(f'Draft images {len(results)}/100', flush=True)
    finally:
        save(OUT/'progress.json', {'completed': len(results), 'elapsed_seconds': prior['elapsed_seconds']+time.monotonic()-started})
    assert len(results) == len(list(IMAGES.glob('*.png'))) == 100
    from PIL import Image
    for r in results:
        with Image.open(IMAGES/r['image']) as image:
            image.verify()
    save(OUT/'summary.json', {'status': 'complete', 'count': 100, 'images': [r['image'] for r in results],
                             'image_folder': str(IMAGES), 'workers': 4, 'threads_per_worker': 1,
                             'pids': sorted({r['pid'] for r in results}), 'all_images_decode': True,
                             'postprocessing': False, 'expansion': False, 'scoring': False,
                             'elapsed_seconds': prior['elapsed_seconds']+time.monotonic()-started})
    print(str(IMAGES), flush=True)


if __name__ == '__main__':
    main()
