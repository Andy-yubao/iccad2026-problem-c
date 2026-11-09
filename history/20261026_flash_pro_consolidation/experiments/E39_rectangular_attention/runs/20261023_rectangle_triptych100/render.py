"""Export fixed dev100 reference/draft/saved-final triptychs; never run a solver."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import html
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
SOURCE = ROOT/'experiments/E39_rectangular_attention/runs/20261023_rectangle_w96d12_r2_rectangle_seed0'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, data):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    os.replace(temp, path)


def initialize():
    global MODEL, DATASET
    import torch
    from datasets.reference_layouts import ReferenceLayouts
    from experiments.E39_rectangular_attention.model import GeometryModel
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    state = torch.load(SOURCE/'checkpoints/best.pt', map_location='cpu', weights_only=False)
    assert state['model_config'] == {'width': 96, 'layers': 12, 'rectangle': True}
    MODEL = GeometryModel(**state['model_config']).eval()
    MODEL.load_state_dict(state['model'], strict=True)
    DATASET = ReferenceLayouts('dev_monitor500')


def render(record):
    import numpy as np
    from experiments.E39_rectangular_attention.inference import predict
    from scoring.evaluation import evaluate_case
    from visualization.layouts import plot_case, geometry_details
    started = time.perf_counter()
    index = record['index']
    item = DATASET[index]
    assert list(item['member']) == record['member']
    case = item['case']
    reference = item['positions'].tolist()
    draft, _ = predict(MODEL, case.solver_inputs)
    raw_metrics = evaluate_case(case, draft.tolist(), record['inference_seconds'])['metrics']
    reference_metrics = evaluate_case(case, reference, 0.)['metrics']
    # Verify the regenerated draft matches the saved evaluation's numeric identity.
    fields = ('cost', 'is_feasible', 'area_violations', 'overlap_violations', 'dimension_violations',
              'boundary_violations', 'grouping_violations', 'mib_violations', 'hpwl_gap', 'area_gap')
    for field in fields:
        assert np.isclose(raw_metrics[field], record['raw_metrics'][field], rtol=1e-9, atol=1e-10), (index, field)
    constraints = np.asarray(case.solver_inputs[5]).astype(int).tolist()
    shard, row = record['member']
    name = f'{index+1:03d}_{Path(shard).parent.name}_{Path(shard).stem}_row{row:03d}_n{record["n"]}.png'
    variants = {}
    for key, positions, metrics in [('reference', reference, reference_metrics), ('raw', draft.tolist(), raw_metrics),
                                    ('final', record['positions'], record['metrics'])]:
        variants[key] = {'positions': positions, 'metrics': metrics,
                         'soft_details': geometry_details(positions, constraints)}
    metadata = plot_case({'panel': 'E39 rectangle 96x12 | frozen dev_monitor500',
        'case_id': f'{index+1:03d} | {shard} row {row}', 'constraints': constraints, 'variants': variants,
        'caption': 'Reference: original supervision (WH may violate v5 mean-area task). Draft: regenerated only. Final: saved evaluation; no solver rerun.'},
        HERE/'images'/name, stages=(('reference', 'Original reference / supervision'),
        ('raw', 'E39 rectangle 96x12 draft'), ('final', f'Saved postprocessing | Q={record["metrics"]["cost"]:.5f}')))
    return {'index': index, 'member': record['member'], 'n': record['n'], 'image': name,
            'Q': record['metrics']['cost'], 'final_seconds_original_evaluation': record['seconds'],
            'saved_raw_metric_parity': True, 'raw_positions': draft.tolist(),
            'plot': metadata, 'render_seconds': time.perf_counter()-started}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pilot', action='store_true')
    args = ap.parse_args()
    started = time.perf_counter()
    (HERE/'images').mkdir(exist_ok=True)
    frozen = read(SOURCE/'frozen.json')
    assert sha(SOURCE/'checkpoints/best.pt') == frozen['checkpoint_sha256']
    assert all(sha(ROOT/name) == value for name, value in frozen['sources'].items())
    records = sorted(read(SOURCE/'terminal/results.json')['records'], key=lambda r:r['index'])[:100]
    assert [r['index'] for r in records] == list(range(100))
    config = {'model': 'E39 rectangle 96x12 seed0 best49', 'checkpoint_sha256': frozen['checkpoint_sha256'],
              'source_terminal': (SOURCE/'terminal/results.json').relative_to(ROOT).as_posix(),
              'source_terminal_sha256': sha(SOURCE/'terminal/results.json'),
              'selection': 'Frozen dev_monitor500 indices 0..99, first100 of evaluated500; no quality selection',
              'members': [{k:r[k] for k in ('index','member','n')} for r in records],
              'stages': ['original reference supervision', 'E39 raw draft', 'saved final postprocessing'],
              'workers': 2, 'threads_per_worker': 2, 'device': 'cpu', 'training': False,
              'solver_calls': 0, 'test_read': False, 'raw_inference_only': True,
              'reference_note': 'Original reference WH may not satisfy derived v5 mean-area constraints; scored as shown without repair.',
              'sources': {str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in
                          [Path(__file__), ROOT/'visualization/layouts.py', ROOT/'experiments/E39_rectangular_attention/model.py',
                           ROOT/'experiments/E39_rectangular_attention/inference.py', ROOT/'datasets/reference_layouts.py']}}
    if (HERE/'config.json').exists():
        assert read(HERE/'config.json') == config, 'Gallery source identity changed'
    else:
        save(HERE/'config.json', config)
    evidence_path = HERE/'drafts_local.json'
    evidence = read(evidence_path) if evidence_path.exists() else []
    done = {r['index'] for r in evidence}
    selected = records[:1] if args.pilot else records
    with ProcessPoolExecutor(max_workers=2, mp_context=mp.get_context('spawn'), initializer=initialize) as pool:
        futures = {pool.submit(render, r):r['index'] for r in selected if r['index'] not in done}
        for future in as_completed(futures):
            evidence.append(future.result())
            evidence.sort(key=lambda r:r['index'])
            save(evidence_path, evidence)
            print(f'Rendered {len(evidence)}/100', flush=True)
    if args.pilot:
        print(f'Pilot complete: {time.perf_counter()-started:.2f} seconds', flush=True)
        return
    assert len(evidence) == 100
    from PIL import Image
    files = []
    for row in evidence:
        path = HERE/'images'/row['image']
        with Image.open(path) as picture:
            size = list(picture.size); picture.verify()
        assert size[0] == 3315 and size[1] in (1988, 1989), size
        files.append({'file': row['image'], 'size': size, 'sha256': sha(path)})
    assert {p.name for p in (HERE/'images').iterdir()} == {r['file'] for r in files}
    cards = '\n'.join(f'<a class="card" href="images/{html.escape(r["image"])}"><img loading="lazy" src="images/{html.escape(r["image"])}"><span>{r["index"]+1:03d} · {r["n"]} blocks · Q {r["Q"]:.5f}</span></a>' for r in evidence)
    (HERE/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>E39 100例三相图</title><style>body{font-family:system-ui;margin:24px;background:#f1f5f9;color:#172033}main{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}.card{background:white;padding:10px;border-radius:10px;color:inherit;text-decoration:none}img{width:100%;display:block}span{display:block;padding:8px}p{max-width:1100px}</style><h1>E39 矩形注意力 96×12：100例三相图</h1><p>固定开发监测500例中的前100例，未按Q挑选。每张：原始参考布局／E39草稿／已保存的后处理结果。参考布局为原始监督，宽高可能不满足v5派生面积约束。Q是开发中性时间指标。点击图片查看原图。</p><main>'+cards+'</main>', encoding='utf-8')
    save(HERE/'summary.json', {'status':'complete', 'count':100, 'all_images_decode':True,
        'saved_raw_metric_parity':all(r['saved_raw_metric_parity'] for r in evidence),
        'image_folder':str(HERE/'images'), 'source_terminal_unchanged':sha(SOURCE/'terminal/results.json') == config['source_terminal_sha256'],
        'solver_calls':0, 'training':False, 'test_read':False, 'seconds_this_invocation':time.perf_counter()-started,
        'workers':2, 'threads_per_worker':2, 'images':files})
    print('Completed and verified 100 three-stage images.', flush=True)


if __name__ == '__main__':
    main()
