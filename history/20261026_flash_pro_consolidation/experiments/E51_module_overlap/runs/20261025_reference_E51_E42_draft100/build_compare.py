"""Reference/current E51 draft/previous-best E42 draft, without ratings or solver."""
import os
for key in ('E51_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['MPLBACKEND'] = 'Agg'
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
import multiprocessing as mp
import time
import torch
from datasets.reference_layouts import ReferenceLayouts
from experiments.E51_module_overlap.io import save, sha, exclusive
from experiments.E51_module_overlap.model import GeometryModel
from experiments.E51_module_overlap.inference import predict
from visualization.layouts import plot_case, geometry_details

OUT = Path(__file__).resolve().parent
HERE = ROOT/'experiments/E51_module_overlap'
SOURCE = HERE/'runs/20261025_flash100_seed20261025'
OLD = ROOT/'experiments/E42_joint_regularization/runs/20261023_rectangle_w96d12_ab_seed0'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def initialize(checkpoint):
    global MODEL, DATA
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    MODEL = GeometryModel(**state['model_config']).eval(); MODEL.load_state_dict(state['model'])
    DATA = ReferenceLayouts('dev_monitor500')


def render(payload):
    index, old_sha = payload
    name = f'case_{index:04d}'
    image = OUT/'images'/(name+'.png')
    meta = OUT/'image_meta'/(name+'.json')
    current_path = SOURCE/f'cases/{index:04d}.json'
    if image.exists() and meta.exists():
        value = read(meta)
        if sha(image) != value['image_sha256'] or sha(current_path) != value['current_case_sha256']:
            raise ValueError('Saved comparison changed')
        return value
    item = DATA[index]; current = read(current_path)
    if list(item['member']) != current['member']: raise ValueError('Sample identity changed')
    cache = OUT/'e42_drafts_local'/(name+'.json')
    if cache.exists():
        previous = read(cache)
        if previous['member'] != current['member'] or previous['checkpoint_sha256'] != old_sha:
            raise ValueError('Cached E42 draft identity changed')
    else:
        positions, seconds = predict(MODEL, item['case'].solver_inputs)
        previous = {'member': current['member'], 'checkpoint_sha256': old_sha,
                    'positions': positions.tolist(), 'generation_seconds': seconds,
                    'purpose': 'Frozen E42 draft reconstruction for model comparison; no postprocessing'}
        save(cache, previous)
    constraints = item['case'].solver_inputs[5].tolist()
    variants = {key: {'positions': p, 'soft_details': geometry_details(p, constraints) if p is not None else {}}
                for key, p in (('reference', item['positions'].tolist()), ('current', current.get('draft_positions')),
                               ('previous', previous['positions']))}
    adapted = {'panel': 'Reference | E51 current draft | E42 previous-best draft', 'case_id': name,
        'constraints': constraints, 'variants': variants,
        'caption': 'Same case, module IDs and coordinate scale. Reference is before MIB mean-area transform. Both drafts use the same public task.'}
    description = plot_case(adapted, image, show_metrics=False, stages=(('reference', 'Reference'),
        ('current', 'E51 best52 draft / current'), ('previous', 'E42 best51 draft / previous best')))
    value = {'id': name, 'source_index': index, 'member': current['member'], 'n': current['n'],
        'image': image.relative_to(OUT).as_posix(), 'image_sha256': sha(image),
        'current_case_source': current_path.relative_to(ROOT).as_posix(), 'current_case_sha256': sha(current_path),
        'previous_draft_source': cache.relative_to(OUT).as_posix(), 'previous_draft_sha256': sha(cache),
        'axes_bounds': description['axes_bounds']}
    save(meta, value); return value


def build():
    started = time.perf_counter()
    source = read(SOURCE/'config.json')['identity']; old = read(OLD/'frozen.json')
    checkpoint = ROOT/old['checkpoint']
    if sha(checkpoint) != old['checkpoint_sha256']: raise ValueError('Frozen E42 checkpoint changed')
    # Only inference semantics matter here: no training/selection code is executed.
    common = ('model.py', 'features.py', 'inference.py', 'batches.py')
    semantics = {}
    for name in common:
        current_path = HERE/name; previous_path = ROOT/'experiments/E42_joint_regularization'/name
        a, b = sha(current_path), sha(previous_path)
        if a != b or b != old['sources'][previous_path.relative_to(ROOT).as_posix()]:
            raise ValueError('Frozen inference semantics mismatch: '+name)
        semantics[name] = a
    indices = source['selected_indices']
    config = {'source_run': SOURCE.relative_to(ROOT).as_posix(), 'source_config_sha256': sha(SOURCE/'config.json'),
        'current_model': 'E51 unified_v2 best52', 'current_checkpoint_sha256': source['checkpoint_sha256'],
        'previous_model': 'E42 joint_AB best51 (accepted previous best)', 'previous_checkpoint_sha256': old['checkpoint_sha256'],
        'count': 100, 'selected_indices': indices, 'selected_members': source['selected_members'], 'seed': 20261025,
        'role': 'v5 dev_monitor500; same exact cases/order as completed flash100', 'inference_semantics': semantics,
        'runner_sha256': sha(Path(__file__)), 'renderer_sha256': sha(ROOT/'visualization/layouts.py'),
        'workers': 4, 'threads_per_worker': 1, 'current_draft_source': 'saved actual E51 flash100 draft, no regeneration',
        'previous_draft_source': 'frozen E42 forward only on the same public inputs',
        'columns': ['reference', 'E51 draft', 'E42 previous-best draft'], 'rating_ui': False,
        'postprocessor_calls': 0, 'training_calls': 0, 'test_read': False}
    with exclusive(OUT/'build.lock'):
        if (OUT/'config.json').exists():
            if read(OUT/'config.json') != config: raise ValueError('Comparison identity changed')
        else: save(OUT/'config.json', config)
        for name in ('images', 'image_meta', 'e42_drafts_local'): (OUT/name).mkdir(exist_ok=True)
        save(OUT/'progress.json', {'status': 'building', 'completed': 0, 'target': 100})
        with ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context('spawn'), initializer=initialize,
                                 initargs=(checkpoint,)) as pool:
            cases = []
            for value in pool.map(render, [(i, old['checkpoint_sha256']) for i in indices]):
                cases.append(value)
                save(OUT/'progress.json', {'status': 'building', 'completed': len(cases), 'target': 100,
                                          'seconds': time.perf_counter()-started})
                if len(cases)%10 == 0: print(f'Comparison images {len(cases)}/100', flush=True)
                if time.perf_counter()-started > 1100: raise TimeoutError('Preparation budget reached; saved images retained')
        save(OUT/'manifest.json', {'title': 'Reference / E51 draft / E42 draft', 'created_utc': datetime.now(timezone.utc).isoformat(),
                                 'selection': config, 'cases': cases, 'rating_ui': False})
        make_page(cases)
        save(OUT/'progress.json', {'status': 'complete', 'completed': 100, 'target': 100,
                                  'seconds': time.perf_counter()-started, 'test_read': False})


def make_page(cases):
    html = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Reference · E51 draft · E42 draft</title><style>
*{box-sizing:border-box}body{margin:0;background:#f4f5f7;color:#17212b;font:15px system-ui,"Microsoft YaHei",sans-serif}
header{padding:14px 20px;background:white;border-bottom:1px solid #dde2e7}h1{font-size:19px;margin:0 0 9px}
.bar{display:flex;align-items:center;gap:12px;flex-wrap:wrap}button,select,a{font:inherit}button,select,.link{padding:7px 12px;border:1px solid #cbd5df;border-radius:7px;background:white;color:#17212b;cursor:pointer;text-decoration:none}
.note{color:#65717d;font-size:13px;margin-top:8px}.viewport{height:calc(100vh - 147px);overflow:auto;text-align:center;padding:10px}
img{display:block;max-width:100%;max-height:100%;width:auto;height:auto;margin:auto;background:white;cursor:zoom-in}.zoom img{max-width:none;max-height:none;width:2200px;cursor:zoom-out}
</style><header><h1>Reference → 本轮 E51 draft → 此前最优 E42 draft</h1><div class="bar">
<button id="prev">← 上一例</button><select id="choose" aria-label="选择样本"></select><button id="next">下一例 →</button><span id="label"></span>
<button id="zoom">放大 / 适应窗口</button><a id="original" class="link" target="_blank">打开原图</a></div>
<div class="note">同一批100个开发样本，同一模块编号与坐标尺度。点击图片放大，方向键切换。Reference 是 MIB 平均面积变换前的原始参考。</div></header>
<main class="viewport" id="viewport"><img id="figure" alt="Reference、本轮草稿与此前最优草稿的同例对比"></main>
<script>const cases=__CASES__;let index=Math.max(0,Math.min(cases.length-1,Number(localStorage.getItem('e51-e42-view-index')||0)));const choose=document.getElementById('choose');
cases.forEach((r,i)=>{const o=document.createElement('option');o.value=i;o.textContent=`${i+1} / ${cases.length} · ${r.id}`;choose.append(o)});
function show(){const r=cases[index];document.getElementById('figure').src=r.image;document.getElementById('original').href=r.image;document.getElementById('label').textContent=`${r.n} 个模块`;choose.value=index;localStorage.setItem('e51-e42-view-index',index);document.getElementById('viewport').scrollTo(0,0)}
document.getElementById('prev').onclick=()=>{index=(index+cases.length-1)%cases.length;show()};document.getElementById('next').onclick=()=>{index=(index+1)%cases.length;show()};choose.onchange=()=>{index=Number(choose.value);show()};
function zoom(){document.getElementById('viewport').classList.toggle('zoom')}document.getElementById('zoom').onclick=zoom;document.getElementById('figure').onclick=zoom;
document.addEventListener('keydown',e=>{if(e.target.tagName==='SELECT')return;if(e.key==='ArrowLeft')document.getElementById('prev').click();if(e.key==='ArrowRight')document.getElementById('next').click()});show();</script></html>'''
    (OUT/'index.html').write_text(html.replace('__CASES__', json.dumps(cases, ensure_ascii=False)), encoding='utf-8')


if __name__ == '__main__':
    build()
