"""Saved reference/current/E42 draft comparison; no ratings or postprocessing."""
import argparse,json,multiprocessing as mp,os,random
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from .dataset_view import ReferenceLayouts
from visualization.layouts import plot_case,geometry_details
from .io import HERE,ROOT,save,sha

PREVIOUS=ROOT/'experiments/E74_collision_32k_posttrain/runs/20261029_e42_topology_seed0/terminal/results.json'

def initialize(folder):
    global ROWS,DATA,PREVIOUS_ROWS
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    evidence=json.loads((Path(folder)/'terminal/results.json').read_text(encoding='utf-8'))
    if 'reused_from' in evidence:
        source=(ROOT/evidence['reused_from']).resolve()
        if not source.is_relative_to(ROOT) or sha(source)!=evidence['source_sha256']:raise ValueError('Reused terminal source changed')
        evidence=json.loads(source.read_text(encoding='utf-8'))
    ROWS={r['index']:r for r in evidence['records']}
    DATA=ReferenceLayouts('dev_monitor500')
    PREVIOUS_ROWS={r['index']:r for r in json.loads(PREVIOUS.read_text(encoding='utf-8'))['records']}

def render(payload):
    index,folder=payload;folder=Path(folder);row=ROWS[index];item=DATA[index]
    if list(item['member'])!=row['member']:raise ValueError('Draft member mismatch')
    old=PREVIOUS_ROWS[index]
    if old['member']!=row['member']:raise ValueError('E74 member mismatch')
    constraints=item['case'].solver_inputs[5].tolist()
    variants={k:{'positions':p,'soft_details':geometry_details(p,constraints) if p is not None else {}} for k,p in
              (('reference',item['positions'].tolist()),('current',row.get('draft_positions')),('previous',old['draft_positions']))}
    target=folder/f'images/case_{index:04d}.png'
    desc=plot_case({'panel':'E91 '+folder.parent.name+' E74 architecture, random32k training, fixed E90 Loss','case_id':f'case_{index:04d}','constraints':constraints,'variants':variants,
        'caption':'Original reference before mean-area transform | Current model draft | Frozen E74 best42 draft'},
        target,show_metrics=False,stages=(('reference','Reference'),('current','Current draft'),('previous','E74 best42 draft')))
    return {'id':index,'n':row['n'],'member':row['member'],'image':target.relative_to(folder).as_posix(),
            'image_sha256':sha(target),'axes_bounds':desc['axes_bounds'],'Q':row['metrics']['cost']}

def run(folder):
    folder=Path(folder).resolve();target=folder/'draft_comparison100'
    if (target/'manifest.json').exists():return
    (target/'images').mkdir(parents=True,exist_ok=True)
    ids=random.Random(20261025).sample(range(500),100)
    save(target/'config.json',{'seed':20261025,'indices':ids,'source_sha256':sha(folder/'terminal/results.json'),
        'columns':['reference','current draft','frozen E74 draft'],'rating_ui':False,'test_read':False})
    os.environ['E74_THREADS']='1'
    try:
        with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=initialize,initargs=(str(folder),)) as pool:
            cases=list(pool.map(render,[(i,str(target)) for i in ids]))
    finally:os.environ['E74_THREADS']='4'
    save(target/'manifest.json',{'cases':cases,'rating_ui':False,'test_read':False})
    images='\n'.join(f'<figure><figcaption>{j+1}/100 · case_{r["id"]:04d} · {r["n"]} modules</figcaption><a href="{r["image"]}" target="_blank"><img loading="lazy" src="{r["image"]}"></a></figure>' for j,r in enumerate(cases))
    (target/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>E91 draft comparison</title><style>body{font:16px system-ui;background:#eef2f5;margin:20px}img{width:100%;background:white}figure{margin:16px 0}figcaption{padding:8px;background:white}</style><h1>Reference → 当前 draft → E74 best42 draft</h1><p>同一批100个开发样本；点击打开原图。没有评分控件。</p>'+images,encoding='utf-8')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);run(p.parse_args().output)
