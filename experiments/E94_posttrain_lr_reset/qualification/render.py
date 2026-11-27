"""Same100 saved reference/E94/E90 drafts, shared renderer, no rating controls."""
import json,multiprocessing as mp,os,random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from ..io import ROOT,save,sha
from ..dataset_view import ReferenceLayouts
from .common import registration

def initialize(folder):
    global CURRENT,PREVIOUS,DATA,TARGET
    import matplotlib
    matplotlib.use('Agg')
    folder=Path(folder);TARGET=folder/'draft_comparison100'
    CURRENT={r['index']:r for r in json.loads((folder/'terminal/results.json').read_text(encoding='utf-8'))['records']}
    reg=registration();name=next(n for n in reg['baseline']['source_files'] if n.replace('\\','/').endswith('terminal/results.json'))
    PREVIOUS={r['index']:r for r in json.loads((ROOT/name).read_text(encoding='utf-8'))['records']}
    DATA=ReferenceLayouts('dev_monitor500')

def render(index):
    from visualization.layouts import plot_case,geometry_details
    sidecar=TARGET/f'case_{index:04d}.json'
    if sidecar.exists():
        row=json.loads(sidecar.read_text(encoding='utf-8'))
        if sha(TARGET/row['image'])!=row['image_sha256']:raise ValueError('Gallery image drift')
        return row
    current,previous=CURRENT[index],PREVIOUS[index];item=DATA[index]
    if current['member']!=previous['member'] or current['member']!=list(item['member']):
        raise ValueError('Gallery member mismatch')
    constraints=item['case'].solver_inputs[5].tolist()
    values=[('reference',item['positions'].tolist()),('current',current.get('draft_positions')),('previous',previous.get('draft_positions'))]
    variants={key:{'positions':p,'soft_details':geometry_details(p,constraints) if p is not None else {}} for key,p in values}
    target=TARGET/f'images/case_{index:04d}.png'
    description=plot_case({'panel':'E94 frozen own-best vs E90 best12','case_id':f'case_{index:04d}',
        'constraints':constraints,'variants':variants,
        'caption':'Original reference before mean-area transform | Frozen E94 own-best draft | Frozen E90 best12 draft'},
        target,show_metrics=False,stages=(('reference','Reference'),('current','E94 layout draft'),('previous','E90 best12 draft')))
    row={'index':index,'member':current['member'],'n':current['n'],'image':target.relative_to(TARGET).as_posix(),
         'image_sha256':sha(target),'axes_bounds':description['axes_bounds'],'rating_ui':False,
         'missing_current_draft':current.get('draft_positions') is None}
    save(sidecar,row);return row

def run(folder,halt=lambda:False):
    folder=Path(folder);target=folder/'draft_comparison100';(target/'images').mkdir(parents=True,exist_ok=True)
    reg=registration();name=next(n for n in reg['baseline']['source_files'] if n.replace('\\','/').endswith('terminal/results.json'))
    ids=random.Random(20261025).sample(range(500),100)
    config={'seed':20261025,'indices':ids,'current_source_sha256':sha(folder/'terminal/results.json'),
            'previous_source_sha256':sha(ROOT/name),'columns':['reference','E94 own-best draft','E90 best12 draft'],
            'rating_ui':False,'test_read':False}
    config_path=target/'config.json'
    if config_path.exists() and json.loads(config_path.read_text(encoding='utf-8'))!=config:
        raise ValueError('Cannot replace gallery identity')
    save(config_path,config)
    if (target/'manifest.json').exists():return
    old=os.environ.get('E94_EVAL_THREADS');os.environ['E94_EVAL_THREADS']='1';cases=[]
    try:
        with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=initialize,initargs=(str(folder),)) as pool:
            # Batches permit budget/STOP checks and resume without redrawing finished rows.
            for start in range(0,100,4):
                if halt():
                    from .run import EvaluationStop
                    raise EvaluationStop()
                cases.extend(pool.map(render,ids[start:start+4]))
    finally:
        if old is None:os.environ.pop('E94_EVAL_THREADS',None)
        else:os.environ['E94_EVAL_THREADS']=old
    save(target/'manifest.json',{'config':config,'cases':cases,'rating_ui':False,'test_read':False})
    figures='\n'.join(f'<figure><figcaption>{j+1}/100 · case_{r["index"]:04d} · {r["n"]} modules</figcaption><a href="{r["image"]}"><img loading="lazy" src="{r["image"]}"></a></figure>' for j,r in enumerate(cases))
    (target/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>E94 draft comparison</title><style>body{font:16px system-ui;background:#eef2f5;margin:20px}img{width:100%;background:white}figure{margin:16px 0}figcaption{padding:8px;background:white}</style><h1>Reference → E94 draft → E90 draft</h1><p>固定同例100张，无评分控件；缺失草稿明确显示，不筛除失败例。</p>'+figures,encoding='utf-8')
