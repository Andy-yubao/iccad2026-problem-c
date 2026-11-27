"""Fixed100 four stages, using saved source P42 and own unique terminal output."""
import argparse,json,multiprocessing as mp,os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from .io import HERE,ROOT,save,sha
from .dataset_view import ReferenceLayouts
from visualization.layouts import plot_case,geometry_details,PLACER_CORRECTOR_STAGES
CACHE=ROOT/'history/20261026_flash_pro_consolidation/experiments/E51_module_overlap/runs/20261025_reference_E51_E42_draft100/e42_drafts_local'
CONTROL=ROOT/'experiments/E81_multisource_refinement/runs/20261029_e42_multisource_seed0/draft_comparison100/manifest.json'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def initialize(folder):
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    global DATA,ROWS,FOLDER
    FOLDER=Path(folder);DATA=ReferenceLayouts('dev_monitor500');ROWS={r['index']:r for r in read(FOLDER/'terminal/results.json')['records']}
def render(index):
    item=DATA[index];r=ROWS[index];old=read(CACHE/f'case_{index:04d}.json');member=list(item['member'])
    assert old['member']==r['member']==member
    assert old['checkpoint_sha256']=='12878108e9a58b8c195b314ba6586686c2a6e2796c1f4b6443e102b4db4224a6'
    c=item['case'].solver_inputs[5].tolist();positions={'reference':item['positions'].tolist(),'placer':old['positions'],'corrected':r['draft_positions'],'final':r['positions']}
    p=np.concatenate([np.asarray(v) for v in positions.values()]);lo=p[:,:2].min(0);hi=(p[:,:2]+p[:,2:]).max(0);margin=.055*float((hi-lo).max())
    limits=[float(lo[0]-margin),float(hi[0]+margin),float(lo[1]-margin),float(hi[1]+margin)]
    target=FOLDER/f'draft_comparison100/images/case_{index:04d}.png';target.parent.mkdir(parents=True,exist_ok=True)
    desc=plot_case({'panel':'E84 same-initialization G-only correction control','case_id':f'case_{index:04d}','constraints':c,
      'axis_limits':limits,'variants':{k:{'positions':v,'soft_details':geometry_details(v,c)} for k,v in positions.items()},
      'caption':'Reference before public area transform | Frozen P42 | G-only trained F including MIB/frame operators | Unique saved repair'},
      target,stages=PLACER_CORRECTOR_STAGES,show_metrics=False)
    return {'id':int(index),'n':int(r['n']),'member':member,'image':target.relative_to(FOLDER/'draft_comparison100').as_posix(),
      'image_sha256':sha(target),'axis_limits':limits,'axes_bounds':desc['axes_bounds']}
def run(folder):
    folder=Path(folder).resolve();target=folder/'draft_comparison100'
    if (target/'manifest.json').exists():return
    fixed=[r['id'] for r in read(CONTROL)['cases']];assert len(fixed)==100
    old=os.environ.get('E84_THREADS');os.environ['E84_THREADS']='1'
    try:
        with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=initialize,initargs=(str(folder),)) as pool:cases=list(pool.map(render,fixed))
    finally:
        if old is None:os.environ.pop('E84_THREADS',None)
        else:os.environ['E84_THREADS']=old
    save(target/'manifest.json',{'cases':cases,'fixed100':fixed,'source_sha256':sha(folder/'terminal/results.json'),'columns':['reference','placer','corrected','final'],'rating_ui':False,'test_read':False})
    body='<h1>E84：参考 → 布局器 → G-only修正 → 后处理</h1><p>原固定100例、同例统一尺度，无评分控件；数据来自唯一best的保存结果。</p>'
    body+=''.join(f'<figure><figcaption>case_{r["id"]:04d} · {r["n"]}模块</figcaption><a href="{r["image"]}"><img loading="lazy" src="{r["image"]}"></a></figure>' for r in cases)
    (target/'index.html').write_text('<!doctype html><meta charset="utf-8"><style>body{font:16px system-ui;background:#eef2f5;margin:20px}img{width:100%}figure{margin:15px 0;background:white}</style>'+body,encoding='utf-8')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);run(p.parse_args().output)
