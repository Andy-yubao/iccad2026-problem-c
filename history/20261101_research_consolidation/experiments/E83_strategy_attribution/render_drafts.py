"""Four-stage fixed100 display from new results and authenticated saved sources."""
import multiprocessing as mp,os
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from visualization.layouts import plot_case,geometry_details,PLACER_CORRECTOR_STAGES
from .io import HERE,ROOT,save,sha
from .prepare import read
from .dataset_view import ReferenceLayouts

OLD=ROOT/'experiments/E81_multisource_refinement/runs/20261029_e42_multisource_seed0'
SOURCE74=ROOT/'experiments/E74_collision_32k_posttrain/runs/20261029_e42_topology_seed0/terminal/results.json'
CACHE42=ROOT/'history/20261026_flash_pro_consolidation/experiments/E51_module_overlap/runs/20261025_reference_E51_E42_draft100/e42_drafts_local'

def initialize():
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    global DATA,ARMS,ROWS,SAVED74
    DATA=ReferenceLayouts('dev_monitor500');ARMS=read(HERE/'plan.json')['arms']
    ROWS={a['name']:{r['index']:r for r in read(HERE/'runs'/a['name']/'terminal/results.json')['records']} for a in ARMS}
    SAVED74={r['index']:r for r in read(SOURCE74)['records']}

def render(index):
    item=DATA[index];member=list(item['member']);constraints=item['case'].solver_inputs[5].tolist();ref=item['positions'].tolist()
    old42=read(CACHE42/f'case_{index:04d}.json')
    assert old42['member']==member and old42['checkpoint_sha256']=='12878108e9a58b8c195b314ba6586686c2a6e2796c1f4b6443e102b4db4224a6'
    assert SAVED74[index]['member']==member
    variants={}
    for a in ARMS:
        row=ROWS[a['name']][index];assert row['member']==member
        variants[a['name']]={'reference':ref,'placer':old42['positions'] if a['placer']=='P42' else SAVED74[index]['draft_positions'],
            'corrected':row['draft_positions'],'final':row['positions']}
    p=np.concatenate([np.asarray(v) for rows in variants.values() for v in rows.values()]);lo=p[:,:2].min(0);hi=(p[:,:2]+p[:,2:]).max(0);margin=.055*float((hi-lo).max())
    limits=[float(lo[0]-margin),float(hi[0]+margin),float(lo[1]-margin),float(hi[1]+margin)];result={}
    for a in ARMS:
        rows=variants[a['name']];target=HERE/'runs'/a['name']/f'four_stage100/images/case_{index:04d}.png'
        target.parent.mkdir(parents=True,exist_ok=True)
        stages=PLACER_CORRECTOR_STAGES if a['learned_residuals'] else tuple((k,'Deterministic operators only' if k=='corrected' else title) for k,title in PLACER_CORRECTOR_STAGES)
        desc=plot_case({'panel':'E83 '+a['name'],'case_id':f'case_{index:04d}','constraints':constraints,'axis_limits':limits,
          'variants':{k:{'positions':v,'soft_details':geometry_details(v,constraints)} for k,v in rows.items()},
          'caption':'Reference before public area transform | Raw placer | Fixed intervention | Unique saved postprocessing; scores hidden'},
          target,stages=stages,show_metrics=False)
        result[a['name']]={'image':target.relative_to(HERE/'runs').as_posix(),'sha256':sha(target),'axes_bounds':desc['axes_bounds']}
    return {'id':index,'member':member,'n':item['case'].block_count,'axis_limits':limits,'models':result}

def main():
    target=HERE/'runs/gallery_manifest.json'
    if target.exists():return
    fixed=[r['id'] for r in read(OLD/'draft_comparison100/manifest.json')['cases']]
    assert len(fixed)==100
    old=os.environ.get('E83_THREADS');os.environ['E83_THREADS']='1'
    try:
        with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=initialize) as pool:cases=list(pool.map(render,fixed))
    finally:
        if old is None:os.environ.pop('E83_THREADS',None)
        else:os.environ['E83_THREADS']=old
    save(target,{'cases':cases,'fixed100':fixed,'scores_hidden':True,'solver_calls':0,'test_read':False,'source74_sha256':sha(SOURCE74)})
    body='<h1>E83策略归因：同100例四阶段图</h1><p>参考 → 布局器 → 固定干预后的草稿 → 后处理。第三列在P42组只含确定性操作，不含神经修正。原固定100例；同例四列、两个组合共用尺度，无评分控件。</p><p><a href="report.md">配对结果报告</a></p>'
    for case in cases:
        body+=f'<h2>case_{case["id"]:04d} · {case["n"]}模块</h2>'
        for name,v in case['models'].items():body+=f'<figure><figcaption>{name}</figcaption><a href="{v["image"]}"><img loading="lazy" src="{v["image"]}"></a></figure>'
    (HERE/'runs/index.html').write_text('<!doctype html><meta charset="utf-8"><title>E83 strategy attribution</title><style>body{font:16px system-ui;background:#eef2f5;margin:20px}img{width:100%}figure{margin:15px 0;background:white}figcaption{padding:10px}</style>'+body,encoding='utf-8')
    print('E83 four-stage fixed100 gallery complete',flush=True)

if __name__=='__main__':main()
