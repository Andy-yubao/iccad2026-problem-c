"""Four fixed requested stages from saved current coordinates; no planner solve."""
import json
from pathlib import Path
import numpy as np
from visualization.layouts import plot_case,geometry_details
from ...deformation import Config as PreviousConfig,preprocess as previous_AR
from ..R16.replay import ROOT,EXPERIMENT,RESEARCH,read,save,sha

PRIOR=EXPERIMENT/'runs/20261102_deform_v3_dev100/config.json'
STAGES=(('raw','Raw C3 draft'),('previous','Previous best AR'),('current','Current retained AR'),('final','Final layout'))


def draw(record,out,index,case_prefix):
    coords=ROOT/record['coordinate_source'];assert sha(coords)==record['coordinates_sha256']
    with np.load(coords) as a:
        raw=a['raw'];inputs=(record['n'],*[a[k] for k in ('areas','b2b','p2b','pins','constraints','specs')])
        cfg=PreviousConfig(**read(PRIOR)['plan']['config']);prior,_=previous_AR(raw,inputs,config=cfg)
        c=a['constraints'].astype(int).tolist()
        positions=dict(raw=raw,previous=prior,current=a['prefix'],final=a['final'])
        variants={k:dict(positions=v.tolist(),soft_details=geometry_details(v,c)) for k,v in positions.items()}
    out.mkdir(parents=True,exist_ok=True);path=out/f'{index:04d}.png'
    plot=plot_case(dict(case_id=f'{case_prefix}:{index}',panel='Raw / previous AR / current AR / final',constraints=c,variants=variants),path,stages=STAGES,show_metrics=False)
    prior_path=out/f'{index:04d}_previous.npz'
    with prior_path.open('wb') as f:np.savez_compressed(f,previous=prior)
    item=dict(index=index,member=record['member'],image=path.name,image_sha256=sha(path),current_coordinates_sha256=record['coordinates_sha256'],previous_config_sha256=sha(PRIOR),previous_coordinates_sha256=sha(prior_path),plot=plot)
    save(path.with_suffix('.json'),item);return item


def render(out,records):
    folder=out/'four_stage100';folder.mkdir(parents=True,exist_ok=True);assets=[]
    for index in range(100):
        if (RESEARCH/'STOP').exists() or (out/'STOP').exists():save(folder/'state.json',dict(status='stopped',completed=len(assets),total=100));return False
        record=records[index];assert record['index']==index;sidecar=folder/f'{index:04d}.json'
        if sidecar.exists():
            item=read(sidecar);assert sha(folder/item['image'])==item['image_sha256'] and item['current_coordinates_sha256']==record['coordinates_sha256'] and item['previous_config_sha256']==sha(PRIOR)
        else:item=draw(record,folder,index,'dev500')
        assets.append(item);save(folder/'state.json',dict(status='running',completed=len(assets),total=100))
        if (index+1)%25==0:print('R23 four-stage images',index+1,'/100',flush=True)
    save(folder/'manifest.json',dict(count=100,stages=[s[0] for s in STAGES],same_IDs_and_scale=True,score_controls=False,no_extra_planner_solve=True,prior_config_sha256=sha(PRIOR),assets=assets))
    html='<!doctype html><meta charset="utf-8"><title>原稿／此前最优AR／当前AR／最终布局</title><style>body{font:16px system-ui;max-width:1800px;margin:auto}img{width:100%}</style><h1>原始草稿／此前最优AR／当前AR／当前完整流程最终布局</h1><p>固定同100例、同模块ID和尺度。此前V3配置在同一新草稿上重放；当前为冻结保留的R15强吸引方案。无评分控件。</p>'
    html+=''.join(f'<p>dev500:{i}</p><img loading="lazy" src="{i:04d}.png">' for i in range(100))
    (folder/'index.html').write_text(html,encoding='utf-8');save(folder/'state.json',dict(status='complete',completed=100,total=100));return True
