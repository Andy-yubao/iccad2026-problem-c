"""Four stages from frozen saved coordinates, shared limits across both arms."""
import argparse,json,random,multiprocessing as mp,os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from visualization.layouts import plot_case,geometry_details,PLACER_CORRECTOR_STAGES
from .dataset_view import ReferenceLayouts
from .io import HERE,ROOT,save,sha
from .pipeline import OUTPUTS
P74=ROOT/'experiments/E74_collision_32k_posttrain/runs/20261029_e42_topology_seed0/terminal/results.json'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def initialize():
    global DATA,BASE,RESULTS
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    DATA=ReferenceLayouts('dev_monitor500');BASE={r['index']:r for r in read(P74)['records']}
    RESULTS={arm:{r['index']:r for r in read(folder/'terminal/results.json')['records']} for arm,folder in OUTPUTS.items()}
def render(payload):
    arm,index,folder=payload;folder=Path(folder);item=DATA[index];record=RESULTS[arm][index];base=BASE[index]
    assert list(item['member'])==record['member']==base['member']
    reference=item['positions'].tolist();all_positions=[reference,base['draft_positions']]
    for rows in RESULTS.values():all_positions += [rows[index]['draft_positions'],rows[index]['positions']]
    all_positions=np.concatenate([np.asarray(a) for a in all_positions]);lo=all_positions[:,:2].min(0);hi=(all_positions[:,:2]+all_positions[:,2:]).max(0);pad=max(hi-lo)*.055
    limits=[float(lo[0]-pad),float(hi[0]+pad),float(lo[1]-pad),float(hi[1]+pad)]
    c=item['case'].solver_inputs[5].tolist()
    variants={k:dict(positions=positions,soft_details=geometry_details(positions,c)) for k,positions in [('reference',reference),('placer',base['draft_positions']),('corrected',record['draft_positions']),('final',record['positions'])]}
    target=folder/('images/case_'+str(index).zfill(4)+'.png')
    desc=plot_case(dict(panel='E85 '+arm,case_id='case_'+str(index).zfill(4),constraints=c,variants=variants,axis_limits=limits,caption='Original reference | Frozen E74 | E85 '+arm+' correction | Saved flash output; adaptive dev, no ratings'),target,show_metrics=False,stages=PLACER_CORRECTOR_STAGES)
    return dict(id=index,n=record['n'],member=record['member'],image=target.relative_to(folder).as_posix(),image_sha256=sha(target),axis_limits=limits,axes_bounds=desc['axes_bounds'])
def run(output):
    output=Path(output).resolve();arm=next(a for a,f in OUTPUTS.items() if f.resolve()==output);folder=output/'four_stage100'
    if (folder/'manifest.json').exists():return
    (folder/'images').mkdir(parents=True,exist_ok=True);ids=random.Random(20261025).sample(range(500),100)
    old=os.environ.get('E85_THREADS');os.environ['E85_THREADS']='1'
    try:
        with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=initialize) as pool:cases=list(pool.map(render,[(arm,i,str(folder)) for i in ids]))
    finally:
        if old is None:os.environ.pop('E85_THREADS',None)
        else:os.environ['E85_THREADS']=old
    save(folder/'manifest.json',dict(cases=cases,seed=20261025,indices=ids,source_sha256=sha(output/'terminal/results.json'),placer_source_sha256=sha(P74),columns=['reference','E74 original output','E85 correction output','saved flash output'],common_scale_across_arms=True,rating_ui=False,test_read=False))
    images=''.join('<figure><figcaption>'+str(j+1)+'/100 · case_'+str(r['id'])+' · '+str(r['n'])+' modules</figcaption><a target="_blank" href="'+r['image']+'"><img loading="lazy" src="'+r['image']+'"></a></figure>' for j,r in enumerate(cases))
    (folder/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>E85 '+arm+'</title><style>body{font:16px system-ui;background:#eef2f5;margin:20px}img{width:100%;background:white}figure{margin:16px 0}figcaption{padding:8px;background:white}</style><h1>Reference → E74 → '+arm+' → 保存的后处理</h1><p>固定同100例；两组共用坐标尺度、模块编号；无评分控件。</p>'+images,encoding='utf-8')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);run(p.parse_args().output)
