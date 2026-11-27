"""Four-stage figures from frozen source/saved correction/saved postprocessing."""
import json,sys,os,multiprocessing as mp
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import numpy as np
from experiments.E81_multisource_refinement.io import sha,save
from experiments.E81_multisource_refinement.dataset_view import ReferenceLayouts
from experiments.E81_multisource_refinement.runs.placer_corrector_audit import OUT,PATHS,SOURCE
from visualization.layouts import plot_case,geometry_details,PLACER_CORRECTOR_STAGES
CACHE=ROOT/'history/20261026_flash_pro_consolidation/experiments/E51_module_overlap/runs/20261025_reference_E51_E42_draft100/e42_drafts_local'
MODELS=('E79','E80','E81','E82')
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def initialize():
    global ROWS,DATA,COORD,METRICS,LOOKUP
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    ROWS={n:{r['index']:r for r in read(PATHS[n]/'terminal/results.json')['records']} for n in MODELS+('E74',)}
    DATA=ReferenceLayouts('dev_monitor500');METRICS=read(OUT/'source_metrics.json')
    LOOKUP={tuple(m):i for i,m in enumerate(METRICS['members'])}
    pack=np.load(OUT/'source_drafts.npz');COORD={n:pack[n] for n in ('E42','E74')}

def render(index):
    item=DATA[index];member=list(item['member']);i=LOOKUP[tuple(member)];n=item['case'].block_count
    cached=CACHE/f'case_{index:04d}.json'
    if cached.exists():
        old=read(cached);assert old['member']==member and old['checkpoint_sha256']=='12878108e9a58b8c195b314ba6586686c2a6e2796c1f4b6443e102b4db4224a6'
        p42=old['positions'];source42={'kind':'existing frozen E42 saved coordinates','path':cached.relative_to(ROOT).as_posix(),'sha256':sha(cached)}
    else:p42=COORD['E42'][i,:n].tolist();source42={'kind':'read-only frozen source inference','path':'source_drafts.npz','index':i}
    p74=ROWS['E74'][index]['draft_positions'];assert ROWS['E74'][index]['member']==member
    constraints=item['case'].solver_inputs[5].tolist();reference=item['positions'].tolist()
    layouts={}
    for name in MODELS:
        saved=ROWS[name][index];assert saved['member']==member and saved['n']==n
        layouts[name]={'reference':reference,'placer':p74 if SOURCE[name]=='E74' else p42,
                       'corrected':saved['draft_positions'],'final':saved['positions']}
    allp=np.concatenate([np.asarray(p) for row in layouts.values() for p in row.values()]);lo=allp[:,:2].min(0);hi=(allp[:,:2]+allp[:,2:]).max(0)
    margin=float((hi-lo).max())*.055;limits=[float(lo[0]-margin),float(hi[0]+margin),float(lo[1]-margin),float(hi[1]+margin)]
    result={}
    for name,stages in layouts.items():
        variants={key:{'positions':p,'soft_details':geometry_details(p,constraints)} for key,p in stages.items()}
        target=OUT/f'four_stage/{name}/images/case_{index:04d}.png';target.parent.mkdir(parents=True,exist_ok=True)
        desc=plot_case({'panel':f'{name}: reference / {SOURCE[name]} placer / correction / saved repair',
            'case_id':f'case_{index:04d}','constraints':constraints,'variants':variants,'axis_limits':limits,
            'caption':'Reference before public area transform | Decoded placer before F | Full correction incl. MIB/frame projection | Existing postprocessed output'},
             target,stages=PLACER_CORRECTOR_STAGES,show_metrics=False)
        assert len(desc['axes_bounds'])==4 and desc['axis_limits']==limits
        result[name]={'image':target.relative_to(OUT).as_posix(),'image_sha256':sha(target),'axes_bounds':desc['axes_bounds'],
           'stage_source_fields':['reference dataset positions',SOURCE[name]+' raw placer','draft_positions','positions'],
           'source_coordinates':{'E42':source42,'E74':{'kind':'existing saved E74 source draft','path':(PATHS['E74']/'terminal/results.json').relative_to(ROOT).as_posix()}},
           'saved_outputs':(PATHS[name]/'terminal/results.json').relative_to(ROOT).as_posix()}
    return {'id':index,'member':member,'n':n,'axis_limits':limits,'models':result}

def main():
    if not (OUT/'audit.json').exists():raise ValueError('Complete numeric source/identity audit first')
    old=read(PATHS['E81']/'draft_comparison100/manifest.json');fixed=[r['id'] for r in old['cases']]
    for name in MODELS:assert [r['id'] for r in read(PATHS[name]/'draft_comparison100/manifest.json')['cases']]==fixed
    targeted=[164,443,329,219,21];ids=list(dict.fromkeys(fixed+targeted))
    os.environ['E81_THREADS']='1'
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=initialize) as pool:
        cases=list(pool.map(render,ids))
    manifest={'role':'New four-stage read-only gallery; historical triptychs unchanged','fixed100':fixed,'targeted':targeted,
      'cases':cases,'models':list(MODELS),'source_for_model':SOURCE,'stages':[list(r) for r in PLACER_CORRECTOR_STAGES],
      'rating_controls':False,'new_model_or_solver_calls':0,'test_read':False,'renderer_sha256':sha(Path(__file__)),
      'shared_visualization_sha256':sha(ROOT/'visualization/layouts.py'),'audit_sha256':sha(OUT/'audit.json')}
    save(OUT/'four_stage_manifest.json',manifest)
    config=json.dumps({'fixed':fixed,'targeted':targeted,'cases':{str(r['id']):r for r in cases}},ensure_ascii=False).replace('</','<\\/')
    html='''<!doctype html><meta charset="utf-8"><title>布局模型与修正模型：四阶段核查</title>
<style>body{font:16px system-ui,sans-serif;background:#eef2f5;color:#172533;margin:20px}header{position:sticky;top:0;background:#eef2f5ed;padding:12px 0;z-index:2}select,button{font:inherit;padding:7px;margin-right:12px}figure{margin:20px 0;background:white;border-radius:6px}figcaption{padding:10px}img{width:100%;display:block}a{color:#125b9c}.note{color:#536575;font-size:14px}</style>
<h1>参考布局 → 布局网络 → 修正网络 → 后处理结果</h1>
<p>实验与自动跟进已暂停。四列共用模块编号和坐标尺度；切换模型时，同一个案例也保持相同尺度。点击图片打开原图。</p>
<p class="note">第三列包含修正模型原有的同形组协调与框内投影；第四列只复用此前保存的后处理结果。参考图为公共面积变换前的原参考。没有评分控件。</p>
<header><label>修正方案 <select id="model"><option>E81</option><option>E82</option><option>E79</option><option>E80</option></select></label>
<label>案例组 <select id="group"><option value="fixed">原固定100例</option><option value="targeted">定向核查5例</option></select></label>
<a href="report.html">核查结论与分层统计</a></header><main id="gallery"></main>
<script>const DATA=CONFIG;const model=document.getElementById('model'),group=document.getElementById('group');function show(){const ids=DATA[group.value];document.getElementById('gallery').innerHTML=ids.map((id,j)=>{const r=DATA.cases[id],v=r.models[model.value];return `<figure><figcaption>${j+1}/${ids.length} · case_${String(id).padStart(4,'0')} · ${r.n} 个模块 · ${model.value}</figcaption><a href="${v.image}" target="_blank"><img loading="lazy" src="${v.image}" alt="参考、布局网络、修正网络、已有后处理四列对比"></a></figure>`}).join('')}model.onchange=show;group.onchange=show;show();</script>'''.replace('CONFIG',config)
    (OUT/'index.html').write_text(html,encoding='utf-8')
    print('Four-stage gallery complete:',len(cases),'unique cases x4 model settings; fixed100 retained.',flush=True)
if __name__=='__main__':main()
