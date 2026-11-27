"""Four saved stages, consistent IDs/scale, no scoring controls or auto-open."""
from html import escape
import json
import numpy as np
from .io import save,sha
from .data import load_rows
from visualization.layouts import plot_case,PLACER_CORRECTOR_STAGES,geometry_details

def render(output,records,terminal_records,halt=lambda:False):
    folder=output/'evaluation/gallery';folder.mkdir(exist_ok=True)
    raw={tuple(r['member']):r for r in records};terminal={tuple(r['member']):r for r in terminal_records}
    rows=load_rows('dev_monitor500',list(range(100)));manifest=[]
    stages=(('reference','Reference'),('placer','Frozen E90 draft'),('corrected','E96 F x1'),('final','Frozen flash'))
    for index,row in enumerate(rows):
        if halt():return
        key=tuple(row['member']);r=raw[key];t=terminal[key];g=row['graph'];constraints=g['constraints'].tolist()
        variants=dict(reference=dict(positions=(g['reference']*g['scale']).tolist()),placer=dict(positions=r['placer']['positions']),
            corrected=dict(positions=r['corrected']['positions']),final=dict(positions=t['positions']))
        for variant in variants.values():variant['soft_details']=geometry_details(variant['positions'],constraints)
        value=dict(case_id=f'dev500:{index}',panel='E96 / '+output.name+' / one F call',constraints=constraints,variants=variants)
        path=folder/f'{index:03d}.png'
        if not path.exists():plot_case(value,path,stages=stages,show_metrics=False)
        manifest.append(dict(index=index,member=list(key),image=path.name,sha256=sha(path),
            source_checkpoint_sha256=sha(output/'checkpoints/best.pt'),stages=['reference','E90','F1','flash']))
    save(folder/'manifest.json',dict(count=100,cases=manifest,score_controls=False,shared_IDs_and_scale=True,test_read=False))
    payload=json.dumps(manifest,ensure_ascii=False).replace('<','\\u003c')
    html='''<!doctype html><meta charset="utf-8"><title>E96: one-call correction</title>
<style>body{font:16px system-ui;background:#eef2f5;margin:20px}nav{position:sticky;top:0;background:white;padding:12px}img{width:100%;background:white}button,select{font:inherit;padding:6px}p{margin:8px}</style>
<nav><b>Reference / E90 draft / F ×1 / flash</b><p>固定同100例；同模块ID、同坐标尺度。原reference为派生面积变换前标签，只作展示。</p><button id="prev">上一例</button><select id="pick"></select><button id="next">下一例</button><span id="member"></span></nav><img id="picture">
<script>const rows=__DATA__;let i=0;const pick=document.getElementById('pick');rows.forEach((r,j)=>pick.add(new Option(j+' / '+r.member.join(':'),j)));function show(){pick.value=i;document.getElementById('picture').src=rows[i].image;document.getElementById('member').textContent=rows[i].member.join(':')}pick.onchange=()=>{i=+pick.value;show()};document.getElementById('prev').onclick=()=>{i=Math.max(0,i-1);show()};document.getElementById('next').onclick=()=>{i=Math.min(99,i+1);show()};show()</script>'''
    (folder/'index.html').write_text(html.replace('__DATA__',payload),encoding='utf-8')
