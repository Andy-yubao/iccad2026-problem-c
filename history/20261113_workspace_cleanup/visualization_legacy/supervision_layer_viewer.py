"""Publish saved same-member layer traces for a pair of supervision models."""
import json
import math
import re
from pathlib import Path
import numpy as np
from .raster_layouts import module_palette
from .layer_trace import axes_layout
from .layer_budget_viewer import TEMPLATE

def publish_pair(models,folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    base=models['baseline']['cases']
    for i,case in enumerate(base):
        rectangles=[np.asarray(case['reference'])]
        for model in models.values():
            other=model['cases'][i]
            if other['member']!=case['member'] or other['n']!=case['n']:
                raise ValueError('Model traces use different train members')
            np.testing.assert_array_equal(other['reference'],case['reference'])
            if len(other['stages'])!=model['depth']+1:raise ValueError('Missing real layer readouts')
            rectangles.extend(np.asarray(s['positions']) for s in other['stages'])
        positions=np.concatenate(rectangles);lo=positions[:,:2].min(0);hi=(positions[:,:2]+positions[:,2:]).max(0)
        center=(lo+hi)/2;span=max(hi-lo)
        limits=[float(center[0]-.54*span),float(center[0]+.54*span),float(center[1]-.54*span),float(center[1]+.54*span)]
        for model in models.values():model['cases'][i]['axis_limits']=limits
    data=dict(models=models,order=list(models),alignment_steps=math.lcm(*(m['depth'] for m in models.values())),
        completed_candidates=2,training=False,test_read=False,postprocessing_calls=0)
    page=TEMPLATE.replace('固定参数预算 · 五例逐层对比','E114 · 五例逐层监督对照')
    page=re.sub(r'<h1>.*?</h1><p class="muted">.*?</p>',
        '<h1>E114：末层监督与密集监督</h1><p class="muted">相同64维/12图块/共享预测头，612307参数；原始面积、无MIB、全模块预测。固定相同五个训练成员、模块ID与颜色，来自各自唯一Loss-best；监督位置不同。</p>',page,count=1)
    page=page.replace('70模块样例高清总览','第3例高清总览')
    page=page.replace('历史MIB与固定规格路径保留以匹配A。','原始面积、无MIB；固定/预放置模块也显示模型真实预测。')
    page=page.replace('__DATA__',json.dumps(data,ensure_ascii=False,separators=(',',':'),allow_nan=False).replace('<','\\u003c'))
    page=page.replace('__PALETTE__',json.dumps(np.round(module_palette(120)*255).astype(int).tolist()))
    (folder/'index.html').write_text(page,encoding='utf-8')
    (folder/'manifest.json').write_text(json.dumps(dict(case_count=5,models={k:v['identity'] for k,v in models.items()},
        training=False,test_read=False,postprocessing_calls=0,scale_policy='Common axes include reference and every real readout of both models; no cropping'),indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,4,figsize=(17,8.6),constrained_layout=True,squeeze=False)
    for row,model in enumerate(models.values()):
        case=model['cases'][2]
        for col,(label,positions) in enumerate([('Reference',case['reference'])]+[(f'Layer {d}/12',case['stages'][d]['positions']) for d in (4,8,12)]):
            axes_layout(axes[row,col],case,positions,model['short_label']+' | '+label)
    fig.suptitle(f'E114 supervision comparison | train case3 N={base[2]["n"]} | common axes | original areas / no MIB',fontsize=12)
    fig.savefig(folder/'case03_summary.png',dpi=130,facecolor='white');plt.close(fig)

