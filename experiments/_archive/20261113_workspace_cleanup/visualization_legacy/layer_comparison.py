"""Display already saved layer layouts; contains no network or optimizer code."""
import json
from pathlib import Path

import numpy as np
from shapely.geometry import box
from shapely.ops import unary_union

from .raster_layouts import module_palette
from .layer_trace import axes_layout


def measures(positions):
    p=np.asarray(positions,dtype=np.float64)
    total=p[:,2:].prod(1).sum()
    union=unary_union([box(x,y,x+w,y+h) for x,y,w,h in p]).area
    bbox=float(np.prod((p[:,:2]+p[:,2:]).max(0)-p[:,:2].min(0)))
    return dict(repeated_coverage_fraction=float(max(0,total-union)/total),bbox_over_area=bbox/float(total))


def build(models, folder):
    folder=Path(folder)
    folder.mkdir(parents=True,exist_ok=False)
    baseline=models['control']['cases']
    for number,base in enumerate(baseline):
        rectangles=[np.asarray(base['reference'])]
        for model in models.values():
            case=model['cases'][number]
            if case['member']!=base['member'] or case['n']!=base['n']:
                raise ValueError('The five members must match across all checkpoints')
            np.testing.assert_allclose(case['reference'],base['reference'],atol=3e-4,rtol=1e-6)
            for stage in case['stages']:rectangles.append(np.asarray(stage['positions']))
        p=np.concatenate(rectangles)
        low=p[:,:2].min(0);high=(p[:,:2]+p[:,2:]).max(0)
        span=max(high-low);center=(low+high)/2
        limits=[float(center[0]-.54*span),float(center[0]+.54*span),float(center[1]-.54*span),float(center[1]+.54*span)]
        for model in models.values():
            model['cases'][number]['axis_limits']=limits
            model['cases'][number]['reference']=base['reference']
    data=dict(models=models,model_order=['control','shared','heads','progressive12','progressive4','progressive8'],
              case_count=5,test_read=False,postprocessing_calls=0,training=False,
              scale_policy='One common square range per case covers reference and every frame of all six checkpoint identities; no cropping')
    compact=json.dumps(data,ensure_ascii=False,separators=(',',':'),allow_nan=False).replace('<','\\u003c')
    (folder/'index.html').write_text(TEMPLATE.replace('__DATA__',compact).replace('__PALETTE__',json.dumps(np.round(module_palette(120)*255).astype(int).tolist())),encoding='utf-8')
    (folder/'manifest.json').write_text(json.dumps(dict(case_count=5,models={key:value['identity'] for key,value in models.items()},
        test_read=False,training=False,postprocessing_calls=0,scale_policy=data['scale_policy']),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    render_summary(models,folder/'case03_comparison.png')
    return data


def render_summary(models,destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(4,4,figsize=(17,17),constrained_layout=True)
    for row,key in enumerate(('control','shared','heads','progressive12')):
        model=models[key];case=model['cases'][2]
        panels=[('Reference',case['reference'])]+[(f'Layer {d}',case['stages'][d]['positions']) for d in (4,8,12)]
        for column,(title,p) in enumerate(panels):
            axes_layout(axes[row,column],case,p,model['short_label']+' | '+title)
    fig.suptitle('Same N=70 training case | common axes and module colors | saved best checkpoints | no postprocessing',fontsize=13)
    fig.savefig(destination,dpi=130,facecolor='white');plt.close(fig)


TEMPLATE=r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>五例逐层布局对比</title>
<style>*{box-sizing:border-box}body{margin:0;background:#f3f5fa;color:#17253a;font:15px/1.6 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1720px;margin:auto;padding:24px}h1{margin:0;font-size:28px}h2{font-size:17px;margin:4px 0 10px}p{margin:8px 0}.muted{color:#5d6c80;font-size:13px}.box{background:white;border:1px solid #dfe6f0;border-radius:12px;padding:16px;margin:14px 0}.controls{display:flex;align-items:center;gap:14px;flex-wrap:wrap}select,button,input[type=number]{font:inherit;padding:7px 10px;background:white;color:inherit;border:1px solid #cbd5e1;border-radius:6px}button{cursor:pointer}button:hover{background:#edf2ff}input[type=range]{min-width:220px;flex:1;accent-color:#2563eb}.views{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}.views.compare{grid-template-columns:repeat(auto-fit,minmax(260px,1fr))}.views section{min-width:0}canvas.layout{width:100%;height:auto;display:block;background:#fbfcff;border:1px solid #e5ebf4;border-radius:6px}.strip{display:grid;grid-template-columns:repeat(13,minmax(0,1fr));gap:5px}.thumb{padding:3px;border:2px solid transparent;border-radius:5px;min-width:0}.thumb.selected{border-color:#2563eb}.thumb canvas{display:block;width:100%;height:auto}.thumb span{display:block;text-align:center;font-size:12px}.note{border-left:4px solid #d29922;background:#fffbeb;padding:10px 14px}.mono{font-variant-numeric:tabular-nums}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:8px;border-bottom:1px solid #e2e8f0;text-align:right}th:first-child,td:first-child{text-align:left}details{margin-top:12px}summary{cursor:pointer}a{color:#1d4ed8}.badge{font-size:12px;background:#eaf0fe;padding:3px 8px;border-radius:5px}.metrics{margin-top:6px;font-size:13px;font-variant-numeric:tabular-nums;min-height:42px}@media(max-width:900px){.views{grid-template-columns:1fr}.strip{grid-template-columns:repeat(7,minmax(0,1fr))}main{padding:12px}}
</style><main><span class="badge">5个相同训练样例 · 冻结best · 真实逐层读出</span><h1>不同训练方法，各层学出了什么布局？</h1>
<p class="muted">沿用之前N=21／45／70／95／120的五例，比较原E98、A、B和分阶段组。模块ID、颜色和坐标尺度在所有模型与层间保持一致；不显示后处理三相图。</p>
<div class="note muted">这是五个训练样例的结构观察，不是未见测试。历史模型的MIB及固定／预放置解码行为忠实保留。A/B的第1–12层有直接监督；原E98与分阶段组只有训练时末层监督。分阶段组的“4层阶段best”“8层阶段best”是不同训练时点，不能与最终12层模型的前缀混为一谈。</div>
<div class="box"><div class="controls"><label>样例 <select id="case"></select></label><label>模型 <select id="model"></select></label><label>查看方式 <select id="view"><option value="detail">当前模型逐层查看</option><option value="compare">原E98／A／B／最终分阶段组并排</option></select></label></div>
<div class="controls" style="margin-top:14px"><button id="prev">上一层</button><button id="play">播放</button><button id="next">下一层</button><strong id="depthLabel"></strong><input type="range" id="depth" min="0" max="12" value="0"></div>
<div class="controls" style="margin-top:12px"><label><input type="checkbox" id="ids"> 模块编号</label><label><input type="checkbox" id="red" checked> 重叠红框</label><label>追踪模块 <input type="number" id="module" min="-1" value="-1" style="width:80px"></label><span class="muted">−1为全部；也可点击模块查看中心轨迹。左右方向键切层。</span></div><p id="source" class="muted"></p><p id="identity" class="muted"></p></div>
<div class="box"><div id="views" class="views"></div><p class="muted">同一模块始终同色；红框表示与其他模块重叠；选中模块的线条连接各次真实中心读出。参考图只用于观察。</p></div>
<div class="box"><h2 id="stripTitle">全部层输出</h2><div id="strip" class="strip"></div><p class="muted">初始读出发生在任何图块之前。层数切换不会改变本例的坐标尺度。分阶段组的4／8层checkpoint只展示其实际开放层。</p></div>
<details class="box"><summary>查看各层重复覆盖与中心步长</summary><div style="overflow:auto"><table><thead><tr><th>层</th><th>重复覆盖</th><th>框面积/模块面积</th><th>解码中心步长</th><th>隐藏特征RMS</th></tr></thead><tbody id="metrics"></tbody></table></div><p class="muted">重复覆盖=(模块面积和−占用并集)/模块面积和；中心步长按√总有效面积归一化。所有数值只描述保存的草稿，不是Q分数。</p></details>
<p class="muted"><a href="case03_comparison.png">查看70模块样例的4／8／12层高清总览</a> · 初始及中间布局来自只读hooks，带／不带hooks最终输出逐位一致，模型和checkpoint均未修改。无训练、修复或评分重跑。</p></main>
<script id="data" type="application/json">__DATA__</script><script>
(()=>{const D=JSON.parse(document.getElementById('data').textContent),M=D.models,palette=__PALETTE__,el=id=>document.getElementById(id);let ci=2,key='shared',depth=0,timer=null;const fixed=['control','shared','heads','progressive12'];
const C=k=>M[k].cases[ci],stage=(k,d)=>C(k).stages[Math.min(d,M[k].depth)],fmt=(x,n=3)=>Number.isFinite(x)?x.toFixed(n):'—';
D.model_order.forEach(k=>{let o=document.createElement('option');o.value=k;o.textContent=M[k].label;el('model').append(o)});M.control.cases.forEach((c,i)=>{let o=document.createElement('option');o.value=i;o.textContent=`样例 ${i+1} · N=${c.n}`;el('case').append(o)});el('model').value=key;el('case').value=ci;
function stop(){if(timer)clearInterval(timer);timer=null;el('play').textContent='播放'}
function bounds(canvas){let L=C(key).axis_limits,pad=canvas.width*.045,s=(canvas.width-2*pad)/(L[1]-L[0]);return{L,pad,s,x:v=>pad+(v-L[0])*s,y:v=>canvas.height-pad-(v-L[2])*s}}
function hits(p){let s=new Set;for(let i=0;i<p.length;i++)for(let j=0;j<i;j++){let a=p[i],b=p[j];if(Math.min(a[0]+a[2],b[0]+b[2])-Math.max(a[0],b[0])>1e-6&&Math.min(a[1]+a[3],b[1]+b[3])-Math.max(a[1],b[1])>1e-6){s.add(i);s.add(j)}}return s}
function draw(canvas,p,k,d,thumb=false){let ctx=canvas.getContext('2d'),v=bounds(canvas),c=C(k),hit=el('red').checked?hits(p):new Set,sel=+el('module').value;ctx.clearRect(0,0,canvas.width,canvas.height);ctx.strokeStyle='#dfe7f1';ctx.lineWidth=1;ctx.strokeRect(v.pad,v.pad,canvas.width-2*v.pad,canvas.height-2*v.pad);p.forEach((r,i)=>{let x=v.x(r[0]),y=v.y(r[1]+r[3]),w=r[2]*v.s,h=r[3]*v.s;ctx.globalAlpha=sel>=0&&i!==sel ? .28 : .78;ctx.fillStyle=`rgb(${palette[i].join(',')})`;ctx.fillRect(x,y,w,h);ctx.strokeStyle=hit.has(i)?'#d32636':'#475569';ctx.lineWidth=thumb?1:hit.has(i)?1.7:.7;ctx.strokeRect(x,y,w,h);if(!thumb&&el('ids').checked){ctx.globalAlpha=1;ctx.font='17px system-ui';ctx.textAlign='center';ctx.fillStyle='#15253a';ctx.fillText(String(i),x+w/2,y+h/2+4)}if(sel===i&&!thumb){ctx.globalAlpha=1;ctx.strokeStyle='#0f172a';ctx.lineWidth=3;ctx.strokeRect(x,y,w,h)}});ctx.globalAlpha=1;if(!thumb&&sel>=0&&sel<c.n){ctx.strokeStyle='#0f172a';ctx.lineWidth=2;ctx.beginPath();c.stages.forEach((f,i)=>{let r=f.positions[sel],x=v.x(r[0]+r[2]/2),y=v.y(r[1]+r[3]/2);i?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.stroke();let r=stage(k,d).positions[sel];ctx.fillStyle='#d32636';ctx.beginPath();ctx.arc(v.x(r[0]+r[2]/2),v.y(r[1]+r[3]/2),6,0,Math.PI*2);ctx.fill()}if(!thumb){ctx.fillStyle='#64748b';ctx.font='14px system-ui';ctx.textAlign='left';ctx.fillText(`${fmt(v.L[0],1)}, ${fmt(v.L[2],1)}`,v.pad,canvas.height-8);ctx.textAlign='right';ctx.fillText(`${fmt(v.L[1],1)}, ${fmt(v.L[3],1)}`,canvas.width-v.pad,24)}}
function pick(canvas,p,event){let r=canvas.getBoundingClientRect(),v=bounds(canvas),x=((event.clientX-r.left)*canvas.width/r.width-v.pad)/v.s+v.L[0],y=(canvas.height-v.pad-(event.clientY-r.top)*canvas.height/r.height)/v.s+v.L[2],found=-1;p.forEach((a,i)=>{if(x>=a[0]&&x<=a[0]+a[2]&&y>=a[1]&&y<=a[1]+a[3])found=i});el('module').value=found;render()}
function panel(title,k,d,reference=false){let section=document.createElement('section'),heading=document.createElement('h2'),cv=document.createElement('canvas'),line=document.createElement('div');heading.textContent=title;cv.width=900;cv.height=900;cv.className='layout';cv.setAttribute('role','img');cv.setAttribute('aria-label',title);line.className='metrics';let p=reference?C(k).reference:stage(k,d).positions;section.append(heading,cv,line);el('views').append(section);draw(cv,p,k,d);cv.onclick=e=>pick(cv,p,e);if(!reference){let m=stage(k,d).metrics;line.textContent=`重复覆盖 ${fmt(m.repeated_coverage_fraction*100,2)}% · 框/面积 ${fmt(m.bbox_over_area)} · 中心步长 ${fmt(m.center_step_mean,4)}`}else line.textContent='原始reference；未进入网络输入'}
function render(){let compare=el('view').value==='compare',max=compare?12:M[key].depth;depth=Math.min(depth,max);el('depth').max=max;el('depth').value=depth;el('depthLabel').textContent=depth===0?'初始读出':`第 ${depth} 层之后`;el('module').max=C(key).n-1;el('source').textContent=`原train索引 ${C(key).train_index} ｜ ${C(key).member[0]} : ${C(key).member[1]} ｜ 所有模型/阶段/层共用同一坐标范围`;el('identity').textContent=`${M[key].label} ｜ own-Loss best epoch ${M[key].identity.best_epoch} ｜ ${M[key].note}`;let views=el('views');views.innerHTML='';views.className='views'+(compare?' compare':'');panel('原始reference',key,depth,true);if(compare){for(let k of fixed)panel(M[k].short_label+' · '+(depth===0?'初始':`第${depth}层`),k,depth)}else{panel(el('depthLabel').textContent,key,depth);panel(`固定最终输出 · 第${M[key].depth}层`,key,M[key].depth)}el('stripTitle').textContent=M[key].short_label+' · 全部层输出';el('strip').innerHTML='';C(key).stages.forEach((f,i)=>{let button=document.createElement('button');button.className='thumb'+(i===depth?' selected':'');let cv=document.createElement('canvas'),label=document.createElement('span');cv.width=210;cv.height=210;label.textContent=i===0?'初始':`层 ${i}`;button.append(cv,label);button.onclick=()=>{stop();depth=i;render()};el('strip').append(button);draw(cv,f.positions,key,i,true)});el('metrics').innerHTML=C(key).stages.map(f=>{let m=f.metrics;return `<tr${f.depth===depth?' style="background:#edf2ff"':''}><td>${f.depth===0?'初始':f.depth}</td><td>${fmt(m.repeated_coverage_fraction*100,2)}%</td><td>${fmt(m.bbox_over_area)}</td><td>${fmt(m.center_step_mean,4)}</td><td>${fmt(m.hidden_rms)}</td></tr>`}).join('')}
el('case').onchange=()=>{stop();ci=+el('case').value;el('module').value=-1;render()};el('model').onchange=()=>{stop();key=el('model').value;if(!fixed.includes(key))el('view').value='detail';depth=Math.min(depth,M[key].depth);render()};el('view').onchange=()=>{stop();if(el('view').value==='compare'&&!fixed.includes(key)){key='progressive12';el('model').value=key}render()};el('depth').oninput=()=>{stop();depth=+el('depth').value;render()};el('prev').onclick=()=>{stop();depth=Math.max(0,depth-1);render()};el('next').onclick=()=>{stop();depth=Math.min(+el('depth').max,depth+1);render()};el('play').onclick=()=>{if(timer){stop();return}let max=+el('depth').max;if(depth===max)depth=0;el('play').textContent='暂停';render();timer=setInterval(()=>{if(depth===max){stop();return}depth++;render()},900)};['ids','red','module'].forEach(id=>el(id).onchange=render);document.addEventListener('keydown',event=>{if(['INPUT','SELECT'].includes(event.target.tagName))return;if(event.key==='ArrowLeft')el('prev').click();if(event.key==='ArrowRight')el('next').click()});render();
})();</script></html>'''
