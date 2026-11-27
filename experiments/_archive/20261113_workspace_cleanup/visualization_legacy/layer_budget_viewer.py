"""Reusable saved-layout viewer for models with different outer block counts."""
import argparse,json,math
from pathlib import Path

import numpy as np
from .raster_layouts import module_palette
from .layer_trace import axes_layout

NAMES=('E112_deep_geometry_budget','E113_deep_head_budget')


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def baseline(root):
    trace=read(root/'experiments/E109_deep_supervision_shared/runs/20261107_shared_seed0/layer_trace5_20261107/trace.json')
    return dict(label='A基线 · 12图块 · 共享头逐层监督',short_label='A baseline',depth=12,parameters=1471539,
                identity=trace['identity'],cases=trace['cases'])


def candidate(plan,trace_path):
    trace=read(trace_path);depth=trace['identity']['active_depth']
    kind=plan['critical_subnetwork']['kind']
    return dict(label=('几何关系网络加深' if kind=='geometry' else '共享预测头加深')+f' · {depth}图块',
                short_label='Deep geometry' if kind=='geometry' else 'Deep head',depth=depth,parameters=plan['parameters'],
                identity=trace['identity'],cases=trace['cases'])


def publish_candidate(root,owner,plan,trace_path):
    root=Path(root)
    models={'baseline':baseline(root),plan['experiment']:candidate(plan,trace_path)}
    write_viewer(models,root/'visualization/critical_budget'/('20261107_'+plan['experiment']))
    publish_available(root)


def publish_available(root,require_complete=False):
    root=Path(root);models={'baseline':baseline(root)}
    for name in NAMES:
        owner=root/'experiments'/name;plan=read(owner/'plan.json')
        path=owner/'runs'/plan['run']/'layer_trace5_20261107/trace.json'
        if path.exists():models[name]=candidate(plan,path)
    if require_complete and len(models)!=3:raise RuntimeError('Both authorized candidate traces required')
    write_viewer(models,root/'visualization/critical_budget/20261107_pair')


def write_viewer(models,folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    base=models['baseline']['cases']
    for index,ref in enumerate(base):
        rectangles=[np.asarray(ref['reference'])]
        for model in models.values():
            case=model['cases'][index]
            if case['member']!=ref['member'] or len(case['stages'])!=model['depth']+1:
                raise ValueError('Saved model member/depth mismatch')
            np.testing.assert_allclose(case['reference'],ref['reference'],rtol=1e-6,atol=3e-4)
            rectangles.extend(np.asarray(frame['positions']) for frame in case['stages'])
        p=np.concatenate(rectangles);lo=p[:,:2].min(0);hi=(p[:,:2]+p[:,2:]).max(0)
        center=(hi+lo)/2;span=max(hi-lo)
        limits=[float(center[0]-.54*span),float(center[0]+.54*span),float(center[1]-.54*span),float(center[1]+.54*span)]
        for model in models.values():
            model['cases'][index]['axis_limits']=limits;model['cases'][index]['reference']=ref['reference']
    data=dict(models=models,order=list(models),alignment_steps=math.lcm(*(m['depth'] for m in models.values())),
              completed_candidates=len(models)-1,training=False,test_read=False,postprocessing_calls=0)
    text=json.dumps(data,ensure_ascii=False,separators=(',',':'),allow_nan=False).replace('<','\\u003c')
    page=TEMPLATE.replace('__DATA__',text).replace('__PALETTE__',json.dumps(np.round(module_palette(120)*255).astype(int).tolist()))
    (folder/'index.html').write_text(page,encoding='utf-8')
    (folder/'manifest.json').write_text(json.dumps(dict(completed_candidates=len(models)-1,case_count=5,
        identities={k:m['identity'] for k,m in models.items()},depths={k:m['depth'] for k,m in models.items()},
        alignment='Real nearest readout at the same outer-block completion fraction; not interpolation or equal FLOPs',
        training=False,test_read=False),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    render_summary(models,folder/'case03_summary.png')


def render_summary(models,destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(len(models),4,figsize=(17,4.3*len(models)),constrained_layout=True,squeeze=False)
    for row,model in enumerate(models.values()):
        case=model['cases'][2];n=model['depth']
        panels=[('Reference',case['reference'])]+[(f'Layer {k}/{n}',case['stages'][k]['positions']) for k in (round(n/3),round(2*n/3),n)]
        for col,(title,p) in enumerate(panels):axes_layout(axes[row,col],case,p,model['short_label']+' | '+title)
    fig.suptitle('Parameter-budget comparison | same N=70 train case | common axes | real readouts at 1/3, 2/3 and full depth',fontsize=12)
    fig.savefig(destination,dpi=130,facecolor='white');plt.close(fig)


TEMPLATE=r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>固定参数预算 · 五例逐层对比</title>
<style>*{box-sizing:border-box}body{margin:0;background:#f4f6fa;color:#17253a;font:15px/1.6 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1640px;margin:auto;padding:24px}h1{margin:0;font-size:27px}h2{font-size:17px;margin:4px 0 8px}.box{padding:16px;border:1px solid #dfe6ef;border-radius:10px;background:white;margin:14px 0}.muted{font-size:13px;color:#5b6b80}.controls{display:flex;flex-wrap:wrap;align-items:center;gap:14px}select,button,input[type=number]{font:inherit;background:white;color:inherit;padding:7px 9px;border:1px solid #cbd5e1;border-radius:6px}button{cursor:pointer}input[type=range]{flex:1;min-width:230px;accent-color:#2563eb}.views{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:14px}.views section{min-width:0}canvas{display:block;width:100%;height:auto;background:#fbfcff}.caption{min-height:46px;margin-top:7px;font-size:13px;font-variant-numeric:tabular-nums}.strip{display:grid;grid-template-columns:repeat(auto-fit,minmax(65px,1fr));gap:5px}.thumb{padding:3px;border:2px solid transparent}.thumb.active{border-color:#2563eb}.thumb span{display:block;font-size:12px;text-align:center}.note{padding:10px 14px;border-left:4px solid #ca8a04;background:#fffbeb}.table{width:100%;border-collapse:collapse;font-size:13px}.table th,.table td{padding:7px;border-bottom:1px solid #e2e8f0;text-align:right}.table td:first-child,.table th:first-child{text-align:left}a{color:#1d4ed8}@media(max-width:650px){main{padding:12px}.views{grid-template-columns:1fr}}
</style><main><h1>同参数预算：把计算放进更深的关键网络</h1><p class="muted">A基线12个外层图块；两个新结构各8个图块，总参数与A相差不到0.4%。沿用之前相同五个训练样例、原任务条件、模块ID和配色；全部来自唯一Loss-best。</p>
<p class="note muted">并排模式按外层图块完成比例对齐，显示最近的真实层，不插值，也不表示等FLOPs。切换模型可逐个实际层查看。模型、层间共用同一坐标尺度；仅展示原始草稿，无后处理或Q控件。</p>
<div class="box"><div class="controls"><label>样例 <select id="case"></select></label><label>模型 <select id="model"></select></label><label>查看方式 <select id="mode"><option value="detail">所选模型逐层</option><option value="compare">按外层完成比例并排</option></select></label></div><div class="controls" style="margin-top:14px"><button id="prev">上一帧</button><button id="play">播放</button><button id="next">下一帧</button><strong id="stepLabel"></strong><input type="range" id="step" min="0" value="0"></div><div class="controls" style="margin-top:12px"><label><input type="checkbox" id="ids"> 模块编号</label><label><input type="checkbox" id="red" checked> 重叠红框</label><label>追踪模块 <input id="module" type="number" value="-1" min="-1" style="width:80px"></label><span class="muted">−1为全部；可点击模块查看轨迹。</span></div><p class="muted" id="source"></p><p class="muted" id="identity"></p></div>
<div class="box"><div class="views" id="views"></div></div><div class="box"><h2 id="stripTitle"></h2><div class="strip" id="strip"></div></div>
<details class="box"><summary>各层几何数值</summary><table class="table"><thead><tr><th>实际层</th><th>重复覆盖</th><th>框/面积</th><th>中心步长</th></tr></thead><tbody id="numbers"></tbody></table></details><p class="muted"><a href="case03_summary.png">70模块样例高清总览</a> · 五例用于结构观察，不是独立泛化结论。历史MIB与固定规格路径保留以匹配A。训练、验证和展示身份分开保存。</p></main>
<script id="data" type="application/json">__DATA__</script><script>
(()=>{const D=JSON.parse(document.getElementById('data').textContent),M=D.models,P=__PALETTE__,el=id=>document.getElementById(id);let ci=2,key=D.order[1]||'baseline',step=0,timer=null;const C=k=>M[k].cases[ci],compare=()=>el('mode').value==='compare',maximum=()=>compare()?D.alignment_steps:M[key].depth,actual=k=>compare()?Math.round(step/D.alignment_steps*M[k].depth):Math.min(step,M[k].depth),fmt=(v,n=3)=>Number.isFinite(v)?v.toFixed(n):'—';
D.order.forEach(k=>{let o=document.createElement('option');o.value=k;o.textContent=M[k].label;el('model').append(o)});M.baseline.cases.forEach((c,i)=>{let o=document.createElement('option');o.value=i;o.textContent=`样例${i+1} · N=${c.n}`;el('case').append(o)});el('case').value=ci;el('model').value=key;
function stop(){if(timer)clearInterval(timer);timer=null;el('play').textContent='播放'}
function coords(canvas){let L=C(key).axis_limits,pad=canvas.width*.045,s=(canvas.width-2*pad)/(L[1]-L[0]);return{L,pad,s,x:v=>pad+(v-L[0])*s,y:v=>canvas.height-pad-(v-L[2])*s}}
function overlaps(p){let hit=new Set;for(let i=0;i<p.length;i++)for(let j=0;j<i;j++){let a=p[i],b=p[j];if(Math.min(a[0]+a[2],b[0]+b[2])-Math.max(a[0],b[0])>1e-6&&Math.min(a[1]+a[3],b[1]+b[3])-Math.max(a[1],b[1])>1e-6){hit.add(i);hit.add(j)}}return hit}
function draw(canvas,p,k,d,thumb=false){let ctx=canvas.getContext('2d'),v=coords(canvas),hit=el('red').checked?overlaps(p):new Set,sel=+el('module').value;ctx.clearRect(0,0,canvas.width,canvas.height);p.forEach((r,i)=>{let x=v.x(r[0]),y=v.y(r[1]+r[3]),w=r[2]*v.s,h=r[3]*v.s;ctx.globalAlpha=sel>=0&&i!==sel?.28:.78;ctx.fillStyle=`rgb(${P[i].join(',')})`;ctx.fillRect(x,y,w,h);ctx.strokeStyle=hit.has(i)?'#d32636':'#475569';ctx.lineWidth=hit.has(i)?1.6:.65;ctx.strokeRect(x,y,w,h);if(!thumb&&el('ids').checked){ctx.font='17px system-ui';ctx.fillStyle='#17253a';ctx.textAlign='center';ctx.fillText(i,x+w/2,y+h/2+4)}});ctx.globalAlpha=1;if(!thumb&&sel>=0&&sel<C(k).n){ctx.strokeStyle='#0f172a';ctx.lineWidth=2;ctx.beginPath();C(k).stages.forEach((f,i)=>{let r=f.positions[sel],x=v.x(r[0]+r[2]/2),y=v.y(r[1]+r[3]/2);i?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.stroke();let r=C(k).stages[d].positions[sel];ctx.fillStyle='#d32636';ctx.beginPath();ctx.arc(v.x(r[0]+r[2]/2),v.y(r[1]+r[3]/2),6,0,Math.PI*2);ctx.fill()}}
function panel(title,k,d,ref=false){let section=document.createElement('section'),h=document.createElement('h2'),cv=document.createElement('canvas'),caption=document.createElement('div');h.textContent=title;cv.width=900;cv.height=900;cv.setAttribute('role','img');cv.setAttribute('aria-label',title);caption.className='caption';let p=ref?C(k).reference:C(k).stages[d].positions;section.append(h,cv,caption);el('views').append(section);draw(cv,p,k,d);if(ref)caption.textContent='原始reference；不进入网络输入';else{let m=C(k).stages[d].metrics;caption.textContent=`真实第${d}/${M[k].depth}层 · 重复覆盖 ${fmt(m.repeated_coverage_fraction*100,2)}% · 框/面积 ${fmt(m.bbox_over_area)} · 步长 ${fmt(m.center_step_mean,4)}`}cv.onclick=e=>{let v=coords(cv),r=cv.getBoundingClientRect(),x=((e.clientX-r.left)*cv.width/r.width-v.pad)/v.s+v.L[0],y=(cv.height-v.pad-(e.clientY-r.top)*cv.height/r.height)/v.s+v.L[2],found=-1;p.forEach((a,i)=>{if(x>=a[0]&&x<=a[0]+a[2]&&y>=a[1]&&y<=a[1]+a[3])found=i});el('module').value=found;render()}}
function render(){step=Math.min(step,maximum());el('step').max=maximum();el('step').value=step;el('module').max=C(key).n-1;el('stepLabel').textContent=compare()?`外层完成比例 ${fmt(step/D.alignment_steps*100,1)}%`:(step===0?'初始读出':`第${step}/${M[key].depth}层`);el('source').textContent=`原train索引 ${C(key).train_index} · ${C(key).member[0]}:${C(key).member[1]}`;el('identity').textContent=`${M[key].label} · ${M[key].parameters.toLocaleString()}参数 · best epoch ${M[key].identity.best_epoch} · 已展示新实验 ${D.completed_candidates}/2`;el('views').innerHTML='';panel('原始reference',key,actual(key),true);if(compare()){D.order.forEach(k=>panel(M[k].short_label+` · 第${actual(k)}/${M[k].depth}层`,k,actual(k)))}else{panel('当前真实读出',key,actual(key));panel('固定最终读出',key,M[key].depth)}el('stripTitle').textContent=M[key].label+' · 全部真实层';el('strip').innerHTML='';C(key).stages.forEach((f,i)=>{let button=document.createElement('button');button.className='thumb'+(i===actual(key)?' active':'');let cv=document.createElement('canvas'),label=document.createElement('span');cv.width=200;cv.height=200;label.textContent=i===0?'初始':`层${i}`;button.append(cv,label);el('strip').append(button);draw(cv,f.positions,key,i,true);button.onclick=()=>{stop();step=compare()?Math.round(i/M[key].depth*D.alignment_steps):i;render()}});el('numbers').innerHTML=C(key).stages.map(f=>{let m=f.metrics;return `<tr><td>${f.depth}</td><td>${fmt(m.repeated_coverage_fraction*100,2)}%</td><td>${fmt(m.bbox_over_area)}</td><td>${fmt(m.center_step_mean,4)}</td></tr>`}).join('')}
el('case').onchange=()=>{stop();ci=+el('case').value;el('module').value=-1;render()};el('model').onchange=()=>{stop();let fraction=step/maximum();key=el('model').value;step=Math.round(fraction*maximum());render()};let lastMode=el('mode').value;el('mode').onchange=()=>{stop();let oldMax=lastMode==='compare'?D.alignment_steps:M[key].depth;step=Math.round(step/oldMax*maximum());lastMode=el('mode').value;render()};el('step').oninput=()=>{stop();step=+el('step').value;render()};el('prev').onclick=()=>{stop();step=Math.max(0,step-1);render()};el('next').onclick=()=>{stop();step=Math.min(maximum(),step+1);render()};el('play').onclick=()=>{if(timer){stop();return}if(step===maximum())step=0;el('play').textContent='暂停';render();timer=setInterval(()=>{if(step===maximum()){stop();return}step++;render()},900)};['ids','red','module'].forEach(id=>el(id).onchange=render);document.addEventListener('keydown',e=>{if(['INPUT','SELECT'].includes(e.target.tagName))return;if(e.key==='ArrowLeft')el('prev').click();if(e.key==='ArrowRight')el('next').click()});render();
})();</script></html>'''


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);args=parser.parse_args()
    publish_available(args.root,require_complete=True)
