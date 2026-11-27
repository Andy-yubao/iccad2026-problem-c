"""Pure display of saved layer traces; no inference, training, repair or Q."""
from pathlib import Path
import json

import numpy as np

from .raster_layouts import module_palette
from .layouts import overlap_blocks


def axes_layout(ax, case, positions, title, labels=True):
    from matplotlib.patches import Rectangle
    overlap = overlap_blocks(positions, max(case['axis_limits'][1] - case['axis_limits'][0], 1) * 1e-9)
    for i, (x, y, w, h) in enumerate(positions):
        ax.add_patch(Rectangle((x, y), w, h, facecolor='#f1f1f1',
                              edgecolor='#d32636' if i in overlap else '#475569',
                              linewidth=.8 if i in overlap else .35, alpha=.77))
        if labels:
            ax.text(x + w / 2, y + h / 2, str(i), fontsize=5.5, ha='center', va='center', clip_on=True,
                    zorder=10, bbox=dict(facecolor='white',edgecolor='none',pad=.15))
    limits = case['axis_limits']
    ax.set(xlim=limits[:2], ylim=limits[2:], title=title)
    ax.set_aspect('equal', adjustable='box')
    ax.tick_params(labelsize=7)
    ax.spines[['top', 'right']].set_visible(False)


def render_case(case, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 4, figsize=(16, 8.8), constrained_layout=True)
    rows = [('Reference', case['reference'])]
    rows += [('Initial readout' if d == 0 else f'After layer {d}', case['stages'][d]['positions'])
             for d in (0, 2, 4, 6, 8, 10, 12)]
    for ax, (title, rects) in zip(axes.flat, rows):
        axes_layout(ax, case, rects, title, labels=case['n'] <= 45)
    fig.suptitle(f"E98 best71 | {case['id']} | N={case['n']} | common axes, stable module colors\n"
                 'True shared-head readouts; intermediate layers have no independent supervision; no repair', fontsize=12)
    fig.savefig(destination, dpi=150, facecolor='white')
    plt.close(fig)


def render_curves(case, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.8), constrained_layout=True)
    metrics = [s['metrics'] for s in case['stages']]
    def series(key):
        return np.asarray([m.get(key, np.nan) for m in metrics])
    axes[0, 0].plot(range(13), 100 * series('repeated_coverage_fraction'), marker='o', label='Repeated coverage / total module area')
    axes[0, 0].set(ylabel='%', title='Exact union-based overlap (lower is less overlap)')
    axes[0, 1].plot(range(13), series('center_step_mean'), marker='o', label='Decoded center change')
    axes[0, 1].plot(range(13), series('raw_center_step_mean'), marker='.', label='Raw center change')
    axes[0, 1].set(ylabel='Mean displacement / sqrt(total effective area)', title='Step size; raw and decoded states differ')
    axes[1, 0].plot(range(13), series('bbox_over_area'), marker='o', label='Bbox / module area')
    axes[1, 0].plot(range(13), series('center_rmse_to_reference'), marker='.', label='Center RMSE to original reference / scale')
    axes[1, 0].set(title='Compactness and reference proximity (distinct objectives)')
    for key, label in [('hidden_rms', 'Hidden state'), ('feedback_rms', 'Position feedback'),
                       ('attention_rms', 'Attention branch'), ('local_rms', 'Graph aggregation'), ('ffn_rms', 'FFN branch')]:
        axes[1, 1].plot(range(13), series(key), marker='.', label=label)
    axes[1, 1].set(yscale='log', title='Activation RMS by branch (log scale)')
    for ax in axes.flat:
        ax.set(xlabel='Layer / readout index', xticks=range(13))
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    fig.suptitle(f"{case['id']} | N={case['n']} | training-case observations, not causal attribution", fontsize=12)
    fig.savefig(destination, dpi=150, facecolor='white')
    plt.close(fig)


def render_overview(cases, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(cases), 4, figsize=(16, 4 * len(cases)), constrained_layout=True)
    for row, case in enumerate(cases):
        layouts = [('Reference', case['reference']), ('Initial', case['stages'][0]['positions']),
                   ('Layer 6', case['stages'][6]['positions']), ('Layer 12', case['stages'][12]['positions'])]
        for col, (label, positions) in enumerate(layouts):
            axes_layout(axes[row, col], case, positions, f"{case['id']} N={case['n']} | {label}")
    fig.suptitle('E98 best71 | five fixed training cases | same axes within each row | no postprocessing', fontsize=14)
    fig.savefig(destination, dpi=140, facecolor='white')
    plt.close(fig)


def write_gallery(result, folder):
    folder = Path(folder)
    data = json.dumps(result, ensure_ascii=False, allow_nan=False).replace('<', '\\u003c')
    colors = np.round(module_palette(120) * 255).astype(int).tolist()
    page = TEMPLATE.replace('__TRACE_DATA__', data).replace('__PALETTE__', json.dumps(colors))
    (folder / 'index.html').write_text(page, encoding='utf-8')


TEMPLATE = r'''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>E98 · 逐层布局观察</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#f4f6fa;color:#17253a;font:15px/1.6 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1560px;margin:auto;padding:24px}h1{margin:0;font-size:28px}h2{font-size:18px;margin:8px 0}.sub{color:#5c6b7e;margin:8px 0 18px}.note{padding:12px 16px;border-left:4px solid #ca8a04;background:#fffbeb;margin:12px 0}.card{background:white;border:1px solid #e0e6ef;border-radius:12px;padding:16px;margin:14px 0}.controls{display:flex;gap:16px;flex-wrap:wrap;align-items:center}select,button,input[type=number]{padding:8px;border:1px solid #cbd5e1;border-radius:6px;background:white;color:inherit}button{cursor:pointer}button:hover{background:#edf2fc}input[type=range]{flex:1;min-width:240px;accent-color:#2563eb}.views{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}canvas.layout{display:block;width:100%;aspect-ratio:1;background:#fafcff;border:1px solid #edf0f5;border-radius:8px}.caption{font-size:13px;color:#64748b;overflow-wrap:anywhere}.stats{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:10px}.stat{background:#f3f6fb;border-radius:8px;padding:12px}.stat strong{display:block;font-size:23px}.stat span{font-size:12px;color:#57667a}.strip{display:grid;grid-template-columns:repeat(13,minmax(0,1fr));gap:5px}.thumb{border:2px solid transparent;border-radius:5px;padding:2px;cursor:pointer}.thumb.active{border-color:#2563eb}.thumb canvas{display:block;width:100%}.thumb small{display:block;text-align:center}.charts{display:grid;grid-template-columns:1fr 1fr;gap:15px}.charts canvas{width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:12px;white-space:nowrap}th,td{border-bottom:1px solid #e2e8f0;padding:7px;text-align:right}td:first-child,th:first-child{text-align:left}.table-scroll{overflow:auto}a{color:#1d4ed8}details{margin:12px 0}summary{cursor:pointer}code{font-family:ui-monospace,monospace}.tag{font-size:12px;padding:4px 8px;background:#eaf0fe;border-radius:6px}.legend{display:flex;gap:18px;flex-wrap:wrap;font-size:13px;color:#526176}.flow{padding:12px;background:#f3f6fb;border-radius:7px;overflow:auto;white-space:nowrap;font-family:ui-monospace,monospace}@media(max-width:900px){.views{grid-template-columns:1fr}.stats{grid-template-columns:repeat(2,1fr)}.strip{grid-template-columns:repeat(7,1fr)}.charts{grid-template-columns:1fr}main{padding:12px}}
</style><main>
<span class="tag">冻结模型 · 五个训练样例 · 只读诊断</span><h1>12层之中，布局究竟发生了什么？</h1>
<p class="sub">E98 增强版 best71 · 96维隐藏特征 · 12个矩形图注意力块 · 同一个两层预测头共读出13次。每一帧来自原始前向计算，无额外修复。</p>
<div class="note">中间读出没有各自独立的训练监督，不能把早期布局直接当作“较浅模型训练后的能力”。历史模型的原登记面积变换和MIB输入保留，以忠实观察其内部计算；本页不计算MIB罚分或Q，不代表新的无MIB模型。</div>
<div class="card"><div class="flow">公开图特征 → h₀ → 共享预测头 → s₀<br>重复12次：[ hᵢ + 位置嵌入(sᵢ) → 注意力/图聚合 → FFN → hᵢ₊₁ → 同一预测头 → sᵢ₊₁ ]</div><p class="caption">h始终保留96维。注意力的几何偏置同时使用上一帧的坐标、形状、间距及穿透信息。位置预测头为96→96→3，偏置小网络为19→24→4；这不等价于只有两层的整体网络。</p></div>
<div class="card"><div class="controls"><label>样例 <select id="case"></select></label><button id="prev">上一层</button><button id="play">播放</button><button id="next">下一层</button><strong id="stageLabel"></strong></div><div class="controls" style="margin-top:12px"><span>初始</span><input id="depth" type="range" min="0" max="12" value="12"><span>第12层</span></div><div class="controls"><label><input type="checkbox" id="ids"> 模块编号</label><label><input type="checkbox" id="overlap" checked> 重叠红框</label><label>追踪模块 <input type="number" id="module" min="-1" value="-1" style="width:80px"></label><span class="caption">−1为不选；点击矩形可选中并显示13次中心轨迹。三个大图及全部缩略图始终共用同一坐标尺度。</span></div><p id="member" class="caption"></p></div>
<div class="card"><div class="views"><section><h2>原始reference</h2><canvas id="ref" class="layout" width="900" height="900"></canvas></section><section><h2 id="currentTitle">当前读出</h2><canvas id="current" class="layout" width="900" height="900"></canvas></section><section><h2>固定最终输出 · 第12层</h2><canvas id="final" class="layout" width="900" height="900"></canvas></section></div><div class="legend"><span>同一模块始终同色</span><span style="color:#d32636">红边：与其他模块重叠</span><span>F/P：固定形状／预放置</span><span>黑线：选中模块的中心轨迹</span></div></div>
<div class="card"><div class="stats" id="stats"></div><p class="caption">重复覆盖＝(各模块面积之和−占用并集面积)／各模块面积之和，未重复累计多重相交。中心步长按√总有效面积归一化。reference仅用于事后观察，接近reference并非唯一合理布局。</p></div>
<div class="card"><h2>13次真实读出</h2><div class="strip" id="strip"></div></div>
<div class="card"><h2>随层数变化</h2><div class="charts"><canvas id="overlapChart" width="1000" height="440"></canvas><canvas id="moveChart" width="1000" height="440"></canvas></div><p class="caption">蓝色：重复覆盖；绿色：外框面积/模块面积。另一张图中紫色为解码中心步长，橙色为原始预测中心步长。解码包含非负坐标及宽高比截断，两种步长可能不同。</p><p><a id="curveLink">查看完整诊断曲线PNG</a> · <a id="stageLink">查看八阶段高清PNG</a> · <a href="overview.png">查看五例总览</a></p></div>
<details class="card"><summary>展开每层数值与分支幅度</summary><p class="caption">RMS是激活幅度，不能直接当成网络贡献率；注意力、图聚合和FFN可相互抵消。早期读出的异常尺度不自动证明隐藏特征缺乏信息。</p><div class="table-scroll"><table><thead><tr><th>层</th><th>重复覆盖%</th><th>框/总面积</th><th>中心步长</th><th>原始步长</th><th>reference中心RMSE</th><th>隐藏RMS</th><th>位置反馈RMS</th><th>注意力RMS</th><th>图聚合RMS</th><th>FFN RMS</th><th>宽高比截断%</th><th>x/y贴零%</th></tr></thead><tbody id="rows"></tbody></table></div></details>
<p class="caption">五例分别按训练名单中最先接近N=21/45/70/95/120的成员确定，未看输出选题；CPU FP32、eval、batch1、无训练、无测试、无后处理。带钩子与不带钩子的最终结果逐位相同，权重未变。观察可以提示问题，删除位置反馈或重新分配深度的因果效果仍需将来的受控实验。</p>
</main><script id="data" type="application/json">__TRACE_DATA__</script><script>
const D=JSON.parse(document.querySelector('#data').textContent),palette=__PALETTE__,el=id=>document.getElementById(id);let ci=2,depth=12,timer=null;
D.cases.forEach((c,i)=>{let o=document.createElement('option');o.value=i;o.textContent=`样例 ${i+1} · N=${c.n}`;el('case').append(o)});el('case').value=ci;
const C=()=>D.cases[ci], num=(x,n=4)=>Number.isFinite(x)?x.toFixed(n):'—';
function overlaps(p){let hit=new Set;for(let i=0;i<p.length;i++)for(let j=0;j<i;j++){let a=p[i],b=p[j];if(Math.min(a[0]+a[2],b[0]+b[2])-Math.max(a[0],b[0])>1e-7&&Math.min(a[1]+a[3],b[1]+b[3])-Math.max(a[1],b[1])>1e-7){hit.add(i);hit.add(j)}}return hit}
function view(canvas){let c=C(),L=c.axis_limits,pad=canvas.width*.055,s=(canvas.width-2*pad)/(L[1]-L[0]);return{L,pad,s,x:v=>pad+(v-L[0])*s,y:v=>canvas.height-pad-(v-L[2])*s}}
function draw(canvas,p,isRef=false,thumb=false){let c=C(),ctx=canvas.getContext('2d'),v=view(canvas),hit=el('overlap').checked?overlaps(p):new Set,sel=+el('module').value;ctx.clearRect(0,0,canvas.width,canvas.height);ctx.fillStyle='#fafcff';ctx.fillRect(0,0,canvas.width,canvas.height);ctx.strokeStyle='#e1e8f2';ctx.lineWidth=1;ctx.strokeRect(v.pad,v.pad,canvas.width-2*v.pad,canvas.height-2*v.pad);p.forEach((r,i)=>{let [x,y,w,h]=r,X=v.x(x),Y=v.y(y+h),W=w*v.s,H=h*v.s;ctx.globalAlpha=sel>=0&&i!==sel?.34:.78;ctx.fillStyle=`rgb(${palette[i].join(',')})`;ctx.fillRect(X,Y,W,H);ctx.strokeStyle=hit.has(i)?'#d32636':'#475569';ctx.lineWidth=thumb?1:hit.has(i)?1.6:.65;ctx.strokeRect(X,Y,W,H);if(!thumb&&el('ids').checked){ctx.globalAlpha=1;ctx.font='13px system-ui';ctx.textAlign='center';ctx.fillStyle='#14243a';ctx.fillText(String(i),X+W/2,Y+H/2+4)}if(!thumb&&(c.constraints[i][0]||c.constraints[i][1])&&Math.min(W,H)>18){ctx.globalAlpha=1;ctx.font='bold 12px system-ui';ctx.fillStyle='#152438';ctx.textAlign='left';ctx.fillText(c.constraints[i][1]?'P':'F',X+2,Y+13)}});ctx.globalAlpha=1;if(!thumb&&sel>=0&&sel<c.n){ctx.strokeStyle='#0f172a';ctx.lineWidth=2;ctx.beginPath();c.stages.forEach((f,j)=>{let r=f.positions[sel],x=v.x(r[0]+r[2]/2),y=v.y(r[1]+r[3]/2);j?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.stroke();c.stages.forEach((f,j)=>{let r=f.positions[sel];ctx.fillStyle=j===depth?'#d32636':'#111827';ctx.beginPath();ctx.arc(v.x(r[0]+r[2]/2),v.y(r[1]+r[3]/2),j===depth?6:3,0,Math.PI*2);ctx.fill()})}if(!thumb){ctx.font='12px system-ui';ctx.fillStyle='#64748b';ctx.textAlign='left';ctx.fillText(`${num(v.L[0],1)}, ${num(v.L[2],1)}`,v.pad,canvas.height-8);ctx.textAlign='right';ctx.fillText(`${num(v.L[1],1)}, ${num(v.L[3],1)}`,canvas.width-v.pad,23)}}
function chart(id,series){let canvas=el(id),ctx=canvas.getContext('2d'),W=canvas.width,H=canvas.height,L=80,R=W-65,T=65,B=H-65;ctx.clearRect(0,0,W,H);series.forEach((s,k)=>{let vals=C().stages.map(f=>f.metrics[s.key]*s.scale),max=Math.max(...vals,1e-9)*1.08;ctx.strokeStyle='#e2e8f0';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(L,T);ctx.lineTo(L,B);ctx.lineTo(R,B);ctx.stroke();ctx.strokeStyle=s.color;ctx.lineWidth=3;ctx.beginPath();vals.forEach((v,i)=>{let x=L+(R-L)*i/12,y=B-(B-T)*v/max;i?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.stroke();ctx.fillStyle=s.color;ctx.font='17px system-ui';ctx.textAlign=k?'right':'left';ctx.fillText(s.label+(s.scale===100?' %':'')+` [0–${num(max,2)}]`,k?R:L,25);vals.forEach((v,i)=>{ctx.beginPath();ctx.arc(L+(R-L)*i/12,B-(B-T)*v/max,i===depth?7:4,0,Math.PI*2);ctx.fill()})});ctx.strokeStyle='#94a3b8';ctx.setLineDash([6,5]);ctx.beginPath();ctx.moveTo(L+(R-L)*depth/12,T);ctx.lineTo(L+(R-L)*depth/12,B);ctx.stroke();ctx.setLineDash([]);ctx.fillStyle='#64748b';ctx.font='16px system-ui';ctx.textAlign='center';for(let i=0;i<13;i++)ctx.fillText(i,L+(R-L)*i/12,B+28)}
function render(){let c=C(),m=c.stages[depth].metrics;el('depth').value=depth;el('stageLabel').textContent=depth===0?'初始读出（第0次）':`第 ${depth} 层之后`;el('currentTitle').textContent=el('stageLabel').textContent;el('module').max=c.n-1;el('member').textContent=`原train索引 ${c.train_index} ｜ ${c.member[0]} : ${c.member[1]} ｜ 所有帧同尺度、无裁剪`;draw(el('ref'),c.reference,true);draw(el('current'),c.stages[depth].positions);draw(el('final'),c.stages[12].positions);let stats=[['重复覆盖',num(m.repeated_coverage_fraction*100,2)+'%'],['框面积 / 模块面积',num(m.bbox_over_area,3)],['解码中心步长',num(m.center_step_mean)],['相交模块对数',m.overlapping_pairs],['隐藏特征RMS',num(m.hidden_rms,2)]];el('stats').innerHTML=stats.map(([k,v])=>`<div class="stat"><span>${k}</span><strong>${v}</strong></div>`).join('');el('strip').innerHTML='';c.stages.forEach((f,i)=>{let d=document.createElement('div');d.className='thumb'+(i===depth?' active':'');let cv=document.createElement('canvas');cv.width=200;cv.height=200;d.append(cv);let label=document.createElement('small');label.textContent=i===0?'初始':`层 ${i}`;d.append(label);d.onclick=()=>{depth=i;render()};el('strip').append(d);draw(cv,f.positions,false,true)});chart('overlapChart',[{key:'repeated_coverage_fraction',scale:100,color:'#2563eb',label:'重复覆盖'},{key:'bbox_over_area',scale:1,color:'#059669',label:'框/总面积'}]);chart('moveChart',[{key:'center_step_mean',scale:1,color:'#7c3aed',label:'解码中心步长'},{key:'raw_center_step_mean',scale:1,color:'#ea580c',label:'原始中心步长'}]);let keys=['repeated_coverage_fraction','bbox_over_area','center_step_mean','raw_center_step_mean','center_rmse_to_reference','hidden_rms','feedback_rms','attention_rms','local_rms','ffn_rms','raw_ratio_clamp_fraction','free_center_at_origin_fraction'];el('rows').innerHTML=c.stages.map(f=>`<tr${f.depth===depth?' style="background:#eaf0fe"':''}><td>${f.depth===0?'初始':f.depth}</td>${keys.map(k=>`<td>${num(f.metrics[k]*(k.includes('fraction')?100:1))}</td>`).join('')}</tr>`).join('');el('curveLink').href=`${c.id}_curves.png`;el('stageLink').href=`${c.id}_stages.png`}
el('case').onchange=()=>{ci=+el('case').value;el('module').value=-1;render()};el('depth').oninput=()=>{depth=+el('depth').value;render()};el('prev').onclick=()=>{depth=Math.max(0,depth-1);render()};el('next').onclick=()=>{depth=Math.min(12,depth+1);render()};el('play').onclick=()=>{if(timer){clearInterval(timer);timer=null;el('play').textContent='播放'}else{if(depth===12)depth=0;render();el('play').textContent='暂停';timer=setInterval(()=>{if(depth===12){clearInterval(timer);timer=null;el('play').textContent='播放';return}depth++;render()},850)}};['ids','overlap','module'].forEach(id=>el(id).onchange=render);['ref','current','final'].forEach(id=>el(id).onclick=e=>{let canvas=el(id),r=canvas.getBoundingClientRect(),v=view(canvas),xx=(e.clientX-r.left)*canvas.width/r.width,yy=(e.clientY-r.top)*canvas.height/r.height,x=(xx-v.pad)/v.s+v.L[0],y=(canvas.height-v.pad-yy)/v.s+v.L[2],p=id==='ref'?C().reference:C().stages[id==='final'?12:depth].positions;let found=-1;p.forEach((a,i)=>{if(x>=a[0]&&x<=a[0]+a[2]&&y>=a[1]&&y<=a[1]+a[3])found=i});el('module').value=found;render()});document.addEventListener('keydown',e=>{if(e.target.tagName==='INPUT')return;if(e.key==='ArrowRight'){depth=Math.min(12,depth+1);render()}if(e.key==='ArrowLeft'){depth=Math.max(0,depth-1);render()}});render();
</script></html>'''
