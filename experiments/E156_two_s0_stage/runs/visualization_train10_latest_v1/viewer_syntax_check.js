
(()=>{
 const el=id=>document.getElementById(id),ids=['reference','draft','final'];
 let manifest,current,selected=-1,requestNumber=0;
 const defaultLossCaption=el('loss-panel').querySelector('caption').textContent;
 const sides=[[1,'L'],[2,'R'],[4,'T'],[8,'B']];
 const fmt=(v,n=3)=>Number.isFinite(v)?v.toFixed(n):'—';
 async function get(url){const response=await fetch(url);if(!response.ok)throw Error(`读取布局失败 (${response.status})`);return response.json()}
 function overlaps(p){const hits=new Set;for(let i=0;i<p.length;i++)for(let j=0;j<i;j++){const a=p[i],b=p[j];if(Math.min(a[0]+a[2],b[0]+b[2])-Math.max(a[0],b[0])>1e-6&&Math.min(a[1]+a[3],b[1]+b[3])-Math.max(a[1],b[1])>1e-6){hits.add(i);hits.add(j)}}return hits}
 function stage(id){return id==='draft'?(current.layers.length?current.layers[+el('layer').value-1]:current.draft):current[id]}
 function draw(id){
  const cv=el(id),ctx=cv.getContext('2d'),c=current,p=stage(id),L=c.bounds,pad=38,s=(cv.width-2*pad)/(L[1]-L[0]);
  const x=v=>pad+(v-L[0])*s,y=v=>cv.height-pad-(v-L[2])*s;
  ctx.clearRect(0,0,cv.width,cv.height);
  if(!p){ctx.fillStyle='#64748b';ctx.font='28px system-ui';ctx.textAlign='center';ctx.fillText('未保存该阶段',cv.width/2,cv.height/2);return}
  const failed=id==='draft'&&c.layers.length?c.boundary_status.layers[+el('layer').value-1]:c.boundary_status[id];
  const hits=id==='draft'&&el('overlap').checked?overlaps(p):new Set;
  const lo=[Math.min(...p.map(r=>r[0])),Math.min(...p.map(r=>r[1]))],hi=[Math.max(...p.map(r=>r[0]+r[2])),Math.max(...p.map(r=>r[1]+r[3]))];
  ctx.strokeStyle='#64748b';ctx.lineWidth=1.3;ctx.setLineDash([8,7]);ctx.strokeRect(x(lo[0]),y(hi[1]),(hi[0]-lo[0])*s,(hi[1]-lo[1])*s);ctx.setLineDash([]);
  for(let i=0;i<p.length;i++){
   const r=p[i],f=c.constraints[i],xx=x(r[0]),yy=y(r[1]+r[3]),w=r[2]*s,h=r[3]*s;
   ctx.globalAlpha=selected>=0&&selected!==i?.18:1;
   ctx.fillStyle=c.group_colors[String(f[3])]||'#e5e7eb';ctx.fillRect(xx,yy,w,h);
   ctx.strokeStyle='#000';ctx.lineWidth=selected===i?3:1.2;ctx.strokeRect(xx,yy,w,h);
   if(hits.has(i)){ctx.strokeStyle='#dc2626';ctx.lineWidth=2;ctx.strokeRect(xx+2,yy+2,Math.max(0,w-4),Math.max(0,h-4))}
  }
  for(let i=0;i<p.length;i++){
   if(selected>=0&&selected!==i)continue;
   const r=p[i],f=c.constraints[i],lines=[String(i)];if(f[3]>0)lines.push('G'+f[3]);if(f[0]||f[1])lines.push([f[0]?'F':'',f[1]?'P':''].filter(Boolean).join(' '));
   const font=Math.max(10,Math.min(25,r[2]*s/(Math.max(...lines.map(t=>t.length))*.65+1),r[3]*s/(lines.length*1.18))),xx=x(r[0]+r[2]/2),yy=y(r[1]+r[3]/2);
   ctx.globalAlpha=1;ctx.font=`${font}px system-ui`;ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillStyle='#111827';
   lines.forEach((line,j)=>ctx.fillText(line,xx,yy+(j-(lines.length-1)/2)*font*1.1));
   const places=[[1,'L',r[0],r[1]+r[3]/2,1,0],[2,'R',r[0]+r[2],r[1]+r[3]/2,-1,0],[4,'T',r[0]+r[2]/2,r[1]+r[3],0,1],[8,'B',r[0]+r[2]/2,r[1],0,-1]];
   for(const [bit,letter,bx,by,dx,dy] of places){if(!(f[4]&bit))continue;const size=Math.max(12,Math.min(21,r[2]*s*.4,r[3]*s*.4)),px=x(bx)+dx*size*.52,py=y(by)+dy*size*.6;ctx.fillStyle=failed[i].includes(bit)?'#b91c1c':'#15803d';ctx.fillRect(px-size*.44,py-size*.52,size*.88,size*1.04);ctx.fillStyle='white';ctx.font=`600 ${size}px system-ui`;ctx.fillText(letter,px,py)}
  }
  ctx.globalAlpha=1;ctx.fillStyle='#64748b';ctx.font='19px system-ui';ctx.textBaseline='alphabetic';ctx.textAlign='left';ctx.fillText(`${fmt(L[0],0)}, ${fmt(L[2],0)}`,pad,cv.height-8);ctx.textAlign='right';ctx.fillText(`${fmt(L[1],0)}, ${fmt(L[3],0)}`,cv.width-pad,24);
 }
 function legend(){
  el('legend').replaceChildren();
  function key(text,fill,border='1px solid #000'){const item=document.createElement('span'),sw=document.createElement('i'),label=document.createElement('span');item.className='key';sw.className='swatch';sw.style.background=fill;sw.style.border=border;label.textContent=text;item.append(sw,label);el('legend').append(item)}
  Object.entries(current.group_colors).forEach(([g,color])=>key('分组 G'+g,color));key('无分组','#e5e7eb');key('L/R/T/B 满足','#15803d');key('L/R/T/B 违反','#b91c1c');if(el('overlap').checked)key('草稿重叠','transparent','2px solid #dc2626');key('当前布局外框','transparent','1px dashed #64748b');const text=document.createElement('span');text.textContent='数字：模块编号 · F：固定形状 · P：预放置';el('legend').append(text);
 }
 function render(){
  if(!current)return;
  const count=current.layers.length,depth=+el('layer').value;
  const omitFinal=Boolean(current.hide_missing_final&&!current.final);
  el('final').closest('section').hidden=omitFinal;el('final').closest('.panels').classList.toggle('two-panels',omitFinal);
  el('reference-title').textContent=current.reference_label||'Reference';
  el('slider-hint').textContent=omitFinal?'滑块切换实际模型阶段；参考栏保持不变':'只切换中栏；左右保持不变';
  const layerIds=current.layer_ids||current.layers.map((_,i)=>i+1),actualDepth=layerIds[depth-1];
  const layerLabel=i=>current.layer_labels?.[i]||'S'+layerIds[i];
  el('depth').textContent=`${layerLabel(depth-1)} / ${layerLabel(count-1)}`;
  el('layer').setAttribute('aria-valuetext',layerLabel(depth-1));
  el('draft-title').textContent=count?`模型草稿 · ${layerLabel(depth-1)}`:current.draft_label||'模型草稿 · 末层';
  const losses=current.layer_losses||[],loss=losses[depth-1];
  el('loss-panel').hidden=!losses.length;el('loss-table').replaceChildren();
  el('objective-detail').hidden=true;
  if(loss){
   const objective=loss.objective_total??loss.total;
   el('draft-loss').textContent=`${current.loss_label||'完整 Loss'}：${fmt(loss.total,6)}（未乘层权重）\n${loss.directly_supervised?'监督目标 '+(loss.objective_name||'完整 Loss')+'：'+fmt(objective,6)+' · 权重 '+fmt(loss.supervision_weight,4):'未直接监督'}`+(Number.isFinite(loss.grouping_violations)?`\n分组断连数：${loss.grouping_violations}`:'');
   if(loss.frontend_components){const f=loss.frontend_components;el('objective-detail').hidden=false;el('objective-detail').textContent=`S0 的 F = 中心容差 ${fmt(f.center_tolerance,6)} + 自由形状×0.05 ${fmt(.05*f.free_shape,6)} + F/P尺寸 ${fmt(f.public_shape,6)} + P位置 ${fmt(f.public_preplaced_position,6)}。完整 Loss 仅作共同诊断，S0 实际按 F 训练。`}
   const columns=current.loss_columns||[
    ['total','完整Loss'],['objective_total','监督目标Loss'],['reference_weighted','Reference'],['overlap_weighted','碰撞'],['wire_compact_weighted','线长/紧凑'],['grouping_weighted','分组'],['boundary_weighted','边界'],['grouping_blockage_weighted','分组阻隔'],['boundary_blockage_weighted','边界阻隔'],['public_shape_weighted','固定尺寸'],['public_preplaced_position_weighted','预放置位置'],['grouping_violations','分组断连数'],['supervision_weight','监督权重']];
   const keys=columns.map(c=>c[0]);el('loss-head').replaceChildren();
   ['层',...columns.map(c=>c[1])].forEach(label=>{const th=document.createElement('th');th.textContent=label;el('loss-head').append(th)});
   el('loss-panel').querySelector('caption').textContent=current.loss_caption||defaultLossCaption;
   losses.forEach((values,i)=>{const row=document.createElement('tr');if(i===depth-1)row.className='current-layer';const layer=document.createElement('td');layer.textContent=layerLabel(i);row.append(layer);keys.forEach(key=>{const cell=document.createElement('td');const value=key==='objective_total'?(values.directly_supervised?(values.objective_total??values.total):undefined):values[key];cell.textContent=fmt(value,key==='grouping_violations'?0:key==='supervision_weight'?4:6);row.append(cell)});el('loss-table').append(row)});
  }else el('draft-loss').textContent='本层 Loss 未保存';
  el('meta').textContent=`${current.label||''} · N=${current.n}`+(current.member?` · ${current.member.join(' : ')}`:'');
  const gs=current.grouping_summary;
  el('reference-stat').textContent='参考布局'+(gs?`\n分组断连数：${gs.reference}`:'');
  el('final-stat').textContent=current.final?'已保存的原预算后处理结果'+(gs?`\n分组断连数：${gs.final}\n仍断开的分组：${gs.final_bad_groups.map(g=>'G'+g.group+'（'+g.components+'片）').join('、')}`:''):'';
  if(current.objective_note){el('objective-detail').hidden=false;el('objective-detail').textContent=current.objective_note}
  ids.forEach(draw);
  const flags=current.constraints[selected];el('selection').textContent=selected<0?'点击模块可在各栏同时定位；再次点击取消。':`模块 ${selected} · ${[flags[3]>0?'G'+flags[3]:'',flags[0]?'F':'',flags[1]?'P':'',sides.filter(([b])=>flags[4]&b).map(([,t])=>t).join('/')].filter(Boolean).join(' · ')||'无附加约束'}`;
  const m=current.metrics||{};el('metrics').textContent=[Number.isFinite(m.Q)?`最终 Q ${fmt(m.Q,6)}`:'',typeof m.hard_legal==='boolean'?`最终硬合法：${m.hard_legal?'是':'否'}`:'',Number.isFinite(m.complete_ms)?`完整耗时 ${fmt(m.complete_ms,2)} ms`:''].filter(Boolean).join(' · ');
 }
 async function load(){
  const ticket=++requestNumber;el('error').textContent='';current=null;ids.forEach(id=>el(id).getContext('2d').clearRect(0,0,1200,1200));el('layer-controls').hidden=true;el('legend').replaceChildren();el('meta').textContent='读取中…';el('metrics').textContent='';el('selection').textContent='';el('draft-loss').textContent='';el('loss-panel').hidden=true;
  const index=+el('case').value;el('previous').disabled=index===0;el('next').disabled=index===manifest.cases.length-1;
  try{const c=await get('/api/case/'+index);if(ticket!==requestNumber)return;current=c;selected=-1;el('layer').max=Math.max(1,c.layers.length);el('layer').value=Math.max(1,c.layers.length);el('layer-controls').hidden=c.layers.length<2;legend();render()}catch(error){if(ticket===requestNumber){el('meta').textContent='';el('error').textContent=error.message}}
 }
 el('case').onchange=load;el('previous').onclick=()=>{el('case').value=Math.max(0,+el('case').value-1);load()};el('next').onclick=()=>{el('case').value=Math.min(manifest.cases.length-1,+el('case').value+1);load()};el('layer').oninput=render;el('overlap').onchange=()=>{if(current){legend();render()}};
 ids.forEach(id=>el(id).onclick=event=>{if(!current)return;const p=stage(id);if(!p)return;const cv=el(id),r=cv.getBoundingClientRect(),L=current.bounds,s=(cv.width-76)/(L[1]-L[0]),x=((event.clientX-r.left)*cv.width/r.width-38)/s+L[0],y=(cv.height-38-(event.clientY-r.top)*cv.height/r.height)/s+L[2];let hit=-1;p.forEach((a,i)=>{if(x>=a[0]&&x<=a[0]+a[2]&&y>=a[1]&&y<=a[1]+a[3])hit=i});selected=selected===hit?-1:hit;render()});
 document.addEventListener('keydown',event=>{if(['INPUT','SELECT'].includes(event.target.tagName)||!current||current.layers.length<2)return;if(['ArrowLeft','ArrowRight'].includes(event.key)){event.preventDefault();el('layer').value=Math.max(1,Math.min(current.layers.length,+el('layer').value+(event.key==='ArrowLeft'?-1:1)));render()}});
 get('/api/manifest').then(data=>{manifest=data;document.title=data.title;el('title').textContent=data.title;el('note').textContent=data.note;el('source').textContent=JSON.stringify(data.identity);const groups=new Map;data.cases.forEach((c,i)=>{const option=new Option(c.label+(c.n?` · N=${c.n}`:''),i);if(c.group){if(!groups.has(c.group)){const group=document.createElement('optgroup');group.label=({ordinary:'普通例子',worst:'最坏例子',fixed:'固定训练例'})[c.group]||c.group;groups.set(c.group,group);el('case').append(group)}groups.get(c.group).append(option)}else el('case').add(option)});return load()}).catch(error=>el('error').textContent=error.message);
})();
