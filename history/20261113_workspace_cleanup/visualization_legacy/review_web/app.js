'use strict';
const $=id=>document.getElementById(id);
let manifest,state,token,index=0,view='all',busy=false,dirty=false,feedbackDirty=false,imageReady=false,dimensionAnswers={},stages=[];
const error=text=>{$('error').textContent=text;$('error').hidden=!text;};
const current=()=>manifest.cases[index];
function savedStatus(text){$('saved').textContent=text;}
function controls(){
  const disabled=busy||!imageReady;
  document.querySelectorAll('#scores button').forEach(b=>b.disabled=disabled);
  for(const id of ['prev','next','unrated','go','save-note','clear','save-feedback','finish'])$(id).disabled=busy;
  for(const id of ['note','tags','feedback','jump','auto'])$(id).disabled=busy;
  document.querySelectorAll('[data-dimension]').forEach(b=>b.disabled=busy);
  $('prev').disabled=busy||index===0;$('next').disabled=busy||index===manifest.cases.length-1;
}
async function update(payload){
  if(busy)return false;
  busy=true;document.body.classList.add('busy');controls();savedStatus('正在保存…');error('');
  try{
    const response=await fetch('/api/update',{method:'POST',headers:{'Content-Type':'application/json','X-Review-Token':token},body:JSON.stringify({...payload,expected_revision:state.revision})});
    const result=await response.json();if(!response.ok)throw Error(result.error||'保存失败');
    state=result;savedStatus('已保存到本机 · '+new Date().toLocaleTimeString('zh-CN'));renderProgress();return true;
  }catch(e){savedStatus('未保存');error(e.message+' 输入仍保留在页面中，请勿关闭。');return false;}
  finally{busy=false;document.body.classList.remove('busy');controls();}
}
function renderProgress(){
  $('progress').textContent=`${state.rated_count} / ${manifest.cases.length} 已评分`;
  $('bar').max=manifest.cases.length;$('bar').value=state.rated_count;
  $('completion').textContent=state.completed?'本轮已完成，原始评价与改分历史均已保存。':'评分逐次保存在本机，可随时离开并继续。';
  if(state.rated_count===manifest.cases.length){
    const counts=[1,2,3].map(n=>`${n}分 ${Object.values(state.ratings).filter(r=>r.score===n).length}张`);
    $('distribution').textContent=manifest.rubric.target==='pairwise_orderliness'?'全部对照已记录，可以填写整体感受。':counts.join(' · ');
  }else $('distribution').textContent='全部评完后显示评分分布。';
}
function highlight(){
  const score=state.ratings[current().id]?.score;
  document.querySelectorAll('[data-score]').forEach(b=>b.classList.toggle('selected',Number(b.dataset.score)===score));
}
function render(){
  const row=current(),rated=state.ratings[row.id]||{};
  $('case').textContent=`第 ${index+1} / ${manifest.cases.length} 张 · ${row.n} 块 · ${row.id}`;
  $('jump').value=index+1;$('jump').max=manifest.cases.length;
  $('note').value=rated.note||'';$('tags').value=(rated.tags||[]).join('，');dirty=false;highlight();
  dimensionAnswers={...(rated.dimensions||{})};renderDimensions();
  $('metrics').open=false;$('numbers').textContent='';
  imageReady=false;$('loading').hidden=false;controls();
  $('image').src='/images/'+encodeURIComponent(row.id);
}
function draw(){
  $('image').hidden=view!=='all';$('focus').hidden=view==='all';
  document.querySelectorAll('[data-view]').forEach(b=>b.classList.toggle('active',b.dataset.view===view));
  const width=Number($('zoom').value)*100+'%';$('image').style.width=width;$('focus').style.width=width;
  if(view!=='all'&&imageReady){
    const im=$('image'),canvas=$('focus'),bounds=current().axes_bounds[stages.findIndex(s=>s.key===view)];
    const [x,y,w,h]=bounds,pad=.025;
    const sx=Math.max(0,(x-pad)*im.naturalWidth),sy=Math.max(0,(1-y-h-pad*2)*im.naturalHeight);
    const sw=Math.min(im.naturalWidth-sx,(w+2*pad)*im.naturalWidth),sh=Math.min(im.naturalHeight-sy,(h+3*pad)*im.naturalHeight);
    canvas.width=Math.round(sw);canvas.height=Math.round(sh);canvas.getContext('2d').drawImage(im,sx,sy,sw,sh,0,0,canvas.width,canvas.height);
  }
}
async function saveNote(score=state.ratings[current().id]?.score??null){
  const tags=$('tags').value.split(/[,，\n]/).map(t=>t.trim()).filter(Boolean);
  const ok=await update({action:'rating',id:current().id,score,note:$('note').value,tags:[...new Set(tags)],dimensions:dimensionAnswers});
  if(ok){dirty=false;highlight();}return ok;
}
async function move(to){
  if(busy||to<0||to>=manifest.cases.length)return;
  if(dirty&&!await saveNote())return;
  index=to;render();$('viewport').scrollTop=0;$('viewport').scrollLeft=0;window.scrollTo(0,0);
}
function nextUnrated(start=index+1){
  for(let n=0;n<manifest.cases.length;n++){const k=(start+n)%manifest.cases.length;if(!state.ratings[manifest.cases[k].id]?.score)return k;}
  return -1;
}
async function rate(score){
  if(busy||!imageReady)return;
  if(await saveNote(score)){
    if($('auto').checked&&index<manifest.cases.length-1)await move(index+1);
    else if(state.rated_count===manifest.cases.length){$('feedback').focus();$('feedback').scrollIntoView({behavior:'smooth',block:'center'});}
  }
}
document.querySelectorAll('[data-score]').forEach(b=>b.onclick=()=>rate(Number(b.dataset.score)));
function configureRubric(){
  const r=manifest.rubric;
  $('rubric-target').textContent=r.target_label||'只评价中间的网络草稿';
  $('rubric-scale').textContent=r.instructions||'1 差 · 2 一般 · 3 好。参考布局与后处理结果用于辅助观察。';
  $('rubric-prompt').textContent=r.prompt||'这张草稿怎么样？';
  document.querySelectorAll('[data-score]').forEach(b=>b.querySelector('span').textContent=r.scale[b.dataset.score]);
  stages=manifest.view_stages||[{key:'reference',label:'原参考'},{key:'raw',label:'草稿放大'},{key:'final',label:'后处理'}];
  const all=[{key:'all',label:stages.length===2?'并排对照':'三相对照'},...stages];
  $('views').replaceChildren(...all.map(s=>{const b=document.createElement('button');b.dataset.view=s.key;b.textContent=s.label;b.onclick=()=>{view=s.key;draw();};return b;}));
  $('image').alt=manifest.image_description||'原参考、网络草稿、常规后处理三相图';
}
function renderDimensions(){
  const questions=manifest.rubric.questions||{};
  $('dimension-questions').replaceChildren(...Object.entries(questions).map(([key,q])=>{
    const box=document.createElement('div'),label=document.createElement('label'),choices=document.createElement('div');
    label.textContent=q.prompt+'（可选）';box.append(label);choices.className='row';
    for(const n of [1,2,3]){
      const b=document.createElement('button');b.type='button';b.dataset.dimension=key;b.textContent=q.scale[String(n)];
      b.classList.toggle('primary',dimensionAnswers[key]===n);
      b.onclick=()=>{dimensionAnswers[key]=dimensionAnswers[key]===n?null:n;dirty=true;renderDimensions();savedStatus('观察维度尚未保存');};choices.append(b);
    }
    box.append(choices);return box;
  }));
}
$('zoom').onchange=draw;
$('image').onload=()=>{imageReady=true;$('loading').hidden=true;draw();controls();};
$('image').onerror=()=>{imageReady=false;$('loading').hidden=true;error('图片未能载入，请刷新后重试。本例评分已禁用。');controls();};
$('note').oninput=$('tags').oninput=()=>{dirty=true;savedStatus('备注尚未保存');};
$('feedback').oninput=()=>{feedbackDirty=true;savedStatus('整轮评价尚未保存');};
$('save-note').onclick=()=>saveNote();
$('clear').onclick=()=>saveNote(null);
$('prev').onclick=()=>move(index-1);$('next').onclick=()=>move(index+1);
$('unrated').onclick=()=>{const k=nextUnrated();if(k>=0)move(k);};
$('go').onclick=()=>{const k=Number($('jump').value)-1;if(Number.isInteger(k))move(k);};
async function saveFeedback(){const ok=await update({action:'feedback',text:$('feedback').value});if(ok)feedbackDirty=false;return ok;}
$('save-feedback').onclick=saveFeedback;
$('finish').onclick=async()=>{
  if(dirty&&!await saveNote())return;
  if(feedbackDirty&&!await saveFeedback())return;
  if(await update({action:'complete'}))savedStatus('本轮评分已完成');
};
$('metrics').ontoggle=async()=>{
  if(!$('metrics').open||!manifest)return;
  const id=current().id;
  const ok=await update({action:'metrics_exposure',id});
  if(!ok||id!==current().id)return;
  $('numbers').textContent=JSON.stringify(current().auxiliary,null,2);
};
document.addEventListener('keydown',e=>{
  if(e.repeat||e.ctrlKey||e.altKey||e.metaKey||['INPUT','TEXTAREA','SELECT'].includes(e.target.tagName)||e.target.isContentEditable)return;
  if(['1','2','3','ArrowLeft','ArrowRight'].includes(e.key))e.preventDefault();
  if(['1','2','3'].includes(e.key))rate(Number(e.key));
  else if(e.key==='ArrowLeft')move(index-1);else if(e.key==='ArrowRight')move(index+1);
});
window.addEventListener('beforeunload',e=>{if(dirty||feedbackDirty||busy){e.preventDefault();e.returnValue='';}});
(async()=>{
  try{
    const response=await fetch('/api/session');if(!response.ok)throw Error('无法连接评分服务');
    const d=await response.json();manifest=d.manifest;state=d.state;token=d.token;
    $('title').textContent=manifest.title;$('scope').textContent=manifest.display_note;configureRubric();
    $('feedback').value=state.feedback||'';renderProgress();
    const k=nextUnrated(0);index=k>=0?k:0;render();savedStatus('已连接 · 自动恢复进度');
  }catch(e){error(e.message);}
})();
