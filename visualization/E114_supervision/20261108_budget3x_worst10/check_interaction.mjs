import fs from 'node:fs';
import vm from 'node:vm';
import crypto from 'node:crypto';
import assert from 'node:assert/strict';
const folder=new URL('.',import.meta.url);
const data=JSON.parse(fs.readFileSync(new URL('data.json',folder),'utf8'));
const fragment=fs.readFileSync('C:/Users/yuche/.codex/visualizations/2026/11/08/01a1baf6-f05b-7b23-b093-fdfe9beb1d00/budget-worst-ten.html','utf8');
const code=fragment.match(/<script>([\s\S]*)<\/script>/)[1];
class Element{
 constructor(id=''){this.id=id;this.value='';this.checked=id==='bw-overlap';this.textContent='';this.children=[];this.style={};this.listeners={};this.dataset={};this.width=1000;this.height=1000;this.clientWidth=320;}
 append(...items){this.children.push(...items);if(this.id==='bw-case'&&this.value==='')this.value=items[0].value;}
 replaceChildren(){this.children=[];}
 addEventListener(type,callback){this.listeners[type]=callback;}
 dispatchEvent(event){this.listeners[event.type]?.(event);}
 getBoundingClientRect(){return{left:0,top:0,width:320,height:320};}
 getContext(){return this.context;}
}
const ids=['case','overlap','layer','depth','meta','draft-label','reference','draft','final','selection','legend','detail'];
const elements=Object.fromEntries(ids.map(id=>['bw-'+id,new Element('bw-'+id)]));
elements['bw-layer'].value='12';
for(const id of ['reference','draft','final']){
 const ctx={ops:[],font:'14px system-ui',clearRect(){this.ops=[];},setLineDash(){},measureText(t){return{width:t.length*(parseFloat(this.font)||14)*.55};}};
 for(const key of ['fillRect','strokeRect','fillText'])ctx[key]=function(...args){this.ops.push([key,...args]);};
 elements['bw-'+id].context=ctx;
}
const root=new Element('iccad-worst-budget');root.querySelector=id=>elements[id.slice(1)];
const palette={foreground:'#202020',background:'#ffffff','muted-foreground':'#666666',muted:'#eeeeee',green:'#15803d',red:'#b91c1c'};
const saved=[];
const sandbox={document:{getElementById:id=>id===root.id?root:null,createElement:()=>new Element(),documentElement:{}},
 getComputedStyle:()=>({fontSize:'14px',color:'#202020',getPropertyValue:key=>palette[key.slice(2)]||''}),
 ResizeObserver:class{observe(){}},MutationObserver:class{observe(){}},
 window:{openai:{setWidgetState:snapshot=>{saved.push(snapshot);return Promise.resolve();}},addEventListener(){}},
 console};
vm.runInNewContext(code,sandbox,{timeout:20000});
assert.equal(elements['bw-case'].children.length,10);
const hash=id=>crypto.createHash('sha256').update(JSON.stringify(elements['bw-'+id].context.ops)).digest('hex');
let checked=0;
for(let i=0;i<10;i++){
 elements['bw-case'].value=String(i);elements['bw-case'].dispatchEvent({type:'change'});
 const left=hash('reference'),right=hash('final'),draftHashes=new Set();
 assert.equal(Number(root.dataset.evaluationIndex),data.cases[i].evaluation_index);
 for(let layer=1;layer<=12;layer++){
  elements['bw-layer'].value=String(layer);elements['bw-layer'].dispatchEvent({type:'input'});
  assert.equal(root.dataset.layer,String(layer));assert.equal(hash('reference'),left);assert.equal(hash('final'),right);
  assert(elements['bw-draft'].context.ops.length>20);draftHashes.add(hash('draft'));checked++;
 }
 assert(draftHashes.size>1);
}
elements['bw-case'].value='0';elements['bw-case'].dispatchEvent({type:'change'});
const c=data.cases[0],r=c.final[0],s=(1000-68)/(c.bounds[1]-c.bounds[0]);
elements['bw-final'].dispatchEvent({type:'click',clientX:(34+(r[0]+r[2]/2-c.bounds[0])*s)*.32,clientY:(1000-34-(r[1]+r[3]/2-c.bounds[2])*s)*.32});
assert(Number(root.dataset.selectedModule)>=0);assert(elements['bw-selection'].textContent.includes('模块'));
assert(saved.at(-1).modelContent.display_id==='e114-budget3x-worst10');
const report={status:'passed',method:'Offline DOM/canvas operation harness plus saved-data and static-preview inspection; no live browser claimed',case_layer_states:checked,
 left_and_right_invariant_when_sliding:true,middle_changes_across_layers:true,module_cross_selection:true,interaction_state_saved:true,
 new_inference:false,new_postprocessing:false};
fs.writeFileSync(new URL('interaction_check.json',folder),JSON.stringify(report,null,2)+'\n');
console.log(JSON.stringify(report));
