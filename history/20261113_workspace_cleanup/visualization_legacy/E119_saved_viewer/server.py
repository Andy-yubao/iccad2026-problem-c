"""Local, read-only browser for the existing E119 dev500 results."""
import json
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = Path(__file__).resolve().parents[2]
CASES = ROOT / 'experiments/E119_global_initial_generator/runs/global2_seed0/evaluation_no_mib_dev500/cases'
PAGE = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>E119 布局浏览</title>
<style>body{font:16px system-ui;margin:24px;background:#f3f5f8;color:#172033}select,button{font:inherit;padding:7px}main{display:flex;gap:20px}section{flex:1;min-width:0}canvas{background:white;width:100%;border:1px solid #ccd3dd}p{line-height:1.6}@media(max-width:700px){main{display:block}}</style>
<h1>E119 · best55 · dev500</h1><p>读取已有评价结果。左侧为真实末层草稿，右侧为原预算后处理结果；两栏使用相同坐标范围和模块编号。</p>
<button id="prev">上一例</button> <select id="pick"></select> <button id="next">下一例</button><p id="info"></p>
<main><section><h2>模型草稿 · L12</h2><canvas id="draft" width="1000" height="1000"></canvas></section><section><h2>后处理布局</h2><canvas id="final" width="1000" height="1000"></canvas></section></main>
<p>本页仅包含已存坐标和模块编号；原评价未保存 reference、约束标记和中间层，因此这里不显示这些信息。</p>
<script>
const pick=document.querySelector('#pick');for(let i=0;i<500;i++)pick.add(new Option('dev '+String(i).padStart(4,'0'),i));
async function show(){let response=await fetch('/case/'+pick.value);if(!response.ok)throw Error('读取失败');let d=await response.json();let all=[...d.draft_positions,...d.positions],lo=[Infinity,Infinity],hi=[-Infinity,-Infinity];all.forEach(p=>{for(let j=0;j<2;j++){lo[j]=Math.min(lo[j],p[j]);hi[j]=Math.max(hi[j],p[j]+p[j+2])}});let span=Math.max(hi[0]-lo[0],hi[1]-lo[1])*1.08,s=920/span,cx=(lo[0]+hi[0])/2,cy=(lo[1]+hi[1])/2;
function draw(id,ps){let c=document.querySelector(id),g=c.getContext('2d');g.clearRect(0,0,1000,1000);ps.forEach((p,i)=>{let x=500+(p[0]-cx)*s,y=500-(p[1]+p[3]-cy)*s,w=p[2]*s,h=p[3]*s;g.fillStyle='#d1d5db';g.fillRect(x,y,w,h);g.strokeStyle='#475569';g.lineWidth=1;g.strokeRect(x,y,w,h);g.fillStyle='#172033';g.font=Math.max(8,Math.min(18,h*.45,w*.5))+'px system-ui';g.textAlign='center';g.textBaseline='middle';g.fillText(i,x+w/2,y+h/2,Math.max(1,w-2));})}draw('#draft',d.draft_positions);draw('#final',d.positions);document.querySelector('#info').textContent=`dev ${d.index} · ${d.member.join(' / ')} · N=${d.n} · 最终 Q=${d.metrics.cost.toFixed(6)} · 完整耗时 ${(d.complete_seconds*1000).toFixed(2)} ms`;}
pick.onchange=()=>show().catch(e=>document.querySelector('#info').textContent=e.message);document.querySelector('#prev').onclick=()=>{pick.value=Math.max(0,+pick.value-1);pick.onchange()};document.querySelector('#next').onclick=()=>{pick.value=Math.min(499,+pick.value+1);pick.onchange()};pick.onchange();
</script></html>'''

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ('/', '/index.html'):
            body, kind = PAGE.encode('utf-8'), 'text/html; charset=utf-8'
        elif self.path.startswith('/case/') and self.path[6:].isdigit() and 0 <= int(self.path[6:]) < 500:
            body, kind = (CASES / f'{int(self.path[6:]):04d}.json').read_bytes(), 'application/json'
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

if __name__ == '__main__':
    ThreadingHTTPServer(('127.0.0.1', 8779), Handler).serve_forever()
