/* Render the reader from the Markdown; no dataset/model/solver access. */
const fs = require('node:fs');
const path = require('node:path');
const {pathToFileURL} = require('node:url');
const crypto = require('node:crypto');

async function main() {
  const packageRoot = process.argv[2];
  if (!packageRoot) throw new Error('Pass the bundled node_modules directory.');
  const {marked} = await import(pathToFileURL(require.resolve('marked', {paths:[packageRoot]})));
  const dir = __dirname, root = path.resolve(dir, '../..');
  const source = fs.readFileSync(path.join(dir, '项目研究全貌报告.md'), 'utf8');
  const toc = [];
  let section = 0;
  let body = marked.parse(source, {gfm:true, breaks:false});
  body = body.replace(/<h2>(.*?)<\/h2>/g, (_, label) => {
    const id = 'section-' + (++section);
    toc.push({id,label});
    return `<h2 id="${id}">${label}</h2>`;
  });
  body = body.replace(/<table>/g, '<div class="table-wrap"><table>').replace(/<\/table>/g, '</table></div>');
  body = body.replace(/<p><img src="([^"]+)" alt="([^"]*)"><\/p>/g,
    '<figure><a href="$1" target="_blank" title="打开原图放大"><img src="$1" alt="$2" loading="lazy"></a><figcaption>点击查看保存的原始四阶段图</figcaption></figure>');
  const nav = toc.map(x=>`<a href="#${x.id}">${x.label}</a>`).join('');
  const html = `<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ICCAD Problem C · 项目研究全貌报告</title>
<style>
:root{--ink:#1b2c3a;--muted:#62717e;--accent:#167071;--line:#dce4e7;--paper:#fff;--wash:#f3f6f6}
*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:28px}body{margin:0;color:var(--ink);background:var(--wash);font:17px/1.9 "Microsoft YaHei UI","Microsoft YaHei","Noto Sans CJK SC",system-ui,sans-serif}
aside{position:fixed;inset:0 auto 0 0;width:292px;padding:32px 22px;overflow:auto;background:#f8faf9;border-right:1px solid var(--line)}
.brand{font-size:13px;letter-spacing:.15em;color:var(--accent);font-weight:700}.aside-title{font-size:24px;line-height:1.45;margin:12px 0 20px;font-weight:700}.nav-label{font-size:12px;color:var(--muted);margin:14px 0}
nav a{display:block;font-size:13px;line-height:1.65;padding:7px 9px;margin:3px 0;border-left:2px solid transparent;color:#405561;text-decoration:none}nav a:hover,nav a.active{background:#e5eeee;border-color:var(--accent);color:#075658}
.reader-links{font-size:12px;padding-top:16px;margin-top:16px;border-top:1px solid var(--line)}main{margin-left:292px;padding:48px 48px 88px}.paper{max-width:1100px;margin:auto;background:var(--paper);padding:54px 62px;border:1px solid var(--line);border-radius:10px}
h1{font-size:36px;line-height:1.4;letter-spacing:-.02em;margin:0 0 24px}h2{font-size:27px;line-height:1.5;margin:64px 0 23px;padding:0 0 12px;border-bottom:2px solid #d8e7e5}h3{font-size:21px;line-height:1.6;margin:34px 0 14px}p{margin:0 0 19px}strong{font-weight:700;color:#153b48}a{color:#096d75;text-underline-offset:3px}a:hover{color:#08454a}li{padding-left:3px;margin:7px 0}ul,ol{padding-left:25px;margin:15px 0 23px}
pre{background:#f3f7f7;border:1px solid #dce7e6;border-left:4px solid #4f8a89;border-radius:5px;padding:20px 23px;overflow-x:auto;font:14px/1.85 Consolas,"Microsoft YaHei",monospace;white-space:pre}code{font-family:Consolas,"Microsoft YaHei",monospace;font-size:.88em}
.table-wrap{overflow:auto;margin:22px 0 27px;border:1px solid var(--line);border-radius:6px}table{width:100%;border-collapse:collapse;font-size:14px;line-height:1.8}th,td{text-align:left;vertical-align:top;padding:12px 14px;border-bottom:1px solid var(--line);min-width:104px}th{background:#eaf1f0;font-weight:700;color:#244d56}tr:last-child td{border-bottom:0}tbody tr:nth-child(even){background:#fafcfc}
figure{margin:30px -24px;padding:12px;background:#f6f8f9;border:1px solid var(--line);border-radius:5px}figure img{display:block;width:100%;height:auto;background:white}figcaption{font-size:12px;color:var(--muted);text-align:center;margin-top:9px}.footer{font-size:12px;color:var(--muted);margin-top:46px;padding-top:15px;border-top:1px solid var(--line)}
@media(max-width:1100px){aside{width:240px;padding:24px 16px}main{margin-left:240px;padding:24px}.paper{padding:36px}h1{font-size:30px}body{font-size:16px}}
@media(max-width:760px){aside{position:static;width:auto;border-right:0;border-bottom:1px solid var(--line);padding:23px}nav{columns:2}nav a{break-inside:avoid}.aside-title{font-size:22px;margin:8px 0}.reader-links{margin-top:8px}main{margin:0;padding:15px}.paper{padding:26px 20px}h1{font-size:27px}h2{font-size:23px;margin-top:44px}figure{margin:23px -12px}pre{font-size:12px;padding:14px}}
@media print{body{background:white;font-size:10pt;line-height:1.65}aside{display:none}main{margin:0;padding:0}.paper{border:0;padding:0;max-width:none}h1{font-size:23pt}h2{font-size:17pt;margin-top:25pt;break-after:avoid}h3{font-size:13pt;break-after:avoid}p{orphans:3;widows:3}table{font-size:8pt}tr,figure,pre{break-inside:avoid}figure{margin:15pt 0}a{color:inherit;text-decoration:none}.table-wrap{overflow:visible}th,td{padding:6pt}.footer{font-size:8pt}@page{size:A4;margin:18mm}}
</style></head><body>
<aside><div class="brand">ICCAD 2026 · PROBLEM C</div><div class="aside-title">项目研究<br>全貌报告</div><div class="nav-label">2026-10-30 · 冻结证据整理</div><nav>${nav}</nav><div class="reader-links"><a href="项目研究全貌报告.md">Markdown 原稿</a> · <a href="README.md">交付说明</a><p>可离线阅读；浏览器打印可保存 PDF。</p></div></aside>
<main><article class="paper">${body}<div class="footer">状态截点：2026-10-30，北京时间。图像和实验数据来自既有保存结果；本报告没有启动训练、推理或求解。完整来源见附录与 sources.json。</div></article></main>
<script>const navLinks=[...document.querySelectorAll('nav a')];const observer=new IntersectionObserver(items=>{for(const x of items){if(x.isIntersecting){navLinks.forEach(a=>a.classList.toggle('active',a.hash==='#'+x.target.id));}}},{rootMargin:'0px 0px -75% 0px'});document.querySelectorAll('h2').forEach(x=>observer.observe(x));</script>
</body></html>`;
  fs.writeFileSync(path.join(dir,'项目研究全貌报告.html'),html,'utf8');

  const links = [...source.matchAll(/\]\((\.\.\/\.\.\/[^)]+)\)/g)].map(x=>x[1]);
  const unique = [...new Set(links)];
  const missing = unique.filter(x=>!fs.existsSync(path.resolve(dir,x)));
  if (missing.length) throw new Error('Missing sources: '+JSON.stringify(missing));
  const sources = unique.map(rel=>{
    const filename=path.resolve(dir,rel), stat=fs.statSync(filename);
    return {path:path.relative(root,filename).split(path.sep).join('/'),bytes:stat.size,
      sha256:crypto.createHash('sha256').update(fs.readFileSync(filename)).digest('hex')};
  });
  const manifest={report:'项目研究全貌报告.md',state_date:'2026-10-30',timezone:'Asia/Shanghai',
    generated_at:new Date().toISOString(),scope:'Read existing project source and frozen run reports; reference existing images; no new inference, training, solver evaluation, or sealed-test reads.',
    theory:'General GNN and Transformer concepts checked against the implementation; no external literature search in this report.',
    sources,verification:{project_links:unique.length,missing_links:missing,top_level_sections:toc.length,
      reused_layout_images:2,reused_figures_visually_inspected:true,new_layout_inference:0,new_solver_evaluations:0,
      html_browser_preview:'Not performed: browser policy disallows local file URLs. Reader checked statically, no browser-rendering claim.'},
    renderer:'build_report.cjs; bundled marked; reader uses no external assets or network calls'};
  fs.writeFileSync(path.join(dir,'sources.json'),JSON.stringify(manifest,null,2)+'\n','utf8');
  const unresolvedCitations=[...body.matchAll(/\[S\d{2}\]/g)].map(x=>x[0]);
  const linkedCitations=(body.match(/>S\d{2}<\/a>/g)||[]).length;
  if (unresolvedCitations.length) throw new Error('Unresolved evidence citations: '+unresolvedCitations);
  console.log(JSON.stringify({sections:toc.length,source_links:unique.length,missing,linked_citations:linkedCitations,
    figures:(body.match(/<figure>/g)||[]).length,markdown_characters:source.length,
    markdown_bytes:Buffer.byteLength(source),html_bytes:Buffer.byteLength(html)}));
}
main().catch(e=>{console.error(e);process.exitCode=1});
