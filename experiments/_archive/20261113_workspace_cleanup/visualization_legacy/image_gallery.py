"""Static reference/generated comparisons, with no scoring controls."""
import html
from pathlib import Path


def write_image_gallery(folder, cases, *, title, note):
    folder = Path(folder)
    cards = []
    for case in cases:
        label = html.escape(f"{case['member'][0]} : {case['member'][1]}")
        reference = html.escape(case['reference'], quote=True)
        generated = html.escape(case['generated'], quote=True)
        cards.append(f'<article><h2>{label}</h2><div class="pair">'
            f'<figure><figcaption>Reference</figcaption><a href="{reference}"><img loading="lazy" src="{reference}"></a></figure>'
            f'<figure><figcaption>生成图片</figcaption><a href="{generated}"><img loading="lazy" src="{generated}"></a></figure>'
            '</div></article>')
    content = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>''' + html.escape(title) + '''</title>
<style>body{margin:0;background:#eef2f6;color:#172033;font:15px system-ui,sans-serif}
header,main{max-width:1180px;margin:auto;padding:24px}header h1{font-size:25px;margin:0 0 12px}
header p{line-height:1.7;margin:0}main{padding-top:0}article{background:white;margin:0 0 24px;padding:18px;border-radius:12px}
h2{font-size:14px;margin:0 0 14px;font-weight:500}.pair{display:grid;grid-template-columns:1fr 1fr;gap:16px}
figure{margin:0}figcaption{margin-bottom:8px;color:#475569}img{display:block;width:100%;height:auto;border:1px solid #dce2ea;box-sizing:border-box}
@media(max-width:600px){header,main{padding:12px}.pair{gap:8px}article{padding:10px}}</style>
<header><h1>''' + html.escape(title) + '</h1><p>' + html.escape(note) + '</p></header><main>' + ''.join(cards) + '</main></html>'
    temporary = folder / "index.html.tmp"
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(folder / "index.html")
