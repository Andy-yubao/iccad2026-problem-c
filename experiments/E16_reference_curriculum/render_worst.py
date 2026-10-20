"""Render ten worst saved model-1 final layouts; no inference or repair."""
import argparse
import html
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageOps
from visualization.layouts import plot_case, geometry_details
from .cache import PackedRows
from .io import HERE, ROOT, save, sha


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', type=Path, default=HERE/'runs/20261019_reference_v5_supervised_small_seed0')
    args = ap.parse_args(); run = args.run.resolve()
    if not run.is_relative_to(HERE/'runs'):
        raise ValueError('Run outside E16')
    source = run/'final_evaluation/results.json'
    result = json.loads(source.read_text(encoding='utf-8'))
    assert result['status'] == 'complete' and len(result['records']) == 500
    worst = sorted(result['records'], key=lambda r: (-r['metrics']['cost'], r['member']))[:10]
    rows = PackedRows('dev_monitor500')
    lookup = {tuple(r['member']): i for i, r in enumerate(rows.records)}
    out = run/'final_evaluation/worst10'; out.mkdir(exist_ok=False)
    font = 'C:/Windows/Fonts/msyh.ttc'
    titlefont, smallfont = ImageFont.truetype(font, 25), ImageFont.truetype(font, 18)
    tiles, manifest = [], []
    for rank, record in enumerate(worst, 1):
        row = rows[lookup[tuple(record['member'])]]
        c = row['graph']['constraints'].tolist()
        positions, m = record['positions'], record['metrics']
        name = f'worst_{rank:02d}.png'
        member = f"{record['member'][0]}:{record['member'][1]}"
        saved = {'case_id': member, 'panel': f'E16 model 1 worst #{rank:02d}', 'constraints': c,
                 'caption': f"Saved final output | HPWL {m['hpwl_gap']:+.1%} | bbox {m['area_gap']:+.1%} | solve {record['solve_seconds']:.2f}s",
                 'variants': {'final': {'positions': positions, 'metrics': m,
                                       'soft_details': geometry_details(positions, c)}}}
        bounds = plot_case(saved, out/name, stages=(('final', 'Model 1 + frozen E14 postprocessing'),))
        with Image.open(out/name) as picture:
            image = picture.convert('RGB')
        w, h = image.size; x,y,ww,hh = bounds['axes_bounds'][0]
        crop = image.crop((int(max(0,x-.055)*w), int(max(0,1-y-hh-.025)*h),
                           int(min(1,x+ww+.025)*w), int(min(1,1-y+.035)*h)))
        tile = Image.new('RGB', (500,650), 'white'); draw = ImageDraw.Draw(tile)
        draw.text((16,12), f"#{rank:02d}  Q = {m['cost']:.4f}", fill='#991b1b', font=titlefont)
        draw.text((16,52), member, fill='#334155', font=smallfont)
        thumb = ImageOps.contain(crop, (472,480), Image.Resampling.LANCZOS)
        tile.paste(thumb, ((500-thumb.width)//2, 88+(480-thumb.height)//2))
        draw.text((16,580), f"HPWL {m['hpwl_gap']:+.1%}  bbox {m['area_gap']:+.1%}", fill='#334155', font=smallfont)
        draw.text((16,615), f"B/G/M {m['boundary_violations']}/{m['grouping_violations']}/{m['mib_violations']}   {record['solve_seconds']:.2f}s", fill='#334155', font=smallfont)
        tiles.append(tile)
        manifest.append({'rank': rank, 'member': record['member'], 'image': name, 'metrics': m,
                         'block_count': record['block_count'], 'fallback': record['fallback'],
                         'stability_triggered': record['stability_triggered'], 'solve_seconds': record['solve_seconds']})
        print(rank, member, m['cost'], flush=True)
    sheet = Image.new('RGB',(2500,1380),'#e2e8f0'); draw = ImageDraw.Draw(sheet)
    draw.text((25,12), 'E16 模型1：固定验证500例中，后处理后Q最差的10例（Q越高越差）', fill='#0f172a', font=ImageFont.truetype(font,30))
    for i, tile in enumerate(tiles): sheet.paste(tile, ((i%5)*500,70+(i//5)*655))
    sheet.save(out/'overview.png')
    save(out/'index.json', {'source': source.relative_to(ROOT).as_posix(), 'source_sha256': sha(source),
        'selection': 'Descending saved final Q; 10 of 500 dev_monitor cases', 'model': 'supervised_small',
        'summary': result['summary'], 'worst10': manifest, 'inference_calls': 0, 'postprocess_calls': 0,
        'test_read': False, 'raw_layouts': 'Not saved by evaluation; not reconstructed or regenerated'})
    lines = ['# 模型1：最终Q最差的10例', '',
             '从supervised_small固定验证500例已保存结果中，按后处理后的Q降序选取。只绘制已保存的最终布局，没有重新推理或后处理。', '',
             'Q越高越差。颜色区分分组G，纹理区分MIB；F/P表示固定/预放置块，L/R/T/B徽记表示边界约束状态。', '',
             '![十例总览](overview.png)', '', '| 排名 | 样本 | Q | 线长变化 | bbox面积变化 | B/G/M | 耗时 |', '|---:|---|---:|---:|---:|---|---:|']
    cards = []
    for r in manifest:
        m=r['metrics']; label=f"{r['member'][0]}:{r['member'][1]}"
        lines.append(f"| {r['rank']} | [{label}]({r['image']}) | {m['cost']:.6f} | {m['hpwl_gap']:+.2%} | {m['area_gap']:+.2%} | {m['boundary_violations']}/{m['grouping_violations']}/{m['mib_violations']} | {r['solve_seconds']:.2f}s |")
        cards.append(f"<section><h2>#{r['rank']}　Q={m['cost']:.6f}　{html.escape(label)}</h2><a href='{r['image']}'><img src='{r['image']}' loading='lazy'></a></section>")
    lines += ['', '10例均硬合法；第1例软违反为零，但线长和面积明显膨胀。现有终末评价未保存网络草稿坐标或E14救援前坐标，因此这些图不能单独定位退化发生在哪个步骤。', '']
    for r in manifest: lines += [f"## 第{r['rank']}例：Q={r['metrics']['cost']:.6f}", '', f"![第{r['rank']}例]({r['image']})", '']
    (out/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    (out/'gallery.html').write_text("<!doctype html><meta charset='utf-8'><title>E16 模型1最差10例</title><style>body{max-width:1300px;margin:30px auto;font:16px/1.6 sans-serif;background:#f1f5f9;padding:20px}img{max-width:100%}section{background:white;padding:20px;margin-bottom:30px}h2{font-size:20px}</style><h1>模型1：最终Q最差10例</h1><p>固定开发500例，Q降序；点击图片查看原图。只读已有布局，无重新推理或后处理。</p>"+''.join(cards),encoding='utf-8')


if __name__ == '__main__':
    main()
