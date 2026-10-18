"""Render a stratified gallery from saved old E7 + E8 dev100 outputs only."""
from pathlib import Path
import html
import json
from PIL import Image
from visualization.layouts import plot_case, geometry_details
from .records import read_rows, save, sha


def main():
    root = Path(__file__).resolve().parents[2]
    sources = {
        'final': root / 'experiments/E8_fast_robust_postprocess/runs/20261018_robust_dev100_seed0/cases.jsonl',
        'draft': root / 'experiments/E8_fast_robust_postprocess/runs/20261018_stageA_dev100_seed0/drafts.jsonl',
        'reference': root / 'experiments/intent_aware_repair/runs/20261017_geometry_only_factorial_seed0/evaluation/cases.jsonl',
    }
    records = {key: {r['case_id']: r for r in read_rows(path)} for key, path in sources.items()}
    assert len(records['final']) == 100
    assert all(set(records['final']) <= set(rows) for rows in records.values())
    ordered = sorted(records['final'].values(), key=lambda r: (r['variants']['conic_robust']['final']['metrics']['cost'], r['case_id']))
    selected = [('low', i) for i in range(5)] + [('middle', i) for i in range(47, 52)] + [('high', i) for i in range(95, 100)]
    out = Path(__file__).resolve().parent / 'runs/20261018_old_model_e8_dev100_gallery15_seed0'
    out.mkdir(parents=True, exist_ok=True)
    items = []
    for band, index in selected:
        row = ordered[index]
        case_id = row['case_id']
        reference = records['reference'][case_id]
        draft = records['draft'][case_id]
        assert row['source']['shard'] == draft['source']['shard'] == reference['source']['shard']
        assert row['source']['sample_index'] == draft['source']['sample_index'] == reference['source']['sample_index']
        final = row['variants']['conic_robust']['final']
        constraints = reference['public_constraints']
        variants = {
            'reference': {'positions': reference['reference'], 'metrics': reference['reference_metrics']},
            'raw': {'positions': draft['draft']},
            'final': {'positions': final['positions'], 'metrics': final['metrics']},
        }
        for value in variants.values():
            value['soft_details'] = geometry_details(value['positions'], constraints)
        q = final['metrics']['cost']
        filename = f'{band}_rank{index + 1:03d}.png'
        metadata = plot_case({'case_id': case_id, 'panel': f'dev100 rank {index + 1}/100 | final Q={q:.6f}',
                              'constraints': constraints, 'variants': variants}, out / filename,
                             stages=(('reference', 'Reference layout (diagnostic)'), ('raw', 'Old E7 model proposal'),
                                     ('final', f'Old E7 + E8 | Q={q:.6f}')))
        with Image.open(out / filename) as image:
            dimensions = list(image.size)
        items.append({'band': band, 'rank': index + 1, 'case_id': case_id, 'Q': q,
                      'metrics': final['metrics'], 'image': filename, 'dimensions': dimensions,
                      'sha256': sha(out / filename), 'render': metadata})
        print(f'Rendered {filename}: Q={q:.6f}', flush=True)
    assert len({item['case_id'] for item in items}) == 15
    labels = {'low': '低 Q：排名 1–5', 'middle': '中间 Q：排名 48–52', 'high': '高 Q：排名 96–100'}
    content = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>旧模型＋E8：dev100 布局观察</title>
<style>body{font-family:system-ui;max-width:1700px;margin:28px auto;padding:0 24px;color:#182330}p{line-height:1.8}img{width:100%;height:auto}article{border:1px solid #ddd;border-radius:10px;margin:22px 0;padding:16px}nav{position:sticky;top:0;background:white;padding:14px;border-bottom:1px solid #ddd}nav a{margin-right:22px}table{border-collapse:collapse}td,th{padding:8px 16px;border:1px solid #ddd}h2{scroll-margin-top:65px}</style>
<h1>旧模型＋E8：15例 dev100 布局观察</h1>
<p>Q 越低越好。按最终 Q 排序，选最低5例、中间5例、最高5例，供观察差异；这15例不是随机评估集。完整 dev100 的平均 Q 为 1.211556。这里展示的是旧 E7 attention_geometry 模型＋E8 conic_robust。</p>
<p>每图从左到右：原参考布局、旧模型原始预测、E8 后处理后的最终布局。同一案例三栏使用相同模块编号、颜色、纹理及坐标尺度。原始预测未重新评分，n/a 表示未提供指标。参考布局只用于观察，未作为本次预测或后处理的输入。</p>
<p>颜色代表分组 G，纹理代表同形组 M；F 为固定形状，P 为预放置，L/R/T/B 为边界要求（绿：满足；红：未满足）。虚线是各自包围盒。红色重叠轮廓仅供视觉检查，其容差比官方硬合法判定更严格；最终15例均通过保存的官方硬合法检查。B/G/M 是官方软约束违规数。Q 使用中性运行时间口径，不是正式比赛总分。</p>
<nav><a href="#low">低 Q 5例</a><a href="#middle">中间 Q 5例</a><a href="#high">高 Q 5例</a></nav>'''
    for band in labels:
        content += f'<h2 id="{band}">{labels[band]}</h2>'
        for item in [r for r in items if r['band'] == band]:
            m = item['metrics']
            assert m['is_feasible'] is True
            content += f'<article><h3>第 {item["rank"]}/100 名 · Q = {item["Q"]:.6f}</h3><p>{html.escape(item["case_id"])} · HPWL gap = {m.get("hpwl_gap", "n/a")} · Area gap = {m.get("area_gap", "n/a")} · B/G/M = {m.get("boundary_violations")}/{m.get("grouping_violations")}/{m.get("mib_violations")}</p><a href="{item["image"]}" target="_blank"><img loading="lazy" src="{item["image"]}" alt="排名{item["rank"]}布局对照"></a><p>点击图片查看原图并放大模块编号。</p></article>'
    (out / 'index.html').write_text(content + '</html>', encoding='utf-8')
    save(out / 'render_manifest.json', {'selection': 'Final Q ascending, ties by case_id; ranks 1-5, 48-52, 96-100',
         'sources': {key: {'path': str(path.relative_to(root)), 'sha256': sha(path)} for key, path in sources.items()},
         'renderer_sha256': sha(root / 'visualization/layouts.py'), 'script_sha256': sha(Path(__file__)),
         'dataset_tensor_reads': 0, 'inference_calls': 0, 'solver_calls': 0, 'images': items})
    print(json.dumps({'images': len(items), 'index': str(out / 'index.html')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
