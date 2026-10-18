"""Render saved E9 diagnostics using the shared layout visualization only."""
import argparse
from pathlib import Path
import json
from visualization.layouts import plot_case, geometry_details
from .records import read_rows, save, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    rows = read_rows(args.run / 'cases.jsonl')
    images = []
    for row in rows:
        worker = int(row['source']['shard'].split('/')[0].split('_')[1])
        variants = {'reference': {'positions': row['reference_positions'], 'metrics': row['reference_metrics']},
                    'current': row['variants']['raw']['final'],
                    'oracle': row['variants']['oracle_layout']['final']}
        for value in variants.values():
            if value['positions'] is not None:
                value['soft_details'] = geometry_details(value['positions'], row['constraints'])
        adapted = {'case_id': row['case_id'], 'panel': 'dev5 diagnostic', 'constraints': row['constraints'], 'variants': variants}
        image = args.run / f'worker_{worker}_comparison.png'
        plot_case(adapted, image, stages=(('reference', 'Reference (diagnostic only)'),
                                         ('current', 'E7 draft + frozen E8'),
                                         ('oracle', 'Reference draft + frozen E8 (oracle)')))
        images.append({'worker': worker, 'path': image.name, 'sha256': sha(image)})
    html = '<!doctype html><html lang="zh-CN"><meta charset="UTF-8"><title>E9 草稿诊断</title><style>body{font-family:system-ui;max-width:1450px;margin:30px auto;padding:0 20px}img{width:100%;margin-bottom:30px}p{line-height:1.7}</style><h1>E9：五例同实例草稿诊断</h1><p>左：原参考；中：旧模型＋冻结 E8；右：参考草稿＋冻结 E8。参考只用于诊断，右栏不是可部署方法。各图保持公共布局图样式。即使参考草稿也会受到公开规格中不可兼容 MIB 的罚项。</p>'
    for item in images:
        html += f'<h2>worker {item["worker"]}</h2><img src="{item["path"]}" alt="worker {item["worker"]} layout comparison">'
    (args.run / 'index.html').write_text(html + '</html>', encoding='utf-8')
    save(args.run / 'render_manifest.json', {'source_cases_sha256': sha(args.run / 'cases.jsonl'),
                                            'renderer': 'visualization.layouts.plot_case', 'images': images,
                                            'dataset_tensor_reads': 0, 'solver_calls': 0})
    print(json.dumps({'images': len(images), 'index': str(args.run / 'index.html')}), flush=True)


if __name__ == '__main__':
    main()
