"""Render five highest-Q saved cases; no dataset access or inference."""
import hashlib
import html
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from visualization.layouts import plot_evaluation_case


def main():
    folder = Path(__file__).resolve().parent
    payload = (folder / 'cases.jsonl').read_bytes()
    rows = [json.loads(line) for line in payload.splitlines() if line.strip()]
    name = 'attention_geometry'
    ranked = sorted(rows, key=lambda row: (-row['variants'][name]['final']['metrics']['cost'], row['case_id']))
    output = folder / 'worst_completed'
    output.mkdir(exist_ok=True)
    records = []
    for rank, row in enumerate(ranked[:5], 1):
        value = row['variants'][name]
        filename = f'{rank:02d}_worst.png'
        plot_evaluation_case(row, name, output / filename)
        record = {'rank': rank, 'case_id': row['case_id'], 'Q': value['final']['metrics']['cost'],
                  'metrics': value['final']['metrics'], 'postprocess_seconds': value.get('postprocess_seconds'),
                  'guard_fallback': value.get('guard_fallback'), 'image': filename}
        records.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
    manifest = {'completed_cases': len(rows), 'selection': 'Top five by descending final Q among saved completed cases; retrospective diagnostic selection.',
                'cases_sha256': hashlib.sha256(payload).hexdigest(),
                'renderer_sha256': hashlib.sha256((ROOT / 'visualization/layouts.py').read_bytes()).hexdigest(),
                'test_tensor_reads': 0, 'inference_calls': 0, 'figures': records}
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    cards = ''.join(f'<section><h2>#{r["rank"]} Q={r["Q"]:.6f} · {html.escape(r["case_id"])}</h2>'
                    f'<a href="{r["image"]}"><img src="{r["image"]}" loading="lazy"></a></section>' for r in records)
    (output / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Highest Q saved layouts</title>'
        '<style>body{font-family:system-ui;margin:24px;background:#f1f5f9}section{background:white;padding:18px;margin:24px 0}img{width:100%}</style>'
        f'<h1>Highest Q: {len(records)} of {len(rows)} completed cases</h1><p>Retrospective diagnostic selection; no new inference.</p>' + cards, encoding='utf-8')


if __name__ == '__main__':
    main()
