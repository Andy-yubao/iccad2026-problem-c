"""Fetch the selected reading set from primary sources without replacing PDFs."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'references/papers'
MANIFEST = DEST / 'manifest.json'
PAPERS = [
    ('floorset', 'FloorSet -- a VLSI Floorplanning Dataset with Design Constraints of Real-World SoCs',
     'FloorSet_2405.05480.pdf', 'https://arxiv.org/abs/2405.05480',
     'https://arxiv.org/pdf/2405.05480', 'Existing local original; preserve bytes'),
    ('trinity', 'Trinity: One Differentiable Physics for Training, Refining and Scoring Generative Floorplanners',
     'Trinity_2610.04957v1.pdf', 'https://arxiv.org/abs/2610.04957v1',
     'https://arxiv.org/pdf/2610.04957v1', 'arXiv v1, 2026-10-04; preprint'),
    ('chipdiffusion', 'Chip Placement with Diffusion Models',
     'ChipDiffusion_ICML2025.pdf', 'https://proceedings.mlr.press/v267/lee25y.html',
     'https://raw.githubusercontent.com/mlresearch/v267/main/assets/lee25y/lee25y.pdf', 'ICML 2025 published proceedings'),
    ('flowplace', 'FlowPlace: Flow Matching for Chip Placement',
     'FlowPlace_2604.23658v2.pdf', 'https://arxiv.org/abs/2604.23658v2',
     'https://arxiv.org/pdf/2604.23658v2', 'arXiv v2, 2026-06-01; metadata lists DAC 2026'),
    ('diffplace', 'DiffPlace: A Conditional Diffusion Framework for Simultaneous VLSI Placement Beyond Sequential Paradigms',
     'DiffPlace_2510.15897v2.pdf', 'https://arxiv.org/abs/2510.15897v2',
     'https://arxiv.org/pdf/2510.15897v2', 'arXiv v2, 2026-04-07'),
    ('maskplace', 'MaskPlace: Fast Chip Placement via Reinforced Visual Representation Learning',
     'MaskPlace_NeurIPS2022.pdf', 'https://papers.nips.cc/paper_files/paper/2022/hash/97c8a8eb0e5231d107d0da51b79e09cb-Abstract-Conference.html',
     'https://papers.nips.cc/paper_files/paper/2022/file/97c8a8eb0e5231d107d0da51b79e09cb-Paper-Conference.pdf', 'NeurIPS 2022 published proceedings'),
    ('chipformer', 'ChiPFormer: Transferable Chip Placement via Offline Decision Transformer',
     'ChiPFormer_ICML2023.pdf', 'https://proceedings.mlr.press/v202/lai23c.html',
     'https://proceedings.mlr.press/v202/lai23c/lai23c.pdf', 'ICML 2023 published proceedings'),
    ('rlfloorplan', 'Learn to Floorplan through Acquisition of Effective Local Search Heuristics',
     'LearnToFloorplan_ICCD2020.pdf', 'https://personal.hkust-gz.edu.cn/yuzhema/papers/C16-ICCD2020-RLF.pdf',
     'https://personal.hkust-gz.edu.cn/yuzhema/papers/C16-ICCD2020-RLF.pdf', 'ICCD 2020; author-hosted copy'),
    ('ruleplanner', 'RulePlanner: All-in-One Reinforcement Learner for Unifying Design Rules in 3D Floorplanning',
     'RulePlanner_2601.22476v2.pdf', 'https://arxiv.org/abs/2601.22476v2',
     'https://arxiv.org/pdf/2601.22476v2', 'arXiv v2, 2026-05-26; metadata lists ICML 2026'),
    ('dc3', 'DC3: A learning method for optimization with hard constraints',
     'DC3_2104.12225.pdf', 'https://arxiv.org/abs/2104.12225',
     'https://arxiv.org/pdf/2104.12225', 'arXiv snapshot; PDF version recorded after extraction; ICLR 2021'),
    ('parsac', 'PARSAC: Fast, Human-quality Floorplanning for Modern SoCs with Complex Design Constraints',
     'PARSAC_2405.05495v3.pdf', 'https://arxiv.org/abs/2405.05495v3',
     'https://arxiv.org/pdf/2405.05495v3', 'arXiv v3, 2024-08-01'),
]


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    previous = json.loads(MANIFEST.read_text(encoding='utf-8')) if MANIFEST.exists() else {}
    old = {p['id']: p for p in previous.get('papers', [])}
    records = []
    for key, title, filename, landing, source, edition in PAPERS:
        path = DEST / filename
        record = dict(old.get(key, {}))
        record.update(id=key, title=title, path=path.relative_to(ROOT).as_posix(),
                      landing_url=landing, source_url=source, edition=edition)
        try:
            if path.exists():
                blob = path.read_bytes()
                sha = hashlib.sha256(blob).hexdigest()
                if record.get('sha256') and record['sha256'] != sha:
                    raise ValueError('Existing PDF differs from manifest; refusing overwrite')
                record.setdefault('acquisition', 'preexisting_local' if key == 'floorset' else 'existing_local')
            else:
                if key == 'floorset':
                    raise FileNotFoundError('Expected existing FloorSet original is missing; not re-downloading it')
                request = urllib.request.Request(source, headers={'User-Agent': 'FloorSet-literature-research/1.0'})
                with urllib.request.urlopen(request, timeout=45) as response:
                    blob = response.read(50 * 1024 * 1024 + 1)
                    record['resolved_url'] = response.url
                    record['content_type'] = response.headers.get('Content-Type')
                if len(blob) > 50 * 1024 * 1024:
                    raise ValueError('PDF exceeds 50 MiB bounded download')
                if not blob.startswith(b'%PDF-'):
                    raise ValueError('Response is not a PDF')
                with path.open('xb') as output:
                    output.write(blob)
                record['acquisition'] = 'downloaded_from_primary_source'
                record['downloaded_at_utc'] = datetime.now(timezone.utc).isoformat()
            if not blob.startswith(b'%PDF-'):
                raise ValueError('Existing file is not a PDF')
            record.update(bytes=len(blob), sha256=hashlib.sha256(blob).hexdigest(), status='available')
            record.pop('error', None)
            print(f'{key}: available, {len(blob)} bytes', flush=True)
        except Exception as exc:
            record.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            print(f'{key}: {record["error"]}', flush=True)
        records.append(record)
        payload = dict(previous)
        payload.update({
            'collection_date_local': '2026-10-16',
            'timezone': 'Asia/Shanghai',
            'scope': 'Selected reading set; no model/data download, training or inference',
            'papers': records,
        })
        MANIFEST.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return int(any(p['status'] != 'available' for p in records))


if __name__ == '__main__':
    raise SystemExit(main())
