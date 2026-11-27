"""Verify the completed image export without loading data/model or rescoring."""
from pathlib import Path
import json
import hashlib
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
config = json.loads((HERE/'config.json').read_text(encoding='utf-8'))
expected = []
for ordinal, record in enumerate(config['members'], 1):
    shard, row = record['member']
    name = f'{ordinal:03d}_{Path(shard).parent.name}_{Path(shard).stem}_row{row:03d}_n{record["n"]}.png'
    expected.append(name)
assert set(expected) == {p.name for p in (HERE/'images').iterdir()}
images = []
for name in expected:
    path = HERE/'images'/name
    with Image.open(path) as image:
        size = list(image.size)
        image.verify()
    images.append({'file': name, 'size': size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest() == h for p, h in config['sources'].items())
drafts = json.loads((HERE/'drafts_local.json').read_text(encoding='utf-8'))
assert all((HERE/'images'/r['image']).exists() for r in drafts)
summary = {'status': 'complete', 'count': len(images), 'image_folder': str(HERE/'images'),
           'all_images_decode': True, 'exact_frozen_member_names': True, 'sources_unchanged': True,
           'workers': 4, 'threads_per_worker': 1, 'postprocessing': False, 'expansion': False, 'scoring': False,
           'free_reference_geometry_indexed': False, 'images': images,
           'coordinate_records_recovered': len(drafts),
           'report_write_interruption': 'All100 images rendered. Parent JSON replacement failed at38 because of Windows file locking. Image-only offline verification recovered completion; no repeated inference.',
           'test_exposure': config['test_exposure']}
(HERE/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
print(json.dumps({k:v for k,v in summary.items() if k!='images'}, ensure_ascii=False))
