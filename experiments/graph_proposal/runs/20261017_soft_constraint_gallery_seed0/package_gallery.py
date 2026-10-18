"""Package the completed visual deliverable, without model or dataset copies."""
from pathlib import Path
import json
import zipfile

run=Path(__file__).resolve().parent
manifest=json.loads((run/'gallery_manifest.json').read_text(encoding='utf-8'))
images=[run/f['image'] for f in manifest['figures']]
assert len(images)==30 and all(p.is_file() for p in images)
names=['index.html','analysis.md','results.json','config.json','gallery_manifest.json',
       'analysis_summary.json','cases.jsonl','ablation_cases.jsonl','soft_penalty_diagnostics.png','source_snapshot.zip']
files=[run/name for name in names]+images
target=run/'soft_constraint_gallery_30.zip'
with zipfile.ZipFile(target,'x',zipfile.ZIP_DEFLATED) as archive:
    for path in files:archive.write(path,path.relative_to(run))
with zipfile.ZipFile(target) as archive:
    assert archive.testzip() is None
    assert len([n for n in archive.namelist() if n.startswith('images/') and n.endswith('.png')])==30
print(json.dumps({'archive':str(target),'bytes':target.stat().st_size,'images':30,'files':len(files)}))
