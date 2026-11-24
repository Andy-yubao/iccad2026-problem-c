"""Build the private Kaggle script folder with source code only."""
import base64
import io
import json
from pathlib import Path
import zipfile

HERE = Path(__file__).resolve().parent
OWNER = HERE.parent
ROOT = OWNER.parents[1]
KERNEL_SLUG = 'iccad-e164-static-relation-l0-36k'
DATASET_SLUG = 'iccad-e161-private-transfer-20261123'


def build(username):
    files = sorted(OWNER.glob('*.py'))
    files += [OWNER / name for name in (
        'README.md', 'DESIGN.md', 'plan.json', 'provenance.json',
        'authorization.json', 'baseline_e149.json',
    )]
    files += [ROOT / name for name in (
        'datasets/__init__.py', 'datasets/random_reference.py', 'datasets/paths.py',
        'datasets/static_tensor_cache.py', 'protocol/paths.json',
        'protocol/research_no_mib_v1.json', 'protocol/model_prediction_boundary_v1.json',
    )]
    bundle = io.BytesIO()
    with zipfile.ZipFile(bundle, 'w', zipfile.ZIP_DEFLATED) as archive:
        for file in files:
            if file.name == '__init__.py' and not file.exists():
                continue  # A namespace package is sufficient for these modules.
            archive.write(file, file.relative_to(ROOT).as_posix())
    folder = HERE / 'submission'
    folder.mkdir(exist_ok=True)
    executable = (HERE / 'entry.py').read_text(encoding='utf-8').replace(
        '__E164_BUNDLE__', base64.b64encode(bundle.getvalue()).decode('ascii'), 1,
    )
    (folder / 'entry.py').write_text(executable, encoding='utf-8')
    metadata = dict(
        id=f'{username}/{KERNEL_SLUG}', title='ICCAD E164 Static Relation L0 36k',
        code_file='entry.py', language='python', kernel_type='script', is_private=True,
        enable_gpu=True, machine_shape='NvidiaTeslaT4', enable_internet=True,
        dataset_sources=[f'{username}/{DATASET_SLUG}'], competition_sources=[],
        kernel_sources=[], model_sources=[],
    )
    (folder / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    return metadata


if __name__ == '__main__':
    import sys
    print(json.dumps(build(sys.argv[1])))
