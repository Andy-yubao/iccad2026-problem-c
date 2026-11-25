"""Publish verified E168 inputs privately and start one Kaggle training job."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
from zipfile import ZIP_DEFLATED, ZipFile

from experiments.E161_kaggle_transfer.deploy_api import (
    check_push_response, ensure_created, error_info, save, wait_dataset,
)
from experiments.E168_s1_progressive_unfreeze.kaggle_entry import PARENT, SOURCE


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE_DATASET = 'andyyubao/iccad-e161-private-transfer-20261123'
DATASET = 'andyyubao/iccad-e168-s1-input'
KERNEL = 'andyyubao/iccad-e168-s1'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def stage_inputs(folder, parent):
    frozen = json.loads((HERE / 'plan.json').read_text(encoding='utf-8'))['parent_initial']
    if digest(parent) != frozen['checkpoint_sha256']:
        raise ValueError('Input checkpoint differs from E167 best57')
    folder.mkdir(parents=True, exist_ok=True)
    files = {
        HERE / name for name in ('README.md', 'plan.json', 'requirements-kaggle.txt')
    }
    files.update(path for path in HERE.rglob('*.py') if 'runs' not in path.relative_to(HERE).parts)
    files.update(ROOT / name for name in (
        'datasets/__init__.py', 'datasets/paths.py', 'datasets/random_reference.py',
        'datasets/static_tensor_cache.py', 'experiments/__init__.py',
        'protocol/paths.json', 'protocol/random_reference40k_v1.json',
        'protocol/research_no_mib_v1.json', 'protocol/model_prediction_boundary_v1.json',
        'protocol/research_final_no_mib_v1.json', 'protocol/research_training36k_v1.json',
    ))
    project = folder / SOURCE
    with ZipFile(project, 'w', ZIP_DEFLATED) as package:
        for path in sorted(files):
            package.write(path, path.relative_to(ROOT).as_posix())
    archive_sha = digest(project)
    (folder / 'e168-project.sha256').write_text(archive_sha + '\n', encoding='utf-8')
    shutil.copyfile(parent, folder / PARENT)
    save(folder / 'dataset-metadata.json', dict(
        id=DATASET, title='ICCAD E168 S1 Input', licenses=[dict(name='other')],
    ))
    return archive_sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--stage-only', action='store_true')
    args = parser.parse_args()
    output = args.work_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    submission = output / 'submission.json'
    if submission.exists():
        print(submission.read_text(encoding='utf-8'), flush=True)
        return
    data, kernel = output / 'dataset', output / 'kernel'
    archive_sha = stage_inputs(data, args.parent)
    kernel.mkdir(exist_ok=True)
    save(kernel / 'E168_Kaggle.ipynb', dict(
        nbformat=4, nbformat_minor=5,
        metadata=dict(kernelspec=dict(name='python3', display_name='Python 3', language='python')),
        cells=[dict(id='e168-training', cell_type='code', metadata={}, execution_count=None,
                    outputs=[], source=(HERE / 'kaggle_entry.py').read_text(encoding='utf-8').splitlines(keepends=True))],
    ))
    save(kernel / 'kernel-metadata.json', dict(
        id=KERNEL, title='ICCAD E168 S1',
        code_file='E168_Kaggle.ipynb', language='python', kernel_type='notebook',
        is_private=True, enable_gpu=True, machine_shape='NvidiaTeslaT4', enable_internet=True,
        dataset_sources=[BASE_DATASET + '/1', DATASET + '/1'],
        competition_sources=[], kernel_sources=[], model_sources=[],
    ))
    if args.stage_only:
        print(json.dumps(dict(staged=str(output), source_sha256=archive_sha)), flush=True)
        return
    from kaggle import api

    username = api.get_config_value(api.CONFIG_NAME_USER)
    if not username or username.lower() != KERNEL.split('/')[0]:
        raise ValueError('Authenticate the Kaggle CLI as the project account before submitting')
    baseline = api.dataset_list_files(BASE_DATASET + '/1', page_size=20)
    if baseline.error_message:
        raise RuntimeError(baseline.error_message)
    available = {entry.name for entry in baseline.files}
    if not {'e161-transfer.tar.gz.bin', 'transfer-manifest.json'} <= available:
        raise RuntimeError('The frozen Kaggle training dataset is missing its archive or manifest')
    ensure_created(api, data, DATASET, archive_sha, output / 'upload_receipt.json')
    expected = {name: (data / name).stat().st_size for name in (SOURCE, PARENT, 'e168-project.sha256')}
    wait_dataset(api, DATASET, expected, wait_seconds=600)
    response = api.kernels_push(str(kernel), timeout='32400', acc='NvidiaTeslaT4')
    check_push_response(response)
    result = dict(
        kernel=KERNEL, remote_ref=response.ref or KERNEL, url=response.url, version=response.version_number,
        submitted_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=archive_sha, gpu_training=True, gpu_terminal_evaluation=False,
    )
    save(submission, result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps(error_info(error)), file=sys.stderr)
        raise SystemExit(1)
