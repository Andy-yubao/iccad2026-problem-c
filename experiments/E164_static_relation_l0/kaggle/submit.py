"""Submit one private E164 kernel against the already-published private dataset."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import sys

from prepare import HERE, ROOT, KERNEL_SLUG, build

# Reuse the deployed Kaggle client's established API and readiness checks.
sys.path.insert(0, str(ROOT / 'experiments/E161_kaggle_transfer'))
from deploy_api import check_push_response, error_info, log, save, wait_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('username')
    parser.add_argument('action', choices=('submit', 'status', 'download'))
    args = parser.parse_args()
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    actual = api.get_config_value(api.CONFIG_NAME_USER)
    if not actual or actual.lower() != args.username.lower():
        raise ValueError('The authenticated Kaggle account does not match the requested owner')
    kernel = f'{actual}/{KERNEL_SLUG}'
    receipt = HERE / 'submission.json'
    if args.action in {'status', 'download'} and receipt.exists():
        submitted = json.loads(receipt.read_bytes())
        if submitted['kernel'] != kernel:
            raise ValueError('The submission receipt belongs to a different Kaggle kernel')
        kernel = f"{kernel}/{submitted['version_number']}"
    if args.action == 'status':
        response = api.kernels_status(kernel)
        log(kernel=kernel, remote_status=str(response.status), failure_message=response.failure_message)
        return
    if args.action == 'download':
        destination = HERE / 'downloaded' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        destination.mkdir(parents=True)
        api.kernels_output(kernel, path=str(destination), quiet=False)
        paths = list(destination.rglob('bootstrap-status.json'))
        if len(paths) != 1 or json.loads(paths[0].read_bytes())['stage'] != 'complete':
            raise RuntimeError(f'Outputs saved to {destination}, but the run has no complete result.')
        result = paths[0].with_name('loss_results.json')
        freeze = json.loads(paths[0].with_name('frozen_best.json').read_bytes())
        checkpoint = paths[0].parent / 'runs/e164_l0_seed0/checkpoints/best.pt'
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != freeze['checkpoint_sha256']:
            raise ValueError('Downloaded best checkpoint differs from the frozen SHA256')
        log(stage='downloaded', result=str(result), **json.loads(result.read_bytes())['frozen'])
        return

    if receipt.exists():
        log(stage='already_submitted', **json.loads(receipt.read_bytes()))
        return
    metadata = build(actual)
    dataset, = metadata['dataset_sources']
    readiness = wait_dataset(api, dataset, {
        'e161-transfer.tar.gz.bin': 982174149, 'transfer-manifest.json': 573742,
    }, wait_seconds=60)
    save(HERE / 'dataset_readiness.json', readiness)
    if readiness['metadata']['current_version_number'] != 1:
        raise ValueError('Expected the previously verified private dataset version 1')

    pending = HERE / 'submission_attempt.json'
    if pending.exists():
        raise RuntimeError('A prior submission may have reached Kaggle. Use -Status to check before submitting again.')
    with pending.open('x', encoding='utf-8') as handle:
        json.dump(dict(kernel=kernel, utc=datetime.now(timezone.utc).isoformat()), handle, indent=2)
    folder = HERE / 'submission'
    response = api.kernels_push(str(folder), timeout='32400', acc='NvidiaTeslaT4')
    check_push_response(response)
    record = dict(kernel=kernel, dataset=dataset, kernel_id=response.kernel_id,
                  version_number=response.version_number, url=response.url,
                  is_private=True, actual_training_updates_confirmed=False,
                  source_sha256=hashlib.sha256((folder / 'entry.py').read_bytes()).hexdigest())
    save(receipt, record)
    try:
        remote = api.kernels_status(kernel)
        record.update(remote_status=str(remote.status), failure_message=remote.failure_message)
    except Exception as error:
        record['status_check_error'] = error_info(error)
    save(receipt, record)
    log(stage='submitted', **record)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        info = error_info(error)
        for name in ('KAGGLE_KEY',):
            if os.environ.get(name):
                info['message'] = info['message'].replace(os.environ[name], '[REDACTED]')
        save(HERE / 'deployment_error.json', info)
        log(stage='stopped', **info)
        sys.exit(1)
