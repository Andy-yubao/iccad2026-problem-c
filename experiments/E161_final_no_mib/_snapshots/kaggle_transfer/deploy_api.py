"""Resume private Kaggle deployment using positive read access and structured results."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent

def save(path, value):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    temp.replace(path)

def log(**value):
    # ASCII output remains readable in both Windows PowerShell5 and PowerShell7.
    print(json.dumps(value,ensure_ascii=True),flush=True)

def error_info(error):
    response = getattr(error,'response',None)
    message = str(error)
    token = os.environ.get('KAGGLE_API_TOKEN')
    if token: message = message.replace(token,'[REDACTED]')
    return dict(type=type(error).__name__,http_status=getattr(response,'status_code',None),message=message)

def assess_dataset(status, metadata, files, expected):
    """A forbidden status endpoint never proves readiness or permission by itself."""
    if status and status.lower() in {'error','failed','failure'}:
        raise RuntimeError('Kaggle reports dataset processing failed')
    if metadata is None or files is None:
        return False
    if not metadata['is_private']:
        raise RuntimeError('Expected a private dataset; refusing to change visibility or submit')
    if metadata['current_version_number'] < 1:
        return False
    if status is not None and status.lower() != 'ready':
        return False
    for name, size in expected.items():
        if name not in files: return False
        if files[name] != size:
            raise RuntimeError('Remote file size differs from the registered package: '+name)
    return True

def ensure_created(api, folder, dataset, archive_sha, receipt_path, use_existing=False):
    if use_existing:
        log(stage='reuse_existing_dataset',dataset=dataset,upload_skipped=True)
        return
    if receipt_path.exists():
        receipt=json.loads(receipt_path.read_text(encoding='utf-8'))
        if receipt['dataset'] != dataset or receipt['archive_sha256'] != archive_sha:
            raise RuntimeError('Previous upload receipt has a different identity; preserve it')
        log(stage='reuse_upload_receipt',dataset=dataset,upload_skipped=True)
        return
    response=api.dataset_create_new(str(folder),public=False,convert_to_csv=False)
    if response.error or str(response.status).lower() != 'ok':
        raise RuntimeError('Dataset creation was not accepted: '+str(response.error))
    # Record acceptance before the first status query, so later failures cannot reupload.
    save(receipt_path,dict(dataset=dataset,archive_sha256=archive_sha,
        status='creation_accepted',is_private=True,url=response.url,
        accepted_utc=datetime.now(timezone.utc).isoformat()))
    log(stage='dataset_creation_accepted',dataset=dataset,url=response.url)

def dataset_metadata(api, dataset):
    from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetRequest
    owner, slug=dataset.split('/')
    request=ApiGetDatasetRequest();request.owner_slug=owner;request.dataset_slug=slug
    with api.build_kaggle_client() as client:
        result=client.datasets.dataset_api_client.get_dataset(request)
    return dict(is_private=result.is_private,current_version_number=result.current_version_number)

def wait_dataset(api, dataset, expected, wait_seconds=1800):
    deadline=time.monotonic()+wait_seconds
    while True:
        status=metadata=files=None
        problems={}
        reads={
            'status':lambda:api.dataset_status(dataset),
            'metadata':lambda:dataset_metadata(api,dataset),
            'files':lambda:api.dataset_list_files(dataset,page_size=20),
        }
        values={}
        for label, read in reads.items():
            try: values[label]=read()
            except Exception as error:
                info=error_info(error)
                # These read failures may reflect an unpublished initial version.
                # No fallback mutation or visibility change is allowed.
                if info['http_status'] not in {403,404,409,429,500,502,503,504}:
                    raise
                problems[label]=info
        status=values.get('status')
        metadata=values.get('metadata')
        response=values.get('files')
        if response is not None:
            if response.error_message: raise RuntimeError(response.error_message)
            files={entry.name:int(entry.total_bytes) for entry in response.files}
        ready=assess_dataset(status,metadata,files,expected)
        log(stage='dataset_readiness',dataset=dataset,status=status,metadata=metadata,
            files=files,ready=ready,read_errors=problems)
        if ready:
            return dict(dataset=dataset,status_endpoint=status,metadata=metadata,
                files=files,status_endpoint_error=problems.get('status'),
                evidence='Published private version and readable files matching both expected sizes')
        remaining=deadline-time.monotonic()
        if remaining <= 0:
            raise RuntimeError('Dataset is not yet readable by this account. Upload was preserved; inspect the private dataset page and rerun with --use-existing-dataset. URL: https://www.kaggle.com/datasets/'+dataset)
        time.sleep(min(15,remaining))

def check_push_response(response):
    if response is None: raise RuntimeError('Kaggle returned no kernel submission result')
    if response.error: raise RuntimeError('Kernel submission rejected: '+str(response.error))
    if response.invalid_dataset_sources:
        raise RuntimeError('Kernel dataset input was rejected: '+str(response.invalid_dataset_sources))
    if int(response.version_number or 0) < 1:
        raise RuntimeError('Kaggle did not confirm a kernel version; do not assume it started')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--username',required=True)
    parser.add_argument('--dataset-folder',type=Path,required=True)
    parser.add_argument('--kernel-folder',type=Path,required=True)
    parser.add_argument('--use-existing-dataset',action='store_true')
    parser.add_argument('--wait-seconds',type=int,default=1800)
    args=parser.parse_args()
    from kaggle.api.kaggle_api_extended import KaggleApi
    api=KaggleApi();api.authenticate()
    actual=api.get_config_value(api.CONFIG_NAME_USER)
    if not actual or actual.lower() != args.username.lower():
        raise RuntimeError('Authenticated account differs from the requested dataset owner: '+str(actual))
    meta=json.loads((args.dataset_folder/'dataset-metadata.json').read_text(encoding='utf-8'))
    kernel_meta=json.loads((args.kernel_folder/'kernel-metadata.json').read_text(encoding='utf-8'))
    dataset=meta['id'];kernel=kernel_meta['id']
    if dataset.split('/')[0].lower()!=actual.lower() or kernel.split('/')[0].lower()!=actual.lower():
        raise RuntimeError('Resource owner does not match the authenticated account')
    if kernel_meta['is_private'] is not True or kernel_meta['dataset_sources'] != [dataset]:
        raise RuntimeError('Kernel privacy/input registration differs')
    manifest=json.loads((args.dataset_folder/'transfer-manifest.json').read_text(encoding='utf-8'))
    expected={name:(args.dataset_folder/name).stat().st_size for name in
              ('e161-transfer.tar.gz.bin','transfer-manifest.json')}
    submission=HERE/'submission.json'
    script_path=args.kernel_folder/kernel_meta['code_file']
    script_sha=hashlib.sha256(script_path.read_bytes()).hexdigest()
    if submission.exists():
        old=json.loads(submission.read_text(encoding='utf-8'))
        if old['kernel'] != kernel or old['dataset'] != dataset:
            raise RuntimeError('Existing kernel submission has a different identity')
        log(stage='already_submitted',kernel=kernel,url=old['url'],new_submission=False)
        return
    ensure_created(api,args.dataset_folder,dataset,manifest['archive_sha256'],
                   HERE/'upload_receipt.json',args.use_existing_dataset)
    readiness=wait_dataset(api,dataset,expected,args.wait_seconds)
    save(HERE/'dataset_readiness.json',readiness)
    log(stage='submitting_private_gpu_kernel',kernel=kernel,machine_shape='NvidiaTeslaT4')
    response=api.kernels_push(str(args.kernel_folder),timeout='32400',acc='NvidiaTeslaT4')
    check_push_response(response)
    record=dict(status='submitted',kernel=kernel,dataset=dataset,url=response.url,
        kernel_id=response.kernel_id,version_number=response.version_number,
        is_private=True,resume_step=7568,archive_sha256=manifest['archive_sha256'],
        kernel_source_sha256=script_sha,actual_training_updates_confirmed=False,
        submitted_utc=datetime.now(timezone.utc).isoformat())
    save(submission,record)
    try:
        status=api.kernels_status(kernel)
        record.update(remote_status=str(status.status),failure_message=status.failure_message)
    except Exception as error:
        record['status_check_error']=error_info(error)
    save(submission,record)
    log(stage='kernel_submitted',**record)

if __name__=='__main__':
    try: main()
    except Exception as error:
        info=error_info(error)
        save(HERE/'deployment_error.json',info)
        log(stage='deployment_stopped',**info)
        sys.exit(1)
