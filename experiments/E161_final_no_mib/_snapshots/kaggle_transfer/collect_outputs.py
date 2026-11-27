"""Read-only, version-pinned Kaggle recovery with bounded parallel downloads."""
import argparse
from collections import Counter
import concurrent.futures
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import uuid

HERE=Path(__file__).resolve().parent

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(4*2**20),b''): h.update(chunk)
    return h.hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kernel',default='andyyubao/iccad-e161-s1-resume-20261123/1')
    parser.add_argument('--reuse',type=Path)
    args=parser.parse_args()
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest,ApiGetAcceleratorQuotaStatisticsRequest
    import requests
    api=KaggleApi();api.authenticate()
    if api.get_config_value(api.CONFIG_NAME_USER)!='andyyubao': raise ValueError('Unexpected account')
    owner,slug,version=args.kernel.split('/')
    state=api.kernels_status(args.kernel)
    target=HERE/'downloaded'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    target.mkdir(parents=True,exist_ok=False)
    inventory=[];token=None
    while True:
        response=api.kernels_list_files(args.kernel,page_size=100,page_token=token)
        for item in response.files or []: inventory.append(dict(name=item.name,reported_size=int(item.size)))
        token=response.next_page_token
        if not token: break
    urls={};token=None;logs=[]
    while True:
        request=ApiListKernelSessionOutputRequest();request.user_name=owner;request.kernel_slug=slug
        request.version_label='v'+version;request.page_size=100
        if token: request.page_token=token
        with api.build_kaggle_client() as client:
            result=client.kernels.kernels_api_client.list_kernel_session_output(request)
        for item in result.files or []: urls[item.file_name]=item.url
        if result.log: logs.append(result.log)
        token=result.next_page_token
        if not token: break
    # ListKernelFiles reports basenames, including repeated names in different
    # directories. ListKernelSessionOutput supplies the complete relative paths.
    (target/'server_inventory.json').write_text(json.dumps(dict(file_sizes=inventory,download_names=sorted(urls)),indent=2)+'\n',encoding='utf-8')
    if len(inventory)!=len(urls): raise ValueError('Kaggle output file counts differ')
    expected_names=Counter(Path(item['name']).name for item in inventory)
    if Counter(Path(name).name for name in urls)!=expected_names:
        raise ValueError('The server output file names differ across listing endpoints')
    (target/'remote_execution.log').write_text('\n'.join(logs),encoding='utf-8')
    print(json.dumps(dict(stage='download',files=len(urls),destination=str(target))),flush=True)
    def fetch(name):
        path=(target/name).resolve()
        if not path.is_relative_to(target.resolve()): raise ValueError('Output name escapes destination')
        path.parent.mkdir(parents=True,exist_ok=True)
        reused=False
        cached=(args.reuse/name) if args.reuse else None
        temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.part')
        with requests.get(urls[name],stream=True,timeout=(20,180),headers={'Accept-Encoding':'identity'}) as response:
            response.raise_for_status()
            length=response.headers.get('Content-Length')
            if length is None: raise ValueError('Missing transfer Content-Length: '+name)
            expected=int(length)
            if cached is not None and cached.is_file() and expected is not None and cached.stat().st_size==expected:
                shutil.copy2(cached,path);reused=True
            else:
                count=0
                with temporary.open('wb') as stream:
                    for blob in response.iter_content(1024*1024):
                        stream.write(blob);count+=len(blob)
                if expected is not None and count!=expected: raise ValueError('Truncated output: '+name)
                temporary.replace(path)
        return dict(path=name,bytes=path.stat().st_size,sha256=digest(path),reused=reused)
    manifest=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        futures={pool.submit(fetch,name):name for name in urls}
        for future in concurrent.futures.as_completed(futures):
            manifest.append(future.result())
            if len(manifest)%10==0 or len(manifest)==len(urls):
                print(json.dumps(dict(completed=len(manifest),total=len(urls))),flush=True)
    manifest.sort(key=lambda item:item['path'])
    if Counter(Path(item['path']).name for item in manifest)!=expected_names:
        raise ValueError('Downloaded file name/count multiset differs from the server inventory')
    (target/'download_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    with api.build_kaggle_client() as client:
        result=client.kernels.kernels_api_client.get_accelerator_quota_statistics(ApiGetAcceleratorQuotaStatisticsRequest())
    gpu=result.gpu_quota
    seconds=lambda value: value.total_seconds() if hasattr(value,'total_seconds') else value
    quota={name:seconds(getattr(gpu,name)) for name in ('time_used','time_reserved','total_time_allowed','minimum_time_allowed')}
    final=api.kernels_status(args.kernel)
    receipt=dict(kernel=args.kernel,status=str(final.status),failure_message=final.failure_message,
        checked_utc=datetime.now(timezone.utc).isoformat(),destination=str(target.relative_to(HERE.parents[1])),
        downloaded_files=len(manifest),total_bytes=sum(item['bytes'] for item in manifest),
        all_output_sizes_verified=True,gpu_quota_seconds=quota,
        gpu_session_terminal=str(final.status).rsplit('.',1)[-1] in {'ERROR','COMPLETE','CANCELLED','CANCELED'})
    (target/'recovery_status.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    (HERE/'latest_recovery.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt),flush=True)

if __name__=='__main__':
    try: main()
    except Exception as error:
        message=str(error)
        token=os.environ.get('KAGGLE_API_TOKEN')
        if token:message=message.replace(token,'[REDACTED]')
        print(json.dumps(dict(error=message,type=type(error).__name__)),flush=True)
        raise SystemExit(1)
