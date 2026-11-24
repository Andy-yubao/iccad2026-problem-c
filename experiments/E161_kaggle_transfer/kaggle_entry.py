"""Kaggle formal session bootstrap. No credentials, model probes, or disposable updates."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time
import traceback

started = time.monotonic()
os.environ.update(CUDA_VISIBLE_DEVICES='0',CUBLAS_WORKSPACE_CONFIG=':4096:8',
                  PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',MPLBACKEND='Agg')
output = Path('/kaggle/working/e161-results')
output.mkdir(exist_ok=True)
status = dict(stage='extracting',actual_training_updates_confirmed=False)
def record(**fields):
    status.update(fields)
    status['session_seconds']=time.monotonic()-started
    (output/'bootstrap-status.json').write_text(json.dumps(status,indent=2)+'\n')
    print(json.dumps(status),flush=True)
try:
    matches = list(Path('/kaggle/input').rglob('e161-transfer.tar.gz.bin'))
    if len(matches) != 1: raise RuntimeError('Exactly one registered private E161 input package is required')
    archive = matches[0]
    manifest = json.loads(archive.with_name('transfer-manifest.json').read_text())
    root = Path('/tmp/iccad-e161-project')
    root.mkdir(exist_ok=False)
    record()
    with tarfile.open(archive, 'r:gz') as source:
        # Upload is a file-only archive created by prepare_bundle; never extract links/devices.
        for member in source:
            destination = (root/member.name).resolve()
            if not member.isfile() or not destination.is_relative_to(root):
                raise ValueError('Unexpected archive member')
            source.extract(member, root)
    extraction_seconds=time.monotonic()-started
    record(stage='verifying_transfer',extraction_seconds=extraction_seconds)
    for item in manifest['files']:
        path = root/item['path']
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for blob in iter(lambda: stream.read(4*2**20),b''): h.update(blob)
        if h.hexdigest() != item['sha256']: raise ValueError('Transfer content changed: '+item['path'])
    install_started=time.monotonic()
    record(stage='installing_dependencies',transfer_verification_seconds=install_started-started-extraction_seconds)
    requirements = root/'experiments/E161_kaggle_transfer/requirements-kaggle.txt'
    subprocess.run([sys.executable,'-m','pip','install','--no-cache-dir','--disable-pip-version-check',
        '--retries','0','-r',str(requirements)],check=True)
    record(stage='formal_resume',dependency_install_seconds=time.monotonic()-install_started)
    env = dict(os.environ,PYTHONPATH=str(root),E161_SESSION_STARTED=str(started))
    subprocess.run([sys.executable,'-u','-B',str(root/'experiments/E161_kaggle_transfer/resume_remote.py')],
                   cwd=root,env=env,check=True)
    record(stage='session_finished')
except BaseException:
    record(stage='error',error=traceback.format_exc())
    raise
