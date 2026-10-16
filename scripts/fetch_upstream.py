"""Archive a commit-pinned FloorSet source snapshot, excluding datasets.

No git transport, third-party packages, or model/dataset downloads are needed.
File contents are checked against GitHub's Git blob SHA-1 and SHA-256 is recorded.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'third_party/FloorSet'
REPO = 'IntelLabs/FloorSet'
EXCLUDED = ('LiteTensorDataTest/', 'data/', '.ipynb_checkpoints/')


def get(url):
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'FloorSet-research-archive/1.0'})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2)


def main():
    manifest_path = ROOT / 'third_party/upstream_manifest.json'
    if manifest_path.exists():
        commit = json.loads(manifest_path.read_text(encoding='utf-8'))['commit']
    else:
        commit = json.loads(get(f'https://api.github.com/repos/{REPO}/commits/main'))['sha']
    tree = json.loads(get(f'https://api.github.com/repos/{REPO}/git/trees/{commit}?recursive=1'))
    if tree.get('truncated'):
        raise RuntimeError('Incomplete GitHub tree; refusing partial archive')
    entries = [x for x in tree['tree'] if x['type'] == 'blob' and not x['path'].startswith(EXCLUDED)]

    def download(x):
        url = f'https://raw.githubusercontent.com/{REPO}/{commit}/{x["path"]}'
        path = DEST / x['path']
        data = path.read_bytes() if path.exists() else get(url)
        blob_hash = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        if blob_hash != x['sha']:
            raise ValueError(f'Git blob hash mismatch: {x["path"]}')
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        return {'path': path.relative_to(ROOT).as_posix(), 'source_url': url, 'bytes': len(data),
                'git_blob_sha1': blob_hash, 'sha256': hashlib.sha256(data).hexdigest()}

    with ThreadPoolExecutor(max_workers=6) as pool:
        records = list(pool.map(download, entries))
    manifest = {'repository': f'https://github.com/{REPO}', 'commit': commit,
                'checked_at_utc': datetime.now(timezone.utc).isoformat(),
                'snapshot_type': 'commit-pinned source files, not a Git checkout',
                'excluded_prefixes': list(EXCLUDED), 'datasets_downloaded': False, 'files': records}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(f'Archived {len(records)} files, {sum(x["bytes"] for x in records)} bytes, commit {commit}')


if __name__ == '__main__':
    main()
