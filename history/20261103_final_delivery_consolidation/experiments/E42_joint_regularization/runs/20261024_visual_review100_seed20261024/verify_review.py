"""Bounded storage/API checks on three disposable copies; never score the real round."""
import csv
import io
import json
from pathlib import Path
import shutil
import sys
import threading
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from visualization.review import ReviewStore, make_server

HERE = Path(__file__).resolve().parent
QA = ROOT / '.cache/review_qa_backend'


def fixture(path):
    assert path.resolve().is_relative_to((ROOT / '.cache').resolve())
    path.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((HERE / 'manifest.json').read_text(encoding='utf-8'))
    manifest['session_id'] = path.name
    manifest['title'] = 'QA temporary review (not human ratings)'
    manifest['cases'] = manifest['cases'][:3]
    (path / 'images').mkdir()
    for row in manifest['cases']:
        shutil.copy2(HERE / row['image'], path / row['image'])
    (path / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False), encoding='utf-8')
    return manifest


def main():
    manifest = fixture(QA)
    server = make_server(QA, 0)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    origin = f'http://127.0.0.1:{server.server_port}'
    def get(path):
        return urllib.request.urlopen(origin + path).read()
    initial = json.loads(get('/api/session'))
    token = initial['token']
    revision = 0
    statuses = []
    def post(payload, expected=200, stale=False, csrf=False):
        nonlocal revision
        request = urllib.request.Request(origin+'/api/update', data=json.dumps({
            **payload, 'expected_revision': 0 if stale else revision}).encode(), headers={
            'Content-Type': 'application/json', 'X-Review-Token': 'wrong' if csrf else token, 'Origin': origin})
        try:
            response = urllib.request.urlopen(request)
            status, value = response.status, json.loads(response.read())
        except urllib.error.HTTPError as error:
            status, value = error.code, json.loads(error.read())
        assert status == expected, (status, value)
        statuses.append(status)
        if status == 200:
            revision = value['revision']
        return value
    ids = [r['id'] for r in manifest['cases']]
    post({'action': 'complete'}, 400)
    post({'action': 'rating', 'id': ids[0], 'score': 4}, 400)
    post({'action': 'rating', 'id': ids[0], 'score': True}, 400)
    post({'action': 'rating', 'id': ids[0], 'score': 1, 'note': '原始观察：还不确定。', 'tags': ['局部拥挤']})
    post({'action': 'rating', 'id': ids[0], 'score': 2}, 409, stale=True)
    post({'action': 'rating', 'id': ids[0], 'score': 3}, 403, csrf=True)
    post({'action': 'rating', 'id': ids[0], 'score': 3, 'note': '=模糊原话', 'tags': ['新维度', '局部拥挤']})
    post({'action': 'rating', 'id': ids[1], 'score': None, 'note': '只有备注'})
    assert server.review_store.state()['rated_count'] == 1
    post({'action': 'rating', 'id': ids[1], 'score': 2})
    post({'action': 'rating', 'id': ids[2], 'score': 1})
    post({'action': 'metrics_exposure', 'id': ids[0]})
    post({'action': 'feedback', 'text': '可能某类模块较拥挤，需要数据验证。'})
    final = post({'action': 'complete'})
    assert final['rated_count'] == 3 and final['completed']
    exported = json.loads(get('/api/export.json'))
    assert len(exported['events']) == 8 and exported['state']['ratings'][ids[0]]['note'] == '=模糊原话'
    csv_rows = list(csv.DictReader(io.StringIO(get('/api/export.csv').decode('utf-8-sig'))))
    assert len(csv_rows) == 3 and csv_rows[0]['note'] == "'=模糊原话"
    assert get('/images/'+ids[0]).startswith(b'\x89PNG')
    for path in ('/../config.json', '/images/no-such-case', '/ratings.sqlite3'):
        try:
            get(path)
            raise AssertionError('Unexpected file exposure')
        except urllib.error.HTTPError as error:
            assert error.code == 404
    server.shutdown()
    server.server_close()
    worker.join()
    server.review_store.db.close()
    reopened = ReviewStore(QA)
    assert reopened.state() == final
    reopened.db.close()
    real = json.loads(urllib.request.urlopen('http://127.0.0.1:8765/api/session').read())
    assert real['state']['rated_count'] == real['state']['revision'] == 0
    (HERE/'verification.json').write_text(json.dumps({
        'scope': 'three disposable image copies; no real human scores inserted',
        'real_round_ratings': 0, 'case_count': 100,
        'passed': ['incomplete rejection', 'invalid score rejection', 'revision conflict', 'cross-origin token rejection',
                   'Chinese notes/tags', 'note-only remains unrated', 'rerating history retained', 'metrics exposure event',
                   'complete after all rated', 'JSON export verbatim', 'CSV formula escaping', 'allowlisted image route',
                   'workspace/database routes blocked', 'database close/reopen durability'],
        'successful_events': len(exported['events']), 'HTTP_statuses': statuses,
        'status': 'passed', 'browser_verification': 'pending'
    }, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    assert QA.resolve().is_relative_to((ROOT/'.cache').resolve())
    shutil.rmtree(QA)
    print('PASS: disposable storage/API checks; actual review remains 0/100')


if __name__ == '__main__':
    main()
