"""Local, reusable review of saved layout images; no model or dataset imports.

python -m visualization.review --session <experiment/runs/review-directory>
Only the manifest, allowlisted images and review API are exposed on loopback.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import secrets
import sqlite3
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

ASSETS = Path(__file__).with_name('review_web')


def encoded(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False).encode('utf-8')


class ReviewStore:
    def __init__(self, folder):
        self.folder = Path(folder).resolve()
        self.manifest_bytes = (self.folder / 'manifest.json').read_bytes()
        self.manifest = json.loads(self.manifest_bytes)
        self.identity = hashlib.sha256(self.manifest_bytes).hexdigest()
        self.cases = {r['id']: r for r in self.manifest['cases']}
        if not self.cases or len(self.cases) != len(self.manifest['cases']):
            raise ValueError('Empty or duplicate review cases')
        for row in self.cases.values():
            path = (self.folder / row['image']).resolve()
            if not path.is_relative_to(self.folder) or not path.is_file():
                raise ValueError('Review image missing or outside session')
        self.lock = threading.RLock()
        self.db = sqlite3.connect(self.folder / 'ratings.sqlite3', check_same_thread=False)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        with self.db:
            self.db.execute('CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)')
            self.db.execute('CREATE TABLE IF NOT EXISTS events (revision INTEGER PRIMARY KEY, timestamp TEXT DEFAULT (strftime(\'%Y-%m-%dT%H:%M:%fZ\',\'now\')), payload TEXT)')
            self.db.execute('CREATE TABLE IF NOT EXISTS ratings (id TEXT PRIMARY KEY, payload TEXT)')
            stored = self.db.execute("SELECT value FROM meta WHERE key='manifest_sha256'").fetchone()
            if stored and stored[0] != self.identity:
                raise ValueError('Manifest changed after review began; create a new session')
            self.db.execute("INSERT OR IGNORE INTO meta VALUES ('manifest_sha256', ?)", (self.identity,))

    def state(self):
        with self.lock:
            ratings = {r[0]: json.loads(r[1]) for r in self.db.execute('SELECT id,payload FROM ratings')}
            meta = dict(self.db.execute('SELECT key,value FROM meta'))
            revision = self.db.execute('SELECT COALESCE(MAX(revision),0) FROM events').fetchone()[0]
            return {'revision': revision, 'ratings': ratings,
                    'feedback': json.loads(meta.get('feedback', '""')),
                    'completed': json.loads(meta.get('completed', 'false')),
                    'rated_count': sum(r.get('score') in (1, 2, 3) for r in ratings.values())}

    def update(self, payload):
        with self.lock:
            state = self.state()
            if type(payload.get('expected_revision')) is not int or payload['expected_revision'] != state['revision']:
                raise RuntimeError('页面版本已改变，请刷新后重试；本次输入尚未保存。')
            action = payload.get('action')
            if action == 'rating':
                if payload.get('id') not in self.cases:
                    raise ValueError('Unknown case')
                score = payload.get('score')
                if score is not None and (type(score) is not int or score not in (1, 2, 3)):
                    raise ValueError('Score must be 1, 2 or 3')
                note, tags = payload.get('note', ''), payload.get('tags', [])
                if not isinstance(note, str) or len(note) > 10000 or not isinstance(tags, list) or len(tags) > 40 or any(not isinstance(t, str) or len(t) > 120 for t in tags):
                    raise ValueError('Invalid note or observation tags')
                value = {'score': score, 'note': note, 'tags': tags,
                         'rubric_version': self.manifest['rubric']['version']}
                answers = payload.get('dimensions', {})
                questions = self.manifest['rubric'].get('questions', {})
                if not isinstance(questions, dict):
                    questions = {}
                if not isinstance(answers, dict) or set(answers)-set(questions):
                    raise ValueError('Unknown observation dimension')
                if any(v is not None and (type(v) is not int or v not in (1, 2, 3)) for v in answers.values()):
                    raise ValueError('Dimension answers must be 1, 2, 3 or null')
                value['dimensions'] = answers
            elif action == 'feedback':
                if not isinstance(payload.get('text'), str) or len(payload['text']) > 40000:
                    raise ValueError('Invalid feedback')
            elif action == 'complete':
                if state['rated_count'] != len(self.cases):
                    raise ValueError('请先完成全部评分；可以暂存后离开。')
            elif action == 'metrics_exposure':
                if payload.get('id') not in self.cases:
                    raise ValueError('Unknown case')
            else:
                raise ValueError('Unknown action')
            with self.db:
                revision = state['revision'] + 1
                self.db.execute('INSERT INTO events(revision,payload) VALUES (?,?)',
                                (revision, encoded(payload).decode()))
                if action == 'rating':
                    self.db.execute('INSERT OR REPLACE INTO ratings VALUES (?,?)', (payload['id'], encoded(value).decode()))
                    self.db.execute("INSERT OR REPLACE INTO meta VALUES ('completed','false')")
                if action == 'feedback':
                    self.db.execute("INSERT OR REPLACE INTO meta VALUES ('feedback',?)", (encoded(payload['text']).decode(),))
                if action == 'complete':
                    self.db.execute("INSERT OR REPLACE INTO meta VALUES ('completed','true')")
            return self.state()

    def export(self):
        with self.lock:
            return {'manifest': self.manifest, 'manifest_sha256': self.identity, 'state': self.state(),
                    'events': [{'revision': r[0], 'timestamp': r[1], 'payload': json.loads(r[2])}
                               for r in self.db.execute('SELECT revision,timestamp,payload FROM events ORDER BY revision')]}


def make_server(folder, port=8765):
    store = ReviewStore(folder)
    token = secrets.token_urlsafe(32)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def reply(self, data, mime='application/json; charset=utf-8', status=200, filename=None):
            self.send_response(status)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; img-src 'self' blob:; style-src 'self'; script-src 'self'; connect-src 'self'; object-src 'none'; frame-ancestors 'none'")
            if filename:
                self.send_header('Content-Disposition', f'attachment; filename="{filename}"')
            self.end_headers()
            self.wfile.write(data)

        def valid_host(self):
            return self.headers.get('Host') == f'127.0.0.1:{self.server.server_port}'

        def do_GET(self):
            if not self.valid_host():
                return self.reply(encoded({'error': 'Invalid host'}), status=403)
            path = urlsplit(self.path).path
            assets = {'/': ('index.html', 'text/html; charset=utf-8'),
                      '/app.js': ('app.js', 'text/javascript; charset=utf-8'),
                      '/style.css': ('style.css', 'text/css; charset=utf-8')}
            if path in assets:
                name, mime = assets[path]
                return self.reply((ASSETS / name).read_bytes(), mime)
            if path == '/api/session':
                return self.reply(encoded({'manifest': store.manifest, 'state': store.state(), 'token': token}))
            if path == '/api/export.json':
                return self.reply(encoded(store.export()), filename='draft_review.json')
            if path == '/api/export.csv':
                content = io.StringIO(newline='')
                writer = csv.writer(content)
                writer.writerow(['ordinal', 'id', 'member', 'score', 'note', 'tags', 'rubric_version', 'dimensions'])
                state = store.state()
                for i, row in enumerate(store.cases.values()):
                    r = state['ratings'].get(row['id'], {})
                    # Spreadsheet formula text is escaped in CSV only; JSON is verbatim.
                    safe = lambda x: "'"+x if isinstance(x, str) and x[:1] in ('=', '+', '-', '@', '\t', '\r', '\n') else x
                    writer.writerow([i+1, row['id'], json.dumps(row.get('member')), r.get('score'),
                                     safe(r.get('note', '')), safe('|'.join(r.get('tags', []))), r.get('rubric_version'), json.dumps(r.get('dimensions', {}), ensure_ascii=False)])
                return self.reply(('\ufeff'+content.getvalue()).encode('utf-8'), 'text/csv; charset=utf-8', filename='draft_ratings.csv')
            if path.startswith('/images/'):
                case = store.cases.get(path.removeprefix('/images/'))
                if case:
                    return self.reply((store.folder / case['image']).read_bytes(), 'image/png')
            return self.reply(encoded({'error': 'Not found'}), status=404)

        def do_POST(self):
            origin = f'http://127.0.0.1:{self.server.server_port}'
            if not self.valid_host() or self.headers.get('Origin') != origin or self.headers.get('X-Review-Token') != token:
                return self.reply(encoded({'error': 'Invalid review origin'}), status=403)
            if urlsplit(self.path).path != '/api/update':
                return self.reply(encoded({'error': 'Not found'}), status=404)
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 100000 or self.headers.get('Content-Type') != 'application/json':
                    raise ValueError('Invalid JSON request')
                value = json.loads(self.rfile.read(size))
                if not isinstance(value, dict):
                    raise ValueError('Expected object')
                return self.reply(encoded(store.update(value)))
            except RuntimeError as error:
                return self.reply(encoded({'error': str(error)}), status=409)
            except (ValueError, TypeError, KeyError) as error:
                return self.reply(encoded({'error': str(error)}), status=400)

    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    server.review_store = store
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    server = make_server(args.session, args.port)
    print(f'Review ready: http://127.0.0.1:{server.server_port}/', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        server.review_store.db.close()


if __name__ == '__main__':
    main()
