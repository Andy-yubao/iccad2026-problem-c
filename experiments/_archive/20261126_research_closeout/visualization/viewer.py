"""Single read-only viewer and PNG entry point for saved layout coordinates."""
import argparse
import json
import sys
from functools import lru_cache
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path

from .style import normalize

ROOT = Path(__file__).resolve().parents[1]
E119 = ROOT/'experiments/E119_global_initial_generator/runs/global2_seed0/evaluation_no_mib_dev500'
E119_GALLERY = ROOT/'experiments/E119_global_initial_generator/visualization/20261113_dev15/data.json'
E120_GALLERY = ROOT/'experiments/E120_node12_collision/visualization/20261113_dev15/data.json'

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

class SavedFile:
    def __init__(self,path):
        self.data = read(path)
        self.rows = self.data['cases']
        if not self.rows:
            raise ValueError('No saved cases')
        self.manifest = dict(title=self.data.get('title','已保存布局'),note=self.data.get('note',''),
                             identity=self.data.get('identity',{}),
                             cases=[dict(label=c.get('label',f'样例 {i}'),n=c.get('n'),group=c.get('group')) for i,c in enumerate(self.rows)])

    @lru_cache(maxsize=16)
    def case(self,index):
        return normalize(self.rows[index])

class E119Results:
    def __init__(self):
        from datasets.random_reference import RandomReference
        self.dataset = RandomReference('dev_monitor500')
        result = read(E119/'results.json')
        if result['status']!='complete' or result['count']!=len(self.dataset):
            raise ValueError('E119 dev evaluation is incomplete')
        self.digest = result['checkpoint_sha256']
        self.manifest = dict(title='E119 · best55 · dev500',
            note='原预算后处理。该评价只保存末层草稿；未保存12层中间坐标。',
            identity=dict(experiment='E119_global_initial_generator',best_epoch=55,checkpoint_sha256=self.digest),
            cases=[dict(label=f'dev {i:04d}') for i in range(len(self.dataset))])

    @lru_cache(maxsize=16)
    def case(self,index):
        item = self.dataset[index]
        row = read(E119/'cases'/f'{index:04d}.json')
        if row['member']!=item['member'] or row['checkpoint_sha256']!=self.digest or row['index']!=index:
            raise ValueError('E119 member/checkpoint/index mismatch')
        metrics = row['metrics']
        return normalize(dict(label=f'dev {index:04d}',member=row['member'],n=row['n'],
            constraints=item['solver_inputs'][5].tolist(),reference=item['supervision']['reference_xywh'].tolist(),
            draft=row['draft_positions'],draft_label='末层草稿 · L12',final=row['positions'],
            metrics=dict(Q=metrics['cost'],hard_legal=bool(metrics['is_feasible']),
                         complete_ms=row['complete_seconds']*1000)))

def make_server(source,port):
    template = Path(__file__).with_name('viewer.html').read_bytes()
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            try:
                path = self.path.split('?',1)[0]
                if path in ('/','/index.html'):
                    body,kind = template,'text/html; charset=utf-8'
                elif path=='/api/manifest':
                    body,kind = json.dumps(source.manifest,ensure_ascii=False).encode(),'application/json; charset=utf-8'
                elif path.startswith('/api/case/') and path[10:].isdigit() and int(path[10:]) < len(source.manifest['cases']):
                    body,kind = json.dumps(source.case(int(path[10:])),ensure_ascii=False,allow_nan=False).encode(),'application/json; charset=utf-8'
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header('Content-Type',kind)
                self.send_header('Content-Length',str(len(body)))
                self.send_header('Cache-Control','no-store')
                self.send_header('X-Content-Type-Options','nosniff')
                self.end_headers()
                self.wfile.write(body)
            except (ValueError,KeyError,IndexError,OSError) as error:
                self.send_error(500,str(error))
    return ThreadingHTTPServer(('127.0.0.1',port),Handler)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--e119',action='store_true',help='E119 saved dev15 layer gallery; falls back to dev500 if absent (default)')
    source.add_argument('--e120',action='store_true',help='E120 saved dev15 layer gallery')
    source.add_argument('--data',type=Path,help='JSON containing title, identity, and saved cases')
    parser.add_argument('--port',type=int,help='Default: 8779 for E119, 8781 for E120')
    parser.add_argument('--log',type=Path,help='Append service output to a log file for hidden background launches')
    parser.add_argument('--png',type=Path,help='Render one case to PNG instead of starting a server')
    parser.add_argument('--case',type=int,default=0)
    parser.add_argument('--layer',type=int,help='Actual saved layer ID for PNG (supports S0 with layer_ids)')
    args = parser.parse_args()
    if args.log:
        args.log.parent.mkdir(parents=True,exist_ok=True)
        sys.stdout=sys.stderr=args.log.open('a',encoding='utf-8',buffering=1)
    if args.data:
        data = SavedFile(args.data)
    elif args.e120:
        data = SavedFile(E120_GALLERY)
    else:
        data = SavedFile(E119_GALLERY) if E119_GALLERY.exists() else E119Results()
    if args.png:
        from .layouts import render
        render(data.case(args.case),args.png,layer=args.layer,title=data.manifest['title'])
        return
    server = make_server(data,args.port if args.port is not None else 8781 if args.e120 else 8779)
    print(f'http://127.0.0.1:{server.server_port}/',flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

if __name__=='__main__':main()
