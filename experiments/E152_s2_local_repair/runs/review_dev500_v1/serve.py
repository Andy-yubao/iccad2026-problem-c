"""Serve the two saved galleries through the existing read-only viewer."""
import json
import os
from pathlib import Path
import threading
from visualization.viewer import SavedFile, make_server

HERE=Path(__file__).resolve().parent
MODELS=(('e152_s2_ref',8812),('e152_s2_no_ref',8813))

def main():
    servers=[];entries=[]
    for arm,port in MODELS:
        path=HERE.parent/f'{arm}_seed0/visualization_normal20_loss5_q5/data.json'
        data=SavedFile(path);assert len(data.rows)==30
        server=make_server(data,port);servers.append(server)
        entries.append(dict(arm=arm,url=f'http://127.0.0.1:{port}/',data=str(path),cases=30))
    (HERE/'servers.json').write_text(json.dumps(dict(status='listening',pid=os.getpid(),galleries=entries),indent=2)+'\n',encoding='utf-8')
    for server in servers[1:]:
        threading.Thread(target=server.serve_forever,daemon=True).start()
    print(json.dumps(entries),flush=True)
    try:servers[0].serve_forever()
    finally:
        for server in servers:server.server_close()

if __name__=='__main__':main()
