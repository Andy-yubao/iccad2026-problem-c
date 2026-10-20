import json
import sys
import time
from .io import np
from .projector import project


def main():
    print(json.dumps({'ready':True}),flush=True)
    for line in sys.stdin:
        request=json.loads(line)
        if request.get('quit'):break
        begin=time.perf_counter()
        try:
            inputs=(int(request['inputs'][0]),*[np.asarray(v) for v in request['inputs'][1:]])
            positions,stats=project(np.asarray(request['draft']),inputs,seconds=request.get('seconds',.35),
                    guidance=request.get('guidance'),snap_mode=request.get('snap_mode','separated'),joint=request.get('joint',False),polished=request.get('polished',False))
            response={'positions':positions.tolist(),'stats':stats,'solve_seconds':time.perf_counter()-begin}
        except Exception as exc:response={'error':repr(exc),'solve_seconds':time.perf_counter()-begin}
        print(json.dumps(response,allow_nan=False),flush=True)


if __name__=='__main__':main()
