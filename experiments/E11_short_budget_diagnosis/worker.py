"""Persistent synchronous worker; no reference metrics or online quality gate."""
import json
import os
import sys
import time

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'

import numpy as np
from .backend.robust import optimize


def main():
    print(json.dumps({'ready': True}), flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        if request.get('quit'):
            break
        start = time.perf_counter()
        try:
            inputs = (int(request['inputs'][0]),
                      *[np.asarray(v) for v in request['inputs'][1:]])
            draft = np.asarray(request['draft'], dtype=float)
            guidance = request.get('guidance')
            prepared = time.perf_counter()
            positions, stats = optimize(draft, inputs, seconds=request['seconds'],
                                       fast_seconds=min(1.3, request['seconds']), guidance=guidance)
            end = time.perf_counter()
            response = {'positions': positions.tolist(), 'stats': stats,
                        'solve_seconds': end - prepared,
                        'request_prepare_seconds': prepared - start}
        except Exception as exc:
            response = {'error': repr(exc), 'solve_seconds': time.perf_counter() - start}
        try:
            import psutil
            info = psutil.Process().memory_info()
            response['peak_memory_bytes'] = getattr(info, 'peak_wset', info.rss)
        except ImportError:
            response['peak_memory_bytes'] = None
        print(json.dumps(response, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
