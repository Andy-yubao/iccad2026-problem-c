"""Synchronous JSON-file worker: public inputs and draft only, no labels/score."""
import json
from pathlib import Path
import sys
import signal
import time
import numpy as np


def main():
    signal.signal(signal.SIGINT,signal.SIG_IGN)
    request, response = map(Path, sys.argv[1:])
    payload = json.loads(request.read_text(encoding='utf-8'))
    if set(payload) != {'inputs', 'draft', 'seconds'}:
        raise ValueError('Public worker contract mismatch')
    inputs = (int(payload['inputs'][0]), *[np.asarray(v) for v in payload['inputs'][1:]])
    from .backend.optimizer import optimize
    start = time.perf_counter()
    try:
        final, stats = optimize(np.asarray(payload['draft'], dtype=float), inputs, seconds=payload['seconds'])
        # Improvement events repeat complete coordinates; retain their metadata only.
        for event in stats.get('improvements', []):
            event.pop('positions', None)
        result = {'positions': final.tolist(), 'stats': stats, 'solve_seconds': time.perf_counter() - start}
    except Exception as exc:
        result = {'error': {'type': type(exc).__name__, 'message': str(exc)}, 'solve_seconds': time.perf_counter() - start}
    response.write_text(json.dumps(result, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
