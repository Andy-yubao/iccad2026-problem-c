"""Spawn-safe CPU preparation workers; each worker uses one library thread."""
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
import json
from pathlib import Path
import signal
import numpy as np
import torch

COMPATIBILITY = Path(__file__).resolve().parent/'runs/20261019_process_preparation_update/compatibility.json'


def compatible_identity(previous, current, kind):
    """Only exact, explicitly recorded source migrations can reuse old work."""
    if previous == current:
        return True
    if not COMPATIBILITY.exists():
        return False
    transitions = json.loads(COMPATIBILITY.read_text(encoding='utf-8'))['transitions']
    return any(t['kind'] == kind and t['previous'] == previous and t['current'] == current
               for t in transitions)


def initialize_cpu(ready_queue=None):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if ready_queue is not None:
        ready_queue.put({'pid': os.getpid(), 'torch_threads': torch.get_num_threads(),
                         'interop_threads': torch.get_num_interop_threads()})


def inspect_candidate(shard, parent, sample_rows=None):
    from datasets.reference_layouts import inspect_shard
    result = inspect_shard(shard, parent, sample_rows=sample_rows)
    result['execution'] = {'pid': os.getpid(), 'torch_threads': torch.get_num_threads(),
                           'interop_threads': torch.get_num_interop_threads()}
    return result


def initialize_cache(role, registry_path):
    global worker_dataset
    initialize_cpu()
    from datasets.reference_layouts import ReferenceLayouts
    worker_dataset = ReferenceLayouts(role, registry_path=registry_path)


def prepare_cache_part(start, indices, path_string):
    """Build and atomically save a whole part; return no graph/tensor payload."""
    from .data import prepare
    from .cache import pack
    from .io import replace
    path = Path(path_string)
    rows = [prepare(worker_dataset[i]) for i in indices]
    columns = pack(rows)
    temp = path.with_suffix('.tmp')
    with temp.open('wb') as handle:
        np.savez(handle, **columns)
    replace(temp, path)
    return {'start': start, 'count': len(indices), 'pid': os.getpid(),
            'torch_threads': torch.get_num_threads(), 'interop_threads': torch.get_num_interop_threads()}
