import json
from .io import HERE
def indices(size,seed):
    if size==4096:return json.loads((HERE/'subsets.json').read_text(encoding='utf-8'))['families']['0']['4096']
    if size in (16000,32000,64000):return list(range(size))
    raise ValueError(size)
