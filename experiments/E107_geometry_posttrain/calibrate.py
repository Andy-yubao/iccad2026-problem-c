"""Freeze three normalization scales from E105 original train_monitor500 only."""
import json
import time
import numpy as np
import torch
from .io import HERE,ROOT,save,sha
from .data import load_rows,register_resident
from .resident import ResidentRows
from .batches import inputs_only
from .runtime import parent_model
from .geometry import measures

@torch.no_grad()
def main():
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if (HERE/'normalization.json').exists():raise RuntimeError('Already frozen; do not refit')
    start=time.perf_counter();rows=load_rows('train_monitor500')
    pack=ResidentRows(rows,'cuda',max_nodes=120);model=parent_model();values=[]
    for first in range(0,500,32):
        batch=pack.batch(list(range(first,min(first+32,500))))
        state=model(inputs_only(batch));m=measures(state,batch)
        keys=('area_excess','tail','whitespace')
        array=torch.stack([m[k] for k in keys],-1)
        if not bool(torch.isfinite(array).all()):raise RuntimeError('Nonfinite calibration')
        values.append(array.cpu().double().numpy())
    array=np.concatenate(values);scales={k:max(float(array[:,j].mean()),1e-4) for j,k in enumerate(keys)}
    plan=json.loads((HERE/'plan.json').read_text(encoding='utf-8'))
    save(HERE/'normalization.json',dict(status='frozen_train500',scales=scales,floor=1e-4,count=500,
        members=[list(r['member']) for r in rows],parent_sha256=plan['parent_checkpoint_sha256'],
        registry_sha256=sha(ROOT/'protocol/reference_workset_v6.json'),
        augmentation=False,student_updates=0,dev_payloads=0,test_read=False,
        elapsed_seconds=time.perf_counter()-start))
    print('Frozen train500 scales',scales,flush=True)

if __name__=='__main__':main()
