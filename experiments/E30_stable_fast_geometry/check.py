"""Five train rows for gradients; saved dev geometry for exact reducer parity."""
from .resources import configure
RESOURCE=configure()
import json
import argparse
import time
import numpy as np
import torch
from .io import HERE,ROOT,save,sha
from .cache import CACHE,verify
from .cases import WEIGHTS
from .data import collate
from .batches import inputs_only
from .model import GeometryModel
from .loss import losses
from .fast_geometry import topology,reduce_edges
from .native_reduce import reduce_edges as native_reduce
from .train import PlateauController,monitor_losses

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--tag',default='20261022_implementation_check');args=ap.parse_args()
    started=time.perf_counter();manifest=verify('train')
    chunk=torch.load(CACHE/'train'/manifest['parts'][0]['name'],map_location='cpu',weights_only=False)
    rows=chunk['rows'][:5];g,_=collate(rows,'cuda')
    checkpoint=torch.load(WEIGHTS,map_location='cpu',weights_only=False)
    model=GeometryModel(**checkpoint['model_config']).cuda();model.load_state_dict(checkpoint['model']);model.train()
    state=model(inputs_only(g));report={}
    for arm in ('geometry_control','relation_risk'):
        values=losses(state,g,None,arm,0,include_quality=True)
        gradient=torch.autograd.grad(values['total'],list(model.parameters()),retain_graph=True,allow_unused=True)
        if not all(v is None or torch.isfinite(v).all() for v in gradient):raise ValueError('Nonfinite gradients')
        report[arm]={k:float(v.detach()) for k,v in values.items()}
    monitored=monitor_losses(model,rows,'cuda','relation_risk',0,32)
    parameter=torch.nn.Parameter(torch.tensor(1.));optimizer=torch.optim.SGD([parameter],lr=1e-3)
    scheduler=PlateauController(optimizer)
    for epoch in range(1,27):
        scheduler.step(1.,epoch)
        if epoch<26 and scheduler.should_stop():raise ValueError('Premature early stop')
    if scheduler.drops!=3 or scheduler.lowest_completed_epochs!=2 or not scheduler.should_stop():raise ValueError('LR schedule failed')
    results=json.loads((HERE/'runs/20261022_baseline_smoke_steady_monitor5/results.json').read_text())
    for row in rows:
        p=np.asarray(row['graph']['reference']);c=np.asarray(row['graph']['constraints'])
        for mode in (0,1):
            edges=topology(p,c,mode=mode,array=True)
            if not np.array_equal(np.asarray(reduce_edges(edges,len(p))).reshape(-1,3),native_reduce(edges,len(p))):raise ValueError('Reducer parity failure')
    # Include a cycle and a 65th-node bit to exercise the second word.
    for n,edges in [(3,[[0,1,0],[1,2,0],[2,0,0]]),(66,[[0,64,0],[64,65,0],[0,65,0]])]:
        if not np.array_equal(np.asarray(reduce_edges(edges,n)),native_reduce(edges,n)):raise ValueError('Reducer boundary parity failure')
    destination=HERE/'runs'/args.tag/'results.json'
    if destination.exists():raise ValueError('Do not overwrite a completed check')
    save(destination,{'five_train_members':[r['member'] for r in rows],
        'gradients_finite':True,'losses':report,'native_reducer_exact_parity':True,'seconds':time.perf_counter()-started,
        'monitor':monitored,'lowest_lr_two_epoch_check':True,
        'resource':RESOURCE,'test_read':False,'checkpoint_sha256':sha(WEIGHTS),'cached_train_manifest':sha(CACHE/'train/manifest.json')})
    print('Five-row gradient and exact reducer checks passed',flush=True)

if __name__=='__main__':main()
