"""Pre-registered post-selection address sensitivity; no training or Q selection."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import json
import signal
import time
import torch
import numpy as np
from .io import HERE, save, sha
from .model import GeometryModel, inputs_only
from .data import load_rows
from .resident import ResidentRows
from .addresses import attach_addresses, DIAGNOSTIC_EPOCH
from .loss import per_case
from .train import seed_all, numerical_check


def main():
    from .run_queue import verify
    parser=argparse.ArgumentParser()
    parser.add_argument('--arm',choices=['standard256'],required=True)
    args=parser.parse_args(); verify(); seed_all(0)
    run=HERE/'runs'/f'{args.arm}_seed0'; output=run/'address_diagnostic.json'
    frozen=json.loads((HERE/'evaluation'/f'frozen_{args.arm}.json').read_text())
    checkpoint=run/'checkpoints/best.pt'
    if sha(checkpoint)!=frozen['checkpoint_sha256']:raise ValueError('Frozen model changed')
    if output.exists() and json.loads(output.read_text())['status']=='complete':return
    requested=[False]
    signal.signal(signal.SIGINT,lambda *_:requested.__setitem__(0,True))
    if hasattr(signal,'SIGBREAK'):signal.signal(signal.SIGBREAK,lambda *_:requested.__setitem__(0,True))
    def halted():
        return requested[0] or any((HERE/name).exists() for name in ('STOP','QUEUE_STOP','PAUSE_REQUEST')) or (run/'STOP').exists()
    start=time.perf_counter()
    rows=load_rows('dev_monitor500',halted=halted)
    if rows is None:
        save(output,dict(status='paused',test_read=False));return
    pack=ResidentRows(rows,'cuda',max_nodes=120,max_pins=145)
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model=GeometryModel(**state['model_config']).cuda().eval();model.load_state_dict(state['model'])
    values={}
    with torch.inference_mode():
        for label,epoch in [('canonical',None),('fixed_renaming',DIAGNOSTIC_EPOCH)]:
            parts=[]
            for start_id in range(0,len(rows),32):
                if halted():save(output,dict(status='paused',test_read=False));return
                ids=list(range(start_id,min(start_id+32,len(rows))))
                batch=pack.batch(ids);attach_addresses(batch,rows,ids,0,epoch)
                predicted=model(inputs_only(batch));loss=per_case(predicted,batch)
                numerical_check(loss,predicted)
                parts.extend(loss['total'].cpu().double().tolist())
            values[label]=np.asarray(parts)
    delta=values['fixed_renaming']-values['canonical']
    save(output,dict(status='complete',checkpoint_sha256=frozen['checkpoint_sha256'],count=500,
        canonical_mean=float(values['canonical'].mean()),renamed_mean=float(values['fixed_renaming'].mean()),
        mean_difference=float(delta.mean()),mean_absolute_case_difference=float(np.abs(delta).mean()),
        max_absolute_case_difference=float(np.abs(delta).max()),per_case={k:v.tolist() for k,v in values.items()},
        fixed_address_domain=20261114137,fixed_address_epoch=DIAGNOSTIC_EPOCH,
        selection_changed=False,postprocessing_calls=0,optimizer_updates=0,test_read=False,
        elapsed_seconds=time.perf_counter()-start))
    print(json.dumps({k:v for k,v in json.loads(output.read_text()).items() if k!='per_case'}),flush=True)


if __name__=='__main__':main()
