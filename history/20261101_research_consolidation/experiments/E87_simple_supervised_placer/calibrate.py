"""User-run, train-only once: fix two auxiliary coefficients before formal training."""
import argparse,json,time
import numpy as np
import torch
from .io import HERE,save,sources,exclusive,sha
from .model import GeometryModel
from .arms import model_config
from .initialization import load,configure_trainability,metadata
from .data import load_rows,collate
from .batches import inputs_only
from .loss import per_case
from .loss_config import set_coefficients,load_coefficients
from .train import seed_all

def run():
    from .preflight import verify
    verify();output=HERE/'runs/registered_loss.json';partial=HERE/'runs/calibration_partial.json'
    with exclusive(HERE/'runs/calibration.lock'):
        if output.exists():load_coefficients(output);print('Existing fixed Loss registration verified');return
        begin=time.perf_counter();seed_all(0);set_coefficients(dict(overlap=1.,task=1.))
        identity=dict(sources=sources(),initialization=metadata(),role='train_monitor500',count=500,device='cuda',seed=0)
        old=json.loads(partial.read_text(encoding='utf-8')) if partial.exists() else dict(identity=identity,records=[],elapsed_seconds=0.)
        if old['identity']!=identity:raise ValueError('Calibration resume identity mismatch')
        model=configure_trainability(GeometryModel(**model_config('simple')).cuda()).eval();load(model,model_config('simple'))
        params=[p for p in model.parameters() if p.requires_grad];rows=load_rows('train_monitor500')
        records=old['records'];start_index=sum(r['count'] for r in records)
        for start in range(start_index,len(rows),32):
            if (HERE/'runs/STOP').exists():print('Calibration stopped; partial saved');return
            batch=rows[start:start+32];g,_=collate(batch,'cuda');state=model(inputs_only(g));v=per_case(state,g,'simple')
            norms={}
            for key in ('reference','overlap_total','task'):
                gradients=torch.autograd.grad(v[key].mean(),params,retain_graph=key!='task',allow_unused=True)
                norms[key]=float(torch.stack([x.square().sum() for x in gradients if x is not None]).sum().sqrt())
            if any(not np.isfinite(x) or x<=1e-10 for x in norms.values()):raise ValueError('Degenerate calibration gradients')
            records.append(dict(start=start,count=len(batch),norms=norms,losses={k:float(v[k].mean().detach()) for k in norms}))
            save(partial,dict(identity=identity,records=records,elapsed_seconds=old['elapsed_seconds']+time.perf_counter()-begin))
            print('train-only calibration',start+len(batch),'/500',flush=True)
        med={key:float(np.median([r['norms'][key] for r in records])) for key in records[0]['norms']}
        coefficients=dict(overlap=.3*med['reference']/med['overlap_total'],task=.2*med['reference']/med['task'])
        set_coefficients(coefficients)
        save(output,dict(status='registered',sources=sources(),initialization=metadata(),panel='frozen train_monitor500',
            members=[list(r['member']) for r in rows],coefficients=coefficients,gradient_medians=med,
            gradient_target_fractions=dict(overlap=.3,task=.2),records=records,
            seconds=old['elapsed_seconds']+time.perf_counter()-begin,parameter_updates=0,dev_read=False,test_read=False))
        print('Frozen auxiliary coefficients:',coefficients,flush=True)

if __name__=='__main__':run()
