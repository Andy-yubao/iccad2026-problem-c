"""One disposable protected update on the same five train cases; no formal run."""
import json
import time
import torch
from .control import OWNER,read,sources,assert_training_authorized,stopped,budget_exhausted
from .io import save
from .data import load_rows
from .model import build_model
from .resident import ResidentRows
from .addresses import attach_addresses
from .runtime import seed_all,optimizer_for
from .flow import training_statistics
from .cuda_step import ProtectedStep

def main():
    started=time.perf_counter();assert_training_authorized()
    if stopped() or budget_exhausted(training=True):raise InterruptedError('Stop/budget before startup update')
    plan=read(OWNER/'plan.json');ids=plan['preflight_train_indices'];rows=load_rows('train',ids)
    seed_all(0);model=build_model().cuda().train();stats=training_statistics(rows)
    model.state_mean.copy_(torch.tensor(stats['mean'],device='cuda'));model.state_std.copy_(torch.tensor(stats['std'],device='cuda'))
    pack=ResidentRows(rows,'cuda');indices=[i%5 for i in range(32)];sample=pack.batch(indices);attach_addresses(sample,rows,indices,epoch=0)
    optimizer=optimizer_for(model);before={k:v.clone() for k,v in model.named_parameters()}
    cap=ProtectedStep(model,optimizer,sample,updates_authorized=True)
    cap.inputs['axis_swap'].copy_(torch.tensor([i%2==0 for i in range(32)],device='cuda'))
    values,norm=cap.replay();torch.cuda.synchronize()
    assert cap.projection.steps==1 and optimizer.state
    assert all(torch.isfinite(p).all() for p in model.parameters())
    changes={k:float((p-before[k]).abs().max()) for k,p in model.named_parameters()}
    assert max(changes.values())>0 and not (OWNER/'runs').exists()
    result=dict(status='passed',optimizer_updates=1,disposable=True,formal_model_or_Adam_reused=False,
        members=[list(r['member']) for r in rows],batch_size=32,total=float(values['total']),max_case=float(cap.max_case),
        raw_gradient_norm=float(norm),max_parameter_change=max(changes.values()),projection=cap.projection.summary(),
        sources=sources(),elapsed_seconds=time.perf_counter()-started,test_read=False,dev_tensor_payloads=0)
    save(OWNER/'startup_check.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ('sources','projection')}),flush=True)
if __name__=='__main__':main()
