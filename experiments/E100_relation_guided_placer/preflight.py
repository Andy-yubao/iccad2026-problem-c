"""Five frozen train members, one actual captured update; no resume tests."""
import json
import time
import torch
from .io import HERE,save,sources
from .model import GeometryModel
from .data import load_rows
from .resident import ResidentRows
from .train import seed_all,optimizer_for,build_capture,MODEL


def verify():
    result = json.loads((HERE/'runs/preflight/results.json').read_text(encoding='utf-8'))
    if result['status']!='passed' or result['sources']!=sources():
        raise ValueError('Current-source minimal preflight required')
    return result


def main():
    started=time.perf_counter()
    seed_all(0)
    plan=json.loads((HERE/'plan.json').read_text(encoding='utf-8'))
    rows=load_rows('train',plan['preflight_train_indices'])
    pack=ResidentRows(rows,'cuda',max_nodes=120,max_pins=512)
    model=GeometryModel(**MODEL).cuda()
    optimizer=optimizer_for(model)
    cap=build_capture(model,optimizer,pack)
    before=model.relation_head[-1].weight.detach().clone()
    values,norm=cap.replay()
    change=(model.relation_head[-1].weight-before).abs().amax().item()
    grad=model.relation_head[-1].weight.grad.norm().item()
    if not change>0 or not grad>0 or not torch.isfinite(torch.tensor(grad)):
        raise RuntimeError('Relation head did not receive a finite actual update')
    result=dict(status='passed',sources=sources(),members=[list(r['member']) for r in rows],
        unique_members=len(rows),actual_updates=1,resume_tests=0,solver_calls=0,test_read=False,
        total=float(values['total']),relation=float(values['relation']),gradient_norm=float(norm),
        relation_head_gradient_norm=grad,relation_head_max_update=change,
        parameters=sum(p.numel() for p in model.parameters()),
        peak_gpu_allocated=torch.cuda.max_memory_allocated(),elapsed_seconds=time.perf_counter()-started)
    save(HERE/'runs/preflight/results.json',result)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
