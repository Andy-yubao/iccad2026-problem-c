"""Bounded CUDA Loss/capture check using the same five prepared train members."""
import json,time,torch
from .io import ROOT,HERE,save,sha
from .train import seed_all
from .data import load_rows,collate
from .batches import inputs_only
from .model import GeometryModel
from .loss import losses
from .loss_config import configuration
from .cuda_step import CapturedStep

def main():
    started=time.perf_counter();seed_all(0)
    ids=[50,36,22,62,45];rows=load_rows('train',ids,use_cache=False)
    graph,_=collate([rows[i%5] for i in range(32)],'cuda')
    path=ROOT/'experiments/E51_module_overlap/runs/20261025_rectangle_w96d12_unified_v2_main_seed0/checkpoints/best.pt'
    state=torch.load(path,map_location='cpu',weights_only=False)
    model=GeometryModel(**state['model_config']).cuda().eval();model.load_state_dict(state['model'])
    output=model(inputs_only(graph));values={}
    for arm in ('geometry','tolerant','clearance','wire_guard','anchor_ablation','count_ablation'):
        a=losses(output,graph,None,arm,0.,include_quality=False)
        b=losses(output,graph,None,arm,100.,include_quality=True)
        assert torch.equal(a['total'],b['total'])
        assert torch.isfinite(torch.stack(list(a.values()))).all()
        grads=torch.autograd.grad(a['total'],list(model.parameters()),retain_graph=True)
        assert all(torch.isfinite(g).all() for g in grads)
        values[arm]={'total':float(a['total'].detach()),'grad_norm':float(sum(g.square().sum() for g in grads).sqrt()),'definition':configuration(arm)}
    # Capture a fresh model, as formal training does. Retained default-stream
    # autograd graphs above must not cross into a different capture stream.
    del output,a,b,grads,model
    torch.cuda.synchronize()
    model=GeometryModel(**state['model_config']).cuda().eval();model.load_state_dict(state['model'])
    optimizer=torch.optim.AdamW(model.parameters(),lr=0.,weight_decay=3e-4)
    captured=CapturedStep(model,optimizer,graph,'geometry');captured.replay()
    assert abs(float(captured.values['total'])-values['geometry']['total'])<1e-6
    save(HERE/'runs/20261026_preparation/results.json',{'status':'passed','seconds':time.perf_counter()-started,
        'unique_train_cases':5,'indices':ids,'arms':values,'geometry_capture_loss_abs_delta':abs(float(captured.values['total'])-values['geometry']['total']),
        'formal_training_started':False,'optimizer_lr':0.,'checkpoint_written':False,'test_read':False})
    print(json.dumps({'status':'passed','seconds':time.perf_counter()-started,'arms':{k:dict(total=v['total'],grad_norm=v['grad_norm']) for k,v in values.items()}},indent=2))

if __name__=='__main__':main()
