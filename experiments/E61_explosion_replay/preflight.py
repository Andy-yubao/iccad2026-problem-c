"""Five unique train cases: unchanged forward/G and real projected-step resume."""
import json
import time
import torch
from .io import ROOT,HERE,save
from .cache import load_rows
from .data import collate
from .batches import inputs_only
from .model import GeometryModel as OriginalModel
from .training_model import GeometryModel
from .loss import losses
from .losses import components,WEIGHTS
from .train import seed_all,parser,run

def main():
    started=time.perf_counter();seed_all(0)
    ids=json.loads((ROOT/'experiments/E60_scaling_diagnosis/runs/20261027_saved_and_probe/five_case_probe.json').read_text(encoding='utf-8'))['train_indices']
    assert len(ids)==len(set(ids))==5
    rows=load_rows('train',ids);batch=[rows[i%5] for i in range(32)];g,_=collate(batch,'cuda')
    old=OriginalModel(width=160,layers=12,rectangle=True,ffn_dropout=.05).cuda().eval()
    new=GeometryModel(width=160,layers=12,rectangle=True,ffn_dropout=.05,graph_prenorm=False).cuda().eval()
    new.load_state_dict(old.state_dict())
    with torch.no_grad():a=old(inputs_only(g));b=new(inputs_only(g))
    assert torch.equal(a,b)
    out=b.detach().requires_grad_();values=losses(out,g,None,'large_original160',0)
    expected=sum(w*v.mean() for k,w in WEIGHTS.items() for v in [components(out,g)[k]])
    assert torch.equal(values['total'],expected)
    assert all(not hasattr(layer,'graph_prenorm') or not layer.graph_prenorm for layer in new.layers)
    del old,new,g,out,values,expected;torch.cuda.empty_cache()
    folder=HERE/'runs/20261027_preflight_projected_resume_r2';direct=folder.with_name(folder.name+'_direct')
    common=['--arm','large_original160','--dataset-size','64000','--max-hours','.2','--max-epochs','2','--engine','staged_cuda_graph']
    args=parser().parse_args(['--output',str(folder),*common]);run(args,implementation_check=True,check_rows=(batch,batch),stop_after_steps=1)
    args.resume=True;run(args,implementation_check=True,check_rows=(batch,batch),stop_after_steps=2)
    args2=parser().parse_args(['--output',str(direct),*common]);run(args2,implementation_check=True,check_rows=(batch,batch),stop_after_steps=2)
    resumed=torch.load(folder/'checkpoints/latest.pt',map_location='cpu',weights_only=False)
    full=torch.load(direct/'checkpoints/latest.pt',map_location='cpu',weights_only=False)
    assert resumed['progress']['step']==full['progress']['step']==2
    assert all(torch.equal(v,full['model'][k]) for k,v in resumed['model'].items())
    for k,v in resumed['optimizer']['state'].items():
        for name,t in v.items():assert torch.equal(t,full['optimizer']['state'][k][name])
    assert resumed['graph_step_control']==full['graph_step_control']
    save(HERE/'runs/20261027_preflight/results.json',{'status':'passed','unique_train_cases':5,'train_indices':ids,
        'original_forward_bit_exact':True,'fixed_G_identical':True,'graph_input_normalization':False,
        'actual_entry_projected_update_exact_resume':True,'model_optimizer_control_equal':True,
        'elapsed_seconds':time.perf_counter()-started,'test_read':False})
    print('E61 five-case preflight passed',flush=True)

if __name__=='__main__':main()
