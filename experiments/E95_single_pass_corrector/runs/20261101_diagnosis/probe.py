"""Read-only frozen checkpoint diagnosis on the original five train members."""
import json,time,copy
from pathlib import Path
import numpy as np
import torch
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[4]))
from experiments.E95_single_pass_corrector.io import HERE,ROOT,save,sha
from experiments.E95_single_pass_corrector.runtime import seed,CONFIG,charge
from experiments.E95_single_pass_corrector.data import load_rows,collate,inputs_only
from experiments.E95_single_pass_corrector.model import GeometryModel,decode
from experiments.E95_single_pass_corrector.initialization import load
from experiments.E95_single_pass_corrector.loss import per_case

def main():
    seed();run=HERE/'runs/20261101_single_call_seed0';out=Path(__file__).parent
    ids=json.loads((HERE/'plan.json').read_text(encoding='utf-8'))['preflight_train_ids']
    rows=load_rows('train',ids);g=inputs_only(collate(rows,'cuda'))
    frozen=torch.load(run/'inputs/train.pt',map_location='cpu',weights_only=False)
    draft=torch.zeros(5,g['valid'].shape[1],3,device='cuda')
    for i,(row,index) in enumerate(zip(rows,ids)):
        assert list(row['member'])==frozen['identity']['members'][index]
        draft[i,:row['graph']['n']]=frozen['states'][index].cuda()
    model=GeometryModel(**CONFIG).cuda();load(model);model.eval()
    initial=copy.deepcopy(model.state_dict());records={}
    for name,path in [('initial',None),('best',run/'checkpoints/best.pt'),('latest',run/'checkpoints/latest.pt')]:
        if path:
            checkpoint=torch.load(path,map_location='cpu',weights_only=False)
            # Current model and objective code must exactly match the source
            # owning this old checkpoint. Protocol prose is intentionally revised.
            for filename in ['model.py','base_model.py','loss.py','public_task.py','group_tree.py']:
                source=f'experiments/E95_single_pass_corrector/{filename}'
                assert sha(HERE/filename)==checkpoint['identity']['sources'][source],filename
            model.load_state_dict(checkpoint['model'])
        else:model.load_state_dict(initial)
        model.eval();layers=[];handles=[]
        def inspect(module,args,value):
            h=value[:,:g['valid'].shape[1]];v=g['valid'][...,None]
            mean=(h*v).sum(1)/v.sum(1)
            variance=((h-mean[:,None]).square()*v).sum((1,2))/(v.sum((1,2))*h.shape[-1])
            rms=((h.square()*v).sum((1,2))/(v.sum((1,2))*h.shape[-1])).sqrt()
            layers.append(dict(rms=float(rms.mean()),between_module_std=float(variance.sqrt().mean()),
                relative_module_std=float((variance.sqrt()/rms.clamp_min(1e-12)).mean())))
        for layer in model.corrector.layers:handles.append(layer.register_forward_hook(inspect))
        with torch.no_grad():
            state=model.corrector(g,draft);values=per_case(state,g,draft)
            physical=decode(state,g);source=decode(draft,g)
            move=((physical[...,:2]+physical[...,2:]/2)-(source[...,:2]+source[...,2:]/2))/g['area'].sqrt()[...,None]
            valid=g['valid'];delta=(state-draft)[valid]
            result=dict(mean_loss={k:float(v.mean()) for k,v in values.items()},hidden_layers=layers,
                center_movement_module_units_mean=float(move.norm(dim=-1)[valid].mean()),
                center_movement_module_units_max=float(move.norm(dim=-1)[valid].max()),
                raw_delta_std=delta.std(0).tolist(),raw_delta_mean=delta.mean(0).tolist(),
                head_last_weight_norm=float(model.corrector.head[-1].weight.norm()),
                head_last_bias=model.corrector.head[-1].bias.tolist())
        for handle in handles:handle.remove()
        state=state.detach().requires_grad_();value=per_case(state,g,draft)['total'].mean()
        grad=torch.autograd.grad(value,state)[0]
        result['output_gradient_norm']=float(grad.norm())
        result['output_gradient_active_fraction']=float((grad[valid].abs()>1e-10).float().mean())
        if path:result.update(checkpoint_sha256=sha(path),checkpoint_step=checkpoint['progress']['step'])
        records[name]=result
    progress=json.loads((run/'progress.json').read_text(encoding='utf-8'))
    save(out/'frozen_probe.json',dict(unique_train_cases=5,indices=ids,records=records,
        history=progress['history'],formal_updates=0,test_read=False))
    for name,value in records.items():
        print(name,json.dumps({k:v for k,v in value.items() if k not in ('hidden_layers','mean_loss')},ensure_ascii=True))
        print('loss',value['mean_loss']['total'],'layer1/12',value['hidden_layers'][0],value['hidden_layers'][-1],flush=True)

if __name__=='__main__':
    started=time.perf_counter()
    try:main()
    finally:charge(time.perf_counter()-started)
