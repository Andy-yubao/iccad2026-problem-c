"""Five training members; inspect saved checkpoints and deterministic CPU paths."""
from pathlib import Path
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import sys
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
import json
import copy
import math
import numpy as np
import torch
from experiments.E57_capacity_data_diagnosis.model import GeometryModel,decode
from experiments.E57_capacity_data_diagnosis.data import load_rows
from experiments.E57_capacity_data_diagnosis.resident import ResidentRows
from experiments.E57_capacity_data_diagnosis.batches import inputs_only
from experiments.E57_capacity_data_diagnosis.loss import losses
from experiments.E57_capacity_data_diagnosis.loss_reference import components as original_components,WEIGHTS
from experiments.E57_capacity_data_diagnosis.io import sha,save

HERE=Path(__file__).resolve().parent
RUN=ROOT/'experiments/E57_capacity_data_diagnosis/runs/20261026_capacity16000_geometry160_seed0'


def trace(model,g):
    records=[]
    valid=g['valid'];free=(g['mask'][...,0]>0)&valid
    def hook(module,args,out):
        v=out.detach();rect=decode(v,g)
        clipped=(v[...,:2]-rect[...,2:]/2)<0
        records.append({'head_readout':len(records),'hidden_abs_max':float(args[0][valid].abs().max()),
            'raw_center_abs_max':float(v[...,:2][valid].abs().max()),
            'raw_shape_abs_max':float(v[...,2][valid].abs().max()),
            'free_center_axes_clipped_fraction':float(clipped[free].float().mean()),
            'free_rectangles_at_origin_fraction':float((rect[...,:2][free]==0).all(-1).float().mean())})
    handle=model.head.register_forward_hook(hook)
    with torch.no_grad():state=model(inputs_only(g));values=losses(state,g,None,'geometry160',0)
    handle.remove()
    return {'losses':{k:float(v) for k,v in values.items()},'layers':records}


def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    checkpoints={name:torch.load(RUN/'checkpoints'/file,map_location='cpu',weights_only=False)
                 for name,file in [('best','best.pt'),('latest','latest.pt')]}
    protected={name:sha(RUN/'checkpoints'/file) for name,file in [('best','best.pt'),('latest','latest.pt')]}
    best=checkpoints['best'];last=checkpoints['latest'];history=last['progress']['batch_history']
    selected=best['identity']['subset_indices'];order=np.random.default_rng(12).permutation(len(selected))
    batch_local=order[(6300-6000-1)*32:(6300-6000)*32]
    ids=[selected[int(batch_local[i])] for i in (0,7,15,23,31)]
    rows=load_rows('train',ids,use_cache=False)
    repeated=[rows[i%5] for i in range(32)]
    packed=ResidentRows(repeated,'cpu',120);g=packed.batch(list(range(32)))
    result={'unique_train_cases':5,'global_train_indices':ids,'member_ids':[list(r['member']) for r in rows],
            'selection':'five fixed positions from reconstructed logged step6300 minibatch; no label/Q selection',
            'sample_is_complete_trigger_batch':False,'GPU_used':False,'test_read':False,
            'source_hashes_match':all(sha(ROOT/n)==h for n,h in best['identity']['sources'].items()),
            'model_source_same_as_E42':sha(ROOT/'experiments/E57_capacity_data_diagnosis/model.py')==sha(ROOT/'experiments/E42_joint_regularization/model.py'),
            'events':[x for x in history if 6000<=x['step']<=6500]}
    spike=next(x for x in history if x['step']==6300)
    result['contact_share_of_spike_total']=.5*spike['losses']['contact']/spike['losses']['total']
    result['spike_global_clip_coefficient_upper_bound']=1./spike['grad_norm']
    result['checkpoints']={}
    for name,checkpoint in checkpoints.items():
        model=GeometryModel(**checkpoint['model_config']).eval();model.load_state_dict(checkpoint['model'])
        value=trace(model,g)
        value['all_parameters_finite']=all(torch.isfinite(v).all() for v in checkpoint['model'].values())
        value['readout_bias']=checkpoint['model']['head.2.bias'].tolist()
        result['checkpoints'][name]=value
    model=GeometryModel(**best['model_config']).eval();model.load_state_dict(best['model'])
    state=model(inputs_only(g)).detach().requires_grad_()
    actual=losses(state,g,None,'geometry160',0)['total'];original=original_components(state,g)
    expected=sum(WEIGHTS[k]*original[k].mean() for k in WEIGHTS)
    ga=torch.autograd.grad(actual,state,retain_graph=True)[0];gb=torch.autograd.grad(expected,state)[0]
    torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
    torch.testing.assert_close(ga,gb,rtol=1e-5,atol=1e-6)
    result['original_loss_parity']={'total_abs_delta':float((actual-expected).detach().abs()),
                                  'state_gradient_max_abs_delta':float((ga-gb).abs().max())}
    del model,state,actual,original,expected,ga,gb
    controls=[]
    for lr in (.001,.0001):
        model=GeometryModel(**best['model_config']).eval();model.load_state_dict(best['model'])
        optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.0003)
        optimizer.load_state_dict(copy.deepcopy(best['optimizer']))
        for group in optimizer.param_groups:group['lr']=lr
        optimizer.zero_grad(set_to_none=True)
        value=losses(model(inputs_only(g)),g,None,'geometry160',0)['total'];value.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        clipped=torch.stack([p.grad.norm().square() for p in model.parameters() if p.grad is not None]).sum().sqrt()
        assert float(clipped)<=1.0001
        optimizer.step()
        with torch.no_grad():after=losses(model(inputs_only(g)),g,None,'geometry160',0)['total']
        delta=sum((p.detach()-best['model'][n]).square().sum() for n,p in model.named_parameters()).sqrt()
        controls.append({'lr':lr,'loss_before':float(value.detach()),'loss_after':float(after),
            'grad_norm_before_clip':float(norm),'grad_norm_after_clip':float(clipped),
            'parameter_delta_norm':float(delta),'all_parameters_finite':all(torch.isfinite(p).all() for p in model.parameters())})
        del model,optimizer,value,after
    result['one_step_CPU_controls']=controls
    result['control_limit']='Eval-mode deterministic five-case batch repeated to32; not the complete original batch, not the original CUDA/Dropout trajectory, not proof that changing LR fixes the incident.'
    assert protected=={name:sha(RUN/'checkpoints'/file) for name,file in [('best','best.pt'),('latest','latest.pt')]}
    result['original_checkpoints_unchanged']=protected
    save(HERE/'cpu_results.json',result)
    print(json.dumps({'sources_match':result['source_hashes_match'],'contact_spike_share':result['contact_share_of_spike_total'],
        'best_final_layer':result['checkpoints']['best']['layers'][-1],
        'latest_final_layer':result['checkpoints']['latest']['layers'][-1],
        'loss_parity':result['original_loss_parity'],'controls':controls}),flush=True)


if __name__=='__main__':main()
