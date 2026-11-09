"""Inspect the failed wide seed1; reuse the same five train diagnostic members."""
import json
import torch
from .io import ROOT,HERE,sha,save
from .cache import load_rows,verify
from .data import collate
from .batches import inputs_only
from .model import GeometryModel
from .loss import losses


def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    old=ROOT/'experiments/E57_capacity_data_diagnosis/runs'
    ids=json.loads((old/'20261026_wide_instability_audit/cpu_results.json').read_text(encoding='utf-8'))['global_train_indices']
    rows=load_rows('train',ids);g,_=collate(rows,'cpu');free=g['valid']&g['mask'][...,0].bool()
    folder=old/'20261026_capacity16000_geometry160_seed1'
    result={'train_indices':ids,'unique_train_cases':5,'test_read':False,'GPU_used':False,'cache_verified':True,'checkpoints':{}}
    for label in ('best','latest'):
        path=folder/'checkpoints'/f'{label}.pt';before=sha(path)
        state=torch.load(path,map_location='cpu',weights_only=False)
        model=GeometryModel(**state['model_config']).eval();model.load_state_dict(state['model'])
        hidden=[];pre=[];act=[]
        handles=[model.head.register_forward_hook(lambda m,a,o:hidden.append(float(a[0].detach()[g['valid']].abs().max()))),
                 model.head[0].register_forward_hook(lambda m,a,o:pre.append(float((o.detach()[free]<-20).float().mean()))),
                 model.head[1].register_forward_hook(lambda m,a,o:act.append(float((o.detach()[free]==0).float().mean())))]
        output=model(inputs_only(g));loss=losses(output,g,None,'original96',0)['total'];loss.backward()
        for h in handles:h.remove()
        batches=state['progress']['batch_history']
        result['checkpoints'][label]={'sha256':before,'step':state['progress']['step'],'loss_on_probe':float(loss.detach()),
            'hidden_max_per_readout':hidden,'final_preact_below_minus20_fraction':pre[-1],
            'final_silu_exact_zero_fraction':act[-1],
            'output_max_delta_from_bias':float((output.detach()[free]-model.head[2].bias.detach()).abs().max()),
            'gradients':{n:float(p.grad.norm()) if p.grad is not None else None for n,p in model.named_parameters() if n in ('input.0.weight','head.0.weight','head.2.weight','head.2.bias')},
            'max_logged_batch_loss':max(x['losses']['total'] for x in batches),
            'max_logged_grad_norm':max(x['grad_norm'] for x in batches),
            'dev_curve':[{'step':x['step'],'loss':x['dev_monitor']['total']} for x in state['progress']['history']]}
        assert sha(path)==before
    latest=result['checkpoints']['latest']
    result['wide_seed1_unstable']=latest['hidden_max_per_readout'][-1]>1e5 or latest['final_silu_exact_zero_fraction']>.99
    result['next_repair']='parameter-free existing LayerNorm before graph aggregation; same seed1/data/Loss/LR, scratch initialization'
    verify('dev')
    save(HERE/'runs/20261027_wide_seed1_audit/results.json',result)
    print(json.dumps({'unstable':result['wide_seed1_unstable'],'latest_hidden':latest['hidden_max_per_readout'][-1],
                      'latest_silu_zero_fraction':latest['final_silu_exact_zero_fraction']},ensure_ascii=False))


if __name__=='__main__':main()
