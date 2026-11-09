"""Fixed actual fault batch: older checkpoint and causal graph/feedback gates."""
import json
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('OMP_NUM_THREADS','4')
import time
import torch
from .io import ROOT,HERE,save
from .cache import load_rows
from .batches import collate
from .model import GeometryModel
from .cuda_step import CapturedStep
from .losses import components,WEIGHTS
from .replay import restore_rng

def main():
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    started=time.perf_counter();folder=HERE/'runs/20261027_seed0_replay'
    after=torch.load(folder/'first_abnormal_before_update.pt',map_location='cpu',weights_only=False)
    before=torch.load(folder/'previous_update_before.pt',map_location='cpu',weights_only=False)
    original=torch.load(ROOT/'experiments/E57_capacity_data_diagnosis/runs/20261026_capacity16000_geometry160_seed0/checkpoints/best.pt',map_location='cpu',weights_only=False)
    ids=after['event']['global_train_indices'];rows=load_rows('train',ids)
    graph=collate([r['graph'] for r in rows],list(range(32)),'cuda',max_nodes=120)
    model=GeometryModel(**after['model_config']).cuda().train();model.load_state_dict(after['model'])
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0003)
    gates=torch.ones(13,device='cuda');hs=[];outputs=[];local_inputs=[];local_outputs=[]
    handles=[model.head.register_forward_pre_hook(lambda m,a:hs.append(a[0])),
        model.head.register_forward_hook(lambda m,a,o:outputs.append(o)),
        model.state_embed.register_forward_hook(lambda m,a,o:o*gates[12])]
    for i,layer in enumerate(model.layers):
        handles.append(layer.local.register_forward_pre_hook(lambda m,a:local_inputs.append(a[0])))
        def hook(m,a,o,index=i):
            local_outputs.append(o)
            return o*gates[index]
        handles.append(layer.local.register_forward_hook(hook))
    restore_rng(after['rng_before_batch']);captured=CapturedStep(model,optimizer,graph,'geometry160')
    for handle in handles:handle.remove()
    hs=hs[-13:];out=outputs[-1];local_inputs=local_inputs[-12:];local_outputs=local_outputs[-12:]
    def evaluate(theta,graph_gain=1.,feedback_gain=1.):
        model.load_state_dict(theta);gates[:12].fill_(graph_gain);gates[12]=feedback_gain
        restore_rng(after['rng_before_batch']);captured.graph.replay()
        with torch.no_grad():
            c=components(out,captured.inputs);per_case=sum(w*c[k] for k,w in WEIGHTS.items())
            return {'total':float(captured.values['total']),'grad_norm':float(captured.norm),
                'losses':{k:float(v) for k,v in captured.values.items()},'case_loss':per_case.cpu().tolist(),
                'hidden_max_per_case_per_layer':[h.detach().abs().amax((1,2)).cpu().tolist() for h in hs],
                'local_input_max_per_case_per_layer':[h.detach().abs().amax((1,2)).cpu().tolist() for h in local_inputs],
                'local_output_max_per_case_per_layer':[h.detach().abs().amax((1,2)).cpu().tolist() for h in local_outputs]}
    result={'original6000':evaluate(original['model']),'before6277':evaluate(before['model']),
        'after6277':evaluate(after['model']),'after6277_graph_off':evaluate(after['model'],graph_gain=0.),
        'after6277_feedback_off':evaluate(after['model'],feedback_gain=0.),'input_cases':[],
        'scope':'Same actual32 fault cases and same CUDA graph Dropout/RNG; no optimizer update or test access.'}
    assert result['after6277']['total']==after['event']['losses']['total']
    for idx in (12,10):
        r=rows[idx];g=r['graph'];n=g['n'];areas=g['area']
        result['input_cases'].append({'batch_index':idx,'train_index':ids[idx],'member':list(r['member']),'n':n,
            'area_min':float(areas.min()),'area_max':float(areas.max()),'area_ratio':float(areas.max()/areas.min()),
            'nonfinite_graph_fields':[k for k,v in g.items() if isinstance(v,torch.Tensor) and not bool(torch.isfinite(v).all())],
            'max_adjacency_row_sum':float(g['adjacency'].sum(1).max()),'max_mib_row_sum':float(g['mib'].sum(1).max()),
            'max_group_row_sum':float(g['group'].sum(1).max()),'target_max':float(g['target'].abs().max())})
    result['elapsed_seconds']=time.perf_counter()-started;result['test_read']=False
    save(folder/'fault_probe.json',result)
    print(json.dumps({k:v['total'] for k,v in result.items() if isinstance(v,dict) and 'total' in v}),flush=True)

if __name__=='__main__':main()
