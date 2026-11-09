"""One failed batch and five-case component restoration; no optimizer updates."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import torch
from .io import HERE,ROOT,save,sha
from .train import seed_all
from .data import load_rows,collate
from .base_model import GeometryModel
from .batches import inputs_only
from .loss import per_case

def evaluate(model,g,ids,grad=False):
    hidden=[];handles=[layer.register_forward_hook(lambda module,args,out:hidden.append(out.detach().abs().amax((1,2)).cpu().tolist())) for layer in model.layers]
    model.eval();v=per_case(model(inputs_only(g)),g,'topology');norm=None
    if grad:
        v['total'].mean().backward();norm=float(torch.stack([p.grad.square().sum() for p in model.parameters() if p.grad is not None]).sum().sqrt())
    result={'mean':float(v['total'].mean().detach()),'gradient_norm':norm,'hidden_max':max(max(layer) for layer in hidden),
        'cases':[{'train_index':index,**{k:float(v[k][i].detach()) for k in ('total','geometry','true_intersection','box_expansion','center_local','shape','contact')}} for i,index in enumerate(ids)]}
    for h in handles:h.remove()
    return result

def main():
    seed_all(0);old=ROOT/'experiments/E74_collision_32k_posttrain/runs/20261029_e42_topology_seed0/checkpoints/best.pt';bad=ROOT/'experiments/E75_collision_box_finetune/runs/20261029_e74_collision_box_seed0/checkpoints/latest.pt'
    source=torch.load(old,map_location='cpu',weights_only=False,mmap=True);latest=torch.load(bad,map_location='cpu',weights_only=False,mmap=True)
    p=latest['progress'];ids=p['order'][p['cursor']:p['cursor']+32].tolist();rows=load_rows('train',ids,use_cache=False);g,_=collate(rows,'cuda');out={}
    for name,state in (('E74_source',source),('E75_pre_rejected',latest)):
        model=GeometryModel(**state['model_config']).cuda().eval();model.load_state_dict(state['model']);out[name]=evaluate(model,g,ids,True);del model;torch.cuda.empty_cache()
    worst=sorted(out['E75_pre_rejected']['cases'],key=lambda r:r['total'],reverse=True)[:5];chosen=[r['train_index'] for r in worst]
    selected=[rows[ids.index(i)] for i in chosen];small,_=collate(selected,'cuda');interventions={}
    groups={'restore_head':lambda k:k.startswith('head.'),'restore_state_embed':lambda k:k.startswith('state_embed.'),
        'restore_attention':lambda k:'.qkv.' in k or '.output.' in k or '.bias.' in k,
        'restore_ffn_norm':lambda k:'.ff.' in k or '.norm' in k}
    for name,match in groups.items():
        model=GeometryModel(**source['model_config']).cuda().eval();model.load_state_dict({k:(source['model'][k] if match(k) else v) for k,v in latest['model'].items()})
        interventions[name]=evaluate(model,small,chosen);del model;torch.cuda.empty_cache()
    drift={k:float((x-source['model'][k]).norm()/source['model'][k].norm().clamp_min(1e-8)) for k,x in latest['model'].items() if x.is_floating_point()}
    out.update(step=p['step'],epoch=p['epoch'],batch_indices=ids,source_sha256=sha(old),failure_sha256=sha(bad),interventions=interventions,relative_parameter_drift=drift,
        test_read=False,parameter_updates=0,scope='One rejected32 batch; component interventions on its5 highest-loss members. Eval diagnostic, not exact rejected Dropout RNG replay.')
    save(HERE/'runs/failure_diagnosis/results.json',out)
    for name in ('E74_source','E75_pre_rejected'):
        v=out[name];print(name,'mean',v['mean'],'grad',v['gradient_norm'],'hidden',v['hidden_max'],'top',sorted(v['cases'],key=lambda r:r['total'],reverse=True)[:2],flush=True)
    for name,v in interventions.items():print(name,v['mean'],v['hidden_max'],flush=True)
if __name__=='__main__':main()
