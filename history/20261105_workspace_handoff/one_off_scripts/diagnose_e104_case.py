import sys;from pathlib import Path;sys.path.insert(0,str(Path.cwd()))
import torch,json
from experiments.E104_scaling32k.train import RUN,seed_all,GeometryModel,restore_rng
from experiments.E104_scaling32k.batches import inputs_only,decode
from experiments.E104_scaling32k.loss import per_case
from experiments.E104_scaling32k.augmentation import axis_swap
from experiments.E104_scaling32k.simple_loss import contact_cost
from experiments.E104_scaling32k.io import save
seed_all(0);s=torch.load(RUN/'checkpoints/latest.pt',map_location='cpu',weights_only=False);f=torch.load(RUN/'failure.pt',map_location='cpu',weights_only=False)
i=1;b={k:v[i:i+1].cuda() for k,v in f['batch'].items()};aug=axis_swap(b);m=GeometryModel(**s['model_config']).cuda();m.load_state_dict(s['model']);m.eval();stats=[]
with torch.no_grad():
    g=inputs_only(aug);h=m.input(g['features'])*g['valid'][...,None];state=m.head(h);stats.append(dict(layer=0,hidden_max=float(h.abs().max()),state_max=float(state.abs().max())))
    for j,layer in enumerate(m.layers):
        h=layer(h+m.state_embed(state*g['mask']),g,state);state=m.head(h);stats.append(dict(layer=j+1,hidden_max=float(h.abs().max()),state_max=float(state.abs().max())))
    vals=per_case(state,aug);ref=per_case(aug['target'],aug);p=decode(state,aug);local=aug['area'].clamp_min(1e-12).sqrt();ps=(local[:,:,None]+local[:,None,:])/2;cost=contact_cost(p,ps);same=(aug['group']>0)&aug['valid'][:,:,None]&aug['valid'][:,None,:]
    n=int(aug['valid'].sum());x=g['features'][0,:n]
    result=dict(member=f['members'][i],n=n,axis_swap=bool(b['axis_swap'][0]),features_min=x.min(0).values.cpu().tolist(),features_max=x.max(0).values.cpu().tolist(),area_min=float(aug['area'][0,:n].min()),area_max=float(aug['area'][0,:n].max()),eval_terms={k:float(v) for k,v in vals.items()},target_terms={k:float(v) for k,v in ref.items()},layer_stats=stats,max_group_pair_contact_cost=float(cost[same].max()),group_labels=aug['constraints'][0,:n,3].cpu().tolist(),positions=p[0,:n].cpu().tolist(),reference=decode(aug['target'],aug)[0,:n].cpu().tolist(),test_read=False,parameter_updates=0)
    save(RUN/'repair_20261105/case_diagnosis.json',result)
print(json.dumps({k:v for k,v in result.items() if k not in ('positions','reference','group_labels')},indent=2))
