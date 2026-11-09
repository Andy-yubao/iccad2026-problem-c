import sys; from pathlib import Path; sys.path.insert(0,str(Path.cwd()))
from pathlib import Path
import json,torch
from experiments.E104_scaling32k.io import save
from experiments.E104_scaling32k.train import RUN,seed_all,restore_rng,GeometryModel,optimizer_for
from experiments.E104_scaling32k.cuda_step import CapturedStep
seed_all(0)
s=torch.load(RUN/'checkpoints/latest.pt',map_location='cpu',weights_only=False)
f=torch.load(RUN/'failure.pt',map_location='cpu',weights_only=False)
print('failure keys',list(f),'checkpoint',s['progress']['step'],s['status'],flush=True)
m=GeometryModel(**s['model_config']).cuda();m.load_state_dict(s['model']);m.train()
o=optimizer_for(m);o.load_state_dict(s['optimizer'])
b={k:v.cuda() for k,v in f['batch'].items()}
cap=CapturedStep(m,o,b,'baseline');cap.inputs['axis_swap'].copy_(b['axis_swap']);restore_rng(f['rng'])
cap.graph.replay();torch.cuda.synchronize()
result=dict(checkpoint_step=s['progress']['step'],batch_members=f['members'],replay_rng='saved post-failure RNG; exact pre-failure CUDA graph offset was not saved',finite=bool(cap.finite),gradient_norm=float(cap.norm),max_case=float(cap.max_case),output_max=float(cap.output_max),values={k:float(v) for k,v in cap.values.items()},per_case_total=cap.per_layout['total'].cpu().tolist(),nonfinite_gradients=[n for n,p in m.named_parameters() if p.grad is not None and not bool(torch.isfinite(p.grad).all())],test_read=False,parameter_updates=0)
save(RUN/'repair_20261105/diagnosis.json',result)
print(json.dumps({k:v for k,v in result.items() if k not in ('batch_members','nonfinite_gradients')},indent=2),flush=True)
print('nonfinite grads',result['nonfinite_gradients'][:20],flush=True)

