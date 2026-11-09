import sys;from pathlib import Path;sys.path.insert(0,str(Path.cwd()))
import torch,json,time
from experiments.E104_scaling32k.train import RUN,seed_all,restore_rng,GeometryModel,optimizer_for
from experiments.E104_scaling32k.cuda_step import CapturedStep
from experiments.E104_scaling32k.batches import inputs_only
from experiments.E104_scaling32k.augmentation import axis_swap
from experiments.E104_scaling32k.loss import per_case
from experiments.E104_scaling32k.io import save,sha
seed_all(0);s=torch.load(RUN/'repair_20261105/before/checkpoints/latest.pt',map_location='cpu',weights_only=False);f=torch.load(RUN/'repair_20261105/before/failure.pt',map_location='cpu',weights_only=False)
config=dict(s['model_config'],normalize_local=True);m=GeometryModel(**config).cuda();m.load_state_dict(s['model']);m.train();o=optimizer_for(m);o.load_state_dict(s['optimizer']);b={k:v.cuda() for k,v in f['batch'].items()}
cap=CapturedStep(m,o,b,'baseline');cap.inputs['axis_swap'].copy_(b['axis_swap']);restore_rng(f['rng']);old=m.layers[0].local.weight.detach().clone();steps=[int(v['step']) for v in o.state.values()]
values,norm=cap.replay();assert not torch.equal(old,m.layers[0].local.weight);assert all(int(v['step'])==step+1 for v,step in zip(o.state.values(),steps));assert all(bool(torch.isfinite(p).all()) for p in m.parameters())
result=dict(status='passed',model_config=config,model_source_sha256=sha(Path('experiments/E104_scaling32k/model.py')),failure_batch_replayed_without_skipping=True,diagnostic_exception_to_five_case_limit='exact saved failing batch32 required to preserve batch-shaped Dropout path; no new cases or panels',unique_failure_batch_members=len(f['members']),actual_updates=1,discarded_diagnostic_update=True,parameters=sum(p.numel() for p in m.parameters()),loss=float(values['total']),max_case=float(cap.max_case),gradient_norm=float(norm),output_max=float(cap.output_max),all_parameters_finite=True,guard_thresholds_unchanged=True,test_read=False)
m.eval();one={k:v[1:2] for k,v in b.items()};one=axis_swap(one)
with torch.no_grad():result['offender_eval_loss_after_diagnostic_update']=float(per_case(m(inputs_only(one)),one)['total'])
save(RUN/'repair_20261105/repair_shortcheck.json',result);print(json.dumps(result,indent=2),flush=True)
