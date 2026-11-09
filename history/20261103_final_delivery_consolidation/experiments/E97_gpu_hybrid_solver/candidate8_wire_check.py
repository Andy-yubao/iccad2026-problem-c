"""Bounded public exact-wire check on the original five; no optimization/solve/Q."""
import json
import numpy as np
from .latency import torch, HERE, ROOT, IDS, registry, public_member_inputs, decode85, save, sha
from .latency_fixed import padded, returned
from .normalized_geometry import DeploymentModel
from .candidate8_io import network_inputs
from .candidate8_objective import raw_terms
from .mechanism import raw_measures, prepare_problem


@torch.no_grad()
def main():
    folder=HERE/'runs/20261102_candidate8_wire_check'
    folder.mkdir(exist_ok=False)
    plan=json.loads((HERE/'candidate8_plan.json').read_text(encoding='utf-8'))
    path=ROOT/plan['source_checkpoint']; assert sha(path)==plan['source_sha256']
    saved=torch.load(path,map_location='cpu',weights_only=False)
    model=DeploymentModel(**saved['model_config']).cuda().eval()
    model.load_state_dict(saved['model'],strict=True)
    records=[]
    for index in IDS:
        inputs=public_member_inputs(registry()['sets']['train']['records'][index],'train')
        g={k:v.cuda() for k,v in padded(inputs,True).items()}
        state=model(network_inputs(g))
        terms=raw_terms(state,g,state,torch.ones(1,device='cuda'))
        mass=.5*g['b2b_weights'].sum()+g['p2b_weights'].sum()
        public_area=(g['area']*g['valid']).sum()
        calculated=float(terms['wire_ratio'][0]*mass*public_area.sqrt()*g['scale'][0])
        geometry=returned(state,g,inputs,decode85)
        area,wire=raw_measures(geometry,prepare_problem(*inputs))
        calculated_area=float(terms['bbox_ratio'][0]*public_area*g['scale'][0].square())
        assert np.isclose(calculated,wire,rtol=1e-5,atol=1e-6),(index,calculated,wire)
        assert np.isclose(calculated_area,area,rtol=1e-5,atol=1e-6),(index,calculated_area,area)
        records.append(dict(train_id=index,n=int(inputs[0]),loss_wire_physical=calculated,independent_wire=wire,relative_wire_error=abs(calculated-wire)/max(abs(wire),1e-12),loss_bbox_physical=calculated_area,independent_bbox=area))
    save(folder/'results.json',dict(status='passed',records=records,rtol=1e-5,atol=1e-6,no_optimizer_updates=True,no_solver_or_Q_calls=True,source_checkpoint_sha256=sha(path),objective_sha256=sha(HERE/'candidate8_objective.py'),test_read=False))
    print(json.dumps(dict(status='passed',records=records),indent=2))


if __name__=='__main__':main()
