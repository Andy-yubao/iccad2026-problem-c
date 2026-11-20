"""Five already-registered train members; public-only input and no-MIB repair checks."""
import json
import numpy as np
import torch
from ..io import HERE,save,sha
from ..data import load_rows
from ..batches import inputs_only,decode
from ..model import GeometryModel
from .run import public_graph,repair_inputs,case_from_item
from .solver import optimize
from .repair.canonical_native import warmup
from datasets.random_reference import RandomReference
from scoring.research_no_mib import evaluate_case

def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    plan=json.loads((HERE/'plan.json').read_text(encoding='utf-8'))
    rows=load_rows('train',plan['preflight_train_indices'])
    dataset=RandomReference('train')
    items=[dataset[i] for i in plan['preflight_train_indices']]
    for row,item in zip(rows,items):
        # Loss ports are not network inputs, so no port padding is required here.
        graph=public_graph(item['solver_inputs'])
        n=row['graph']['n']
        for key in inputs_only(graph):
            if key=='valid':assert graph[key][0,:n].all() and not graph[key][0,n:].any()
            elif key=='scale':assert torch.equal(graph[key][0],row['graph'][key])
            elif key in ('adjacency','group'):assert torch.equal(graph[key][0,:n,:n],row['graph'][key])
            else:assert torch.equal(graph[key][0,:n],row['graph'][key]),key
        repaired=repair_inputs(item['solver_inputs'])
        assert not repaired[5][:,2].any()
        for k in (1,2,3,4,6):np.testing.assert_array_equal(repaired[k],item['solver_inputs'][k])
    p=HERE/'runs/terminal_seed0/checkpoints/best.pt';digest=sha(p)
    saved=torch.load(p,map_location='cpu',weights_only=False)
    model=GeometryModel(**saved['model_config']).eval();model.load_state_dict(saved['model'])
    item=items[0];g=public_graph(item['solver_inputs'])
    with torch.no_grad():draft=decode(model(g)[-1],g)[0,:int(item['solver_inputs'][0])].numpy().astype(float)*float(g['scale'][0])
    warmup();final,info=optimize(draft,repair_inputs(item['solver_inputs']))
    scored=evaluate_case(case_from_item(item),final,info['seconds'])
    assert scored['metrics']['is_feasible'] and scored['metrics']['mib_violations']==0
    assert sha(p)==digest
    save(HERE/'evaluation/adapter_check.json',dict(status='passed',unique_train_cases=5,
        train_postprocessing_calls=1,dev_payloads=0,test_read=False,
        public_input_matches_training=True,original_areas_and_specs_unchanged=True,
        copied_repair_ignores_MIB=True,final_hard_legal=True,checkpoint_unchanged=True,
        optimizer_updates=0))
    print('E114 public-only adapter and no-MIB repair check passed',flush=True)

if __name__=='__main__':main()
