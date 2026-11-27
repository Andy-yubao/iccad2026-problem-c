"""Own E114 readout only; reuse all existing finals and train-five layers."""
import json
import copy
import numpy as np
import torch
from .io import ROOT,HERE,save,sha
from .evaluation_staged.run import Predictor
from .batches import decode
from datasets.random_reference import RandomReference
from visualization.layer_comparison import measures

def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    folder=ROOT/'visualization/E117_node26/20261109_vs_E114'
    selection=json.loads((folder/'selection.json').read_text())
    previous=json.loads((ROOT/'visualization/E114_supervision/20261108_staged_three_phase10/data.json').read_text(encoding='utf-8'))
    lookup={tuple(c['member']):c for c in previous['cases']}
    run=HERE/'runs/staged_seed0';checkpoint=run/'checkpoints/best.pt'
    result=json.loads((run/'evaluation_no_mib_dev500/results.json').read_text())
    assert sha(checkpoint)==result['checkpoint_sha256']
    train=RandomReference('train');dev=RandomReference('dev_monitor500');cases=[];predictor=None;inferences=0
    for rank,index in enumerate(selection['train_indices']):
        member=train.records[train.indices[index]]['member']
        case=copy.deepcopy(lookup[tuple(member)]);case.update(group='fixed',label='固定例 '+str(rank+1),evaluation_index=None)
        cases.append(case)
    with torch.inference_mode():
        for rank,index in enumerate(selection['dev_indices']):
            record=json.loads((run/f'evaluation_no_mib_dev500/cases/{index:04d}.json').read_text())
            if tuple(record['member']) in lookup:
                case=copy.deepcopy(lookup[tuple(record['member'])])
                np.testing.assert_allclose(case['final'],record['positions'],rtol=0,atol=1e-10)
            else:
                item=dev[index];inputs=item['solver_inputs'];n=int(inputs[0]);assert item['member']==record['member']
                if predictor is None:predictor=Predictor(checkpoint,inputs)
                draft=predictor.predict(inputs)
                np.testing.assert_allclose(draft,record['draft_positions'],rtol=1e-6,atol=1e-5)
                layers=[(decode(s,predictor.g)[0,:n]*predictor.g['scale'][0]).cpu().numpy().astype(float) for s in predictor.states]
                case=dict(member=item['member'],n=n,constraints=np.asarray(inputs[5]).astype(int).tolist(),
                    reference=np.asarray(item['supervision']['reference_xywh']).tolist(),layers=[r.tolist() for r in layers],
                    final=record['positions'],coverage=[measures(r)['repeated_coverage_fraction'] for r in layers],
                    Q=record['metrics']['cost'],boundary=record['metrics']['boundary_violations'],grouping=record['metrics']['grouping_violations'])
                inferences+=1
            case.update(group='worst',label='Node26最差第 '+str(rank+1),evaluation_index=index);cases.append(case)
    save(folder/'e114_evidence.json',dict(model='e114',best_epoch=48,checkpoint_sha256=result['checkpoint_sha256'],cases=cases,
        train_layer_cache_reused=5,new_dev_inference_count=inferences,new_postprocessing_count=0,test_read=False))
    assert sha(checkpoint)==result['checkpoint_sha256']
    print('E114 cache reused; missing dev layer readouts:',inferences,flush=True)

if __name__=='__main__':main()
