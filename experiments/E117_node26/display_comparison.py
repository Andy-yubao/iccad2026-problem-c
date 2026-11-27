"""Own frozen model: cached train5, saved finals, at most five dev layer readouts."""
import json
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from .io import ROOT,HERE,save,sha
from .evaluation.run import Predictor,repair_inputs,case_from_item
from .evaluation.solver import optimize
from .evaluation.repair.canonical_native import warmup
from .batches import decode
from datasets.random_reference import RandomReference
from datasets.reference_layouts import public_hard_reference
from scoring.checks import independent_hard_checks
from scoring.research_no_mib import evaluate_case
from visualization.layer_comparison import measures

def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    import numba
    numba.set_num_threads(1)
    folder=ROOT/'visualization/E117_node26/20261109_vs_E114'
    selection=json.loads((folder/'selection.json').read_text())
    run=HERE/'runs/node26_seed0';checkpoint=run/'checkpoints/best.pt'
    frozen=json.loads((run/'evaluation_no_mib_dev500/results.json').read_text())
    assert sha(checkpoint)==frozen['checkpoint_sha256']
    manifest=json.loads((ROOT/'visualization/E117_node26/20261109_train5/manifest.json').read_text())
    assert manifest['checkpoint_sha256']==frozen['checkpoint_sha256']
    cache=np.load(ROOT/'visualization/E117_node26/20261109_train5/layers.npz')
    train=RandomReference('train');dev=RandomReference('dev_monitor500');cases=[]
    with torch.inference_mode(),threadpool_limits(limits=1):
        warmup()
        for i,index in enumerate(selection['train_indices']):
            item=train[index];inputs=item['solver_inputs'];n=int(inputs[0])
            assert item['member']==manifest['members'][i]
            layers=[p[i,:n].astype(float) for p in cache['layers']]
            final,info=optimize(layers[-1],repair_inputs(inputs))
            metrics=evaluate_case(case_from_item(item),final,info['seconds'])['metrics']
            hard=independent_hard_checks(final.tolist(),inputs[1],inputs[5],public_hard_reference(inputs[1],inputs[5],inputs[6]))
            assert hard['is_feasible'] and metrics['is_feasible']
            save(folder/f'node26_train{i+1}_postprocessing.json',dict(member=item['member'],positions=final.tolist(),metrics=metrics,trace=info['trace'],work=info['work'],hard_checks=hard))
            cases.append(pack(item,layers,final,metrics,'fixed',i,None))
            print('Node26 cached train',i+1,flush=True)
        predictor=Predictor(checkpoint,dev[selection['dev_indices'][0]]['solver_inputs'])
        for rank,index in enumerate(selection['dev_indices']):
            item=dev[index];record=json.loads((run/f'evaluation_no_mib_dev500/cases/{index:04d}.json').read_text())
            assert item['member']==record['member']
            draft=predictor.predict(item['solver_inputs']);n=int(item['solver_inputs'][0])
            np.testing.assert_allclose(draft,record['draft_positions'],rtol=1e-6,atol=1e-5)
            layers=[(decode(s,predictor.g)[0,:n]*predictor.g['scale'][0]).cpu().numpy().astype(float) for s in predictor.states]
            cases.append(pack(item,layers,np.asarray(record['positions']),record['metrics'],'worst',rank,index))
            print('Node26 saved dev final + layers',index,flush=True)
    save(folder/'node26_evidence.json',dict(model='node26',best_epoch=64,checkpoint_sha256=frozen['checkpoint_sha256'],cases=cases,
        train_layer_cache_reused=5,new_dev_inference_count=5,new_train_postprocessing_count=5,new_dev_postprocessing_count=0,test_read=False))
    assert sha(checkpoint)==frozen['checkpoint_sha256']

def pack(item,layers,final,metrics,group,rank,index):
    return dict(group=group,label=('固定例 ' if group=='fixed' else 'Node26最差第 ')+str(rank+1),member=item['member'],
        n=int(item['solver_inputs'][0]),constraints=np.asarray(item['solver_inputs'][5]).astype(int).tolist(),
        reference=np.asarray(item['supervision']['reference_xywh']).tolist(),layers=[r.tolist() for r in layers],
        final=final.tolist(),coverage=[measures(r)['repeated_coverage_fraction'] for r in layers],Q=metrics['cost'],
        boundary=metrics['boundary_violations'],grouping=metrics['grouping_violations'],evaluation_index=index)

if __name__=='__main__':main()
