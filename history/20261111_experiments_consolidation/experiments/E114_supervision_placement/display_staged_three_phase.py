"""Authorized fixed train5 + saved dev worst5, frozen staged best48."""
import json
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from .io import ROOT,HERE,save,sha
from .batches import decode
from .evaluation_staged.run import Predictor,repair_inputs,case_from_item
from .evaluation_staged.solver import optimize
from .evaluation_staged.repair.canonical_native import warmup
from datasets.random_reference import RandomReference
from datasets.reference_layouts import public_hard_reference
from scoring.checks import independent_hard_checks
from scoring.research_no_mib import evaluate_case
from visualization.layer_comparison import measures
from visualization.three_phase_viewer import publish,render_preview

def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    import numba
    numba.set_num_threads(1)
    run=HERE/'runs/staged_seed0';checkpoint=run/'checkpoints/best.pt';digest=sha(checkpoint)
    frozen=json.loads((HERE/'evaluation_staged/plan.json').read_text(encoding='utf-8'))['models']['staged']
    assert digest==frozen['checkpoint_sha256']
    out=run/'three_phase10_20261108';assert not out.exists();out.mkdir()
    evalrows=[json.loads(p.read_text(encoding='utf-8')) for p in sorted((run/'evaluation_no_mib_dev500/cases').glob('*.json'))]
    assert len(evalrows)==500
    worst=sorted(evalrows,key=lambda x:(-x['metrics']['cost'],x['index']))[:5]
    plan=json.loads((HERE/'plan.json').read_text(encoding='utf-8'))
    train=RandomReference('train');dev=RandomReference('dev_monitor500')
    selected=[('fixed',i,train[idx],None) for i,idx in enumerate(plan['preflight_train_indices'])]
    selected += [('worst',i,dev[row['index']],row) for i,row in enumerate(worst)]
    save(out/'state.json',dict(status='running',count=0,test_read=False,train_postprocessing_limit=5,dev_postprocessing_calls=0))
    cases=[]
    with torch.inference_mode(),threadpool_limits(limits=1):
        warmup();predictor=Predictor(checkpoint,selected[0][2]['solver_inputs'])
        for group,i,item,record in selected:
            inputs=item['solver_inputs'];n=int(inputs[0]);draft=predictor.predict(inputs)
            layers=[(decode(s,predictor.g)[0,:n]*predictor.g['scale'][0]).cpu().numpy().astype(float) for s in predictor.states]
            assert len(layers)==12;np.testing.assert_array_equal(layers[-1],draft)
            if record is None:
                final,info=optimize(draft,repair_inputs(inputs))
                metrics=evaluate_case(case_from_item(item),final,info['seconds'])['metrics']
                save(out/f'fixed{i+1}_postprocessing.json',dict(member=item['member'],metrics=metrics,positions=final.tolist(),trace=info['trace'],work=info['work']))
            else:
                assert item['member']==record['member']
                np.testing.assert_allclose(draft,record['draft_positions'],rtol=1e-6,atol=1e-5)
                final=np.asarray(record['positions']);metrics=record['metrics']
            hard=independent_hard_checks(final.tolist(),inputs[1],inputs[5],public_hard_reference(inputs[1],inputs[5],inputs[6]))
            assert hard['is_feasible'] and metrics['is_feasible']
            case=dict(group=group,label=('固定例 ' if group=='fixed' else '最差第 ')+str(i+1)+('' if group=='fixed' else ' 名'),member=item['member'],n=n,constraints=np.asarray(inputs[5]).astype(int).tolist(),reference=np.asarray(item['supervision']['reference_xywh']).tolist(),layers=[p.tolist() for p in layers],final=final.tolist(),coverage=[measures(p)['repeated_coverage_fraction'] for p in layers],Q=metrics['cost'],boundary=metrics['boundary_violations'],grouping=metrics['grouping_violations'],evaluation_index=record['index'] if record else None)
            cases.append(case);save(out/'state.json',dict(status='running',count=len(cases),test_read=False));print(case['label'],n,case['Q'],flush=True)
    assert sha(checkpoint)==digest
    identity=dict(experiment='E114_supervision_placement',arm='staged',best_epoch=48,checkpoint_sha256=digest,test_read=False,training=False,dev_postprocessing_calls=0,train_postprocessing_calls=5,selection='fixed original train5 plus largest final per-case Q in existing dev500; all500 retained in scores',inference='same GPU FP32 eval CUDA-graph public adapter as completed evaluation; saved worst5 final coordinates reused; regenerated final drafts match saved evaluation')
    data=dict(identity=identity,cases=cases)
    folder=ROOT/'visualization/E114_supervision/20261108_staged_three_phase10'
    inline=Path('C:/Users/yuche/.codex/visualizations/2026/11/08/01a1b914-5f0c-7cd2-b03f-eb4544f68ade/staged-three-phase.html')
    publish(data,folder,inline);save(out/'identity.json',identity);save(out/'state.json',dict(status='complete',count=10,test_read=False,display=folder.relative_to(ROOT).as_posix(),inline=str(inline)))
    render_preview(data,folder)
    print('Complete:',inline,flush=True)

if __name__=='__main__':main()
