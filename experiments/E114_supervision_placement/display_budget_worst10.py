"""Current increased-budget worst10; reuse saved finals/layers, infer five missing."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[key]='1'
import json
import copy
import numpy as np
import torch
from pathlib import Path
from threadpoolctl import threadpool_limits
from .io import ROOT,HERE,save,sha
from .batches import decode
from .evaluation_budget500ms.run import Predictor
from datasets.random_reference import RandomReference
from visualization.layer_comparison import measures
from visualization.three_phase_viewer import publish,render_preview


def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    evaluation=HERE/'evaluation_budget500ms/output'
    result=json.loads((evaluation/'results.json').read_text(encoding='utf-8'))
    assert result['status']=='complete' and result['count']==500
    records=[json.loads(p.read_text(encoding='utf-8')) for p in sorted((evaluation/'cases').glob('*.json'))]
    assert len(records)==500
    worst=sorted(records,key=lambda r:(-r['metrics']['cost'],r['index']))[:10]
    folder=ROOT/'visualization/E114_supervision/20261108_budget3x_worst10'
    assert not folder.exists();folder.mkdir(parents=True)
    cache_path=ROOT/'visualization/E114_supervision/20261108_staged_three_phase10/data.json'
    cache_data=json.loads(cache_path.read_text(encoding='utf-8'))
    checkpoint=HERE/'runs/staged_seed0/checkpoints/best.pt'
    digest=sha(checkpoint)
    assert digest==result['checkpoint_sha256']==cache_data['identity']['checkpoint_sha256']
    cache={tuple(c['member']):c for c in cache_data['cases']}
    dev=RandomReference('dev_monitor500')
    cases=[];reused=[];inferred=[];predictor=None
    save(folder/'state.json',dict(status='running',completed=0,total=10,
         new_postprocessing_calls=0,test_read=False))
    with torch.inference_mode(),threadpool_limits(limits=1):
        for rank,record in enumerate(worst,1):
            member=tuple(record['member']);n=record['n']
            assert record['independent_hard_checks']['is_feasible'] and record['metrics']['is_feasible']
            if member in cache:
                old=cache[member]
                layers=copy.deepcopy(old['layers'])
                reference=copy.deepcopy(old['reference'])
                constraints=copy.deepcopy(old['constraints'])
                coverage=copy.deepcopy(old['coverage'])
                reused.append(record['index'])
            else:
                item=dev[record['index']]
                assert tuple(item['member'])==member
                inputs=item['solver_inputs']
                if predictor is None:predictor=Predictor(checkpoint,inputs)
                draft=predictor.predict(inputs)
                positions=[(decode(s,predictor.g)[0,:n]*predictor.g['scale'][0]).cpu().numpy().astype(float) for s in predictor.states]
                assert len(positions)==12
                np.testing.assert_array_equal(positions[-1],draft)
                layers=[p.tolist() for p in positions]
                reference=np.asarray(item['supervision']['reference_xywh']).tolist()
                constraints=np.asarray(inputs[5]).astype(int).tolist()
                coverage=[measures(p)['repeated_coverage_fraction'] for p in positions]
                inferred.append(record['index'])
            np.testing.assert_allclose(layers[-1],record['draft_positions'],rtol=1e-6,atol=1e-6)
            case=dict(group='worst',rank=rank,label=f'最差第 {rank} 名',
                member=record['member'],evaluation_index=record['index'],n=n,
                reference=reference,layers=layers,constraints=constraints,coverage=coverage,
                final=copy.deepcopy(record['positions']),Q=record['metrics']['cost'],
                old_Q=record['baseline_cost'],complete_seconds=record['complete_seconds'],
                boundary=record['metrics']['boundary_violations'],
                grouping=record['metrics']['grouping_violations'])
            cases.append(case)
            save(folder/'state.json',dict(status='running',completed=len(cases),total=10,
                reused_layers=len(reused),new_inference_cases=len(inferred),new_postprocessing_calls=0,test_read=False))
            print(f'rank={rank} dev_index={record["index"]} N={n} Q={case["Q"]:.6f}',flush=True)
    assert len(reused)==5 and len(inferred)==5
    assert sha(checkpoint)==digest
    identity=dict(experiment='E114_supervision_placement',arm='staged',best_epoch=48,
        checkpoint_sha256=digest,postprocessor='E114.staged.depth_budget_x3.target500ms.v1',
        evaluation=evaluation.relative_to(ROOT).as_posix(),evaluation_results_sha256=sha(evaluation/'results.json'),
        selection='Top10 descending current final per-case Q, stable evaluation-index tie break; all500 retained',
        dev_indices=[r['index'] for r in worst],reused_layer_cases=reused,new_inference_cases=inferred,
        reused_layer_cache_sha256=sha(cache_path),new_postprocessing_calls=0,new_Q_calls=0,
        test_read=False,training=False,layout='reference | actual draft slider1..12 | fixed current3x-budget final',
        visual_encoding='Confirmed E6: numeric IDs, group-only Set3 fills, G/F/P, L/R/T/B status, bottom legend; no MIB')
    data=dict(identity=identity,cases=cases)
    save(folder/'collected.json',data)
    inline=Path('C:/Users/yuche/.codex/visualizations/2026/11/08/01a1baf6-f05b-7b23-b093-fdfe9beb1d00/budget-worst-ten.html')
    inline.parent.mkdir(parents=True,exist_ok=True)
    publish(data,folder,inline,update=True,template_path=ROOT/'visualization/three_phase_worst_viewer_template.html')
    render_preview(data,folder,case_index=0)
    save(folder/'identity.json',identity)
    save(folder/'state.json',dict(status='complete',completed=10,total=10,inline=str(inline),
        reused_layer_cases=5,new_inference_cases=5,new_postprocessing_calls=0,new_Q_calls=0,test_read=False))
    print('Complete:',inline,flush=True)


if __name__=='__main__':main()
