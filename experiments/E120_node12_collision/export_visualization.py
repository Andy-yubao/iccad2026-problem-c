"""User-requested dev15 layer capture; frozen E120, saved final layouts, no scoring."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[key]='1'

import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from datasets.random_reference import RandomReference,REGISTRY
from visualization.style import normalize
from .batches import decode
from .evaluation.run import Predictor

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
RUN=HERE/'runs/node12_seed0'
EVALUATION=RUN/'evaluation_no_mib_dev500'
DESTINATION=HERE/'visualization/20261113_dev15'

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def save(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def main():
    started=time.perf_counter()
    if DESTINATION.exists():
        raise FileExistsError('Existing capture is preserved; do not overwrite or automatically retry')
    dataset=RandomReference('dev_monitor500')
    result=read(EVALUATION/'results.json')
    checkpoint=RUN/'checkpoints/best.pt'
    digest=sha(checkpoint)
    assert result['status']=='complete' and result['count']==len(dataset)==500
    assert result['best_epoch']==79 and result['checkpoint_sha256']==digest
    rows=[read(EVALUATION/'cases'/f'{i:04d}.json') for i in range(500)]
    for i,row in enumerate(rows):
        assert row['index']==i and row['checkpoint_sha256']==digest
        assert row['member']==dataset.records[dataset.indices[i]]['member']
        assert np.isfinite(row['metrics']['cost'])
    worst=sorted(range(500),key=lambda i:(-rows[i]['metrics']['cost'],i))[:5]
    ordinary=random.Random(20261113).sample([i for i in range(500) if i not in worst],10)
    indices=ordinary+worst
    assert len(set(indices))==15
    # Freeze selection and identity before reading selected payloads or model inference.
    DESTINATION.mkdir(parents=True)
    selection=dict(date='2026-11-13',checkpoint=checkpoint.relative_to(ROOT).as_posix(),
        checkpoint_sha256=digest,best_epoch=79,registry_sha256=sha(REGISTRY),
        source_results_sha256=sha(EVALUATION/'results.json'),seed=20261113,
        ordinary_rule='Uniform sample without replacement from dev500 excluding worst5; no quality filtering',
        worst_rule='Final saved per-case Q descending, stable dev index ascending tie-break',
        ordinary_indices=ordinary,worst_indices=worst,
        members=[dict(dev_index=i,member=rows[i]['member'],final_Q=rows[i]['metrics']['cost'],
                      evaluation_sha256=sha(EVALUATION/'cases'/f'{i:04d}.json')) for i in indices],
        layers='Actual model readouts after geometry blocks1..12; no interpolation or fixed-spec overwrite',
        inference='Original evaluation Predictor, CUDA graph FP32 batch1 padded120, eval/inference_mode',
        final_layouts='Reuse original E98.depth_budget.v1 evaluation coordinates without changes',
        training=False,optimizer_updates=0,postprocessing_calls=0,Q_calls=0,test_payloads_read=0)
    save(DESTINATION/'selection.json',selection)
    save(DESTINATION/'state.json',dict(status='capturing',completed=0,count=15))
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    cases=[]
    try:
        items=[dataset[i] for i in indices]
        with torch.inference_mode():
            predictor=Predictor(checkpoint,items[0]['solver_inputs'])
            before={k:v.detach().clone() for k,v in predictor.model.state_dict().items()}
            for j,(i,item) in enumerate(zip(indices,items)):
                row=rows[i]
                assert item['member']==row['member']
                predicted=predictor.predict(item['solver_inputs'])
                saved=np.asarray(row['draft_positions'])
                # Same original CUDA path must reproduce the saved terminal draft exactly.
                np.testing.assert_array_equal(predicted,saved)
                n=row['n']
                layers=[(decode(state,predictor.g)[0,:n]*predictor.g['scale'][0]).cpu().numpy().astype(float).tolist()
                        for state in predictor.states]
                assert len(layers)==12
                np.testing.assert_array_equal(layers[-1],saved)
                group='ordinary' if j<10 else 'worst'
                rank=j+1 if j<10 else j-9
                metrics=row['metrics']
                case=normalize(dict(label=f'{"普通例" if j<10 else "最坏第"} {rank}{"" if j<10 else " 名"} · dev {i:04d}',
                    group=group,dev_index=i,member=row['member'],n=n,
                    constraints=item['solver_inputs'][5].tolist(),reference=item['supervision']['reference_xywh'].tolist(),
                    layers=layers,final=row['positions'],
                    metrics=dict(Q=metrics['cost'],hard_legal=bool(metrics['is_feasible']),
                                 complete_ms=row['complete_seconds']*1000)))
                assert case['final']==row['positions']
                cases.append(case)
                save(DESTINATION/'state.json',dict(status='capturing',completed=j+1,count=15,dev_index=i))
                print(f'{j+1}/15 dev{i:04d}: 12 layers, terminal exact, Q={metrics["cost"]:.6f}',flush=True)
            assert all(torch.equal(v,before[k]) for k,v in predictor.model.state_dict().items())
        assert sha(checkpoint)==digest
        payload=dict(title='E120 · best79 · 10个普通例＋5个最坏例',
            note='普通10例：固定随机抽样并排除最坏5例。最坏5例：原dev500最终Q降序，同分按dev编号。滑块为真实12层，后处理复用原预算结果。',
            identity=dict(experiment='E120_global_initial_generator',best_epoch=79,checkpoint_sha256=digest,
                          source_evaluation=EVALUATION.relative_to(ROOT).as_posix(),selection='selection.json',
                          seed=20261113,terminal_drafts_bitwise_equal=True,training=False,
                          postprocessing_calls=0,Q_calls=0,test_payloads_read=0),cases=cases)
        save(DESTINATION/'data.json',payload)
        save(DESTINATION/'state.json',dict(status='complete',completed=15,count=15,actual_layers=12,
            terminal_drafts_bitwise_equal=True,model_state_unchanged=True,checkpoint_unchanged=True,
            final_layouts_reused_exactly=True,elapsed_seconds=time.perf_counter()-started,
            training=False,postprocessing_calls=0,Q_calls=0,test_payloads_read=0))
    except Exception as error:
        save(DESTINATION/'state.json',dict(status='failed',completed=len(cases),error=repr(error),automatic_retry=False))
        raise

if __name__=='__main__':main()
